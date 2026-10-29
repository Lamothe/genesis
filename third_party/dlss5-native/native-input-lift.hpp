// Native Linux CPU input-lift slice: RGB frame -> 16 input features -> 32 lifted
// channels, with explicit, deterministic arithmetic. No Python/PyTorch runtime, no
// GPU, no proprietary payload. See vr/NATIVE-INPUT-LIFT.md for the full specification,
// attribution and the qualification that this establishes synthetic arithmetic
// correctness only, not vendor correspondence.
//
// Structure (feature order, conditioning derivation, noise generator, stored 16->32
// lift permutation, constants) adapted from mauri870/dlssnr-pytorch commit
// 3beef7935a8ef976b47ab78b906e75600c52d2de, MIT Licence, Copyright (c) 2026
// Mauri de Souza Meneguzzo. Test frames and test weights are original synthetic data.
#ifndef VR_NATIVE_INPUT_LIFT_HPP
#define VR_NATIVE_INPUT_LIFT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dlss5_nr {

// ---- documented constants ------------------------------------------------------------------
inline constexpr int kFeatureCount = 16;        // input features per pixel
inline constexpr int kLiftedChannels = 32;      // lifted channels per pixel
inline constexpr int kStoredLiftValues = 512;   // 32 * 16 stored lift entries
inline constexpr int kNoiseFeatures = 3;
inline constexpr int kConditioningFeatures = 5;
inline constexpr float kColourScale = 0.125f;   // centred colour scale into the lift range
inline constexpr float kConstantFeature = 1.0f;
inline constexpr float kNoiseGain = 1.0f;       // gain of the three noise features
inline constexpr float kHalfMinusZero = 0.5f;   // colour centre

// Bounded working extent. These limits bound validation arithmetic and the output
// footprint the caller must provide (32 * rows * cols floats); they are checked with
// widened arithmetic before any multiplication in int.
inline constexpr int kMaxRows = 16384;
inline constexpr int kMaxCols = 16384;
inline constexpr long long kMaxPixels = 1LL << 24;

enum class Status : int {
    kOk = 0,
    kNullPointer = 1,
    kInvalidDimensions = 2,
    kInvalidStride = 3,
    kInvalidValues = 4,
    kInvalidControls = 5,
    kInvalidBuffer = 6,
};

// The controls the lift slice consumes. Mirrors the reference Controls subset that the
// pre-block reads; intensity/depth/motion/history/reset belong to later stages.
struct Controls {
    int style = 0;                  // 0..2
    float local_tone = 1.0f;        // 0..2
    float local_structure = 1.0f;   // 0..2
    float skin_structure = -1.0f;   // -1..2; any negative value follows local_structure
    bool auto_mask = true;
    std::uint32_t seed = 0;         // noise seed; pure function of (x, y, seed)
};

// Interleaved RGB frame, row-major, natural orientation (x = column, y = row), values
// in [0, 1] (the reference path feeds 8-bit frames as k/255). row_stride is in floats
// between consecutive rows and must be >= 3 * cols.
struct FrameView {
    const float* rgb = nullptr;
    int rows = 0;
    int cols = 0;
    std::ptrdiff_t row_stride = 0;
};

// ---- float16 rounding ------------------------------------------------------------------------
// Round a float32 to the nearest IEEE 754 binary16 (ties to even) and return the float32
// value of that binary16. Software conversion: no F16C/AVX512FP16 requirement. NaN input
// returns a canonical quiet NaN; infinities round to infinities.

inline std::uint16_t f16_bits(float value) {
    std::uint32_t u = 0;
    std::memcpy(&u, &value, sizeof u);
    const std::uint32_t sign = (u >> 16) & 0x8000u;
    const std::uint32_t magnitude = u & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return static_cast<std::uint16_t>(sign | 0x7e00u);  // NaN -> canonical quiet NaN
    }
    if (magnitude > 0x477fefffu) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);  // >= 65520 rounds to infinity
    }
    if (magnitude < 0x38800000u) {
        // Below 2^-14 the binary16 grid is the 2^-24 subnormal grid (2^-25 rounds to
        // zero: the midpoint tie goes to the even value zero).
        if (magnitude < 0x33000000u) {
            return static_cast<std::uint16_t>(sign);
        }
        std::uint32_t significand = (magnitude & 0x007fffffu) | 0x00800000u;  // 24-bit
        const int shift = 126 - static_cast<int>(magnitude >> 23);
        const std::uint32_t half_bit = 1u << (shift - 1);
        const std::uint32_t rest = significand & ((half_bit << 1) - 1u);
        std::uint32_t rounded = significand >> shift;
        if (rest > half_bit || (rest == half_bit && (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);  // 1024 encodes the min normal
    }
    // Normal range: round the 24-bit significand to 11 bits (implicit bit + 10 stored).
    const std::uint32_t significand = (magnitude & 0x007fffffu) | 0x00800000u;
    const std::uint32_t half_bit = 1u << 12;
    const std::uint32_t rest = significand & 0x1fffu;
    std::uint32_t rounded = significand >> 13;
    if (rest > half_bit || (rest == half_bit && (rounded & 1u) != 0u)) {
        ++rounded;
    }
    // rounded is [1024, 2048]; drop the implicit bit, carrying the rounding overflow
    // into the exponent field via bit 10 of the sum below.
    std::uint32_t exponent = static_cast<std::uint32_t>(static_cast<int>(magnitude >> 23) - 127 + 15);
    std::uint32_t mantissa = rounded - 1024u;
    if (mantissa == 1024u) {
        mantissa = 0u;
        ++exponent;
    }
    return static_cast<std::uint16_t>(sign | (exponent << 10) | mantissa);
}

inline float f16_from_bits(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) ? 0x80000000u : 0u;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t u = 0;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            u = sign;
        } else {
            // Subnormal: renormalise into the binary32 normal range.
            std::uint32_t m = mantissa;
            std::uint32_t e = 127u - 15u + 1u;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            u = sign | (e << 23) | (m << 13);
        }
    } else if (exponent == 31u) {
        u = sign | 0x7f800000u | (mantissa << 13);
    } else {
        u = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &u, sizeof value);
    return value;
}

inline float f16(float value) { return f16_from_bits(f16_bits(value)); }

inline bool is_f16(float value) {
    std::uint32_t u = 0;
    std::memcpy(&u, &value, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return false;  // NaN
    }
    return f16(value) == value;
}

// ---- deterministic noise --------------------------------------------------------------------
// Integer hashing is exact uint32 arithmetic and the uniform mapping v * 2^-24 is exact
// in float32 (v <= 2^24). The gaussian stage uses platform logf/sqrtf/cosf/sinf with each
// operation rounded to float32; those transcendental results are deterministic per
// platform/libm but are NOT claimed bit-exact across platforms.

inline std::uint32_t noise_pcg(std::uint32_t value) {
    value = (value >> ((value >> 28) + 4u)) ^ value;
    return value * 0x108ef2d9u;
}

inline float noise_uniform(std::uint32_t stream) {
    const std::uint32_t t = noise_pcg(stream);
    const std::uint32_t v = ((t >> 30) ^ (t >> 8)) + 1u;
    return static_cast<float>(v) * 0x1p-24f;  // (0, 1], exact
}

inline Status noise_uniforms(std::uint32_t x, std::uint32_t y, std::uint32_t seed, float out[4]) {
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    const std::uint32_t base = (x * 0x8da6b343u) ^ (seed * 0x9e3779b9u)
                             ^ (y * 0xd8163841u) ^ 0x243f6a88u;
    const std::uint32_t t = noise_pcg(base);
    const std::uint32_t h = (t >> 22) ^ t;
    out[0] = noise_uniform(h * 0x2c9277b5u + 0xac564b05u);
    out[1] = noise_uniform(h * 0xfa6dc5f9u + 0x4712a88eu);
    out[2] = noise_uniform(h * 0xcaa5b80du + 0x21dd796bu);
    out[3] = noise_uniform(h * 0x83232c31u + 0x3463e0acu);
    return Status::kOk;
}

// The three f16 noise features of one pixel: two Box-Muller halves of the first uniform
// pair and the cosine half of the second pair, as a pure function of (x, y, seed).
inline Status noise_features(std::uint32_t x, std::uint32_t y, std::uint32_t seed, float out[3]) {
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    float uniforms[4] = {};
    const Status status = noise_uniforms(x, y, seed, uniforms);
    if (status != Status::kOk) {
        return status;
    }
    using std::cos;
    using std::log;
    using std::sin;
    using std::sqrt;
    const float two_pi = 6.28318530718f;
    const float radius_a = sqrt(-2.0f * log(uniforms[0]));
    const float radius_c = sqrt(-2.0f * log(uniforms[2]));
    const float angle_a = uniforms[1] * two_pi;
    const float angle_c = uniforms[3] * two_pi;
    out[0] = f16(kNoiseGain * (radius_a * cos(angle_a)));
    out[1] = f16(kNoiseGain * (radius_a * sin(angle_a)));
    out[2] = f16(kNoiseGain * (radius_c * cos(angle_c)));
    return Status::kOk;
}

// ---- input features --------------------------------------------------------------------------
// Centred colour: a = f16(rgb); b = f16(a - 0.5); c = f16(b * 0.125), each step evaluated
// in float32 and rounded to the f16 grid.
inline Status centred_colour(const float rgb[3], float out[3]) {
    if (rgb == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    for (int channel = 0; channel < 3; ++channel) {
        const float a = f16(rgb[channel]);
        const float b = f16(a - kHalfMinusZero);
        out[channel] = f16(b * kColourScale);
    }
    return Status::kOk;
}

inline Status validate(const Controls& controls) {
    if (controls.style < 0 || controls.style > 2) {
        return Status::kInvalidControls;
    }
    const float values[3] = {controls.local_tone, controls.local_structure, controls.skin_structure};
    for (float value : values) {
        if (!(value == value)) {
            return Status::kInvalidControls;  // NaN
        }
    }
    if (values[0] < 0.0f || values[0] > 2.0f || values[1] < 0.0f || values[1] > 2.0f
        || values[2] < -1.0f || values[2] > 2.0f) {
        return Status::kInvalidControls;
    }
    return Status::kOk;
}

// The five f16 conditioning features (style, tone, structure, skin, other). With the
// automatic mask the structure feature is 1 and the mask carries the strengths (skin
// takes skin_structure, or local_structure when skin_structure is negative; everything
// else takes local_structure). Without the mask both mask features are -1 and the
// structure feature carries local_structure.
inline Status conditioning(const Controls& controls, float out[5]) {
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    const Status status = validate(controls);
    if (status != Status::kOk) {
        return status;
    }
    const float style = static_cast<float>(controls.style) * 0x1p-7f;  // style / 128, exact
    if (!controls.auto_mask) {
        out[0] = f16(style);
        out[1] = f16(controls.local_tone);
        out[2] = f16(controls.local_structure);
        out[3] = -1.0f;
        out[4] = -1.0f;
        return Status::kOk;
    }
    const float skin = controls.skin_structure < 0.0f ? controls.local_structure
                                                      : controls.skin_structure;
    out[0] = f16(style);
    out[1] = f16(controls.local_tone);
    out[2] = 1.0f;
    out[3] = f16(skin);
    out[4] = f16(controls.local_structure);
    return Status::kOk;
}

// ---- stored lift permutation ------------------------------------------------------------------
// Stored record order: rows in groups of 16 channels; inside a group the channel's low
// three bits select a 32-element stride and the 16 inputs are split into halves and
// pairs. This is the stored 16 -> 32 input-lift permutation of the reference pipeline.
inline std::size_t lift_offset(int channel, int k) {
    return static_cast<std::size_t>((channel / 16) * 256 + 32 * (channel % 8)
        + 8 * ((k % 8) / 2) + 4 * ((channel / 8) % 2) + (k % 2) + 2 * (k / 8));
}

inline Status decode_lift(const float* stored, std::size_t stored_count, float* lift) {
    if (stored == nullptr || lift == nullptr) {
        return Status::kNullPointer;
    }
    if (stored_count != static_cast<std::size_t>(kStoredLiftValues)) {
        return Status::kInvalidBuffer;
    }
    for (int channel = 0; channel < kLiftedChannels; ++channel) {
        for (int k = 0; k < kFeatureCount; ++k) {
            lift[channel * kFeatureCount + k] = stored[lift_offset(channel, k)];
        }
    }
    return Status::kOk;
}

inline Status validate_weights(const float* stored, std::size_t stored_count) {
    if (stored == nullptr) {
        return Status::kNullPointer;
    }
    if (stored_count != static_cast<std::size_t>(kStoredLiftValues)) {
        return Status::kInvalidBuffer;
    }
    for (std::size_t i = 0; i < stored_count; ++i) {
        // f16-representable and finite: infinities are representable but are not valid
        // weights (inf * 0.0 would poison the accumulation with NaN).
        if (!is_f16(stored[i]) || !std::isfinite(stored[i])) {
            return Status::kInvalidValues;
        }
    }
    return Status::kOk;
}

// ---- frame validation --------------------------------------------------------------------------
inline Status validate(const FrameView& frame) {
    if (frame.rgb == nullptr) {
        return Status::kNullPointer;
    }
    if (frame.rows <= 0 || frame.cols <= 0 || frame.rows > kMaxRows || frame.cols > kMaxCols) {
        return Status::kInvalidDimensions;
    }
    const long long pixels = static_cast<long long>(frame.rows) * frame.cols;
    if (pixels > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    if (frame.row_stride < static_cast<std::ptrdiff_t>(3 * frame.cols)
        || frame.row_stride > static_cast<std::ptrdiff_t>(3 * kMaxCols)) {
        return Status::kInvalidStride;
    }
    for (int y = 0; y < frame.rows; ++y) {
        const float* row = frame.rgb
            + static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.row_stride);
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(3 * frame.cols); ++i) {
            const float value = row[i];
            if (!(value == value) || value < 0.0f || value > 1.0f) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

// ---- the lift slice ------------------------------------------------------------------------------
// lifted[c * rows * cols + y * cols + x] = f16( sum_{k=0..15} lift[c][k] * features[k] ),
// the sum accumulated left to right in float32 with one final rounding to the f16 grid.
// That order is the specified semantics: the regression runner compiles with
// -ffp-contract=off to pin the build contract. Every product is exact in float32 (both
// operands are f16-grid values, <= 11-bit mantissas), so FMA contraction could not
// change any result here; reassociation of the sum would, and the runner's flags
// exclude it.
// The 16 input features of pixel (x, y) are, in order:
//   0..2   three f16 noise fields at (x, y, seed),
//   3      the constant feature 1.0,
//   4..6   the centred colour (R, G, B),
//   7..9   the centred history colour: the previous frame's colour in temporal
//          mode (the carrier view of native-history.hpp), or the centred colour
//          again as the reference's no-history stand-in (the single-frame stub),
//   10..14 five conditioning features (style, tone, structure, skin, other),
//   15     a zero pad.
// Allocation-free: the caller provides lifted (>= 32 * rows * cols floats). Validation
// completes before any output write, so on any error the output buffer is untouched.
inline Status input_lift(const FrameView& frame, const FrameView& history,
                         const Controls& controls,
                         const float* lift, std::size_t lift_count,
                         float* lifted, std::size_t lifted_count) {
    Status status = validate(frame);
    if (status != Status::kOk) {
        return status;
    }
    if (history.rows != frame.rows || history.cols != frame.cols) {
        return Status::kInvalidDimensions;
    }
    status = validate(history);
    if (status != Status::kOk) {
        return status;
    }
    status = validate(controls);
    if (status != Status::kOk) {
        return status;
    }
    status = validate_weights(lift, lift_count);
    if (status != Status::kOk) {
        return status;
    }
    if (lifted == nullptr) {
        return Status::kNullPointer;
    }
    const std::size_t pixels = static_cast<std::size_t>(frame.rows) * frame.cols;
    if (lifted_count < static_cast<std::size_t>(kLiftedChannels) * pixels) {
        return Status::kInvalidBuffer;
    }
    float condition[kConditioningFeatures] = {};
    status = conditioning(controls, condition);
    if (status != Status::kOk) {
        return status;
    }
    for (int y = 0; y < frame.rows; ++y) {
        const float* row = frame.rgb
            + static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.row_stride);
        for (int x = 0; x < frame.cols; ++x) {
            float features[kFeatureCount] = {};
            status = noise_features(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
                                   controls.seed, &features[0]);
            if (status != Status::kOk) {
                return status;
            }
            features[3] = kConstantFeature;
            const float* rgb = row + 3 * static_cast<std::ptrdiff_t>(x);
            float colour[3] = {};
            status = centred_colour(rgb, colour);
            if (status != Status::kOk) {
                return status;
            }
            float history_colour[3] = {};
            const float* history_rgb = history.rgb
                + static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(history.row_stride)
                + 3 * static_cast<std::ptrdiff_t>(x);
            status = centred_colour(history_rgb, history_colour);
            if (status != Status::kOk) {
                return status;
            }
            for (int channel = 0; channel < 3; ++channel) {
                features[4 + channel] = colour[channel];
                features[7 + channel] = history_colour[channel];
            }
            for (int i = 0; i < kConditioningFeatures; ++i) {
                features[10 + i] = condition[i];
            }
            features[15] = 0.0f;
            const std::size_t pixel = static_cast<std::size_t>(y) * frame.cols + x;
            for (int c = 0; c < kLiftedChannels; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < kFeatureCount; ++k) {
                    sum += lift[c * kFeatureCount + k] * features[k];
                }
                lifted[static_cast<std::size_t>(c) * pixels + pixel] = f16(sum);
            }
        }
    }
    return Status::kOk;
}

// The single-frame stub: the current frame stands in for the history colour
// (the reference's no-history path). Same behaviour as before this overload.
inline Status input_lift(const FrameView& frame, const Controls& controls,
                         const float* lift, std::size_t lift_count,
                         float* lifted, std::size_t lifted_count) {
    return input_lift(frame, frame, controls, lift, lift_count, lifted,
                      lifted_count);
}

}  // namespace dlss5_nr

#endif  // VR_NATIVE_INPUT_LIFT_HPP
