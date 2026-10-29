// Native post-block blend stage: the 2x replication with replicate padding and the
// per-channel f16 mix of the upsampled half-resolution input with the full-resolution
// skip. CPU-only, typed, allocation-free; no GPU, headset, proprietary payload,
// Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/pre_post.py "_mix" and "PostBlock.blend"). The full MIT licence
// notice is retained in vr/NATIVE-BLEND.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_BLEND_HPP
#define DLSS5_NR_NATIVE_BLEND_HPP

#include "native-input-lift.hpp"

namespace dlss5_nr {

// ---- double -> f16 rounding --------------------------------------------------------------------
// Round a double to the nearest IEEE 754 binary16 (ties to even) and return the float32
// value of that binary16. This is the rounding the reference performs when it sends its
// float64 fused sum through .to(torch.float16); overflow (>= 65520 in magnitude, the
// tie rounds up) becomes an infinity, matching f16_bits().
//
// The rounding happens entirely in double arithmetic: a double -> float32 -> f16 chain
// is NOT equivalent to a direct double -> f16 rounding (the float32 step can round onto
// a binary16 midpoint and then break that tie the wrong way, e.g. 1037.5 - 1.5 * 2^-23
// rounds to 1038 through float32 but to 1037 directly). The scaling by powers of two
// below is exact, so the tie comparisons are exact. Zones: |value| > 65504 saturates
// toward the overflow boundary (the (65504, 65520) band rounds to 65504; >= 65520 to
// infinity) with explicit comparisons; below 2^-14 the grid is the 2^-24 subnormal
// grid; in [2^-14, 65504] the 53-bit significand is rounded to the 11-bit normal grid.
inline float f16_from_double(double value) {
    if (std::isnan(value)) {
        float canonical = 0.0f;
        const std::uint32_t bits = 0x7fc00000u;
        std::memcpy(&canonical, &bits, sizeof canonical);
        return canonical;
    }
    const double magnitude = std::fabs(value);
    const bool negative = std::signbit(value);
    if (magnitude >= 65520.0) {
        return negative ? -f16_from_bits(0x7c00u) : f16_from_bits(0x7c00u);
    }
    if (magnitude > 65504.0) {
        return negative ? -f16_from_bits(0x7bffu) : f16_from_bits(0x7bffu);
    }
    if (magnitude < 0x1p-14) {
        // Subnormal grid k * 2^-24 (k in 0..1023; a rounding carry to 1024 encodes the
        // minimum normal 2^-14, which the code space does naturally).
        const double scaled = magnitude * 0x1p24;
        if (scaled < 0.5) {
            return negative ? -0.0f : 0.0f;
        }
        std::uint32_t k = static_cast<std::uint32_t>(scaled);
        const double fraction = scaled - static_cast<double>(k);
        if (fraction > 0.5 || (fraction == 0.5 && (k & 1u) != 0u)) {
            ++k;
        }
        return f16_from_bits(static_cast<std::uint16_t>((negative ? 0x8000u : 0u) | k));
    }
    // Normal grid: 11-bit significand (implicit bit + 10 stored) with ties to even.
    int exponent = std::ilogb(magnitude);  // -14..15 here
    const double scaled = std::ldexp(magnitude, 10 - exponent);  // in [1024, 2048)
    std::uint32_t k = static_cast<std::uint32_t>(scaled);
    const double fraction = scaled - static_cast<double>(k);
    if (fraction > 0.5 || (fraction == 0.5 && (k & 1u) != 0u)) {
        ++k;
    }
    if (k == 2048u) {
        // Rounding carry at the top of a binade (e.g. 32761 -> 32768): renormalise onto
        // the next binade's grid. Routine at every binade top; the carry in the topmost
        // binade (65520) is already mapped to infinity above.
        k = 1024u;
        ++exponent;
    }
    return f16_from_bits(static_cast<std::uint16_t>((negative ? 0x8000u : 0u)
        | (static_cast<std::uint32_t>(exponent + 15) << 10) | (k - 1024u)));
}

// ---- the elementwise mix ------------------------------------------------------------------------
// The reference's _mix, per element:
//   main = f16(up * main_gain)                 (float32 product, one f16 rounding)
//   mix  = f16(skip * skip_gain + main)        (the skip product computed exactly in
//                                               float64 and fused into one float64 add,
//                                               then rounded to f16)
// main_gain and skip_gain are the per-channel f16 gains of the post-block blend. The
// float64 path matters: a float32 form of the fused sum is NOT rounding-equivalent (it
// can round onto an f16 midpoint and break that tie wrongly), so the skip product and
// add are done in float64 exactly as the reference does. NaN maps to the canonical f16
// NaN (the reference's conversion preserves NaN signs and payloads; inputs are
// validated finite upstream).
inline float mix_element(float up, float skip, float main_gain, float skip_gain) {
    const float main = f16(up * main_gain);
    const double fused = static_cast<double>(skip) * static_cast<double>(skip_gain)
        + static_cast<double>(main);
    return f16_from_double(fused);
}

// ---- the post-block blend -------------------------------------------------------------------------
// A channels-first feature batch: one plane of rows x cols floats per channel, planes in
// channel order, natural orientation (x = column, y = row). row_stride is in floats
// between consecutive rows and must satisfy cols <= row_stride <= kMaxCols;
// plane_stride is in floats between consecutive planes and must satisfy
// rows * row_stride <= plane_stride <= kMaxRows * kMaxCols. Any stride outside those
// bounds is rejected with Status::kInvalidStride before any output write.
struct FeatureBatchView {
    const float* data = nullptr;
    int channels = 0;
    int rows = 0;
    int cols = 0;
    std::ptrdiff_t plane_stride = 0;
    std::ptrdiff_t row_stride = 0;
};

// The post-block's input blend of the half-resolution map `up` with the full-resolution
// skip. `up` is replicated 2x (each value repeated over a 2x2 block); where the skip is
// larger than 2x the replicated map, the last row/column is repeated (replicate
// padding); where the skip is smaller, the replicated map is cropped. In full:
//   up2x(y, x) = up(min(y / 2, up_rows - 1), min(x / 2, up_cols - 1))
//   dst(c, y, x) = mix_element(up2x(c, y, x), skip(c, y, x), main_gain[c], skip_gain[c])
// for y in [0, skip_rows), x in [0, skip_cols), c in [0, channels). `up` and `skip` must
// have the same channel count (1..kLiftedChannels; the pipeline uses 32); their spatial
// extents need not match. main_gain/skip_gain hold exactly one f16 gain per channel.
//
// dst receives channels * skip_rows * skip_cols floats, planes in channel order,
// row-major within a plane. Allocation-free and validated (including finite-value scans
// of both batches and the gains) before any output write, so on any error the output
// buffer is untouched. The output buffer must not overlap either input batch;
// overlapping writes can silently corrupt the mix (they are never rejected: all access
// stays in bounds).
inline Status blend(const FeatureBatchView& up, const FeatureBatchView& skip,
                    const float* main_gain, const float* skip_gain, std::size_t gain_count,
                    float* dst, std::size_t dst_count) {
    if (up.data == nullptr || skip.data == nullptr) {
        return Status::kNullPointer;
    }
    if (main_gain == nullptr || skip_gain == nullptr) {
        return Status::kNullPointer;
    }
    // The two batches share the channel count; up keeps its own spatial extent.
    if (up.channels != skip.channels || up.channels <= 0 || up.channels > kLiftedChannels) {
        return Status::kInvalidDimensions;
    }
    const long long up_pixels = static_cast<long long>(up.rows) * up.cols;
    const long long skip_pixels = static_cast<long long>(skip.rows) * skip.cols;
    if (up.rows <= 0 || up.cols <= 0 || up.rows > kMaxRows || up.cols > kMaxCols
        || up_pixels > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    if (skip.rows <= 0 || skip.cols <= 0 || skip.rows > kMaxRows || skip.cols > kMaxCols
        || skip_pixels > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    const long long plane_span = static_cast<long long>(kMaxRows) * kMaxCols;
    if (up.row_stride < up.cols || up.row_stride > kMaxCols
        || up.plane_stride < static_cast<long long>(up.rows) * up.row_stride
        || up.plane_stride > plane_span) {
        return Status::kInvalidStride;
    }
    if (skip.row_stride < skip.cols || skip.row_stride > kMaxCols
        || skip.plane_stride < static_cast<long long>(skip.rows) * skip.row_stride
        || skip.plane_stride > plane_span) {
        return Status::kInvalidStride;
    }
    if (gain_count != static_cast<std::size_t>(up.channels)) {
        return Status::kInvalidBuffer;
    }
    for (int c = 0; c < up.channels; ++c) {
        // f16-representable and finite gains, matching the lift weights' contract.
        if (!is_f16(main_gain[c]) || !std::isfinite(main_gain[c])
            || !is_f16(skip_gain[c]) || !std::isfinite(skip_gain[c])) {
            return Status::kInvalidValues;
        }
    }
    if (dst == nullptr) {
        return Status::kNullPointer;
    }
    const std::size_t out_plane = static_cast<std::size_t>(skip.rows) * skip.cols;
    if (dst_count < static_cast<std::size_t>(up.channels) * out_plane) {
        return Status::kInvalidBuffer;
    }
    // Finite-value scans complete all validation before the first output write.
    for (int c = 0; c < up.channels; ++c) {
        for (int y = 0; y < up.rows; ++y) {
            const float* row = up.data
                + static_cast<std::size_t>(c) * static_cast<std::size_t>(up.plane_stride)
                + static_cast<std::size_t>(y) * static_cast<std::size_t>(up.row_stride);
            for (int x = 0; x < up.cols; ++x) {
                if (!std::isfinite(row[x])) {
                    return Status::kInvalidValues;
                }
            }
        }
        for (int y = 0; y < skip.rows; ++y) {
            const float* row = skip.data
                + static_cast<std::size_t>(c) * static_cast<std::size_t>(skip.plane_stride)
                + static_cast<std::size_t>(y) * static_cast<std::size_t>(skip.row_stride);
            for (int x = 0; x < skip.cols; ++x) {
                if (!std::isfinite(row[x])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    for (int y = 0; y < skip.rows; ++y) {
        const int up_y = (y / 2 < up.rows) ? y / 2 : up.rows - 1;
        for (int x = 0; x < skip.cols; ++x) {
            const int up_x = (x / 2 < up.cols) ? x / 2 : up.cols - 1;
            for (int c = 0; c < up.channels; ++c) {
                const float* up_plane = up.data
                    + static_cast<std::size_t>(c) * static_cast<std::size_t>(up.plane_stride);
                const float* skip_plane = skip.data
                    + static_cast<std::size_t>(c) * static_cast<std::size_t>(skip.plane_stride);
                const float up_value = up_plane[static_cast<std::size_t>(up_y)
                    * static_cast<std::size_t>(up.row_stride) + up_x];
                const float skip_value = skip_plane[static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(skip.row_stride) + x];
                dst[static_cast<std::size_t>(c) * out_plane
                    + static_cast<std::size_t>(y) * static_cast<std::size_t>(skip.cols) + x]
                    = mix_element(up_value, skip_value, main_gain[c], skip_gain[c]);
            }
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_BLEND_HPP
