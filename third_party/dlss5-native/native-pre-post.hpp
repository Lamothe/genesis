// Native pre/post-block arithmetic stages: the e4m3 rounding boundary and the 2x2 f16
// pooling. CPU-only, typed, allocation-free; no GPU, headset, proprietary payload,
// Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/quant.py "q8", dlssnr/layers/swin.py "_pool"/"pool2x2"). The full MIT
// licence notice is retained in vr/NATIVE-INPUT-LIFT.md and vr/NATIVE-PRE-POST.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_PRE_POST_HPP
#define DLSS5_NR_NATIVE_PRE_POST_HPP

#include "native-input-lift.hpp"

namespace dlss5_nr {

// ---- e4m3 rounding boundary ---------------------------------------------------------------------
// The network computes in FP8 e4m3 (4 exponent bits, 3 mantissa bits; e4m3fn: no
// infinities, the codes 0x7f/0xff are NaN). Every operation boundary rounds to the
// nearest e4m3 value (ties to even), saturating at +-448, and the next operation reads
// it back exactly. Subnormal grid: fixed 2^-9 step below 2^-6. Every e4m3 value is
// exactly representable in float32.
inline constexpr float kE4m3Max = 448.0f;
inline constexpr float kE4m3MinNormal = 0x1p-6f;      // 2^-6
inline constexpr float kE4m3SubnormalStep = 0x1p-9f;  // 2^-9

// Round a float32 to the nearest e4m3 value (ties to even) and return the float32
// value. The reference saturates (clamps to +-448) before rounding, so infinities
// saturate rather than becoming NaN. NaN maps to the canonical e4m3 NaN (sign dropped;
// the reference does not preserve NaN).
inline std::uint8_t e4m3_bits(float value) {
    if (std::isnan(value)) {
        return 0x7fu;
    }
    const bool negative = std::signbit(value);
    const float magnitude = std::fabs(value);
    if (!(magnitude < kE4m3Max)) {  // >= 448 or infinity: saturating clamp
        return negative ? 0xfeu : 0x7eu;  // +-448 are the codes 0x7e/0xfe
    }
    if (magnitude < kE4m3MinNormal) {
        // Subnormal grid m * 2^-9, m in 0..8 (8 carries into the min normal 2^-6).
        const float scaled = magnitude * 0x1p9f;  // exact power-of-two scaling
        std::uint32_t m = static_cast<std::uint32_t>(scaled);
        const float fraction = scaled - static_cast<float>(m);
        if (fraction > 0.5f || (fraction == 0.5f && (m & 1u) != 0u)) {
            ++m;
        }
        if (m >= 8u) {
            return negative ? 0x88u : 0x08u;
        }
        return static_cast<std::uint8_t>((negative ? 0x80u : 0u) | m);
    }
    // Normal grid: the 24-bit float32 significand rounded to 3 mantissa bits with
    // ties to even; the rounding carry increments the exponent field.
    std::uint32_t bits = 0;
    std::memcpy(&bits, &magnitude, sizeof bits);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;  // -6..8 here
    const std::uint32_t significand = 0x800000u | (bits & 0x7fffffu);
    std::uint32_t rounded = significand >> 20;          // in [8, 15]
    const std::uint32_t rest = significand & 0xfffffu;
    const std::uint32_t half_bit = 0x80000u;
    if (rest > half_bit || (rest == half_bit && (rounded & 1u) != 0u)) {
        ++rounded;
    }
    int e_field = exponent + 7;
    if (rounded == 16u) {
        rounded = 8u;
        ++e_field;
    }
    // For finite pre-clamp inputs e_field is bounded to [1, 15] and rounded to
    // [8, 15] (e.g. 256.0 -> e_field 15, 240.0 -> rounded 15); the guards below
    // keep the encoding total anyway (15/7 is the NaN code, and 16 would collide
    // with the sign bit).
    if (e_field > 15 || (e_field == 15 && rounded >= 15u)) {
        return negative ? 0xfeu : 0x7eu;
    }
    return static_cast<std::uint8_t>((negative ? 0x80u : 0u)
        | (static_cast<std::uint32_t>(e_field) << 3) | (rounded - 8u));
}

// Decode an e4m3 code to its float32 value (0x7f and 0xff decode to the canonical
// quiet NaN).
inline float e4m3_from_bits(std::uint8_t raw) {
    // Bit-exact construction (no ldexp/branches on the value paths): e4m3 is
    // sign + 4 exponent bits (bias 7) + 3 mantissa bits, e=15/m=7 the sole
    // NaN. Normal codes (e >= 1): (8+m) * 2^(e-10) = 1.m * 2^(e-7) -> float
    // exponent field e+120, mantissa m<<20. Subnormals (e == 0): m * 2^-9 =
    // bits((float)m) - (9 << 23) for m >= 1; m == 0 is the sign-only zero.
    // The test suite pins all 256 codes against two independently derived
    // oracles (an ldexpf reference and the format formula in double), so
    // the construction is exact by exhaustive check, not by appeal to a
    // previous implementation.
    static const std::uint32_t kSub[8] = {
        0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u,
        0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u};
    const std::uint32_t s = static_cast<std::uint32_t>(raw & 0x80u) << 24;
    const std::uint32_t e = (raw >> 3) & 0xFu;
    const std::uint32_t m = raw & 7u;
    if (e == 0xFu && m == 7u) {
        const std::uint32_t canonical = 0x7fc00000u;
        float nan_value = 0.0f;
        std::memcpy(&nan_value, &canonical, sizeof nan_value);
        return nan_value;
    }
    const std::uint32_t bits = e == 0u
        ? (kSub[m] | s)
        : (s | ((e + 120u) << 23) | (m << 20));
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

inline float e4m3(float value) { return e4m3_from_bits(e4m3_bits(value)); }

inline bool is_e4m3(float value) {
    return !std::isnan(value) && e4m3(value) == value;
}

// ---- 2x2 f16 pooling ----------------------------------------------------------------------------
// A single planar float32 map, row-major, natural orientation (x = column, y = row).
// row_stride is in floats between consecutive rows and must be >= cols.
struct PlaneView {
    const float* data = nullptr;
    int rows = 0;
    int cols = 0;
    std::ptrdiff_t row_stride = 0;
};

// The network's 2x2 mean of a block's unrounded output: pixel pairs across the columns
// are added in float32 and narrowed to f16, the two row sums are added in f16
// (computed in float32 and narrowed; for f16 addends the double rounding is benign
// and equals a strict binary16 add), the quarter is exact (a power of two in float32),
// and the result is rounded to the e4m3 grid by e4m3(). dst receives rows/2 * cols/2
// values in row-major order; extents must be even (the pipeline pads to even working
// extents first). Allocation-free and validated before any output write.
inline Status pool2x2(const PlaneView& src, float* dst, std::size_t dst_count) {
    if (src.data == nullptr) {
        return Status::kNullPointer;
    }
    const long long pixels = static_cast<long long>(src.rows) * src.cols;
    if (src.rows <= 0 || src.cols <= 0 || (src.rows & 1) != 0 || (src.cols & 1) != 0
        || src.rows > kMaxRows || src.cols > kMaxCols || pixels > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    if (src.row_stride < src.cols || src.row_stride > kMaxCols) {
        return Status::kInvalidStride;
    }
    if (dst == nullptr) {
        return Status::kNullPointer;
    }
    const std::size_t out_cols = static_cast<std::size_t>(src.cols / 2);
    const std::size_t out_rows = static_cast<std::size_t>(src.rows / 2);
    if (dst_count < out_rows * out_cols) {
        return Status::kInvalidBuffer;
    }
    for (int oy = 0; oy < src.rows / 2; ++oy) {
        const float* top = src.data
            + static_cast<std::size_t>(2 * oy) * static_cast<std::size_t>(src.row_stride);
        const float* bottom = src.data
            + static_cast<std::size_t>(2 * oy + 1) * static_cast<std::size_t>(src.row_stride);
        for (int ox = 0; ox < src.cols / 2; ++ox) {
            const float* a = top + 2 * ox;
            const float* b = bottom + 2 * ox;
            const float s0 = f16(a[0] + a[1]);
            const float s1 = f16(b[0] + b[1]);
            const float total = f16(s0 + s1);
            dst[static_cast<std::size_t>(oy) * out_cols + ox] = e4m3(total * 0.25f);
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_PRE_POST_HPP
