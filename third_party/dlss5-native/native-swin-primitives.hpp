// Native Swin body scalar primitives: the cubic SiLU activation, the double-computed
// fused multiply-add, the deterministic reciprocal square root and the head-dimension
// reciprocal L2 norm of the reference's windowed-attention blocks. CPU-only, typed,
// allocation-free; no GPU, headset, proprietary payload, Python or PyTorch runtime
// required. This is the first executable slice of the learned-weight execution graph
// (the Swin body); the windowed attention, the e4m3 linears and the window
// partitioning are later slices.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "_fma", "activation", "_rsqrt" and "_norm_scale"). The full
// MIT licence notice is retained in vr/NATIVE-SWIN-PRIMITIVES.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_PRIMITIVES_HPP
#define DLSS5_NR_NATIVE_SWIN_PRIMITIVES_HPP

#include <cmath>

#include "native-blend.hpp"

namespace dlss5_nr {

// ---- documented constants ----------------------------------------------------------------------
inline constexpr float kActLimit = 4.0f;                // clamp of the activation argument t
inline constexpr float kActCubic = -0.055908203125f;    // -229 / 4096
inline constexpr float kActQuadratic = 0.447265625f;    // 229 / 512
inline constexpr float kActLinear = 0.89453125f;        // 229 / 256
inline constexpr float kNormFloor = 0x1.04p-14f;        // f16(0.000062) = 6.198883056640625e-05
inline constexpr int kNormDim = 32;                     // head dimensions per norm

// ---- the double-computed fused multiply-add ------------------------------------------------------
// a * b + c computed exactly as the reference's `_fma`: the product and the sum are
// taken in float64 and the result is rounded once to float32. This is NOT std::fma
// (which rounds the fused expression only once, in float32 precision): the reference's
// float64 intermediate is part of the semantics. For float32 `a` and the f16-exact
// constants `b` and `c` the product is exact in float64; the sum is rounded to
// float64 and then to float32, a deliberate double rounding that this routine
// reproduces. The build contract is -ffp-contract=off, so no compiler may substitute
// a hardware fused multiply-add; the explicit double casts pin the operations anyway.
inline float swin_fma(float a, float b, float c) {
    return static_cast<float>(static_cast<double>(a) * static_cast<double>(b)
        + static_cast<double>(c));
}

// ---- the cubic SiLU activation -------------------------------------------------------------------
// The network's activation, `x * (kActLinear + t * (kActQuadratic + kActCubic * |t|))`
// with `t = clamp(x, -kActLimit, kActLimit)`, evaluated as the reference writes it:
//   t     = clamp(x, -4, 4)                       (float32)
//   inner = swin_fma(|t|, kActCubic, kActQuadratic)
//   out   = x * swin_fma(t, inner, kActLinear)    (float32 multiply, one rounding)
// The outer multiply uses the unclamped x: beyond the clamp the ramp saturates but the
// outer factor does not (activation(10) = 17.890625, not 4 * 1.7890625). Sign follows
// x: activation(-0.0) = -0.0 and the cancellation at x = -4 gives activation(-4) =
// -0.0 (the ramp fma — the second of the two — cancels to +0.0 at t = -4, since
// t * inner = -kActLinear exactly there, and the outer multiply carries x's sign).
// Precondition: x is finite.
inline float swin_activation(float x) {
    const float t = (x < -kActLimit) ? -kActLimit : ((x > kActLimit) ? kActLimit : x);
    const float magnitude = (t < 0.0f) ? -t : t;
    const float inner = swin_fma(magnitude, kActCubic, kActQuadratic);
    return x * swin_fma(t, inner, kActLinear);
}

// ---- the deterministic reciprocal square root ------------------------------------------------------
// float32 `1 / sqrt(x)` with the reference's exact bit pattern: the square root is
// taken in float64 (one rounding, to float64), rounded to float32, then a correctly
// rounded float32 division. A GPU's rsqrt approximation is NOT reproduced and is not
// claimed; these are the CPU bits. Precondition: x is positive and finite, or x =
// +infinity (swin_norm_scale can feed an overflowed sum); the zero and negative
// inputs are outside the contract (the formula gives +inf for 0 and NaN below it).
inline float swin_rsqrt(float x) {
    const float root = static_cast<float>(std::sqrt(static_cast<double>(x)));
    return 1.0f / root;
}

// ---- the head-dimension reciprocal L2 norm ---------------------------------------------------------
// The attention head's normalisation scale from one vector of kNormDim (32) head
// dimensions, exactly as `_norm_scale` builds it:
//   1. Each value is squared in float64 (exact for float32 operands). The 32 slots
//      decode to (fragment f, component group g, chain c, half h) as
//      i = f * 16 + g * 8 + c * 2 + h.
//   2. For each (c, h) the four squares of that chain accumulate in the (g, f)
//      order (g=0, f=0), (g=0, f=1), (g=1, f=0), (g=1, f=1), interleaved across
//      the chains; after every step the running total is rounded to float32 and
//      widened again (the reference's `.float().double()` at each fragment
//      accumulation).
//   3. The two halves h are each ((c0 + c1) + (c2 + c3)) in float32 and narrowed to
//      binary16; the halves are added in binary16 and clamped at kNormFloor.
//   4. The scale is swin_rsqrt of that binary16 sum. A sum that overflows binary16
//      (or float32 in the chains) becomes an infinity and the scale is zero.
// The reference documents the chain/half structure as its f16 fragment-accumulator
// order; the exact decomposition above is this implementation's reading of it and the
// tests pin it. Precondition: values are finite (NaN is outside the contract).
inline float swin_norm_scale(const float values[kNormDim]) {
    double squares[2][2][4][2];
    for (int i = 0; i < kNormDim; ++i) {
        const int f = (i >> 4) & 1;
        const int g = (i >> 3) & 1;
        const int c = (i >> 1) & 3;
        const int h = i & 1;
        const double v = static_cast<double>(values[i]);
        squares[f][g][c][h] = v * v;
    }
    double totals[4][2] = {};
    for (int g = 0; g < 2; ++g) {
        for (int f = 0; f < 2; ++f) {
            for (int c = 0; c < 4; ++c) {
                for (int h = 0; h < 2; ++h) {
                    const double sum = squares[f][g][c][h] + totals[c][h];
                    totals[c][h] = static_cast<double>(static_cast<float>(sum));
                }
            }
        }
    }
    float halves[2];
    for (int h = 0; h < 2; ++h) {
        const float chains[4] = {
            static_cast<float>(totals[0][h]), static_cast<float>(totals[1][h]),
            static_cast<float>(totals[2][h]), static_cast<float>(totals[3][h]),
        };
        const float pairs_a = chains[0] + chains[1];
        const float pairs_b = chains[2] + chains[3];
        const float sum = pairs_a + pairs_b;
        halves[h] = f16_from_double(static_cast<double>(sum));
    }
    // The binary16 add of the two halves: each half is a binary16 value, so their
    // double sum is exact and one f16 rounding is the IEEE binary16 add.
    float total = f16_from_double(static_cast<double>(halves[0])
        + static_cast<double>(halves[1]));
    if (total < kNormFloor) {
        total = kNormFloor;
    }
    return swin_rsqrt(total);
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_PRIMITIVES_HPP
