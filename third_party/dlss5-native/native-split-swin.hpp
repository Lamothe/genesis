// SPDX-License-Identifier: MIT
// Native CPU "split Swin" layer family (blocks 23-30 of the encoder bottom and
// 40-47 of the decoder start): grouped feed-forward, 1x1 projections with
// gained residuals, windowed multi-head attention on the shifted 8x8 grid,
// the pooled projection and the final 512 -> 1024 head — typed, deterministic
// CPU kernels.
//
// Re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch` commit
// 3beef7935a8ef976b47ab78b906e75600c52d2de (`dlssnr/layers/split_swin.py`);
// full MIT attribution in vr/NATIVE-SPLIT-SWIN.md. Not vendor code. The
// reference's experimental int4/int8 quantisation hooks are out of scope (the
// native deliverable executes the e4m3/float16 dataflow; `int8.mm` with the
// default empty site set is a plain float32 matrix product).
//
// Semantics carried over exactly:
//   - maps are token-major [rows * cols][channels], token = r * cols + c (the
//     reference's `_channels_last`); every stored result is rounded twice,
//     float32 -> float16 -> e4m3 (`_store`).
//   - `SplitFfwd` is eight groups of 512 -> 64 -> 256 -> 64 with the cubic
//     SiLU computed in float16 arithmetic (a rounding after every operation).
//   - the projections are `q8(f16(W x + gain * residual))` per channel
//     (`_Projection.project`); `SplitProjPool` additionally pools the f16
//     projection values (before their e4m3 rounding) with pairwise float16
//     sums, and `SplitFinalHead` projects the pooled copy without a residual.
//   - `SplitQKVAttn` windows over 8x8 pixels of a grid shifted by 0 or -1
//     tiles; a window's 64 tokens run over its four 4x4 tiles (top-left,
//     top-right, bottom-left, bottom-right), each tile row-major. Q and K are
//     normalised per 32-channel head with the network's float16 butterfly
//     reduction, Q is scaled by the head's gain, both are rounded to e4m3;
//     logits are Q.K^T plus the per-head position bias; the softmax uses the
//     network's own float16 exponential and the pinned denominator reduction;
//     padded zero tokens take part as keys (their logits still carry the bias)
//     and their outputs are dropped.

#ifndef DLSS5_NR_NATIVE_SPLIT_SWIN_HPP
#define DLSS5_NR_NATIVE_SPLIT_SWIN_HPP

#include "native-matmul-simd.hpp"
#include "native-vit.hpp"  // E4m3Matrix, Status; f16/e4m3/f16_from_double via its includes

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace dlss5_nr {

// ---- family constants -----------------------------------------------------------------------------
inline constexpr int kSplitChannels = 512;        // channels of every split map
inline constexpr int kSplitWindow = 8;            // pixels per window side
inline constexpr int kSplitTile = 4;              // pixels per tile side (shift unit)
inline constexpr int kSplitTokensPerWindow = kSplitWindow * kSplitWindow;
inline constexpr int kSplitHeads = 16;            // attention heads
inline constexpr int kSplitHeadDim = 32;          // channels per head
inline constexpr int kSplitGroups = 8;            // feed-forward groups
inline constexpr int kSplitGroupIn = 64;          // channels per group edge
inline constexpr int kSplitGroupHidden = 256;     // hidden width per group
inline constexpr int kSplitQkvOut = kSplitHeads * 3 * kSplitHeadDim;  // 1536

// The attention's float16 exponential: y = f16(fma(f16(logit), slope, offset))
// clamped to [low, high]; the ten mantissa bits of y shifted left by five then
// select the float16 result (an affine map into the exponent field).
inline constexpr float kSplitExpSlope = 0.044921875f;      // f16-exact (23 / 512)
inline constexpr float kSplitExpOffset = 1.30078125f;      // f16-exact (1332 / 1024)
inline constexpr float kSplitExpLow = 1.03125f;            // f16-exact (1056 / 1024)
inline constexpr float kSplitExpHigh = 1.5693359375f;      // f16-exact (1607 / 1024)

inline constexpr float kSplitSumFloor = 6.103515625e-05f;  // 2^-14 (denominator floor)
inline constexpr float kSplitNormFloor = 0.000062f;        // -> f16 6.198883056640625e-05

// Kernel envelopes (checked with plain comparisons and widened products
// before any offset maths). 16448 is the padded-frame bound (16384 frame plus
// the 64-pixel alignment of the plan's working extent).
inline constexpr int kSplitMaxExtent = 16448;
inline constexpr long long kSplitMaxTokens = 1LL << 29;    // rows * columns
inline constexpr int kSplitMaxChannels = 1024;             // projection c_in / c_out

// ---- float16 fused multiply-add and store ----------------------------------------------------------

// `a * b + c` with the exact product and sum in binary64 and a single rounding
// to float16 (the reference's `_fma_half`, `f16_from_double`'s documented
// zones). a, b and c are float16 values, so the binary64 expression is exact
// and the one rounding is the reference's float16 result.
inline float split_fma_half(float a, float b, float c) {
    return f16_from_double(static_cast<double>(a) * static_cast<double>(b)
        + static_cast<double>(c));
}

// The network store (`_store`): float32 -> float16 -> e4m3, the double rounding
// of every stored result. Direct e4m3 rounding differs at float16 half-steps
// (e.g. 68.0001f stores as 64 via the float16 hop but rounds to 72 directly).
inline float split_store(float value) { return e4m3(f16(value)); }

// ---- feed-forward activation -----------------------------------------------------------------------

// The cubic SiLU of `SplitFfwd.activation`, computed in float16 arithmetic
// (a rounding after every operation): x is rounded to float16 first,
// t = clamp(x, +-4), u = fma(-0.055908203125, |t|, 0.447265625),
// v = fma(t, u, 0.89453125) and the result is f16(x * v). The outer multiply
// uses the unclamped x.
inline float split_ffn_activation(float x) {
    const float xf = f16(x);
    const float t = xf < -4.0f ? -4.0f : (xf > 4.0f ? 4.0f : xf);
    const float abs_t = t < 0.0f ? -t : t;
    const float u = split_fma_half(f16(-0.055908203125f), f16(abs_t), f16(0.447265625f));
    const float v = split_fma_half(t, u, f16(0.89453125f));
    return f16(xf * v);
}

// ---- the attention's exponential --------------------------------------------------------------------

// `SplitQKVAttn.exponential`: y = clamp(fma(f16(logit), slope, offset), low,
// high) in float16; the result is the float16 whose bit pattern is the ten
// mantissa bits of y shifted left by five (y stays in [1, 2), so the mask
// extracts exactly the mantissa). A NaN logit maps deterministically to the
// clamp floor (the reference's conversion there is garbage-but-defined).
inline float split_exp(float logit) {
    float y = split_fma_half(f16(logit), f16(kSplitExpSlope), f16(kSplitExpOffset));
    if (!(y >= kSplitExpLow)) {
        y = kSplitExpLow;
    }
    if (y > kSplitExpHigh) {
        y = kSplitExpHigh;
    }
    const std::uint16_t mantissa = static_cast<std::uint16_t>(f16_bits(y) & 0x3ffu);
    return f16_from_bits(static_cast<std::uint16_t>(mantissa << 5));
}

// ---- per-head normalisation --------------------------------------------------------------------------

// `SplitQKVAttn.normalise`: divide one 32-channel head by its length with the
// network's float16 reduction (squares summed as d with d + 16, then 8, then
// the pairs (0+2, 1+3) joined). `values` must already be float16 values; the
// squares, every sum, the reciprocal and the per-channel products round to
// float16. The magnitude is floored at f16(0.000062) = 6.198883056640625e-05.
inline void split_normalise(const float* values, float* out) {
    float total[kSplitHeadDim / 2];
    for (int c = 0; c < kSplitHeadDim / 2; ++c) {
        const float a = f16(values[c] * values[c]);
        const float b = f16(values[c + kSplitHeadDim / 2] * values[c + kSplitHeadDim / 2]);
        total[c] = f16(a + b);
    }
    for (int half = kSplitHeadDim / 4; half >= 1; half /= 2) {
        for (int c = 0; c < half; ++c) {
            total[c] = f16(total[c] + total[c + half]);
        }
    }
    float magnitude = total[0];
    const float floor_value = f16(kSplitNormFloor);
    if (magnitude < floor_value) {
        magnitude = floor_value;
    }
    // torch computes rsqrt in float32 and rounds the result to float16.
    const float norm = f16(1.0f / std::sqrt(magnitude));
    for (int d = 0; d < kSplitHeadDim; ++d) {
        out[d] = f16(values[d] * norm);
    }
}

// ---- softmax denominator -------------------------------------------------------------------------------

// The pinned reduction of the 64 exponential leaves (key j = tile * 16 +
// half * 8 + c): tiles sequential ((0+1)+2)+3 per (half, c), adjacent c pairs,
// (0+1)+(2+3) per half, then the two halves. The reference accumulates these
// sums in float32 (only the normalisation/exp/activation are float16); over
// the exponential's leaf range every intermediate is exact in float32, so the
// pinned order and the exact sum agree.
inline float split_softmax_denominator(const float* leaves) {
    float columns[2][8];
    for (int half = 0; half < 2; ++half) {
        for (int c = 0; c < 8; ++c) {
            const int j0 = half * 8 + c;
            float acc = leaves[j0];
            acc = acc + leaves[16 + j0];
            acc = acc + leaves[32 + j0];
            acc = acc + leaves[48 + j0];
            columns[half][c] = acc;
        }
    }
    float halves[2];
    for (int half = 0; half < 2; ++half) {
        const float p0 = columns[half][0] + columns[half][1];
        const float p1 = columns[half][2] + columns[half][3];
        const float p2 = columns[half][4] + columns[half][5];
        const float p3 = columns[half][6] + columns[half][7];
        halves[half] = (p0 + p1) + (p2 + p3);
    }
    return halves[0] + halves[1];
}

// ---- matrix product -----------------------------------------------------------------------------------

// out[t][n] = sum_k in[t][k] * weight[n][k]: e4m3 weights held exactly in
// float32, float32 accumulation in key order (the family's `int8.mm` with the
// default plain-product semantics; identical arithmetic to `matmul_e4m3` with
// the split family's envelopes).
inline Status split_matmul(const float* in, long long tokens, int c_in,
                           const E4m3Matrix& weight, float* out) {
    if (in == nullptr || weight.bits == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || tokens > kSplitMaxTokens || c_in < 1 || c_in > kSplitMaxChannels
        || weight.rows < 1 || weight.rows > kSplitQkvOut || weight.cols != c_in) {
        return Status::kInvalidDimensions;
    }
    // Long keys only (see native-matmul-simd.hpp): the AVX2 dot8 keeps each
    // output's key-ordered accumulation exactly (bit-identical).
    const bool simd = matmul_simd_available() && c_in >= 256;
    for (long long t = 0; t < tokens; ++t) {
        const float* row = in + t * c_in;
        float* dst = out + t * weight.rows;
        int n = 0;
#if defined(__x86_64__) || defined(__i386__)
        if (simd) {
            const std::uint8_t* wrows[8];
            for (; n + 8 <= weight.rows; n += 8) {
                for (int i = 0; i < 8; ++i) {
                    wrows[i] = weight.bits
                        + static_cast<std::size_t>(n + i) * c_in;
                }
                e4m3_dot8(row, c_in, wrows, dst + n);
            }
        }
#endif
        for (; n < weight.rows; ++n) {
            const std::uint8_t* w = weight.bits + static_cast<std::size_t>(n) * c_in;
            float acc = 0.0f;
            for (int k = 0; k < c_in; ++k) {
                acc += row[k] * e4m3_from_bits(w[k]);
            }
            dst[n] = acc;
        }
    }
    return Status::kOk;
}

// ---- SplitFfwd ------------------------------------------------------------------------------------------

// The grouped feed-forward's matrices, in natural [out, in] orientation:
// `expand_in` is [512, 512] with row g * 64 + j = A_g row j (512 keys);
// `expand_hidden` is [8 * 256, 64] with row g * 256 + j = Q0_g row j;
// `contract` is [8 * 64, 256] with row g * 64 + j = Q2_g row j. (The pinned
// reference stores the transposed products of its packed weight record; the
// record decode is a weight-file concern outside this slice.)
struct SplitFfwdWeight {
    E4m3Matrix expand_in;
    E4m3Matrix expand_hidden;
    E4m3Matrix contract;
};

// SplitFfwd: eight groups of 512 -> 64 -> 256 (cubic SiLU in float16) -> 64,
// every stage stored (float32 -> float16 -> e4m3). `in` and `out` are
// token-major [rows * cols][512]; the block's residual input is `in` itself
// (the reference returns it unchanged alongside the feed-forward map).
inline Status split_ffwd(const float* in, int rows, int cols,
                         const SplitFfwdWeight& w, float* out) {
    if (in == nullptr || out == nullptr || w.expand_in.bits == nullptr
        || w.expand_hidden.bits == nullptr || w.contract.bits == nullptr) {
        return Status::kNullPointer;
    }
    if (rows < 1 || rows > kSplitMaxExtent || cols < 1 || cols > kSplitMaxExtent) {
        return Status::kInvalidDimensions;
    }
    const long long tokens = static_cast<long long>(rows) * cols;
    if (tokens > kSplitMaxTokens) {
        return Status::kInvalidDimensions;
    }
    if (w.expand_in.rows != kSplitChannels || w.expand_in.cols != kSplitChannels
        || w.expand_hidden.rows != kSplitGroups * kSplitGroupHidden
        || w.expand_hidden.cols != kSplitGroupIn
        || w.contract.rows != kSplitGroups * kSplitGroupIn
        || w.contract.cols != kSplitGroupHidden) {
        return Status::kInvalidDimensions;
    }
    // Token-parallel (native-parallel.hpp): each token's stack-locals and
    // key-ordered accumulations are thread-invariant; exact parity.
    parallel_for(0LL, tokens, [&](long long t) {
        const float* row = in + t * kSplitChannels;
        float* dst = out + t * kSplitChannels;
        float reduced[kSplitChannels];
        for (int n = 0; n < kSplitChannels; ++n) {
            const std::uint8_t* wrow = w.expand_in.bits
                + static_cast<std::size_t>(n) * kSplitChannels;
            float acc = 0.0f;
            for (int k = 0; k < kSplitChannels; ++k) {
                acc += row[k] * e4m3_from_bits(wrow[k]);
            }
            reduced[n] = split_store(acc);
        }
        for (int g = 0; g < kSplitGroups; ++g) {
            float hidden[kSplitGroupHidden];
            for (int j = 0; j < kSplitGroupHidden; ++j) {
                const std::uint8_t* wrow = w.expand_hidden.bits
                    + static_cast<std::size_t>(g * kSplitGroupHidden + j) * kSplitGroupIn;
                float acc = 0.0f;
                for (int k = 0; k < kSplitGroupIn; ++k) {
                    acc += reduced[g * kSplitGroupIn + k] * e4m3_from_bits(wrow[k]);
                }
                hidden[j] = split_store(split_ffn_activation(acc));
            }
            for (int j = 0; j < kSplitGroupIn; ++j) {
                const std::uint8_t* wrow = w.contract.bits
                    + static_cast<std::size_t>(g * kSplitGroupIn + j) * kSplitGroupHidden;
                float acc = 0.0f;
                for (int k = 0; k < kSplitGroupHidden; ++k) {
                    acc += hidden[k] * e4m3_from_bits(wrow[k]);
                }
                dst[g * kSplitGroupIn + j] = split_store(acc);
            }
        }
    });
    return Status::kOk;
}

// ---- projections -----------------------------------------------------------------------------------------

// The `_Projection` weights: `weight` is [c_out, c_in] e4m3; `gain` holds
// c_out float16 values (the per-channel residual gain). `gain == nullptr`
// selects the residual-free form (`SplitFinalHead`) and then `residual` must
// be nullptr too.
struct SplitProjectionWeight {
    E4m3Matrix weight;
    const float* gain = nullptr;
};

// `SplitFfwdProj`/`SplitProj`/`SplitProjPool`/`SplitFinalHead` share
// `value[t][j] = f16(sum_k x[t][k] * W[j][k] + gain[j] * residual[t][j])`
// (`_Projection.project`); the layers below store or pool this float16 value.
// `value` receives tokens * c_out floats.
// The per-element post chain, shared by the CPU kernel's fused loop and any
// external raw-projection executor (the Vulkan backend): identical
// operations in identical order. The split gain takes NO binary16 hop (the
// pinned's f32 semantics); the result rounds to binary16 once.
inline float split_projection_post(float product, float gain_value,
                                   float skip_value, bool has_skip) {
    float acc = product;
    if (has_skip) {
        acc += gain_value * skip_value;
    }
    return f16(acc);
}

// An external raw-projection site for the split rows: when present, the raw
// projection (out[t][j] = sum_k in[t][k] * w[j][k]) runs through the
// executor ONCE for all tokens into `products` (caller-owned, [tokens,
// c_out] floats), and the per-element post chain runs on the CPU —
// bit-identical to the fused CPU loop when every intermediate is in the
// fp32 normal range (the Vulkan backend's documented denormal tolerance).
// Null keeps the CPU kernel (the deterministic reference).
struct SplitProjectionSite {
    const ProjectionExecutor* ex = nullptr;
    float* products = nullptr;
    std::size_t products_capacity = 0;  // in floats
};

inline Status split_projection_value(const float* x, const float* residual,
                                     long long tokens, int c_in, int c_out,
                                     const SplitProjectionWeight& w, float* value) {
    if (x == nullptr || value == nullptr || w.weight.bits == nullptr) {
        return Status::kNullPointer;
    }
    if ((residual == nullptr) != (w.gain == nullptr)) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || tokens > kSplitMaxTokens
        || c_in < 1 || c_in > kSplitMaxChannels
        || c_out < 1 || c_out > kSplitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    if (w.weight.rows != c_out || w.weight.cols != c_in) {
        return Status::kInvalidDimensions;
    }
    // Token-parallel (native-parallel.hpp): every element keeps its
    // key-ordered accumulation exactly; bit-identical at any thread count.
    parallel_for(0LL, tokens, [&](long long t) {
        const float* row = x + t * c_in;
        const float* skip = residual == nullptr ? nullptr : residual + t * c_out;
        float* dst = value + t * c_out;
        for (int j = 0; j < c_out; ++j) {
            const std::uint8_t* wrow = w.weight.bits + static_cast<std::size_t>(j) * c_in;
            float acc = 0.0f;
            for (int k = 0; k < c_in; ++k) {
                acc += row[k] * e4m3_from_bits(wrow[k]);
            }
            dst[j] = split_projection_post(acc, w.gain == nullptr
                                                   ? 0.0f
                                                   : w.gain[j],
                                           skip == nullptr ? 0.0f : skip[j],
                                           skip != nullptr);
        }
    });
    return Status::kOk;
}

// SplitFfwdProj: the stored projection of the feed-forward map; the residual
// is the block input. `x` is the feed-forward output, `block_input` the
// unchanged layer input the feed-forward returned alongside it.
inline Status split_ffwd_proj(const float* x, const float* block_input, int rows,
                              int cols, const SplitProjectionWeight& w, float* out, float* side_out = nullptr,
                              const SplitProjectionSite* site = nullptr) {
    // x/block_input/out are mandatory and the residual pairing rule then
    // requires gain: both the CPU and site paths reject the same inputs.
    if (x == nullptr || block_input == nullptr || out == nullptr
        || w.gain == nullptr) {
        return Status::kNullPointer;
    }
    if (rows < 1 || rows > kSplitMaxExtent || cols < 1 || cols > kSplitMaxExtent) {
        return Status::kInvalidDimensions;
    }
    const long long tokens = static_cast<long long>(rows) * cols;
    if (tokens > kSplitMaxTokens) {
        return Status::kInvalidDimensions;
    }
    const int c_in = w.weight.cols;
    const int c_out = w.weight.rows;
    if (c_in < 1 || c_in > kSplitMaxChannels || c_out < 1 || c_out > kSplitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    if (site != nullptr
        && (site->ex == nullptr || site->ex->matmul == nullptr
            || site->products == nullptr
            || site->products_capacity
                   < static_cast<std::size_t>(rows) * cols
                       * static_cast<std::size_t>(c_out))) {
        return Status::kInvalidBuffer;
    }
    if (site != nullptr) {
        const Status st = site->ex->matmul(site->ex->ctx, x,
                                           static_cast<int>(tokens), c_in,
                                           w.weight, site->products);
        if (st != Status::kOk) {
            return st;
        }
    }
    // Token-parallel with a recorded failure status (the earlier serial
    // loop could return mid-flight): tokens are independent and every
    // element keeps its exact arithmetic, so any thread count is
    // bit-identical to the serial reference.
    std::atomic<int> fail{static_cast<int>(Status::kOk)};
    parallel_for(0LL, tokens, [&](long long t) {
        if (fail.load(std::memory_order_relaxed) != 0) {
            return;
        }
        float value[kSplitMaxChannels];
        Status status = Status::kOk;
        if (site != nullptr) {
            const float* product_row = site->products + t * c_out;
            for (int j = 0; j < c_out; ++j) {
                value[j] = split_projection_post(
                    product_row[j], w.gain == nullptr ? 0.0f : w.gain[j],
                    block_input[t * c_out + j], true);
            }
        } else {
            status = split_projection_value(
                x + t * c_in, block_input + t * c_out, 1, c_in, c_out, w,
                value);
        }
        if (status != Status::kOk) {
            fail.store(static_cast<int>(status), std::memory_order_relaxed);
            return;
        }
        float* dst = out + t * c_out;
        for (int j = 0; j < c_out; ++j) {
            if (side_out != nullptr) {
                // The pinned `int4.side[("attn", block)] = value`: the
                // projection's value at its pre-store precision.
                side_out[t * c_out + j] = value[j];
            }
            dst[j] = e4m3(value[j]);
        }
    });
    return static_cast<Status>(fail.load(std::memory_order_relaxed));
}

// SplitProj: the stored projection of the attention map; the residual is the
// SplitFfwdProj output.
inline Status split_proj(const float* x, const float* residual, int rows, int cols,
                         const SplitProjectionWeight& w, float* out,
                         const SplitProjectionSite* site = nullptr) {
    return split_ffwd_proj(x, residual, rows, cols, w, out, nullptr, site);
}

// SplitProjPool: `SplitProj` plus the half-resolution pooled copy. The pooled
// map is the 2x2 mean of the float16 projection values (before they are
// rounded to e4m3): pairwise column sums, then row sums, all float16, scaled
// by 0.25 and rounded to e4m3. Its extent is ceil(rows / 2) x ceil(cols / 2)
// (the reference only executes even extents; a trailing odd row or column
// pairs with an implicit zero) padded with zeros to a multiple of 4. `out_pooled`
// receives pooled_rows * pooled_cols * c_out floats where `*pooled_rows` and
// `*pooled_cols` are the padded extents written on success.
inline Status split_proj_pool(const float* x, const float* residual, int rows,
                              int cols, const SplitProjectionWeight& w,
                              float* out_proj, float* out_pooled,
                              int* pooled_rows, int* pooled_cols,
                              const SplitProjectionSite* site = nullptr) {
    if (x == nullptr || residual == nullptr || out_proj == nullptr
        || out_pooled == nullptr || pooled_rows == nullptr || pooled_cols == nullptr
        || w.gain == nullptr) {
        return Status::kNullPointer;
    }
    if (rows < 1 || rows > kSplitMaxExtent || cols < 1 || cols > kSplitMaxExtent) {
        return Status::kInvalidDimensions;
    }
    const long long tokens = static_cast<long long>(rows) * cols;
    if (tokens > kSplitMaxTokens) {
        return Status::kInvalidDimensions;
    }
    const int c_in = w.weight.cols;
    const int c_out = w.weight.rows;
    if (c_in < 1 || c_in > kSplitMaxChannels || c_out < 1 || c_out > kSplitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    if (site != nullptr
        && (site->ex == nullptr || site->ex->matmul == nullptr
            || site->products == nullptr
            || site->products_capacity
                   < static_cast<std::size_t>(rows) * cols
                       * static_cast<std::size_t>(c_out))) {
        return Status::kInvalidBuffer;
    }
    const int half_rows = (rows + 1) / 2;
    const int half_cols = (cols + 1) / 2;
    const int pad_rows = (half_rows + 3) & ~3;
    const int pad_cols = (half_cols + 3) & ~3;
    const long long pooled_slots = static_cast<long long>(pad_rows) * pad_cols;
    for (long long r = 0; r < pooled_slots * c_out; ++r) {
        out_pooled[r] = 0.0f;
    }
    if (site != nullptr) {
        const Status st = site->ex->matmul(
            site->ex->ctx, x, static_cast<int>(rows) * cols, w.weight.cols,
            w.weight, site->products);
        if (st != Status::kOk) {
            return st;
        }
    }
    for (int r = 0; r < half_rows; ++r) {
        for (int c = 0; c < half_cols; ++c) {
            float block[4 * kSplitMaxChannels];
            for (int br = 0; br < 2; ++br) {
                for (int bc = 0; bc < 2; ++bc) {
                    const int src_r = 2 * r + br;
                    const int src_c = 2 * c + bc;
                    float* slot = block + (br * 2 + bc) * c_out;
                    if (src_r < rows && src_c < cols) {
                        const long long t = static_cast<long long>(src_r) * cols + src_c;
                        Status status = Status::kOk;
                        if (site != nullptr) {
                            const float* product_row = site->products + t * c_out;
                            for (int j = 0; j < c_out; ++j) {
                                slot[j] = split_projection_post(
                                    product_row[j], w.gain[j],
                                    residual[t * c_out + j], true);
                            }
                        } else {
                            status = split_projection_value(
                                x + t * c_in, residual + t * c_out, 1, c_in,
                                c_out, w, slot);
                        }
                        if (status != Status::kOk) {
                            return status;
                        }
                        float* dst = out_proj + t * c_out;
                        for (int j = 0; j < c_out; ++j) {
                            dst[j] = e4m3(slot[j]);
                        }
                    } else {
                        for (int j = 0; j < c_out; ++j) {
                            slot[j] = 0.0f;
                        }
                    }
                }
            }
            float* pooled = out_pooled + (static_cast<std::size_t>(r) * pad_cols + c) * c_out;
            for (int j = 0; j < c_out; ++j) {
                const float top = f16(block[j] + block[c_out + j]);
                const float bottom = f16(block[2 * c_out + j] + block[3 * c_out + j]);
                const float sum = f16(top + bottom);
                pooled[j] = e4m3(f16(sum * 0.25f));
            }
        }
    }
    *pooled_rows = pad_rows;
    *pooled_cols = pad_cols;
    return Status::kOk;
}

// SplitFinalHead: 512 -> 1024 projection of the pooled copy without residual
// (the transformer stage's input). `pooled` is token-major
// [pooled_rows * pooled_cols][c_in] (the padded pooled map); `out` receives
// [pooled_rows * pooled_cols][c_out].
inline Status split_final_head(const float* pooled, int pooled_rows, int pooled_cols,
                               const SplitProjectionWeight& w, float* out,
                               const SplitProjectionSite* site = nullptr) {
    if (pooled == nullptr || out == nullptr || w.gain != nullptr) {
        return Status::kNullPointer;
    }
    if (pooled_rows < 1 || pooled_rows > kSplitMaxExtent
        || pooled_cols < 1 || pooled_cols > kSplitMaxExtent) {
        return Status::kInvalidDimensions;
    }
    if (site != nullptr
        && (site->ex == nullptr || site->ex->matmul == nullptr
            || site->products == nullptr
            || site->products_capacity
                   < static_cast<std::size_t>(pooled_rows) * pooled_cols
                       * static_cast<std::size_t>(w.weight.rows))) {
        return Status::kInvalidBuffer;
    }
    const long long tokens = static_cast<long long>(pooled_rows) * pooled_cols;
    if (tokens > kSplitMaxTokens) {
        return Status::kInvalidDimensions;
    }
    const int c_in = w.weight.cols;
    const int c_out = w.weight.rows;
    if (c_in < 1 || c_in > kSplitMaxChannels || c_out < 1 || c_out > kSplitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    if (site != nullptr) {
        const Status st = site->ex->matmul(site->ex->ctx, pooled,
                                           static_cast<int>(tokens), c_in,
                                           w.weight, site->products);
        if (st != Status::kOk) {
            return st;
        }
    }
    for (long long t = 0; t < tokens; ++t) {
        float value[kSplitMaxChannels];
        Status status = Status::kOk;
        if (site != nullptr) {
            const float* product_row = site->products + t * c_out;
            for (int j = 0; j < c_out; ++j) {
                value[j] = split_projection_post(product_row[j], 0.0f, 0.0f,
                                                 false);
            }
        } else {
            status = split_projection_value(pooled + t * c_in, nullptr, 1,
                                            c_in, c_out, w, value);
        }
        if (status != Status::kOk) {
            return status;
        }
        float* dst = out + t * c_out;
        for (int j = 0; j < c_out; ++j) {
            dst[j] = e4m3(value[j]);
        }
    }
    return Status::kOk;
}

// ---- SplitQKVAttn ------------------------------------------------------------------------------------------

// The window attention's weights: `qkv` is [1536, 512] e4m3 with row
// h * 96 + v * 32 + d (v: 0 query, 1 key, 2 value) reading the 512 map
// channels; `bias` holds the unpacked per-head [64, 64] position biases
// (query i, key j, window-token order) as float16 values; `gain` holds the
// 16 per-head float32 gains (rounded to float16 at use). (The pinned
// reference's accumulator-fragment bias unpack is a weight-file concern
// outside this slice.)
struct SplitQkvWeight {
    E4m3Matrix qkv;
    const float* bias = nullptr;
    const float* gain = nullptr;
    // The int4 miring (the pinned `attention_matrix`): when `qkv_products`
    // is set the per-token projections come from [rows * cols, kSplitQkvOut]
    // (the site's quantised-accumulated output, rows in the h * 96 + v * 32
    // + d order above) instead of the e4m3 matmul, and padded zero-valued
    // tokens project from `padded_product` ([kSplitQkvOut]) — the pinned
    // projects padded tokens through the matrix, so for the int4 form that
    // is the zero-input projection through the matrix — generally not zero
    // (it is zero when the quantiser's zero point gives codes(0) = 0 and the
    // row constant is 0).
    const float* qkv_products = nullptr;
    const float* padded_product = nullptr;
};

// SplitQKVAttn: window attention over 8x8 pixel windows of a grid shifted by
// `shift_x`/`shift_y` tiles (0 or -1; `shift_x` moves the grid along the
// columns, `shift_y` along the rows). `in` and `out` are token-major
// [rows * cols][512]. Tiles of a shifted window that lie outside the map are
// zero tokens: they take part as keys (their logits still carry the position
// bias) and their outputs are dropped.
inline Status split_qkv_attn(const float* in, int rows, int cols,
                             int shift_x, int shift_y,
                             const SplitQkvWeight& w, float* out) {
    if (in == nullptr || out == nullptr || w.bias == nullptr
        || w.gain == nullptr
        || (w.qkv.bits == nullptr && w.qkv_products == nullptr)
        || (w.qkv_products != nullptr && w.padded_product == nullptr)) {
        return Status::kNullPointer;
    }
    if (rows < 1 || rows > kSplitMaxExtent || cols < 1 || cols > kSplitMaxExtent) {
        return Status::kInvalidDimensions;
    }
    if ((shift_x != 0 && shift_x != -1) || (shift_y != 0 && shift_y != -1)) {
        return Status::kInvalidControls;
    }
    if (w.qkv_products == nullptr
        && (w.qkv.rows != kSplitQkvOut || w.qkv.cols != kSplitChannels)) {
        return Status::kInvalidDimensions;
    }
    const long long tokens = static_cast<long long>(rows) * cols;
    if (tokens > kSplitMaxTokens) {
        return Status::kInvalidDimensions;
    }
    const int top = -shift_y * kSplitTile;
    const int left = -shift_x * kSplitTile;
    const int window_rows = (rows + top + kSplitWindow - 1) / kSplitWindow;
    const int window_cols = (cols + left + kSplitWindow - 1) / kSplitWindow;
    // Window-parallel with a recorded failure status (the earlier serial
    // loops could return mid-flight): windows write disjoint output tokens
    // and share read-only inputs, so any thread count is bit-identical.
    std::atomic<int> fail{static_cast<int>(Status::kOk)};
    parallel_for(0, window_rows * window_cols, [&](int wi) {
        if (fail.load(std::memory_order_relaxed) != 0) {
            return;
        }
        const int wr = wi / window_cols;
        const int wc = wi % window_cols;
        {
            std::uint8_t q_code[kSplitTokensPerWindow][kSplitHeadDim];
            std::uint8_t k_code[kSplitTokensPerWindow][kSplitHeadDim];
            std::uint8_t v_code[kSplitTokensPerWindow][kSplitHeadDim];
            for (int head = 0; head < kSplitHeads; ++head) {
                // QKV projection of this window's 64 tokens for this head's
                // 96 output rows (the shared key-ascending float32 product);
                // padded tokens project from zero on the e4m3 path and from
                // the zero-input projection on the products path.
                for (int i = 0; i < kSplitTokensPerWindow; ++i) {
                    const int pr = wr * kSplitWindow + (i / 32) * kSplitTile
                        + (i / 4) % kSplitTile;
                    const int pc = wc * kSplitWindow + ((i / 16) & 1) * kSplitTile
                        + i % kSplitTile;
                    const bool inside = pr >= top && pr < top + rows
                        && pc >= left && pc < left + cols;
                    const float* token = inside
                        ? in + (static_cast<std::size_t>(pr - top) * cols + (pc - left))
                                  * kSplitChannels
                        : nullptr;
                    float projected[3 * kSplitHeadDim];
                    if (w.qkv_products != nullptr) {
                        // The int4 site's precomputed projections (already in
                        // the matrix-row order); the same binary16 round as
                        // the e4m3 path's post-matmul round.
                        const std::size_t token_index = token != nullptr
                            ? static_cast<std::size_t>(pr - top) * cols
                                  + static_cast<std::size_t>(pc - left)
                            : 0;
                        const float* src = token != nullptr
                            ? w.qkv_products
                                  + token_index * kSplitQkvOut
                            : w.padded_product;
                        const float* head_src = src
                            + static_cast<std::size_t>(head) * 3
                                  * kSplitHeadDim;
                        for (int n = 0; n < 3 * kSplitHeadDim; ++n) {
                            projected[n] = f16(head_src[n]);
                        }
                    } else if (token != nullptr) {
                        // Built only here: w.qkv.bits may be null when the
                        // products branch is active (no null-base arithmetic).
                        const E4m3Matrix head_rows{
                            w.qkv.bits
                                + static_cast<std::size_t>(head) * 3
                                      * kSplitHeadDim * kSplitChannels,
                            3 * kSplitHeadDim, kSplitChannels};
                        const Status status = split_matmul(token, 1, kSplitChannels,
                                                           head_rows, projected);
                        if (status != Status::kOk) {
                            fail.store(static_cast<int>(status),
                                       std::memory_order_relaxed);
                            return;
                        }
                        for (int n = 0; n < 3 * kSplitHeadDim; ++n) {
                            projected[n] = f16(projected[n]);
                        }
                    } else {
                        for (int n = 0; n < 3 * kSplitHeadDim; ++n) {
                            projected[n] = 0.0f;
                        }
                    }
                    float q_norm[kSplitHeadDim];
                    float k_norm[kSplitHeadDim];
                    split_normalise(projected, q_norm);
                    split_normalise(projected + kSplitHeadDim, k_norm);
                    const float gain = f16(w.gain[head]);
                    for (int d = 0; d < kSplitHeadDim; ++d) {
                        q_code[i][d] = e4m3_bits(f16(f16(q_norm[d] * gain)));
                        k_code[i][d] = e4m3_bits(k_norm[d]);
                        v_code[i][d] = e4m3_bits(projected[2 * kSplitHeadDim + d]);
                    }
                }
                // Attention: logits = Q.K^T + bias, the network's exponential,
                // the pinned denominator and e4m3 probabilities; the context
                // accumulates in float32 and is stored (float16 -> e4m3).
                for (int i = 0; i < kSplitTokensPerWindow; ++i) {
                    const int pr = wr * kSplitWindow + (i / 32) * kSplitTile
                        + (i / 4) % kSplitTile;
                    const int pc = wc * kSplitWindow + ((i / 16) & 1) * kSplitTile
                        + i % kSplitTile;
                    const bool inside = pr >= top && pr < top + rows
                        && pc >= left && pc < left + cols;
                    if (!inside) {
                        continue;  // padded queries' outputs are dropped
                    }
                    float leaves[kSplitTokensPerWindow];
                    for (int j = 0; j < kSplitTokensPerWindow; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < kSplitHeadDim; ++d) {
                            dot += e4m3_from_bits(q_code[i][d]) * e4m3_from_bits(k_code[j][d]);
                        }
                        const float bias = w.bias[
                            (static_cast<std::size_t>(head) * kSplitTokensPerWindow + i)
                                * kSplitTokensPerWindow + j];
                        leaves[j] = split_exp(dot + bias);
                    }
                    const float total = split_softmax_denominator(leaves);
                    const float inverse = 1.0f / (total < kSplitSumFloor ? kSplitSumFloor : total);
                    float accum[kSplitHeadDim] = {0};
                    for (int j = 0; j < kSplitTokensPerWindow; ++j) {
                        const float p = split_store(leaves[j] * inverse);
                        for (int d = 0; d < kSplitHeadDim; ++d) {
                            accum[d] += p * e4m3_from_bits(v_code[j][d]);
                        }
                    }
                    float* dst = out
                        + (static_cast<std::size_t>(pr - top) * cols + (pc - left))
                              * kSplitChannels;
                    for (int d = 0; d < kSplitHeadDim; ++d) {
                        dst[head * kSplitHeadDim + d] = split_store(accum[d]);
                    }
                }
            }
        }
    });
    return static_cast<Status>(fail.load(std::memory_order_relaxed));
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SPLIT_SWIN_HPP
