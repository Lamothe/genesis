// SPDX-License-Identifier: MIT
// Native CPU vision-transformer bottleneck family (blocks 31-38): the Vit
// layer kinds of the network plan — FFN expand/contract, QKV, global
// attention and projection — as typed, deterministic kernels.
//
// Re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch` commit
// 3beef7935a8ef976b47ab78b906e75600c52d2de (`dlssnr/layers/vit.py`); full MIT
// attribution in vr/NATIVE-VIT.md. Not vendor code; the int4 experimental
// quantisation hooks of the reference are out of scope here (the native
// deliverable executes the e4m3/float16 dataflow).
//
// Semantics carried over exactly:
//   - tokens are token-major [token][channel]; a feature map [1, C, h, w] is
//     read as h*w tokens in row-major order, no positional term anywhere.
//   - matmul inputs are e4m3 values held exactly in float32; accumulation is
//     float32 in key order; every layer output goes float32 -> float16 -> e4m3.
//   - the FFN activation is the clipped cubic SiLU of the reference.
//   - QKV normalises k and q per 32-channel head in float16 with the
//     reference's butterfly norm order (d with d+16, then 8, 4, 2, 1) and a
//     per-head float16 temperature.
//   - attention is global; probabilities come from the network's piecewise
//     linear exponential (a float16 fused multiply-add and a bit trick); the
//     denominator is the float16 grouped sum over 64-key groups with padding
//     keys counted as exp(0) and subtracted at the end.

#ifndef VR_NATIVE_VIT_HPP
#define VR_NATIVE_VIT_HPP

#include "native-blend.hpp"      // f16 helpers (via native-input-lift) + Status
#include "native-matmul-simd.hpp"
#include "native-parallel.hpp"
#include "native-pre-post.hpp"   // e4m3 rounding

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace dlss5_nr {

inline constexpr int kVitHeadDim = 32;    // channels per attention head
inline constexpr int kVitKeyGroup = 64;   // keys summed as one group

// The network's exponential: y = round((x * slope + offset) * 1024) clamped to
// [1474, 2025]; the mantissa of y then selects the float16 value
// ldexp(1 + (y-1024 & 63)/64, (y-1024 >> 6) - 15).
inline constexpr float kVitExpSlope = 0.08953857421875f;    // 1467 / 2^14
inline constexpr float kVitExpOffset = 1.708984375f;        // 1750 / 1024
inline constexpr double kVitExpClampLow = 1474.0;           // round(1.439453125 * 1024)
inline constexpr double kVitExpClampHigh = 2025.0;          // round(1.9775390625 * 1024)

inline constexpr float kVitDenominatorFloor = 6.198883056640625e-05f;  // f32 floor
inline constexpr float kVitNormFloorF16 = 0.000062f;  // f16 value 6.198883056640625e-05

// Kernel envelopes (checked with plain comparisons before any offset maths).
inline constexpr int kVitMaxTokens = 1 << 28;     // 16384 * 16384 padded extent
inline constexpr int kVitMaxChannels = 1 << 14;   // per-tensor channel count

// A dense weight matrix of e4m3 values, [rows][cols] row-major (out x key).
struct E4m3Matrix {
    const std::uint8_t* bits = nullptr;
    int rows = 0;
    int cols = 0;
};

// ---- float16 / e4m3 helpers used throughout --------------------------------------------------------

inline float vit_f16(float value) { return f16(value); }

// ---- matmul ---------------------------------------------------------------------------------------

// out[t][n] = sum_k in[t][k] * weight[n][k], e4m3 inputs held exactly in
// float32, float32 accumulation in key order.
inline Status matmul_e4m3(const float* in, int tokens, int c_in,
                          const E4m3Matrix& weight, float* out) {
    if (in == nullptr || weight.bits == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || c_in <= 0 || weight.rows <= 0 || weight.cols != c_in
        || tokens > kVitMaxTokens || c_in > kVitMaxChannels
        || weight.rows > kVitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    // Token-parallel (native-parallel.hpp): each token's output rows keep
    // the single-threaded key-ordered accumulation exactly, so any thread
    // count is bit-identical to the serial reference. The AVX2 dot8 path
    // (native-matmul-simd.hpp) keeps that per-lane order exactly too; it is
    // only used for long keys (gather-bound below ~256).
    const bool simd = matmul_simd_available() && c_in >= 256;
    parallel_for(0, tokens, [&](int t) {
        const float* row = in + static_cast<std::size_t>(t) * c_in;
        float* dst = out + static_cast<std::size_t>(t) * weight.rows;
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
    });
    return Status::kOk;
}

// ---- FFN activation -------------------------------------------------------------------------------

// The reference's cubic SiLU: x * (-0.055908203125 * |clip| * clip
// + 0.447265625 * clip + 0.89453125) with clip = clamp(x, -4, 4).
inline float ffn_activation(float x) {
    const float clipped = x < -4.0f ? -4.0f : (x > 4.0f ? 4.0f : x);
    const float abs_c = clipped < 0.0f ? -clipped : clipped;
    return x * (-0.055908203125f * abs_c * clipped + 0.447265625f * clipped
                + 0.89453125f);
}

// ---- the network's exponential --------------------------------------------------------------------

// Round to nearest, ties to even, without depending on the ambient FP rounding
// mode (std::nearbyint would follow it).
inline double round_half_even(double v) {
    const double floor_v = std::floor(v);
    const double diff = v - floor_v;
    if (diff > 0.5) {
        return floor_v + 1.0;
    }
    if (diff < 0.5) {
        return floor_v;
    }
    return std::fmod(floor_v, 2.0) == 0.0 ? floor_v : floor_v + 1.0;
}

inline float approximate_exp(float logit) {
    const double x = static_cast<double>(vit_f16(logit));
    double y = round_half_even((x * static_cast<double>(kVitExpSlope)
                                + static_cast<double>(kVitExpOffset)) * 1024.0);
    // A NaN logit leaves y NaN; map it deterministically to the clamp floor so
    // the mantissa arithmetic below is always defined (the reference's
    // conversion there is garbage-but-defined too).
    if (!(y >= kVitExpClampLow)) {
        y = kVitExpClampLow;
    }
    if (y > kVitExpClampHigh) {
        y = kVitExpClampHigh;
    }
    const long long mantissa = static_cast<long long>(y) - 1024;
    const long long exponent = (mantissa >> 6) - 15;
    const float fraction = static_cast<float>(mantissa & 63) * 0x1p-6f;
    return std::ldexp(1.0f + fraction, static_cast<int>(exponent));
}

// ---- per-head inverse norm ------------------------------------------------------------------------

// 1 / max(sqrt(sum of squares), floor) over one 32-channel head of float16
// values. The squares and every sum round to float16 in the network's order:
// channel d with d + 16 first, then a butterfly over distances 8, 4, 2, 1.
inline float inverse_norm_head(const float* head_values) {
    float total[kVitHeadDim / 2];
    for (int c = 0; c < kVitHeadDim / 2; ++c) {
        const float a = vit_f16(head_values[c] * head_values[c]);
        const float b = vit_f16(head_values[c + kVitHeadDim / 2]
                                * head_values[c + kVitHeadDim / 2]);
        total[c] = vit_f16(a + b);
    }
    for (int half = kVitHeadDim / 4; half >= 1; half /= 2) {
        for (int c = 0; c < half; ++c) {
            total[c] = vit_f16(total[c] + total[c + half]);
        }
    }
    float magnitude = total[0];
    const float floor_value = vit_f16(kVitNormFloorF16);
    if (magnitude < floor_value) {
        magnitude = floor_value;
    }
    return vit_f16(1.0f / std::sqrt(magnitude));
}

// ---- QKV ------------------------------------------------------------------------------------------

// Query, key and value projections with per-head normalisation. Weights are
// e4m3 [channels][channels] for q, k and v; `temperature` is one float32
// value per head (rounded to float16 at use). Output [tokens][3*channels]:
// [q | k | v], every value rounded float32 -> float16 -> e4m3.
// (The reference's int4 hooks are wired in native-eval.hpp; only this post
// chain is shared with them.)
// The QKV post chain shared by the e4m3 and int4 projection paths: the
// per-head binary16 projection round, the inverse norm, the temperature and
// the e4m3 stores. `products` is [tokens, 3 * channels] raw projections (the
// q block, k block, v block per token; head-major rows within each block)
// and may alias `out`.
inline Status vit_qkv_post(const float* products, int tokens, int channels,
                           const float* temperature, float* out) {
    if (products == nullptr || temperature == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || channels <= 0 || channels % kVitHeadDim != 0
        || tokens > kVitMaxTokens || channels > kVitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    const int heads = channels / kVitHeadDim;
    const float sqrt_head = vit_f16(std::sqrt(static_cast<double>(kVitHeadDim)));
    for (int t = 0; t < tokens; ++t) {
        const float* prod = products + static_cast<std::size_t>(t) * 3
            * channels;
        float* dst = out + static_cast<std::size_t>(t) * 3 * channels;
        float projected_q[kVitHeadDim];
        float projected_k[kVitHeadDim];
        float projected_v[kVitHeadDim];
        for (int h = 0; h < heads; ++h) {
            const int base = h * kVitHeadDim;
            for (int d = 0; d < kVitHeadDim; ++d) {
                projected_q[d] = vit_f16(prod[base + d]);
                projected_k[d] = vit_f16(prod[channels + base + d]);
                projected_v[d] = vit_f16(prod[2 * channels + base + d]);
            }
            const float inv_q = inverse_norm_head(projected_q);
            const float inv_k = inverse_norm_head(projected_k);
            const float temp = vit_f16(temperature[h]);
            for (int d = 0; d < kVitHeadDim; ++d) {
                const float q = vit_f16(vit_f16(vit_f16(projected_q[d] * sqrt_head)
                                                * inv_q) * temp);
                const float k = vit_f16(vit_f16(projected_k[d] * inv_k));
                dst[base + d] = e4m3(q);
                dst[channels + base + d] = e4m3(k);
                dst[2 * channels + base + d] = e4m3(projected_v[d]);
            }
        }
    }
    return Status::kOk;
}

inline Status vit_qkv(const float* in, int tokens, int channels,
                      const E4m3Matrix& q_weight, const E4m3Matrix& k_weight,
                      const E4m3Matrix& v_weight, const float* temperature,
                      float* out) {
    if (in == nullptr || q_weight.bits == nullptr || k_weight.bits == nullptr
        || v_weight.bits == nullptr || temperature == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || channels <= 0 || channels % kVitHeadDim != 0
        || tokens > kVitMaxTokens || channels > kVitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    if (q_weight.rows != channels || q_weight.cols != channels
        || k_weight.rows != channels || k_weight.cols != channels
        || v_weight.rows != channels || v_weight.cols != channels) {
        return Status::kInvalidDimensions;
    }
    // Token-parallel (native-parallel.hpp): every projection keeps its
    // key-ordered accumulation exactly; bit-identical at any thread count.
    parallel_for(0, tokens, [&](int t) {
        const float* row = in + static_cast<std::size_t>(t) * channels;
        float* dst = out + static_cast<std::size_t>(t) * 3 * channels;
        // The raw float32 projections land in `out` first; the shared post
        // chain (vit_qkv_post) applies the reference's float16 round and the
        // per-head normalisation in place.
        for (int r = 0; r < 3 * channels; ++r) {
            const std::uint8_t* w = (r < channels ? q_weight.bits
                     : r < 2 * channels ? k_weight.bits : v_weight.bits)
                + static_cast<std::size_t>(r % channels) * channels;
            float acc = 0.0f;
            for (int k = 0; k < channels; ++k) {
                acc += row[k] * e4m3_from_bits(w[k]);
            }
            dst[r] = acc;
        }
    });
    return vit_qkv_post(out, tokens, channels, temperature, out);
}

// ---- attention ------------------------------------------------------------------------------------

// One 64-key group's probability sum in the network's order: key c is added to
// key c + 8 inside each 16-key block; the four blocks accumulate one after the
// other; the eight partials then add as ((p0+p2)+p4)+p6 and ((p1+p3)+p5)+p7,
// and the two results add. Every addition is a float16 addition.
inline float group_sum64(const float* probabilities) {
    float accumulated[8];
    for (int block = 0; block < 4; ++block) {
        for (int c = 0; c < 8; ++c) {
            const float pair = vit_f16(probabilities[block * 16 + c]
                                       + probabilities[block * 16 + 8 + c]);
            if (block == 0) {
                accumulated[c] = pair;
            } else {
                accumulated[c] = vit_f16(accumulated[c] + pair);
            }
        }
    }
    const float even = vit_f16(vit_f16(vit_f16(accumulated[0] + accumulated[2])
                                       + accumulated[4]) + accumulated[6]);
    const float odd = vit_f16(vit_f16(vit_f16(accumulated[1] + accumulated[3])
                                      + accumulated[5]) + accumulated[7]);
    return vit_f16(even + odd);
}

// Global multi-head attention over all tokens of the [tokens][3*channels] QKV
// tensor (e4m3 values). No weights. Keys are padded to a multiple of 64 with
// zero keys (logit 0, counted as exp(0) and subtracted at the end). Output
// [tokens][channels], every value rounded float32 -> float16 -> e4m3.
inline Status vit_attention(const float* qkv, int tokens, int channels,
                            float* out) {
    if (qkv == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || channels <= 0 || channels % kVitHeadDim != 0
        || tokens > kVitMaxTokens || channels > kVitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    const int heads = channels / kVitHeadDim;
    const int padded = ((tokens + kVitKeyGroup - 1) / kVitKeyGroup) * kVitKeyGroup;
    const float padding_keys = static_cast<float>(padded - tokens);
    const float padding = vit_f16(padding_keys * approximate_exp(0.0f));
    for (int h = 0; h < heads; ++h) {
        const int base = h * kVitHeadDim;
        for (int t = 0; t < tokens; ++t) {
            const float* q = qkv + static_cast<std::size_t>(t) * 3 * channels + base;
            // Probabilities of all keys (zero keys included), one group of 64
            // at a time; the numerator accumulates in float32.
            float probabilities[kVitKeyGroup];
            float numerator[kVitHeadDim] = {0.0f};
            float denominator = 0.0f;
            for (int j = 0; j < padded; ++j) {
                float logit = 0.0f;
                if (j < tokens) {
                    const float* k_row = qkv
                        + static_cast<std::size_t>(j) * 3 * channels + channels + base;
                    for (int d = 0; d < kVitHeadDim; ++d) {
                        logit += q[d] * k_row[d];
                    }
                }
                const float p = approximate_exp(logit);
                probabilities[j % kVitKeyGroup] = p;
                if (j < tokens) {
                    const float* v_row = qkv
                        + static_cast<std::size_t>(j) * 3 * channels
                        + 2 * channels + base;
                    const float p_quant = e4m3(p);
                    for (int d = 0; d < kVitHeadDim; ++d) {
                        numerator[d] += p_quant * v_row[d];
                    }
                }
                if (j % kVitKeyGroup == kVitKeyGroup - 1) {
                    // Groups add one after the other in float16.
                    denominator = vit_f16(denominator + group_sum64(probabilities));
                }
            }
            const float adjusted = vit_f16(denominator - padding);
            float magnitude = adjusted;
            if (magnitude < kVitDenominatorFloor) {
                magnitude = kVitDenominatorFloor;
            }
            const float inverse = vit_f16(1.0f / magnitude);
            float* dst = out + static_cast<std::size_t>(t) * channels + base;
            for (int d = 0; d < kVitHeadDim; ++d) {
                dst[d] = e4m3(vit_f16(vit_f16(numerator[d]) * inverse));
            }
        }
    }
    return Status::kOk;
}

// ---- projection executors --------------------------------------------------------------------------

// An external raw-projection executor: out[t*rows+n] = sum_k in[t*c_in+k] *
// w[n][k], bit-identical to matmul_e4m3's element order on normal-range
// intermediates (the Vulkan backend's documented tolerance applies). The
// kernels' post chains stay on the CPU, so executor results flow through the
// reference arithmetic unchanged. Null keeps the CPU kernel (the reference).
struct ProjectionExecutor {
    void* ctx = nullptr;
    // The float-weight variant (the swin family's decoded weights): raw
    // float32 rows [rows, cols], no e4m3 codes. Same product semantics
    // (bit-parity assumes grid-exact products and the normal-range
    // contract; see NATIVE-VULKAN.md).
    Status (*matmul_f32)(void* engine, const float* in, int tokens, int c_in,
                         const float* weights, int rows, float* out) = nullptr;
    Status (*matmul)(void* ctx, const float* in, int tokens, int c_in,
                     const E4m3Matrix& w, float* out) = nullptr;
};

// ---- shared post chains ----------------------------------------------------------------------------

// The per-element post chains, shared by the CPU kernels' inner loops and by
// any external projection executor (e.g. the Vulkan backend in
// native-vulkan.hpp): identical operations in identical order, so an
// executor's results flow through exactly the reference arithmetic.
inline float vit_ffn_expand_post(float product, float* stash_slot) {
    const float hidden = ffn_activation(product);
    if (stash_slot != nullptr) {
        *stash_slot = hidden;
    }
    return e4m3(vit_f16(hidden));
}

inline float vit_residual_post(float product, float skip_value,
                               float gain_value, float* stash_slot) {
    // The binary16 value is the pinned's stash (`int4.side`); the store
    // rounds it to e4m3. The gain takes its own binary16 hop first.
    const float value = vit_f16(product + vit_f16(gain_value) * skip_value);
    if (stash_slot != nullptr) {
        *stash_slot = value;
    }
    return e4m3(value);
}

// ---- the layer kinds ------------------------------------------------------------------------------

// VitFfnExpand: 1024 -> 4096 with the cubic activation, no residual.
// out = e4m3(f16(activation(matmul(in)))).
// `stash`, when non-null, receives the UNROUNDED activation (the pinned
// `int4.side[("hidden", block)]`: "the activation stays unrounded for the
// contraction's int4 channels, which quantise it themselves") alongside the
// `e4m3(f16(...))` store in `out`.
inline Status vit_ffn_expand(const float* in, int tokens, int c_in, int c_out,
                             const E4m3Matrix& weight, float* out,
                             float* stash = nullptr) {
    if (in == nullptr || weight.bits == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || c_in <= 0 || c_out <= 0 || weight.rows != c_out
        || weight.cols != c_in || tokens > kVitMaxTokens || c_in > kVitMaxChannels
        || c_out > kVitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    // Token-parallel (native-parallel.hpp): every element keeps its
    // key-ordered accumulation exactly; bit-identical at any thread count.
    parallel_for(0, tokens, [&](int t) {
        const float* row = in + static_cast<std::size_t>(t) * c_in;
        float* dst = out + static_cast<std::size_t>(t) * c_out;
        for (int n = 0; n < c_out; ++n) {
            const std::uint8_t* w = weight.bits + static_cast<std::size_t>(n) * c_in;
            float acc = 0.0f;
            for (int k = 0; k < c_in; ++k) {
                acc += row[k] * e4m3_from_bits(w[k]);
            }
            dst[n] = vit_ffn_expand_post(
                acc, stash != nullptr
                         ? &stash[static_cast<std::size_t>(t) * c_out + n]
                         : nullptr);
        }
    });
    return Status::kOk;
}

// VitFfnContract / VitProjection: c_out -> c_out plus gain * skip, where the
// gain is per output channel (float16) and the skip is the block's earlier
// stream. value = f16(product + gain * skip); out = e4m3(value).
inline Status vit_residual_linear(const float* in, const float* skip, int tokens,
                                  int c_in, int c_out,
                                  const E4m3Matrix& weight, const float* gain,
                                  float* out, float* stash = nullptr) {
    if (in == nullptr || skip == nullptr || weight.bits == nullptr
        || gain == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || c_in <= 0 || c_out <= 0 || weight.rows != c_out
        || weight.cols != c_in || tokens > kVitMaxTokens || c_in > kVitMaxChannels
        || c_out > kVitMaxChannels) {
        return Status::kInvalidDimensions;
    }
    // Token-parallel (native-parallel.hpp): as above, exact parity.
    parallel_for(0, tokens, [&](int t) {
        const float* row = in + static_cast<std::size_t>(t) * c_in;
        const float* skip_row = skip + static_cast<std::size_t>(t) * c_out;
        float* dst = out + static_cast<std::size_t>(t) * c_out;
        for (int n = 0; n < c_out; ++n) {
            const std::uint8_t* w = weight.bits + static_cast<std::size_t>(n) * c_in;
            float acc = 0.0f;
            for (int k = 0; k < c_in; ++k) {
                acc += row[k] * e4m3_from_bits(w[k]);
            }
            dst[n] = vit_residual_post(
                acc, skip_row[n], gain[n],
                stash != nullptr
                    ? &stash[static_cast<std::size_t>(t) * c_out + n]
                    : nullptr);
        }
    });
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // VR_NATIVE_VIT_HPP
