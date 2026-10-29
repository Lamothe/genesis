// SPDX-License-Identifier: MIT
// Native CPU residual block core primitives (foundation slice for the
// remaining layer families and full network assembly).
//
// Typed, deterministic, allocation-free-in-the-core helpers shared by the
// residual blocks of the pinned DLSSnR-style generator:
//   - `LayerNorm`          normalisation over the channel axis (token-major).
//   - `conv2d_kxk`         k*k convolution by explicit im2col + GEMM, with
//                          `group == in_channels` depthwise specialisation.
//   - `pixel_unshuffle`    space-to-depth (stride-4 patchification inverse).
//   - `pixel_shuffle`      depth-to-space (the output boundary unfold).
//   - `ConvNormAct`        conv + norm + GELU activation chain semantics.
//   - `residual_split` / `residual_join`  additive residual wiring with
//                          optional channel projection for the join.
//
// The earlier Swin slices (`native-swin-linear/attention/block/...`) implement
// the windowed-transformer families; this slice adds the convolutional and
// normalisation core the conv layers and the pixel-shuffle boundary blocks
// need. Like every slice in this campaign it is a clean CPU-only C++
// re-derivation of published open-source semantics (pinned MIT reference,
// see vr/NATIVE-GRAPH-PLAN.md) — not vendor code — and performs no
// proprietary routing, addressing or probing.
//
// Layout conventions (shared with the Swin slices):
//   - activation tokens are token-major: [token][channel].
//   - weights are `[out_channel][in_channel*k*k]` row-major.
//   - the layer norm output tensor is channel-major: [channel][token].

#ifndef VR_NATIVE_BLOCK_HPP
#define VR_NATIVE_BLOCK_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <new>

namespace vr::native_block {

// ---------------------------------------------------------------------------
// float16 helpers (identical semantics to the Swin slices).
// ---------------------------------------------------------------------------

struct F16 {
    std::uint16_t bits;
};

inline constexpr F16 f16_from_bits(std::uint16_t b) {
    return F16{b};
}

inline float f16_to_f32(F16 h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h.bits & 0x8000u) << 16;
    const std::uint32_t exp = static_cast<std::uint32_t>(h.bits & 0x7C00u) >> 10;
    const std::uint32_t man = static_cast<std::uint32_t>(h.bits & 0x03FFu);
    if (exp == 0x1Fu) {
        const std::uint32_t f = sign | 0x7F800000u | (man << 13);
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }
    if (exp == 0u) {
        if (man == 0u) {
            const std::uint32_t f = sign;
            float out;
            std::memcpy(&out, &f, sizeof(out));
            return out;
        }
        std::uint32_t m = man;
        std::uint32_t e = 0u;
        while ((m & 0x0400u) == 0u) {
            m <<= 1;
            ++e;
        }
        m &= 0x03FFu;
        const std::uint32_t f = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }
    const std::uint32_t f = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline F16 f32_to_f16(float v) {
    std::uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    const std::int32_t exp = static_cast<std::int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    const std::uint32_t man = f & 0x7FFFFFu;
    if (((f >> 23) & 0xFFu) == 0xFFu) {
        return f16_from_bits(static_cast<std::uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u)));
    }
    if (exp >= 0x1F) {
        return f16_from_bits(static_cast<std::uint16_t>(sign | 0x7C00u));
    }
    if (exp <= 0) {
        if (exp < -10) {
            return f16_from_bits(static_cast<std::uint16_t>(sign));
        }
        const std::uint32_t m = (man | 0x800000u) >> (14 - exp);
        return f16_from_bits(static_cast<std::uint16_t>(sign | (m >> 13)));
    }
    return f16_from_bits(static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (man >> 13)));
}

// ---------------------------------------------------------------------------
// LayerNorm over the channel axis.
//
// Token-major input [tokens][channels]; channel-major output [channels][tokens]
// (the layout the Swin MLP family consumes). gamma/beta are per-channel
// (identity/zero when null). Variance is the biased (population) variance with
// the pinned reference's epsilon.
// ---------------------------------------------------------------------------

constexpr float kNormEps = 1e-5f;

inline void layer_norm(const float* input,
                       std::size_t tokens,
                       std::size_t channels,
                       const float* gamma,
                       const float* beta,
                       float* output_c_major,
                       bool relu_half = false) {
    for (std::size_t t = 0; t < tokens; ++t) {
        const float* row = input + t * channels;
        float mean = 0.0f;
        for (std::size_t c = 0; c < channels; ++c) {
            mean += row[c];
        }
        mean /= static_cast<float>(channels);
        float var = 0.0f;
        for (std::size_t c = 0; c < channels; ++c) {
            const float d = row[c] - mean;
            var += d * d;
        }
        var /= static_cast<float>(channels);
        const float inv = 1.0f / std::sqrt(var + kNormEps);
        for (std::size_t c = 0; c < channels; ++c) {
            float v = (row[c] - mean) * inv;
            if (gamma != nullptr) {
                v *= gamma[c];
            }
            if (beta != nullptr) {
                v += beta[c];
            }
            if (relu_half) {
                v = v > 0.0f ? v : 0.0f;
            }
            output_c_major[c * tokens + t] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// pixel_unshuffle: space-to-depth (exact inverse of `pixel_shuffle` above).
// Input tokens [tokens][in_ch] with the given (w, h) extent; output tokens
// [tokens/(sx*sy)][in_ch*sx*sy]: output token (ox, oy) channel
// c*(sx*sy) + i*sx + j reads input channel c at token (ox*sx + j, oy*sy + i).
// ---------------------------------------------------------------------------

inline void pixel_unshuffle(const float* input,
                            std::size_t w,
                            std::size_t h,
                            std::size_t channels,
                            std::size_t sx,
                            std::size_t sy,
                            float* output) {
    const std::size_t ow = w / sx;
    const std::size_t oh = h / sy;
    for (std::size_t oy = 0; oy < oh; ++oy) {
        for (std::size_t ox = 0; ox < ow; ++ox) {
            float* out = output + ((oy * ow + ox) * (channels * sx * sy));
            for (std::size_t c = 0; c < channels; ++c) {
                for (std::size_t fy = 0; fy < sy; ++fy) {
                    for (std::size_t fx = 0; fx < sx; ++fx) {
                        const std::size_t ix = ox * sx + fx;
                        const std::size_t iy = oy * sy + fy;
                        out[c * (sx * sy) + fy * sx + fx] =
                            input[(iy * w + ix) * channels + c];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pixel_shuffle: depth-to-space (the output boundary unfold), per PyTorch
// nn.PixelShuffle: output token (x', y') channel c reads input channel
// c*(sx*sy) + i*sx + j at token (x'/sx, y'/sy), with j = x' % sx, i = y' % sy.
// ---------------------------------------------------------------------------

inline void pixel_shuffle(const float* input,
                          std::size_t w,
                          std::size_t h,
                          std::size_t channels,
                          std::size_t sx,
                          std::size_t sy,
                          float* output) {
    const std::size_t ow = w * sx;
    const std::size_t oh = h * sy;
    const std::size_t in_c = channels / (sx * sy);
    for (std::size_t oy = 0; oy < oh; ++oy) {
        for (std::size_t ox = 0; ox < ow; ++ox) {
            const std::size_t ix = ox / sx;
            const std::size_t iy = oy / sy;
            const std::size_t fx = ox % sx;
            const std::size_t fy = oy % sy;
            const float* in = input + (iy * w + ix) * channels;
            float* out = output + (oy * ow + ox) * in_c;
            for (std::size_t c = 0; c < in_c; ++c) {
                out[c] = in[c * (sx * sy) + fy * sx + fx];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// k*k convolution via explicit im2col + GEMM.
//
// Input tokens [tokens][in_channels] at extent (w, h); weights
// [out_channels][in_channels*group_ratio*k*k] row-major (group == in_channels
// selects the depthwise path where weights are [out=in][k*k]); bias optional.
// Padding is symmetric `pad`; stride is 1 (the plan's Conv rows keep the
// extent — downsampling appears as explicit strided folds elsewhere).
// Output [out_tokens][out_channels].
// ---------------------------------------------------------------------------

inline void conv2d_kxk(const float* input,
                       std::size_t w,
                       std::size_t h,
                       std::size_t in_channels,
                       std::size_t out_channels,
                       std::size_t k,
                       std::size_t pad,
                       std::size_t group,
                       const float* weights,
                       const float* bias,
                       float* output,
                       float* im2col_scratch) {
    const std::size_t ow = w;
    const std::size_t oh = h;
    const std::size_t per_group_in = in_channels / group;
    const std::size_t per_group_out = out_channels / group;
    const std::size_t window = per_group_in * k * k;

    if (group == in_channels && per_group_out == 1) {
        // Depthwise: one output channel per input channel, k*k taps each.
        for (std::size_t y = 0; y < oh; ++y) {
            for (std::size_t x = 0; x < ow; ++x) {
                float* out = output + (y * ow + x) * out_channels;
                for (std::size_t c = 0; c < in_channels; ++c) {
                    float acc = bias != nullptr ? bias[c] : 0.0f;
                    const float* wgt = weights + c * (k * k);
                    std::size_t tap = 0;
                    for (std::size_t ky = 0; ky < k; ++ky) {
                        for (std::size_t kx = 0; kx < k; ++kx, ++tap) {
                            const long ix = static_cast<long>(x + kx) - static_cast<long>(pad);
                            const long iy = static_cast<long>(y + ky) - static_cast<long>(pad);
                            if (ix < 0 || iy < 0 || ix >= static_cast<long>(w) ||
                                iy >= static_cast<long>(h)) {
                                continue;
                            }
                            acc += wgt[tap] *
                                   input[(static_cast<std::size_t>(iy) * w +
                                          static_cast<std::size_t>(ix)) * in_channels + c];
                        }
                    }
                    out[c] = acc;
                }
            }
        }
        return;
    }

    // Grouped convolution by im2col + per-group GEMM.
    for (std::size_t y = 0; y < oh; ++y) {
        for (std::size_t x = 0; x < ow; ++x) {
            float* patch = im2col_scratch;
            for (std::size_t g = 0; g < group; ++g) {
                float* col = patch;
                for (std::size_t ic = 0; ic < per_group_in; ++ic) {
                    const std::size_t ch = g * per_group_in + ic;
                    for (std::size_t ky = 0; ky < k; ++ky) {
                        for (std::size_t kx = 0; kx < k; ++kx) {
                            const long ix = static_cast<long>(x + kx) - static_cast<long>(pad);
                            const long iy = static_cast<long>(y + ky) - static_cast<long>(pad);
                            float v = 0.0f;
                            if (ix >= 0 && iy >= 0 && ix < static_cast<long>(w) &&
                                iy < static_cast<long>(h)) {
                                v = input[(static_cast<std::size_t>(iy) * w +
                                           static_cast<std::size_t>(ix)) * in_channels + ch];
                            }
                            *col++ = v;
                        }
                    }
                }
                (void)0;
                float* out = output + (y * ow + x) * out_channels + g * per_group_out;
                for (std::size_t oc = 0; oc < per_group_out; ++oc) {
                    const float* wgt =
                        weights + (g * per_group_out + oc) * window;
                    float acc = bias != nullptr ? bias[g * per_group_out + oc] : 0.0f;
                    for (std::size_t i = 0; i < window; ++i) {
                        acc += wgt[i] * patch[i];
                    }
                    out[oc] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GELU (exact erf formulation, matching the Swin MLP family).
// ---------------------------------------------------------------------------

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

inline void gelu_apply(const float* input, float* output, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = gelu(input[i]);
    }
}

// ---------------------------------------------------------------------------
// ConvNormAct: conv + norm + activation chain semantics. Norm is applied
// channel-wise over the output tokens (LayerNorm over channels), then GELU.
// `output_c_major` receives the normalised+activated values channel-major when
// non-null (the layout the Swin MLP consumes); `output_token_major` receives
// token-major values when non-null.
// ---------------------------------------------------------------------------

inline void conv_norm_act(const float* input,
                          std::size_t w,
                          std::size_t h,
                          std::size_t in_channels,
                          std::size_t out_channels,
                          std::size_t k,
                          std::size_t pad,
                          std::size_t group,
                          const float* weights,
                          const float* bias,
                          const float* gamma,
                          const float* beta,
                          float* im2col_scratch,
                          float* conv_scratch,
                          float* norm_scratch,
                          float* output_c_major,
                          float* output_token_major) {
    conv2d_kxk(input, w, h, in_channels, out_channels, k, pad, group, weights,
               bias, conv_scratch, im2col_scratch);
    const std::size_t tokens = w * h;
    layer_norm(conv_scratch, tokens, out_channels, gamma, beta, norm_scratch);
    for (std::size_t c = 0; c < out_channels; ++c) {
        for (std::size_t t = 0; t < tokens; ++t) {
            const float v = gelu(norm_scratch[c * tokens + t]);
            if (output_c_major != nullptr) {
                output_c_major[c * tokens + t] = v;
            }
            if (output_token_major != nullptr) {
                output_token_major[t * out_channels + c] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Residual split / join.
//
// `residual_split` copies the identity stream for a later join (optionally
// projecting channels with a 1*1 weight set). `residual_join` adds the
// transformed stream back to the stored identity, projecting the identity when
// the channels differ.
// ---------------------------------------------------------------------------

inline void residual_split(const float* input,
                           std::size_t tokens,
                           std::size_t channels,
                           float* identity_out) {
    std::memcpy(identity_out, input, tokens * channels * sizeof(float));
}

inline void residual_project(const float* input,
                             std::size_t tokens,
                             std::size_t in_channels,
                             std::size_t out_channels,
                             const float* weights,
                             float* output) {
    for (std::size_t t = 0; t < tokens; ++t) {
        const float* in = input + t * in_channels;
        float* out = output + t * out_channels;
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            const float* wgt = weights + oc * in_channels;
            float acc = 0.0f;
            for (std::size_t i = 0; i < in_channels; ++i) {
                acc += wgt[i] * in[i];
            }
            out[oc] = acc;
        }
    }
}

inline void residual_join(const float* identity,
                          const float* branch,
                          std::size_t tokens,
                          std::size_t channels,
                          float* output) {
    const std::size_t n = tokens * channels;
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = identity[i] + branch[i];
    }
}

}  // namespace vr::native_block

#endif  // VR_NATIVE_BLOCK_HPP
