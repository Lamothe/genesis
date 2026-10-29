// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Native Linux DLSS-style neural rendering — the int4 mixed-network operand
// path.
//
// The int4 mixed network's semantics are re-derived from the MIT reference
// mauri870/dlssnr-pytorch 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/int4.py), itself following DLSSNR-AMD's i4_obtain and shaders
// (MIT, Copyright (c) 2026 mochizuki0323). The int4 weights are GPTQ
// solutions that cannot ship: what ships are recovery tables (.fix/.mu/.aq)
// holding no weight value. Those tables and their file decode remain the
// host's weight-file boundary (the same declared boundary as native-eval.hpp's
// `ffwd` matrices); this header takes the decoded records and runs them:
//
//   y = s * (q . x_int4) + c        x_int4 = quantise(x)
//   quantise:  code = clamp(rint(H(x) * gain + zero), 0, 15) - 8
//
// where H is the unnormalised Walsh-Hadamard of each `block` consecutive
// values (the shaders' ascending butterfly), gain = (1/step) * (1/sqrt(block))
// in float32 (the half form keeps 1/step alone and rounds it to binary16),
// and zero is the zero point rounded to an integer by the 1.5 * 2^23 trick
// (ties to even for |value| < 2^22; the half form keeps its binary16 rounding
// instead). The rint is round-half-to-even of the fused product-plus-sum
// taken exactly in float64 (the pinned form). The accumulator's integer dot
// is exact in float32 in any order when |q * x| <= 1016 and cols <= 2^14
// (partial sums stay below 2^24 — the bound enforced below; the pinned's q
// is int8 after its escape corrections wrap); the affine is evaluated in
// float64 and stored once as float32 (the half form rounds s and c through
// binary16 first).
//
// The mixed ViT contraction adds the int4 channels' product (from the
// unrounded activation) to the e4m3 tail's product (exact e4m3 values),
// summed in float32. The Swin sites' hard-swish is x * clamp(fma(x, A, B),
// 0, 1) with (A, B) = (0.187367, 0.500139) at C=32 and (0.182412, 0.500465)
// wide; the scale folded into the weights that read the hidden follows the
// settings chain NR_HS_FOLDPP (blocks 0 and 70) / NR_HS_FOLD32 (C=32) /
// NR_HS_FOLD (wide), each level defaulting to the wide value below it,
// gated by NR_HS_FOLD_BLOCKS.
//
// Allocation-free; validation completes before any output write. CPU-only;
// no GPU, headset, proprietary payload, Python or PyTorch runtime required.
// No vendor code and no vendor correspondence.
#ifndef DLSS5_NR_NATIVE_INT4_HPP
#define DLSS5_NR_NATIVE_INT4_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "native-pre-post.hpp"

namespace dlss5_nr {

inline constexpr float kInt4Magic = 12582912.0f;  // 1.5 * 2^23

// One decoded int4 matrix with its input quantiser (the host's weight-file
// boundary record, like `ffwd`): q is [rows, cols] int8 in [-8, 7], scale and
// constant [rows] float32, aq [cols, 2] float32 (1/step, zero), block the
// Hadamard block size (1 for the Swin sites), half the Swin sites' binary16
// quantiser/affine form.
struct Int4MatrixView {
    const std::int8_t* q;
    const float* scale;
    const float* constant;
    const float* aq;
    int rows;
    int cols;
    int block;
    bool half;
};

// The per-site record for the network miring (the host's weight-file
// boundary): the decoded int4 matrix plus, for the ViT feed-forward site, the
// hidden permutation (perm_count = matrix.cols + the tail's columns; the
// first matrix.cols entries order the int4 channels, the rest order the e4m3
// tail channels). The tail matrix itself arrives through the eval records'
// `e4m3` callback as "tail" ([rows, tail_cols]).
struct Int4Site {
    Int4MatrixView matrix;
    const int* perm = nullptr;
    int perm_count = 0;
    // The vit FFN's expansion matrix (the pinned `VitFfn.expand`, block 16
    // with the same permutation), present when the tables have it; the
    // expansion reads the ("stream", block) side stash through it.
    Int4MatrixView ex;
    bool has_ex = false;
};

// The bounds keep every index and accumulation in range: cols <= 2^14 makes
// the integer dot exact for int8 q and |x| <= 8 (partial sums <= 2^24).
inline constexpr int kInt4MaxRows = 16384;
inline constexpr int kInt4MaxCols = 16384;

namespace int4_detail {

inline Status validate_matrix(const Int4MatrixView& matrix) {
    if (matrix.q == nullptr || matrix.scale == nullptr
        || matrix.constant == nullptr || matrix.aq == nullptr) {
        return Status::kNullPointer;
    }
    if (matrix.rows < 1 || matrix.cols < 1 || matrix.rows > kInt4MaxRows
        || matrix.cols > kInt4MaxCols) {
        return Status::kInvalidDimensions;
    }
    if (matrix.block < 1 || matrix.block > matrix.cols
        || matrix.cols % matrix.block != 0) {
        return Status::kInvalidValues;
    }
    return Status::kOk;
}

// The 1.5 * 2^23 trick: adding the magic to a float32 rounds the value to an
// integer (ties to even, |value| < 2^22; beyond that the sum's ulp grows past
// 1 and the trick degrades — same expression as the pinned, documented here).
inline float round_to_integer(float value) {
    return (value + kInt4Magic) - kInt4Magic;
}

// Round-half-to-even of a finite double, independent of the ambient rounding
// mode (std::rint honours the mode). Valid for |value| < 2^31.
inline long round_half_even(double value) {
    const double below = std::floor(value);
    const double frac = value - below;
    long whole = static_cast<long>(below);
    if (frac > 0.5) {
        whole += 1;
    } else if (frac == 0.5 && (whole % 2 != 0)) {
        whole += 1;
    }
    return whole;
}

}  // namespace int4_detail

// The unnormalised Walsh-Hadamard transform of every `block` consecutive
// values (float32, the butterfly stages in ascending order — the shaders'
// order). in/out may be the same buffer; `count` values total
// (`count % block == 0`).
inline Status int4_fwht(const float* in, float* out, int count, int block) {
    if (in == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (count < 1 || block < 1 || block > count || count % block != 0) {
        return Status::kInvalidDimensions;
    }
    // memmove: partial in/out overlap is safe, exact aliasing skipped.
    if (in != out) {
        std::memmove(out, in, static_cast<std::size_t>(count) * sizeof(float));
    }
    const int groups = count / block;
    for (int h = 1; h < block; h *= 2) {
        for (int g = 0; g < groups; ++g) {
            float* x = out + static_cast<std::size_t>(g) * block;
            for (int base = 0; base < block; base += 2 * h) {
                for (int j = 0; j < h; ++j) {
                    const float a = x[base + j];
                    const float b = x[base + h + j];
                    x[base + j] = a + b;
                    x[base + h + j] = a - b;
                }
            }
        }
    }
    return Status::kOk;
}

// The quantiser parameters as Int4Matrix derives them: gain = (1/step) *
// (1/sqrt(block)) in float32 and the magic-rounded zero point; the half form
// keeps 1/step and zero at their binary16 roundings (and is only paired with
// block 1 in the pinned network). `gain`/`zero` take `cols` values each.
inline Status int4_quantiser_params(const Int4MatrixView& matrix,
                                    float* gain, float* zero) {
    const Status status = int4_detail::validate_matrix(matrix);
    if (status != Status::kOk) {
        return status;
    }
    if (gain == nullptr || zero == nullptr) {
        return Status::kNullPointer;
    }
    const float block_gain = matrix.block > 1
        ? 1.0f / std::sqrt(static_cast<float>(matrix.block)) : 1.0f;
    for (int k = 0; k < matrix.cols; ++k) {
        const float inverse = matrix.aq[2 * k];
        const float point = matrix.aq[2 * k + 1];
        if (matrix.half) {
            gain[k] = f16(inverse);
            zero[k] = f16(point);
        } else {
            gain[k] = inverse * block_gain;
            zero[k] = int4_detail::round_to_integer(point);
        }
    }
    return Status::kOk;
}

// quantise: value [tokens, cols] float32 -> codes [tokens, cols] float32
// integers in [-8, 7]:
//     code = clamp(rint(H(x) * gain + zero), 0, 15) - 8
// with the round-half-to-even of the fused product-plus-sum taken exactly.
// Validation completes before any output write.
inline Status int4_quantise(const Int4MatrixView& matrix, const float* value,
                            int tokens, float* codes) {
    Status status = int4_detail::validate_matrix(matrix);
    if (status != Status::kOk) {
        return status;
    }
    if (value == nullptr || codes == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1) {
        return Status::kInvalidDimensions;
    }
    const float block_gain = matrix.block > 1
        ? 1.0f / std::sqrt(static_cast<float>(matrix.block)) : 1.0f;
    for (int t = 0; t < tokens; ++t) {
        const float* row = value + static_cast<std::size_t>(t) * matrix.cols;
        float* out = codes + static_cast<std::size_t>(t) * matrix.cols;
        status = int4_fwht(row, out, matrix.cols, matrix.block);
        if (status != Status::kOk) {
            return status;
        }
        for (int k = 0; k < matrix.cols; ++k) {
            const float inverse = matrix.aq[2 * k];
            const float point = matrix.aq[2 * k + 1];
            const float a = matrix.half ? f16(inverse) : inverse * block_gain;
            const float z = matrix.half ? f16(point)
                : int4_detail::round_to_integer(point);
            const double exact = static_cast<double>(out[k]) * a + z;
            long code;
            if (!(exact > 0.0)) {          // also maps NaN to code 0
                code = 0;
            } else if (exact >= 15.0) {
                code = 15;
            } else {
                code = int4_detail::round_half_even(exact);
            }
            out[k] = static_cast<float>(code - 8);
        }
    }
    return Status::kOk;
}

// accumulate: codes [tokens, cols] float32 integers in [-8, 7] ->
// y [tokens, rows] = f32(f64(s * (q . codes) + c)): the integer dot is exact
// in float32 in any order; the affine is evaluated in float64 and stored
// once (the half form rounds s and c through binary16 first). Validation
// completes before any output write.
inline Status int4_accumulate(const Int4MatrixView& matrix, const float* codes,
                              int tokens, float* out) {
    Status status = int4_detail::validate_matrix(matrix);
    if (status != Status::kOk) {
        return status;
    }
    if (codes == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1) {
        return Status::kInvalidDimensions;
    }
    for (int t = 0; t < tokens; ++t) {
        const float* x = codes + static_cast<std::size_t>(t) * matrix.cols;
        for (int n = 0; n < matrix.rows; ++n) {
            float total = 0.0f;
            for (int k = 0; k < matrix.cols; ++k) {
                total += static_cast<float>(
                             matrix.q[static_cast<std::size_t>(n) * matrix.cols
                                      + k])
                    * x[k];
            }
            const float s = matrix.half ? f16(matrix.scale[n])
                : matrix.scale[n];
            const float c = matrix.half ? f16(matrix.constant[n])
                : matrix.constant[n];
            out[static_cast<std::size_t>(t) * matrix.rows + n] =
                static_cast<float>(static_cast<double>(total) * s + c);
        }
    }
    return Status::kOk;
}

// The mixed ViT contraction (VitFfn.contraction): the int4 channels' product
// from `codes` (the quantiser's output over hidden[:, perm[:cols]], from
// int4_quantise) plus the e4m3 tail's product from `hidden_e4m3`, summed in
// float32. `perm` maps the hidden order (perm[0..cols) the int4 channels,
// perm[cols..cols + tail_cols) the tail channels); hidden_e4m3 is
// [tokens, cols + tail_cols] (the exact e4m3 values in the hidden order) and
// tail is [rows, tail_cols] float32. out is [tokens, rows]; the tail product
// accumulates left to right in float32 (the native deterministic contract;
// the pinned's matmul order is unspecified) and is added to the int4 affine
// in one float32 operation, as the pinned's `total + tail_product`.
inline Status int4_mixed_contraction(const Int4MatrixView& matrix,
                                     const float* codes,
                                     const float* tail, int tail_cols,
                                     const float* hidden_e4m3,
                                     const int* perm, int tokens, float* out) {
    Status status = int4_detail::validate_matrix(matrix);
    if (status != Status::kOk) {
        return status;
    }
    if (codes == nullptr || tail == nullptr || hidden_e4m3 == nullptr
        || perm == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || tail_cols < 1 || tail_cols > kInt4MaxCols) {
        return Status::kInvalidDimensions;
    }
    for (int t = 0; t < tokens; ++t) {
        const float* x = codes + static_cast<std::size_t>(t) * matrix.cols;
        const float* e4m3_row = hidden_e4m3
            + static_cast<std::size_t>(t) * (matrix.cols + tail_cols);
        for (int n = 0; n < matrix.rows; ++n) {
            float total = 0.0f;
            for (int k = 0; k < matrix.cols; ++k) {
                total += static_cast<float>(
                             matrix.q[static_cast<std::size_t>(n) * matrix.cols
                                      + k])
                    * x[k];
            }
            const float s = matrix.half ? f16(matrix.scale[n])
                : matrix.scale[n];
            const float c = matrix.half ? f16(matrix.constant[n])
                : matrix.constant[n];
            const float affine = static_cast<float>(
                static_cast<double>(total) * s + c);
            float tail_acc = 0.0f;
            for (int j = 0; j < tail_cols; ++j) {
                tail_acc += e4m3_row[perm[matrix.cols + j]]
                    * tail[static_cast<std::size_t>(n) * tail_cols + j];
            }
            out[static_cast<std::size_t>(t) * matrix.rows + n] = affine
                + tail_acc;
        }
    }
    return Status::kOk;
}

// The Swin sites' hard-swish: x * gate with gate = clamp(f32(f64(x * A) + B),
// 0, 1) — the pinned form. The double product of two binary32 operands is
// exact; the sum is exact whenever the gate does not saturate, so after the
// clamp this equals the single-rounding fma form. A/B come from the pinned
// tables: (0.187367, 0.500139) at C=32, (0.182412, 0.500465) wide.
// The scalar form with explicit parameters (the pinned `hard_swish`):
// ``x * clamp(fma(x, A, B), 0, 1)`` — the product and sum of the fma rounded
// once (binary64 then binary32), as the pinned's double->float round.
inline float hard_swish_value(float x, float a, float b) {
    float gate = static_cast<float>(static_cast<double>(x) * a
                                    + static_cast<double>(b));
    gate = gate < 0.0f ? 0.0f : (gate > 1.0f ? 1.0f : gate);
    return x * gate;
}

inline Status hard_swish(const float* in, int count, int channels,
                         float* out) {
    if (in == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (count < 1 || channels < 1) {
        return Status::kInvalidDimensions;
    }
    const bool narrow = channels == 32;
    const float a = narrow ? 0.187367f : 0.182412f;
    const float b = narrow ? 0.500139f : 0.500465f;
    for (int i = 0; i < count; ++i) {
        out[i] = hard_swish_value(in[i], a, b);
    }
    return Status::kOk;
}

// The Swin site's fold constant (SwinSite.__init__): the settings chain
// NR_HS_FOLDPP (the pre and post blocks 0/70) / NR_HS_FOLD32 (C=32) /
// NR_HS_FOLD (wide); narrow defaults to wide and first/last to narrow,
// gated by NR_HS_FOLD_BLOCKS (an ascending block list keeps the fold on
// those blocks only; the host maps a setting that is absent OR EMPTY to
// fold_block_count -1, which keeps it everywhere, as the pinned parsing
// does). A null fold_blocks with a positive count is treated as absent.
struct SwinFoldSettings {
    float wide = 0.0f;
    float narrow = 0.0f;
    float first_last = 0.0f;
    bool has_narrow = false;
    bool has_first_last = false;
    const int* fold_blocks = nullptr;  // NR_HS_FOLD_BLOCKS, ascending
    int fold_block_count = -1;         // -1: the setting was absent
};

inline float swin_fold_constant(const SwinFoldSettings& settings, int block,
                                int channels) {
    const float wide = settings.wide;
    const float narrow = settings.has_narrow ? settings.narrow : wide;
    const float first_last = settings.has_first_last ? settings.first_last
                                                     : narrow;
    float fold = block == 0 || block == 70
        ? first_last
        : (channels == 32 ? narrow : wide);
    if (settings.fold_block_count >= 0 && settings.fold_blocks != nullptr) {
        bool listed = false;
        for (int i = 0; i < settings.fold_block_count; ++i) {
            if (settings.fold_blocks[i] == block) {
                listed = true;
                break;
            }
        }
        if (!listed) {
            fold = 0.0f;
        }
    }
    return fold;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_INT4_HPP
