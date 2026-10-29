// Native Swin windowed attention: the `_attention` body of the Swin blocks on the
// e4m3 grid — position-bias deswizzle (`_deswizzle_bias`) and its exponent ramp,
// QKV normalisation (`_normalise_qkv`, with the head scale), the attention logits
// (`einsum("ntgd,nsgd->ngts")`), the exponent-ramp softmax (`_softmax_numerators`,
// `_denominator`, `_softmax`), the context (`einsum("ngts,nsgd->ntgd")` through
// `store`) and the residual output line. CPU-only, typed, allocation-free; no GPU,
// headset, proprietary payload, Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "_deswizzle_bias"/"_normalise_qkv"/"_softmax_numerators"/
// "_denominator"/"_softmax"/"_attention"). The full MIT licence notice is retained
// in vr/NATIVE-SWIN-ATTENTION.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_ATTENTION_HPP
#define DLSS5_NR_NATIVE_SWIN_ATTENTION_HPP

#include "native-swin-mlp.hpp"

namespace dlss5_nr {

// ---- documented constants ----------------------------------------------------------------------
inline constexpr int kWindowTokens = 64;               // TOKENS = WINDOW * WINDOW
inline constexpr int kBiasTable = kWindowTokens * kWindowTokens;  // 64x64 bias per head
// EXP_SLOPE / EXP_OFFSET: the affine exponent ramp (23 / 512 and 1 + 77 / 256).
inline constexpr float kExpSlope = 0.044921875f;
inline constexpr float kExpOffset = 1.30078125f;
// EXP_LOW / EXP_HIGH: the clamp of the ramp before the exponent transplant
// (1 + 32 / 1024 and 1 + 583 / 1024; both exactly binary16-representable).
inline constexpr float kExpLow = 1.03125f;
inline constexpr float kExpHigh = 1.5693359375f;
// SUM_FLOOR: lower bound of the softmax denominator (2^-14).
inline constexpr float kSumFloor = 0x1p-14f;
inline constexpr int kMaxHeads = kMaxLinearDim / kHeadDim;  // 32 heads at HEAD_DIM 32
// _CHUNK_ELEMENTS: the reference bounds its largest temporary (the attention
// logits) to 2^22 elements per chunk; the tensor strides and spans here obey the
// same bound.
inline constexpr std::ptrdiff_t kMaxAttentionSpan = 1 << 22;

// A [windows, heads, 64, 64] float32 tensor (logits, softmax numerators or e4m3
// probabilities): element (w, h, i, j) at
// data + w * window_stride + h * head_stride + i * 64 + j, strides in floats with
// kBiasTable <= head_stride and heads * head_stride <= window_stride.
struct AttentionTensorView {
    const float* data = nullptr;
    int windows = 0;
    int heads = 0;
    std::ptrdiff_t window_stride = 0;
    std::ptrdiff_t head_stride = 0;
};

struct MutableAttentionTensorView {
    float* data = nullptr;
    int windows = 0;
    int heads = 0;
    std::ptrdiff_t window_stride = 0;
    std::ptrdiff_t head_stride = 0;
};

// ---- validation shared by the entry points -------------------------------------------------------
// A [heads, 4096] table (one deswizzled or raw 64x64 bias per head) or its output:
// dimensions, strides and, where asked, the binary16 operand grid. The tables are
// wider than the linear slice's bounds on purpose: 4096 entries per head.
inline Status validate_bias_table(const MatrixView& table) {
    if (table.data == nullptr) {
        return Status::kNullPointer;
    }
    if (table.rows < 1 || table.rows > kMaxHeads || table.cols != kBiasTable) {
        return Status::kInvalidDimensions;
    }
    if (table.row_stride < static_cast<std::ptrdiff_t>(table.cols)
        || table.row_stride > kMaxAttentionSpan) {
        return Status::kInvalidStride;
    }
    return Status::kOk;
}

inline Status validate_bias_table_f16(const MatrixView& table) {
    const Status status = validate_bias_table(table);
    if (status != Status::kOk) {
        return status;
    }
    for (int r = 0; r < table.rows; ++r) {
        const float* row = table.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(table.row_stride);
        for (int c = 0; c < table.cols; ++c) {
            if (!std::isfinite(row[c]) || !is_f16(row[c])) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

inline Status validate_bias_table_finite(const MatrixView& table) {
    const Status status = validate_bias_table(table);
    if (status != Status::kOk) {
        return status;
    }
    for (int r = 0; r < table.rows; ++r) {
        const float* row = table.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(table.row_stride);
        for (int c = 0; c < table.cols; ++c) {
            if (!std::isfinite(row[c])) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

inline Status validate_bias_table_output(const MutableMatrixView* out,
                                         std::size_t out_capacity) {
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows < 1 || out->rows > kMaxHeads || out->cols != kBiasTable) {
        return Status::kInvalidDimensions;
    }
    if (out->row_stride < static_cast<std::ptrdiff_t>(out->cols)
        || out->row_stride > kMaxAttentionSpan) {
        return Status::kInvalidStride;
    }
    const std::size_t needed = static_cast<std::size_t>(out->rows - 1)
            * static_cast<std::size_t>(out->row_stride)
        + static_cast<std::size_t>(out->cols);
    if (out_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

inline Status validate_tensor(const AttentionTensorView& t) {
    if (t.data == nullptr) {
        return Status::kNullPointer;
    }
    if (t.windows < 1 || t.windows > kMaxLinearRows / kWindowTokens
        || t.heads < 1 || t.heads > kMaxHeads) {
        return Status::kInvalidDimensions;
    }
    if (t.head_stride < static_cast<std::ptrdiff_t>(kBiasTable)
        || t.head_stride > kMaxAttentionSpan
        || t.window_stride < static_cast<std::ptrdiff_t>(t.heads) * t.head_stride
        || t.window_stride > kMaxAttentionSpan
        || static_cast<std::size_t>(t.windows) * static_cast<std::size_t>(t.window_stride)
            > static_cast<std::size_t>(kMaxAttentionSpan)) {
        return Status::kInvalidStride;
    }
    return Status::kOk;
}

inline Status validate_tensor_output(const MutableAttentionTensorView* t,
                                     std::size_t out_capacity) {
    if (t == nullptr || t->data == nullptr) {
        return Status::kNullPointer;
    }
    if (t->windows < 1 || t->windows > kMaxLinearRows / kWindowTokens
        || t->heads < 1 || t->heads > kMaxHeads) {
        return Status::kInvalidDimensions;
    }
    if (t->head_stride < static_cast<std::ptrdiff_t>(kBiasTable)
        || t->head_stride > kMaxAttentionSpan
        || t->window_stride < static_cast<std::ptrdiff_t>(t->heads) * t->head_stride
        || t->window_stride > kMaxAttentionSpan
        || static_cast<std::size_t>(t->windows) * static_cast<std::size_t>(t->window_stride)
            > static_cast<std::size_t>(kMaxAttentionSpan)) {
        return Status::kInvalidStride;
    }
    const std::size_t needed = static_cast<std::size_t>(t->windows - 1)
            * static_cast<std::size_t>(t->window_stride)
        + static_cast<std::size_t>(t->heads - 1) * static_cast<std::size_t>(t->head_stride)
        + static_cast<std::size_t>(kBiasTable);
    if (out_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// A float32 matrix whose values are arbitrary finite (accumulator outputs, skip
// residuals, exp-bias ramps): dimensions and strides like the linear slice.
inline Status validate_finite_matrix(const MatrixView& m) {
    if (m.data == nullptr) {
        return Status::kNullPointer;
    }
    if (m.rows < 1 || m.rows > kMaxLinearRows || m.cols < 1 || m.cols > kMaxLinearDim) {
        return Status::kInvalidDimensions;
    }
    if (m.row_stride < static_cast<std::ptrdiff_t>(m.cols)
        || m.row_stride > static_cast<std::ptrdiff_t>(kMaxLinearDim)) {
        return Status::kInvalidStride;
    }
    for (int r = 0; r < m.rows; ++r) {
        const float* row = m.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(m.row_stride);
        for (int c = 0; c < m.cols; ++c) {
            if (!std::isfinite(row[c])) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

// The head-scale row's magnitude bound: `_norm_scale` is at most ~127.02 (the
// kNormFloor clamp), so `|_norm_scale * head_scale| <= 127.02 * 65504 < 8.4e6`
// keeps the per-head scale product finite for every accepted row. Larger finite
// values are rejected at validation instead of overflowing that product to an
// infinity whose zero-dimension products would be NaN and would then fail the
// e4m3 operand checks mid-write.
inline constexpr float kMaxHeadScale = 65504.0f;  // the binary16 maximum magnitude

// A one-row head-scale matrix: finite float32 within the magnitude bound above
// (the reference's `head_scale` is float32; the magnitude bound is this
// implementation's accepted-domain restriction, not an arithmetic change).
inline Status validate_head_scale(const MatrixView& m) {
    const Status status = validate_finite_matrix(m);
    if (status != Status::kOk) {
        return status;
    }
    for (int r = 0; r < m.rows; ++r) {
        const float* row = m.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(m.row_stride);
        for (int c = 0; c < m.cols; ++c) {
            const float magnitude = row[c] < 0.0f ? -row[c] : row[c];
            if (magnitude > kMaxHeadScale) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

// A one-row matrix of finite, exactly binary16-representable values (the raw bias
// table entries, the attn_scale row): the reference stores both as f16.
inline Status validate_f16_matrix(const MatrixView& m) {
    if (m.data == nullptr) {
        return Status::kNullPointer;
    }
    if (m.rows < 1 || m.rows > kMaxLinearRows || m.cols < 1 || m.cols > kMaxLinearDim) {
        return Status::kInvalidDimensions;
    }
    if (m.row_stride < static_cast<std::ptrdiff_t>(m.cols)
        || m.row_stride > static_cast<std::ptrdiff_t>(kMaxLinearDim)) {
        return Status::kInvalidStride;
    }
    for (int r = 0; r < m.rows; ++r) {
        const float* row = m.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(m.row_stride);
        for (int c = 0; c < m.cols; ++c) {
            if (!std::isfinite(row[c]) || !is_f16(row[c])) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

inline Status validate_tensor_f16_grid(const AttentionTensorView& t) {
    const Status status = validate_tensor(t);
    if (status != Status::kOk) {
        return status;
    }
    for (int w = 0; w < t.windows; ++w) {
        for (int h = 0; h < t.heads; ++h) {
            const float* block = t.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(t.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(t.head_stride);
            for (int p = 0; p < kBiasTable; ++p) {
                if (!std::isfinite(block[p]) || !is_f16(block[p])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    return Status::kOk;
}

inline Status validate_tensor_e4m3_grid(const AttentionTensorView& t) {
    const Status status = validate_tensor(t);
    if (status != Status::kOk) {
        return status;
    }
    for (int w = 0; w < t.windows; ++w) {
        for (int h = 0; h < t.heads; ++h) {
            const float* block = t.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(t.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(t.head_stride);
            for (int p = 0; p < kBiasTable; ++p) {
                if (!is_e4m3(block[p])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    return Status::kOk;
}

// ---- the exponent-ramp numerator (`_softmax_numerators` core) ------------------------------------
// One numerator from one (logit, bias) pair. `ramp` is swin_fma(logit, EXP_SLOPE,
// bias) — one float32 rounding from the double intermediate — narrowed to binary16
// (ties to even) and clamped to [EXP_LOW, EXP_HIGH]; the clamp forces the value into
// [1, 2), so its binary16 pattern is 0_01111_mmmmmmmmmm. The mantissa bits are then
// shifted into the exponent field ((bits & 0x3FF) << 5): the result is
// 2^((m >> 5) - 15) * (1 + (m & 31) / 32) for mantissa m, strictly increasing over
// the clamp range, spanning [2^-14, 9.75]. The reference builds exactly this ramp:
// an affine approximation of exp carried in the exponent field. Overflow of the
// float32 ramp gives +-inf, which the clamp saturates to EXP_LOW/EXP_HIGH exactly as
// the reference's `torch.clamp` on inf. NaN is excluded by validation (finite
// inputs), matching the reference's float32 data path.
inline float swin_softmax_numerator(float logit, float bias) {
    const float ramp = f16(swin_fma(logit, kExpSlope, bias));
    const float clamped = ramp < kExpLow ? kExpLow
        : (ramp > kExpHigh ? kExpHigh : ramp);
    return f16_from_bits(static_cast<std::uint16_t>((f16_bits(clamped) & 0x3FFu) << 5));
}

// ---- the position-bias deswizzle (`_deswizzle_bias`) ---------------------------------------------
// The index map of one head's 64x64 table from its stored fragment order to
// [query, key]. The table is stored as sixteen 16x16 (query block, key block) tiles,
// each tile in the lane order of an accumulator fragment: 32 lanes of eight values.
// For i (query) and j (key) in [0, 64):
//   lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2
//   slot = (j % 2) | (((i % 16) / 8) << 1) | (((j % 16) / 8) << 2)
//   index = (4 * (i / 16) + (j / 16)) * 256 + lane * 8 + slot
// index runs over all of [0, 4096) exactly once (a permutation; pinned by the
// tests). Out-of-range i or j returns -1.
inline int swin_deswizzle_index(int i, int j) {
    if (i < 0 || i >= kWindowTokens || j < 0 || j >= kWindowTokens) {
        return -1;
    }
    const int lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2;
    const int slot = (j % 2) | (((i % 16) / 8) << 1) | (((j % 16) / 8) << 2);
    return (4 * (i / 16) + (j / 16)) * 256 + lane * 8 + slot;
}

// ---- one query row's softmax (`_softmax_numerators`/`_denominator`/`_softmax`) --------------------
// 64 numerators for one (window, head, query) row: the ramp per (logit, bias[j]).
inline void swin_softmax_numerators_row(const float logits_row[kWindowTokens],
                                        const float* bias_row,
                                        float out_row[kWindowTokens]) {
    for (int j = 0; j < kWindowTokens; ++j) {
        out_row[j] = swin_softmax_numerator(logits_row[j], bias_row[j]);
    }
}

// The softmax denominator: the sum over the 64 keys in the order the reference adds
// them, each add a binary16 add (the sum of two binary16 values is exact in
// float32, so `f16(a + b)` is the correctly rounded binary16 add). The keys of a
// query sit in two lane halves, each holding eight of every 16-key tile block as
// four pairs: key = 16 block + 4 pair + 2 member + half (the reference's
// `reshape(4, 4, 2, 2)`). Blocks are added first ((b0 + b1) + b2) + b3, then pairs
// ((p0 + p1) + p2) + p3, then the two halves of a member, then the two members.
// Binary16 overflow gives +-inf, which passes through.
inline float swin_softmax_denominator_row(const float keys[kWindowTokens]) {
    float blocks[4][2][2];
    for (int p = 0; p < 4; ++p) {
        for (int m = 0; m < 2; ++m) {
            for (int h = 0; h < 2; ++h) {
                float sum = keys[0 * 16 + p * 4 + m * 2 + h];
                sum = f16(sum + keys[1 * 16 + p * 4 + m * 2 + h]);
                sum = f16(sum + keys[2 * 16 + p * 4 + m * 2 + h]);
                sum = f16(sum + keys[3 * 16 + p * 4 + m * 2 + h]);
                blocks[p][m][h] = sum;
            }
        }
    }
    float pairs[2][2];
    for (int m = 0; m < 2; ++m) {
        for (int h = 0; h < 2; ++h) {
            float sum = blocks[0][m][h];
            sum = f16(sum + blocks[1][m][h]);
            sum = f16(sum + blocks[2][m][h]);
            sum = f16(sum + blocks[3][m][h]);
            pairs[m][h] = sum;
        }
    }
    const float half0 = f16(pairs[0][0] + pairs[0][1]);
    const float half1 = f16(pairs[1][0] + pairs[1][1]);
    return f16(half0 + half1);
}

// e4m3 probabilities for one query row: numerators -> denominator -> reciprocal
// (float32 division of the denominator clamped to SUM_FLOOR) -> swin_store of each
// float32 numerator * reciprocal product (one float32 rounding, then the stored
// f16 -> e4m3 double rounding). Through the full softmax the denominator is at
// least 64 * 2^-14 = 2^-8, so the SUM_FLOOR clamp never binds there; it is kept for
// exactness with the reference.
inline void swin_softmax_row(const float logits_row[kWindowTokens],
                             const float* bias_row, float out_row[kWindowTokens]) {
    float numerators[kWindowTokens];
    swin_softmax_numerators_row(logits_row, bias_row, numerators);
    const float denominator = swin_softmax_denominator_row(numerators);
    const float clamped = denominator < kSumFloor ? kSumFloor : denominator;
    const float reciprocal = 1.0f / clamped;
    for (int j = 0; j < kWindowTokens; ++j) {
        out_row[j] = swin_store(numerators[j] * reciprocal);
    }
}

// ---- one head's QKV normalisation (`_normalise_qkv`) ----------------------------------------------
// One (token, head) slice of 32 dimensions. Q and K are scaled straight from the
// float32 accumulator and rounded ONCE to e4m3: the per-head scale product
// `_norm_scale(query) * head_scale` is one float32 product, each scaled dimension
// is one float32 product, and `q8` rounds that product directly to the e4m3 grid.
// V takes the stored-result double rounding (`q8(_f16(value))`, swin_store). The
// q/v paths therefore differ observably (367.875 scales to 352 on the q path and
// stores to 384 on the v path; pinned by the tests).
inline void swin_normalise_head_row(const float query32[kHeadDim],
                                    const float key32[kHeadDim],
                                    const float value32[kHeadDim],
                                    float head_scale,
                                    float out_q[kHeadDim], float out_k[kHeadDim],
                                    float out_v[kHeadDim]) {
    const float query_scale = swin_norm_scale(query32) * head_scale;
    const float key_scale = swin_norm_scale(key32);
    for (int d = 0; d < kHeadDim; ++d) {
        out_q[d] = e4m3(query32[d] * query_scale);
        out_k[d] = e4m3(key32[d] * key_scale);
        out_v[d] = swin_store(value32[d]);
    }
}

// ---- one attention logit (`_attention` einsum core) -----------------------------------------------
// The dot product of one (query, key) head pair over the 32 head dimensions: both
// operands are e4m3-grid values (every product exact in float32), the products are
// added to a float32 accumulator in increasing d order starting at +0.0 — the
// explicit summation convention the linears slice documents. The reference's
// einsum states no summation-order contract; no vendor correspondence is claimed
// for the order.
inline float swin_logit_dot(const float query32[kHeadDim],
                            const float key32[kHeadDim]) {
    float sum = 0.0f;
    for (int d = 0; d < kHeadDim; ++d) {
        sum += query32[d] * key32[d];
    }
    return sum;
}

// ---- the batched deswizzle (`_deswizzle_bias`) ----------------------------------------------------
// raw [heads, 4096] -> out [heads, 4096]: out[h][i * 64 + j] = raw[h][index(i, j)]
// with the swin_deswizzle_index map. `raw` holds f16-grid values (the stored bias
// table); the copy is index arithmetic only. Validation (tables, the binary16
// operand grid, output capacity) completes before the first write; on any error
// `out` is untouched.
inline Status swin_deswizzle_bias(const MatrixView& raw,
                                  MutableMatrixView* out,
                                  std::size_t out_capacity) {
    {
        const Status status = validate_bias_table_f16(raw);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows != raw.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_bias_table_output(out, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int h = 0; h < raw.rows; ++h) {
        const float* raw_row = raw.data + static_cast<std::size_t>(h)
            * static_cast<std::size_t>(raw.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(h)
            * static_cast<std::size_t>(out->row_stride);
        for (int i = 0; i < kWindowTokens; ++i) {
            for (int j = 0; j < kWindowTokens; ++j) {
                out_row[i * kWindowTokens + j] = raw_row[swin_deswizzle_index(i, j)];
            }
        }
    }
    return Status::kOk;
}

// ---- the exponent-ramp bias (`SwinParams.exp_bias`) -----------------------------------------------
// The deswizzled position bias carried through the exponent ramp:
// exp_bias[h][i * 64 + j] = swin_fma(deswizzled(h, i, j), EXP_SLOPE, EXP_OFFSET) —
// one float32 rounding from the double intermediate per entry. `raw` holds f16-grid
// values. Validation completes before the first write; on any error `out` is
// untouched.
inline Status swin_exp_bias(const MatrixView& raw, MutableMatrixView* out,
                            std::size_t out_capacity) {
    {
        const Status status = validate_bias_table_f16(raw);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows != raw.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_bias_table_output(out, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int h = 0; h < raw.rows; ++h) {
        const float* raw_row = raw.data + static_cast<std::size_t>(h)
            * static_cast<std::size_t>(raw.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(h)
            * static_cast<std::size_t>(out->row_stride);
        for (int i = 0; i < kWindowTokens; ++i) {
            for (int j = 0; j < kWindowTokens; ++j) {
                out_row[i * kWindowTokens + j] =
                    swin_fma(raw_row[swin_deswizzle_index(i, j)], kExpSlope, kExpOffset);
            }
        }
    }
    return Status::kOk;
}

// ---- the softmax numerators (`_softmax_numerators`) ------------------------------------------------
// logits [windows, heads, 64, 64] finite float32 and exp_bias [heads, 4096] finite
// float32 -> out [windows, heads, 64, 64] binary16-grid values (the numerator
// values, carried as float32). Validation completes before the first write; on any
// error `out` is untouched.
inline Status swin_softmax_numerators(const AttentionTensorView& logits,
                                      const MatrixView& exp_bias,
                                      MutableAttentionTensorView* out,
                                      std::size_t out_capacity) {
    {
        const Status status = validate_tensor(logits);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_bias_table_finite(exp_bias);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (exp_bias.rows != logits.heads) {
        return Status::kInvalidDimensions;
    }
    for (int w = 0; w < logits.windows; ++w) {
        for (int h = 0; h < logits.heads; ++h) {
            const float* block = logits.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(logits.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(logits.head_stride);
            for (int p = 0; p < kBiasTable; ++p) {
                if (!std::isfinite(block[p])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->windows != logits.windows || out->heads != logits.heads) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_tensor_output(out, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int w = 0; w < logits.windows; ++w) {
        for (int h = 0; h < logits.heads; ++h) {
            const float* logits_block = logits.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(logits.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(logits.head_stride);
            const float* bias_row_base = exp_bias.data + static_cast<std::size_t>(h)
                * static_cast<std::size_t>(exp_bias.row_stride);
            float* out_block = out->data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(out->window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(out->head_stride);
            for (int i = 0; i < kWindowTokens; ++i) {
                swin_softmax_numerators_row(logits_block + i * kWindowTokens,
                                            bias_row_base + i * kWindowTokens,
                                            out_block + i * kWindowTokens);
            }
        }
    }
    return Status::kOk;
}

// ---- the softmax denominator (`_denominator`) -------------------------------------------------------
// numerators [windows, heads, 64, 64] binary16-grid values -> out
// [windows * heads * 64] (one denominator per (window, head, query), indexed
// (w * heads + h) * 64 + i), each the reference's binary16 add tree. Validation
// completes before the first write; on any error `out` is untouched.
inline Status swin_softmax_denominator(const AttentionTensorView& numerators,
                                       float* out, std::size_t out_capacity) {
    {
        const Status status = validate_tensor_f16_grid(numerators);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    const std::size_t needed = static_cast<std::size_t>(numerators.windows)
        * static_cast<std::size_t>(numerators.heads) * kWindowTokens;
    if (out_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    for (int w = 0; w < numerators.windows; ++w) {
        for (int h = 0; h < numerators.heads; ++h) {
            const float* block = numerators.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(numerators.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(numerators.head_stride);
            float* out_row = out + (static_cast<std::size_t>(w) * numerators.heads + h)
                * kWindowTokens;
            for (int i = 0; i < kWindowTokens; ++i) {
                out_row[i] = swin_softmax_denominator_row(block + i * kWindowTokens);
            }
        }
    }
    return Status::kOk;
}

// ---- the softmax (`_softmax`) -----------------------------------------------------------------------
// logits [windows, heads, 64, 64] finite float32 and exp_bias [heads, 4096] finite
// float32 -> out [windows, heads, 64, 64] e4m3 probabilities, via one fixed
// internal 64-float row buffer per query (no allocation). Validation completes
// before the first write; on any error `out` is untouched.
inline Status swin_softmax(const AttentionTensorView& logits,
                           const MatrixView& exp_bias,
                           MutableAttentionTensorView* out,
                           std::size_t out_capacity) {
    {
        const Status status = validate_tensor(logits);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_bias_table_finite(exp_bias);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (exp_bias.rows != logits.heads) {
        return Status::kInvalidDimensions;
    }
    for (int w = 0; w < logits.windows; ++w) {
        for (int h = 0; h < logits.heads; ++h) {
            const float* block = logits.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(logits.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(logits.head_stride);
            for (int p = 0; p < kBiasTable; ++p) {
                if (!std::isfinite(block[p])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->windows != logits.windows || out->heads != logits.heads) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_tensor_output(out, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int w = 0; w < logits.windows; ++w) {
        for (int h = 0; h < logits.heads; ++h) {
            const float* logits_block = logits.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(logits.window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(logits.head_stride);
            const float* bias_row_base = exp_bias.data + static_cast<std::size_t>(h)
                * static_cast<std::size_t>(exp_bias.row_stride);
            float* out_block = out->data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(out->window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(out->head_stride);
            for (int i = 0; i < kWindowTokens; ++i) {
                swin_softmax_row(logits_block + i * kWindowTokens,
                                 bias_row_base + i * kWindowTokens,
                                 out_block + i * kWindowTokens);
            }
        }
    }
    return Status::kOk;
}

// ---- the QKV normalisation (`_normalise_qkv`) -------------------------------------------------------
// query/key/value [rows, heads * 32] finite float32 (the raw qkv accumulator
// slices, token-major: row = window * 64 + token, head g at columns [g * 32,
// (g + 1) * 32)) -> out_q/out_k/out_v [rows, heads * 32] e4m3-grid values, with the
// swin_normalise_head_row arithmetic per (row, head). `head_scale` is [1, heads]
// finite float32 within the `kMaxHeadScale` magnitude bound (see
// `validate_head_scale`), which keeps the scale product finite so the outputs
// stay on the e4m3 grid. Validation (dimensions, strides, value classes,
// output capacities) completes before the first write; on any error all three
// outputs are untouched.
inline Status swin_normalise_qkv(const MatrixView& query, const MatrixView& key,
                                 const MatrixView& value,
                                 const MatrixView& head_scale, int heads,
                                 MutableMatrixView* out_q, std::size_t out_q_capacity,
                                 MutableMatrixView* out_k, std::size_t out_k_capacity,
                                 MutableMatrixView* out_v, std::size_t out_v_capacity) {
    {
        const Status status = validate_finite_matrix(query);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_finite_matrix(key);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_finite_matrix(value);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_head_scale(head_scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (heads < 1 || heads > kMaxHeads
        || query.cols != heads * kHeadDim || key.cols != query.cols
        || value.cols != query.cols || value.rows != query.rows
        || key.rows != query.rows
        || head_scale.rows != 1 || head_scale.cols != heads) {
        return Status::kInvalidDimensions;
    }
    if (out_q == nullptr || out_q->data == nullptr || out_k == nullptr
        || out_k->data == nullptr || out_v == nullptr || out_v->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out_q->rows != query.rows || out_k->rows != query.rows
        || out_v->rows != query.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out_q, query.cols, out_q_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_output(out_k, query.cols, out_k_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_output(out_v, query.cols, out_v_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int r = 0; r < query.rows; ++r) {
        const float* q_row = query.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(query.row_stride);
        const float* k_row = key.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(key.row_stride);
        const float* v_row = value.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(value.row_stride);
        float* oq_row = out_q->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out_q->row_stride);
        float* ok_row = out_k->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out_k->row_stride);
        float* ov_row = out_v->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out_v->row_stride);
        for (int g = 0; g < heads; ++g) {
            swin_normalise_head_row(q_row + g * kHeadDim, k_row + g * kHeadDim,
                                    v_row + g * kHeadDim, head_scale.data[g],
                                    oq_row + g * kHeadDim, ok_row + g * kHeadDim,
                                    ov_row + g * kHeadDim);
        }
    }
    return Status::kOk;
}

// ---- the attention logits (`_attention` einsum) ------------------------------------------------------
// query/key [rows, heads * 32] e4m3-grid values (the normalised Q and K), row =
// window * 64 + token -> out [windows, heads, 64, 64] float32 logits with
// out[w][h][i][j] = swin_logit_dot(query[w * 64 + i], key[w * 64 + j]) over head h.
// Validation completes before the first write; on any error `out` is untouched.
inline Status swin_logits(const MatrixView& query, const MatrixView& key, int heads,
                          MutableAttentionTensorView* out,
                          std::size_t out_capacity) {
    {
        const Status status = validate_linear_operand(query);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(key);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (heads < 1 || heads > kMaxHeads || query.cols != heads * kHeadDim
        || key.cols != query.cols || key.rows != query.rows
        || query.rows % kWindowTokens != 0) {
        return Status::kInvalidDimensions;
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->windows * kWindowTokens != query.rows || out->heads != heads) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_tensor_output(out, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int w = 0; w < out->windows; ++w) {
        for (int h = 0; h < heads; ++h) {
            float* block = out->data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(out->window_stride)
                + static_cast<std::size_t>(h) * static_cast<std::size_t>(out->head_stride);
            for (int i = 0; i < kWindowTokens; ++i) {
                const float* q_row = query.data
                    + static_cast<std::size_t>(w * kWindowTokens + i)
                        * static_cast<std::size_t>(query.row_stride)
                    + h * kHeadDim;
                for (int j = 0; j < kWindowTokens; ++j) {
                    const float* k_row = key.data
                        + static_cast<std::size_t>(w * kWindowTokens + j)
                            * static_cast<std::size_t>(key.row_stride)
                        + h * kHeadDim;
                    block[i * kWindowTokens + j] = swin_logit_dot(q_row, k_row);
                }
            }
        }
    }
    return Status::kOk;
}

// ---- the attention context (`_attention` einsum through `store`) --------------------------------------
// probabilities [windows, heads, 64, 64] e4m3-grid and value [rows, heads * 32]
// e4m3-grid (the normalised V), row = window * 64 + token -> out [rows, heads * 32]
// with out[w * 64 + i][g * 32 + d] = swin_store(sum_j probabilities[w][g][i][j] *
// value[w * 64 + j][g * 32 + d]), the products added to a float32 accumulator in
// increasing j order starting at +0.0 (each product exact). Validation completes
// before the first write; on any error `out` is untouched.
inline Status swin_context(const AttentionTensorView& probabilities,
                           const MatrixView& value,
                           MutableMatrixView* out, std::size_t out_capacity) {
    {
        const Status status = validate_tensor_e4m3_grid(probabilities);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(value);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (value.cols != probabilities.heads * kHeadDim
        || value.rows != probabilities.windows * kWindowTokens) {
        return Status::kInvalidDimensions;
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows != value.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out, value.cols, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int w = 0; w < probabilities.windows; ++w) {
        for (int g = 0; g < probabilities.heads; ++g) {
            const float* block = probabilities.data + static_cast<std::size_t>(w)
                    * static_cast<std::size_t>(probabilities.window_stride)
                + static_cast<std::size_t>(g) * static_cast<std::size_t>(probabilities.head_stride);
            for (int i = 0; i < kWindowTokens; ++i) {
                const float* probs_row = block + i * kWindowTokens;
                float* out_row = out->data
                    + static_cast<std::size_t>(w * kWindowTokens + i)
                        * static_cast<std::size_t>(out->row_stride)
                    + g * kHeadDim;
                for (int d = 0; d < kHeadDim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < kWindowTokens; ++j) {
                        const float* v_row = value.data
                            + static_cast<std::size_t>(w * kWindowTokens + j)
                                * static_cast<std::size_t>(value.row_stride)
                            + g * kHeadDim;
                        sum += probs_row[j] * v_row[d];
                    }
                    out_row[d] = swin_store(sum);
                }
            }
        }
    }
    return Status::kOk;
}

// ---- the `_attention` assembly (the e4m3 path, or the int4 QKV site's projection) -------------------------------------------------------
// out [rows, C] = skip * attn_scale + out_proj . attention(qkv(quantised)) with
// C = heads * kHeadDim and rows = windows * kWindowTokens, mirroring the
// reference's `_attention` with the int4 QKV site threaded as the projection override:
// the qkv projection `swin_linear(quantised, qkv)` (raw float32 accumulator),
// `_normalise_qkv`, the logits einsum, `_softmax`, the context einsum through
// `store`, and the residual line reusing the mlp slice's tail arithmetic
// (`residual * scale + product` as two separate float32 roundings). The return
// stays float32; the caller-side final e4m3 rounding is a later composition stage.
// `scratch` is caller-owned workspace of 3 * rows * C (the qkv accumulator) + 4 *
// rows * C (normalised q, k, v and the context) + 2 * heads * 4096 (one window's
// logits and probabilities) floats. Every parameter — dimensions, strides, the e4m3
// operand grids, the finite/f16 value classes and both capacities — is validated
// before the first write: on any error `out` and `scratch` are both untouched. Each
// internal stage is then called with exactly its own scratch region as the capacity.
// The buffers must not overlap (caller discipline, never rejected). The qkv width
// 3 * C must stay within the linear slice's bounds (heads <= 10 here; the stage
// entries allow up to 32).
inline Status swin_attention(const MatrixView& quantised, const MatrixView& qkv,
                             const MatrixView& head_scale, const MatrixView& exp_bias,
                             const MatrixView& skip, const MatrixView& attn_scale,
                             const MatrixView& out_proj,
                             MutableMatrixView* out, std::size_t out_capacity,
                             float* scratch, std::size_t scratch_capacity,
                             const Int4MatrixView* int4_qkv = nullptr,
                             const SwinProjectionSite* site = nullptr) {
    if (quantised.data == nullptr || qkv.data == nullptr || head_scale.data == nullptr
        || exp_bias.data == nullptr || skip.data == nullptr
        || attn_scale.data == nullptr || out_proj.data == nullptr || scratch == nullptr) {
        return Status::kNullPointer;
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_linear_operand(quantised);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(qkv);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(out_proj);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (quantised.cols % kHeadDim != 0) {
        return Status::kInvalidDimensions;
    }
    {
        const int heads = quantised.cols / kHeadDim;
        if (heads < 1 || heads > kMaxLinearDim / (3 * kHeadDim)) {
            return Status::kInvalidDimensions;
        }
    }
    const int heads = quantised.cols / kHeadDim;
    const int channels = quantised.cols;
    if (quantised.rows % kWindowTokens != 0
        || qkv.rows != 3 * channels || qkv.cols != quantised.cols
        || out_proj.rows != channels || out_proj.cols != channels) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_head_scale(head_scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_bias_table_finite(exp_bias);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_finite_matrix(skip);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_f16_matrix(attn_scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (head_scale.rows != 1 || head_scale.cols != heads
        || exp_bias.rows != heads
        || skip.rows != quantised.rows || skip.cols != channels
        || attn_scale.rows != 1 || attn_scale.cols != channels) {
        return Status::kInvalidDimensions;
    }
    if (out->rows != quantised.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out, channels, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    const std::size_t rows_c = static_cast<std::size_t>(quantised.rows)
        * static_cast<std::size_t>(channels);
    const std::size_t qkv_span = static_cast<std::size_t>(quantised.rows)
        * static_cast<std::size_t>(qkv.rows);
    const std::size_t needed = qkv_span
        + 4 * rows_c
        + 2 * static_cast<std::size_t>(heads) * kBiasTable;
    if (scratch_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    // All validation is complete: from here every stage is guaranteed kOk and out
    // and scratch are written.
    float* qkv_buffer = scratch;
    float* q_buffer = qkv_buffer + static_cast<std::size_t>(quantised.rows) * qkv.rows;
    float* k_buffer = q_buffer + rows_c;
    float* v_buffer = k_buffer + rows_c;
    float* context_buffer = v_buffer + rows_c;
    float* window_logits = context_buffer + rows_c;
    float* window_probs = window_logits + static_cast<std::size_t>(heads) * kBiasTable;
    MutableMatrixView qkv_mat;
    qkv_mat.data = qkv_buffer;
    qkv_mat.rows = quantised.rows;
    qkv_mat.cols = qkv.rows;  // the projection writes 3 * C outputs per token
    qkv_mat.row_stride = qkv.rows;
    {
        // The int4 site's QKV matrix replaces the e4m3 projection (the
        // pinned SwinSite's qkv); the accumulator is the same raw float32.
        const Status status = int4_qkv != nullptr
            ? swin_linear_int4(quantised, *int4_qkv, &qkv_mat)
            // The int4_qkv branch above keeps the CPU int4 path and
            // ignores the site (a set site must not change it).
            : swin_linear(quantised, qkv, &qkv_mat, qkv_span, site);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int r = 0; r < quantised.rows; ++r) {
        const float* qkv_row = qkv_buffer + static_cast<std::size_t>(r) * qkv.rows;
        float* q_row = q_buffer + static_cast<std::size_t>(r) * channels;
        float* k_row = k_buffer + static_cast<std::size_t>(r) * channels;
        float* v_row = v_buffer + static_cast<std::size_t>(r) * channels;
        for (int g = 0; g < heads; ++g) {
            swin_normalise_head_row(qkv_row + g * 3 * kHeadDim,
                                    qkv_row + g * 3 * kHeadDim + kHeadDim,
                                    qkv_row + g * 3 * kHeadDim + 2 * kHeadDim,
                                    head_scale.data[g],
                                    q_row + g * kHeadDim, k_row + g * kHeadDim,
                                    v_row + g * kHeadDim);
        }
    }
    const int windows = quantised.rows / kWindowTokens;
    for (int w = 0; w < windows; ++w) {
        MatrixView q_window;
        q_window.data = q_buffer + static_cast<std::size_t>(w * kWindowTokens) * channels;
        q_window.rows = kWindowTokens;
        q_window.cols = channels;
        q_window.row_stride = channels;
        MatrixView k_window = q_window;
        k_window.data = k_buffer + static_cast<std::size_t>(w * kWindowTokens) * channels;
        MatrixView v_window = q_window;
        v_window.data = v_buffer + static_cast<std::size_t>(w * kWindowTokens) * channels;
        MutableAttentionTensorView logits_window;
        logits_window.data = window_logits;
        logits_window.windows = 1;
        logits_window.heads = heads;
        logits_window.window_stride = static_cast<std::ptrdiff_t>(heads) * kBiasTable;
        logits_window.head_stride = kBiasTable;
        {
            const Status status = swin_logits(q_window, k_window, heads,
                                              &logits_window,
                                              static_cast<std::size_t>(heads)
                                                  * kBiasTable);
            if (status != Status::kOk) {
                return status;
            }
        }
        AttentionTensorView logits_read;
        logits_read.data = window_logits;
        logits_read.windows = 1;
        logits_read.heads = heads;
        logits_read.window_stride = static_cast<std::ptrdiff_t>(heads) * kBiasTable;
        logits_read.head_stride = kBiasTable;
        MutableAttentionTensorView probs_window;
        probs_window.data = window_probs;
        probs_window.windows = 1;
        probs_window.heads = heads;
        probs_window.window_stride = static_cast<std::ptrdiff_t>(heads) * kBiasTable;
        probs_window.head_stride = kBiasTable;
        {
            const Status status = swin_softmax(logits_read, exp_bias,
                                               &probs_window,
                                               static_cast<std::size_t>(heads)
                                                   * kBiasTable);
            if (status != Status::kOk) {
                return status;
            }
        }
        AttentionTensorView probs_read;
        probs_read.data = window_probs;
        probs_read.windows = 1;
        probs_read.heads = heads;
        probs_read.window_stride = static_cast<std::ptrdiff_t>(heads) * kBiasTable;
        probs_read.head_stride = kBiasTable;
        MutableMatrixView context_window;
        context_window.data = context_buffer
            + static_cast<std::size_t>(w * kWindowTokens) * channels;
        context_window.rows = kWindowTokens;
        context_window.cols = channels;
        context_window.row_stride = channels;
        {
            const Status status = swin_context(probs_read, v_window,
                                               &context_window,
                                               static_cast<std::size_t>(
                                                   kWindowTokens) * channels);
            if (status != Status::kOk) {
                return status;
            }
        }
    }
    MatrixView context_mat;
    context_mat.data = context_buffer;
    context_mat.rows = quantised.rows;
    context_mat.cols = channels;
    context_mat.row_stride = channels;
    return swin_mlp_tail(context_mat, out_proj, skip, attn_scale, out, out_capacity);
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_ATTENTION_HPP
