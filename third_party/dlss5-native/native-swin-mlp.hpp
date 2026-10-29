// Native Swin MLP assembly: the `_mlp` body of the Swin blocks on the e4m3 grid —
// the e4m3 hidden (`swin_hidden` from the linears slice), the grouped middle
// projection narrowed through `store` (float32 -> f16 -> e4m3), the contract
// projection and the f16 residual scale. CPU-only, typed, allocation-free; no GPU,
// headset, proprietary payload, Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "_mlp"/"store" and the `_hidden`/`_linear` it composes).
// The full MIT licence notice is retained in vr/NATIVE-SWIN-MLP.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_MLP_HPP
#define DLSS5_NR_NATIVE_SWIN_MLP_HPP

#include "native-swin-linear.hpp"

namespace dlss5_nr {

// ---- documented constants ----------------------------------------------------------------------
inline constexpr int kHeadDim = 32;  // HEAD_DIM: per-head width of the mid projection

// ---- the stored-result double rounding (`store`) -----------------------------------------------
// float32 -> f16 -> e4m3: the reference's `store` is `q8(_f16(x))` — the float32
// accumulator is first narrowed to binary16 (ties to even), then the f16 value is
// rounded to the e4m3 grid (ties to even, saturating at +-448; the f16 infinities
// saturate through the e4m3 clamp). The double rounding is the reference's own and
// is observable: 367.875 stores as 384 (the f16 tie lands on 368 and the e4m3 tie
// on the even code 384) where a direct e4m3 rounding would give 352.
inline float swin_store(float value) { return e4m3(f16(value)); }

// ---- residual and scale value classes -----------------------------------------------------------
// The tail's auxiliary operands: `residual` is arbitrary finite float32 (the block
// feature maps); `scale` is one row of finite, exactly binary16-representable values
// (the reference's f16-exact residual scales). Stride bounds match the linear slice.
inline Status validate_residual_scale(const MatrixView& residual,
                                      const MatrixView& scale) {
    if (residual.row_stride < static_cast<std::ptrdiff_t>(residual.cols)
        || residual.row_stride > static_cast<std::ptrdiff_t>(kMaxLinearDim)
        || scale.row_stride < static_cast<std::ptrdiff_t>(scale.cols)
        || scale.row_stride > static_cast<std::ptrdiff_t>(kMaxLinearDim)) {
        return Status::kInvalidStride;
    }
    for (int r = 0; r < residual.rows; ++r) {
        const float* row = residual.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(residual.row_stride);
        for (int c = 0; c < residual.cols; ++c) {
            if (!std::isfinite(row[c])) {
                return Status::kInvalidValues;
            }
        }
    }
    for (int c = 0; c < scale.cols; ++c) {
        const float value = scale.data[c];
        if (!std::isfinite(value) || !is_f16(value)) {
            return Status::kInvalidValues;
        }
    }
    return Status::kOk;
}

// ---- one token's grouped middle projection ------------------------------------------------------
// line [g, j] -> out_row [g, d]: out_row[g * kHeadDim + d] =
// swin_store(sum_j hidden_row[g * group_width + j] * mid[g * kHeadDim + d][j]),
// the products added to a float32 accumulator in increasing j order starting at
// +0.0 (each product exact: both operands are e4m3 grid values). Shared by
// swin_mid and the swin_mlp assembly so both use one arithmetic path.
inline void swin_mid_row(const float* hidden_row, int heads, int group_width,
                         const MatrixView& mid, float* out_row) {
    for (int g = 0; g < heads; ++g) {
        const float* group = hidden_row + g * group_width;
        for (int d = 0; d < kHeadDim; ++d) {
            const float* m_row = mid.data + static_cast<std::size_t>(g * kHeadDim + d)
                * static_cast<std::size_t>(mid.row_stride);
            float sum = 0.0f;
            for (int j = 0; j < group_width; ++j) {
                sum += group[j] * m_row[j];
            }
            out_row[g * kHeadDim + d] = swin_store(sum);
        }
    }
}

// ---- one token's contract projection and residual scale ------------------------------------------
// out_row [c] = residual_row[c] * scale[c] + sum_j line[j] * contract[c][j]. The
// contract product uses the linears slice's k-order float32 accumulator; the
// residual scale multiply and the add are two separate float32 operations with one
// rounding each, exactly as the reference's eager `residual * mlp_scale +
// _linear(...)` — NOT a fused multiply-add. Shared by swin_mlp_tail and swin_mlp.
inline void swin_mlp_row_tail(const float* line, int width,
                              const MatrixView& contract,
                              const float* residual_row, const MatrixView& scale,
                              float* out_row) {
    for (int c = 0; c < contract.rows; ++c) {
        const float* w_row = contract.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(contract.row_stride);
        float sum = 0.0f;
        for (int j = 0; j < width; ++j) {
            sum += line[j] * w_row[j];
        }
        const float scaled = residual_row[c] * scale.data[c];
        out_row[c] = scaled + sum;
    }
}

// ---- the grouped middle projection (`_mlp` mid path) ---------------------------------------------
// hidden [rows, heads * group_width] -> out [rows, heads * kHeadDim]: per token and
// head, the `store`-narrowed product of the head's group of the hidden with that
// head's slice of `mid`. `mid` is the block-diagonal per-head weight laid out flat
// [heads * kHeadDim, group_width]; `hidden` and `mid` are e4m3-grid operands.
// Validation (dimensions and strides, the e4m3 operand grids, output capacity)
// completes before the first output write; on any error `out` is untouched.
inline Status swin_mid(const MatrixView& hidden, const MatrixView& mid, int heads,
                       MutableMatrixView* out, std::size_t out_capacity) {
    if (hidden.data == nullptr || mid.data == nullptr) {
        return Status::kNullPointer;
    }
    if (heads < 1 || heads > kMaxLinearDim / kHeadDim) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_operand(hidden);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(mid);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (mid.rows != heads * kHeadDim || hidden.cols != heads * mid.cols) {
        return Status::kInvalidDimensions;
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows != hidden.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out, mid.rows, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int r = 0; r < hidden.rows; ++r) {
        const float* hidden_row = hidden.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(hidden.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out->row_stride);
        swin_mid_row(hidden_row, heads, mid.cols, mid, out_row);
    }
    return Status::kOk;
}

// ---- the contract projection and residual scale (`_mlp` return line) ------------------------------
// out [rows, n] = residual [rows, n] * scale [1, n] + in [rows, k] @ contract [n, k]^T,
// with the arithmetic of swin_mlp_row_tail above. `in` and `contract` are e4m3-grid
// operands; `residual` is finite float32; `scale` is finite and f16-exact. Extreme
// residual magnitudes can overflow the float32 product or sum (IEEE +-inf), which
// passes through. Validation (dimensions, strides, operand classes, output
// capacity) completes before the first output write; on any error `out` is
// untouched.
inline Status swin_mlp_tail(const MatrixView& in, const MatrixView& contract,
                            const MatrixView& residual, const MatrixView& scale,
                            MutableMatrixView* out, std::size_t out_capacity) {
    if (in.data == nullptr || contract.data == nullptr || residual.data == nullptr
        || scale.data == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_linear_operand(in);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(contract);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (in.cols != contract.cols || residual.rows != in.rows
        || residual.cols != contract.rows || scale.rows != 1
        || scale.cols != contract.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_residual_scale(residual, scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows != in.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out, contract.rows, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int r = 0; r < in.rows; ++r) {
        const float* in_row = in.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(in.row_stride);
        const float* residual_row = residual.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(residual.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out->row_stride);
        swin_mlp_row_tail(in_row, in.cols, contract, residual_row, scale, out_row);
    }
    return Status::kOk;
}

// ---- the `_mlp` assembly ---------------------------------------------------------------------------
// out [rows, n] = residual * scale + contract_line(line) where the hidden is
// swin_hidden(x, expand) [rows, expand.rows] and `line` is that hidden directly
// (single-head case: `mid` null, contract.cols == expand.rows) or its
// `store`-narrowed grouped mid projection (contract.cols == heads * kHeadDim).
// `scratch` is caller-owned workspace of rows x expand.rows floats holding the
// hidden between stages (one fixed internal row buffer holds a mid row). Every
// parameter — dimensions, strides, the e4m3 operand grids, the residual/scale
// classes and both capacities — is validated before the first write: on any error
// `out` and `scratch` are both untouched. The buffers must not overlap (caller
// discipline, never rejected). `heads` is ignored when `mid` is null.
inline Status swin_mlp(const MatrixView& x, const MatrixView& expand,
                       const MatrixView* mid, int heads,
                       const MatrixView& contract, const MatrixView& residual,
                       const MatrixView& scale,
                       MutableMatrixView* out, std::size_t out_capacity,
                       MutableMatrixView* scratch, std::size_t scratch_capacity,
                       const SwinSiteOverlay* overlay = nullptr,
                       const SwinProjectionSite* site = nullptr) {
    if (x.data == nullptr || expand.data == nullptr || contract.data == nullptr
        || residual.data == nullptr || scale.data == nullptr
        || (mid != nullptr && mid->data == nullptr)) {
        return Status::kNullPointer;
    }
    if (out == nullptr || out->data == nullptr || scratch == nullptr
        || scratch->data == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_linear_operand(x);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(expand);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (mid != nullptr) {
        const Status status = validate_linear_operand(*mid);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(contract);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (x.cols != expand.cols || out->rows != x.rows) {
        return Status::kInvalidDimensions;
    }
    if (mid != nullptr) {
        if (heads < 1 || heads > kMaxLinearDim / kHeadDim
            || mid->rows != heads * kHeadDim || expand.rows != heads * mid->cols
            || contract.cols != mid->rows) {
            return Status::kInvalidDimensions;
        }
    } else if (contract.cols != expand.rows) {
        return Status::kInvalidDimensions;
    }
    if (residual.rows != x.rows || residual.cols != contract.rows
        || scale.rows != 1 || scale.cols != contract.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_residual_scale(residual, scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (scratch->rows != x.rows || scratch->cols != expand.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out, contract.rows, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_output(scratch, expand.rows,
                                                    scratch_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    // All validation is complete: from here every stage is guaranteed kOk and out
    // and scratch are written.
    MutableMatrixView hidden;
    hidden.data = scratch->data;
    hidden.rows = x.rows;
    hidden.cols = expand.rows;
    hidden.row_stride = scratch->row_stride;
    {
        const Status status = swin_hidden(x, expand, &hidden, scratch_capacity,
                                          overlay, site);
        if (status != Status::kOk) {
            return status;
        }
    }
    float mid_row[kMaxLinearDim];
    for (int r = 0; r < x.rows; ++r) {
        const float* hidden_row = scratch->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(scratch->row_stride);
        const float* line = hidden_row;
        int width = expand.rows;
        if (mid != nullptr) {
            swin_mid_row(hidden_row, heads, mid->cols, *mid, mid_row);
            line = mid_row;
            width = mid->rows;
        }
        const float* residual_row = residual.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(residual.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out->row_stride);
        swin_mlp_row_tail(line, width, contract, residual_row, scale, out_row);
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_MLP_HPP
