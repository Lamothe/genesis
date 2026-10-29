// Native Swin block composition: the `swin_block`/`swin_core` body of the
// windowed-attention blocks — the e4m3 input rounding (`round8`), the shifted
// window passes (`_to_windows`), the per-window MLP -> stored-result rounding ->
// attention chain with the one-head unrounded-skip rule, the map reassembly
// (`_from_windows`) and the f16 `wide` output. CPU-only, typed, allocation-free;
// no GPU, headset, proprietary payload, Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "swin_block"/"swin_core"). The full MIT licence notice
// is retained in vr/NATIVE-SWIN-BLOCK.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_BLOCK_HPP
#define DLSS5_NR_NATIVE_SWIN_BLOCK_HPP

#include "native-swin-attention.hpp"
#include "native-swin-windows.hpp"

namespace dlss5_nr {

// ---- the block's weight set ------------------------------------------------------------------------
// The learned operands of one block, all passed explicitly (original synthetic
// values in tests; no learned weights are used or implied). `mid` is the grouped
// middle projection (null on the single-head path, exactly as the MLP slice).
// `head_scale` is the attention slice's bounded head-scale row (finite, magnitude
// at most `kMaxHeadScale` = 65504).
struct SwinBlockParams {
    MatrixView expand;
    const MatrixView* mid = nullptr;
    int heads = 1;
    MatrixView contract;
    MatrixView mlp_scale;
    MatrixView qkv;
    MatrixView head_scale;
    MatrixView exp_bias;
    MatrixView attn_scale;
    MatrixView out_proj;
    // The int4 mixed network's per-site overlay (the pinned SwinSite); null
    // keeps the e4m3 dataflow. The callers keep the overlay (and its pointed
    // matrices) alive for the call.
    const SwinSiteOverlay* site = nullptr;
};

// ---- pre-validation helpers ------------------------------------------------------------------------

// The block-level pairing checks, mirroring the staged entries' pairing exactly so
// every staged call below is guaranteed kOk once the block's validation returns.
inline Status swin_block_pairings(const FeatureBatchView& x,
                                  const SwinBlockParams& p) {
    if (p.heads < 1 || p.heads > kMaxLinearDim / (3 * kHeadDim)) {
        return Status::kInvalidDimensions;
    }
    const int channels = x.channels;
    if (channels != p.heads * kHeadDim) {
        return Status::kInvalidDimensions;
    }
    if (p.expand.cols != channels || p.qkv.rows != 3 * channels
        || p.qkv.cols != channels || p.out_proj.rows != channels
        || p.out_proj.cols != channels) {
        return Status::kInvalidDimensions;
    }
    if (p.mid != nullptr) {
        if (p.mid->cols < 1 || p.mid->cols > kMaxLinearDim) {
            return Status::kInvalidDimensions;
        }
        if (p.mid->rows != p.heads * kHeadDim
            || p.expand.rows != p.heads * p.mid->cols
            || p.contract.cols != p.mid->rows) {
            return Status::kInvalidDimensions;
        }
    } else if (p.contract.cols != p.expand.rows) {
        return Status::kInvalidDimensions;
    }
    // The MLP scratch width bound (swin_mlp validates its scratch with
    // validate_linear_output(..., expand.rows, ...)), mirrored here so the
    // staged call cannot reject after the first write.
    if (p.expand.rows < 1 || p.expand.rows > kMaxLinearDim) {
        return Status::kInvalidDimensions;
    }
    if (p.contract.rows != channels || p.mlp_scale.rows != 1
        || p.mlp_scale.cols != channels) {
        return Status::kInvalidDimensions;
    }
    if (p.head_scale.rows != 1 || p.head_scale.cols != p.heads
        || p.exp_bias.rows != p.heads
        || p.attn_scale.rows != 1 || p.attn_scale.cols != channels) {
        return Status::kInvalidDimensions;
    }
    return Status::kOk;
}

// ---- the G8 batched seam's scratch (chunked site projections) ------------------------------------
// The swin windows are arithmetically independent and the two site-projected
// stages (the MLP expand/hidden projection and the attention QKV projection)
// are row-independent matrix products, so a chunk of kSwinBatchWindows
// windows computes exactly the per-window results in ONE swin_mlp and ONE
// swin_attention call — one executor call per stage per chunk instead of one
// per 64-token window (the per-call staging thrash that made the per-window
// GPU seam measure a regression; see NATIVE-PERFORMANCE.md G6). The chunk
// work is carried by the SwinProjectionSite's `batch_work` buffer: a null
// buffer or any capacity below the formulas below keeps the per-window path
// exactly as before.
inline constexpr int kSwinBatchWindows = 32;  // windows per batched chunk

// The chunk work (tightly packed, in order): y (rows * channels), y_q
// (rows * channels), the MLP hidden (rows * expand_rows) and the attention
// scratch (rows * 3 * channels + 4 * rows * channels + 2 * heads * kBiasTable).
inline std::size_t swin_batch_work_needed(int rows, int channels,
                                          int expand_rows, int heads) {
    const std::size_t r = static_cast<std::size_t>(rows);
    return r * static_cast<std::size_t>(channels)       // y
        + r * static_cast<std::size_t>(channels)        // y_q
        + r * static_cast<std::size_t>(expand_rows)     // the MLP hidden
        + r * (static_cast<std::size_t>(3) * channels)  // the qkv accumulator
        + 4 * r * static_cast<std::size_t>(channels)    // q, k, v, context
        + 2 * static_cast<std::size_t>(heads) * kBiasTable;
}

// The full site scratch for one chunk: swin_linear's dense input and
// products (products cover max(expand_rows, 3 * channels) rows per token —
// the expand and QKV projections' larger height) plus the chunk work above.
// Callers size for rows = kSwinBatchWindows * kWindowTokens, the largest
// possible chunk; the last chunk of a row is never larger.
inline std::size_t swin_site_scratch_needed(int rows, int channels,
                                            int expand_rows, int heads) {
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t expand = static_cast<std::size_t>(expand_rows);
    const std::size_t qkv = static_cast<std::size_t>(3) * channels;
    const std::size_t product_rows = expand > qkv ? expand : qkv;
    return r * static_cast<std::size_t>(channels)  // swin_linear's dense input
        + r * product_rows                         // swin_linear's products
        + swin_batch_work_needed(rows, channels, expand_rows, heads);
}

// ---- the block body (`swin_block`) ---------------------------------------------------------------------
// One block on the channels-first map `x [C, rows, cols]`, exactly as the
// reference's `swin_block`:
//   quantised, grid = _to_windows(round8(x), shift_rows, shift_cols)
//   residual        = quantised, or _to_windows(x_wide, ...)[0] when x_wide is given
//   per window:     y = _mlp(quantised_w, residual_w);  y_q = store(y)
//                   skip = y (heads == 1) or y_q (several heads)
//                   out_w = _attention(skip, y_q)
//   acc  = _from_windows(out, grid, rows, cols)      (tight [C, rows, cols])
//   wide = _f16(acc)
// `shift_rows`/`shift_cols` are the window shifts in tiles (0 or -1 only).
// `wide_dst` receives the f16 `wide` map; `acc_dst`, when given, receives the
// pre-f16 `acc` map (the accumulator mode of the reference). Both are tight
// channels-first buffers of channels * rows * cols floats. When `acc_dst` is
// null the `acc` map is kept in `scratch` and `acc_count` is ignored.
//
// Operand classes: `x` is finite float32 (rounded to e4m3 per element before the
// products); `x_wide`, when given, has `x`'s dimensions and holds finite,
// exactly binary16-representable values (the f16 data the MLP residual uses; the
// reference's `swin_core` passes f16-exact `x` here). The weight classes are the
// MLP and attention slices' (e4m3-grid matrices, f16-exact scale rows, the
// bounded head-scale row and finite exp_bias).
//
// The scratch buffer is caller-owned and holds, in order (all tightly packed):
//   the e4m3 map (C*rows*cols), the quantised token batch (windows*64*C),
//   the residual token batch (only when x_wide is given), the out token batch,
//   one window's y and y_q (64*C each), the MLP scratch (64*expand.rows),
//   the attention scratch (64*3C + 4*64*C + 2*heads*4096), and the acc map
//   (only when acc_dst is null).
// Validation (nulls, dimensions, strides, pairings, value classes, capacities)
// completes before the first write: on any error `wide_dst`, `acc_dst` and
// `scratch` are all untouched. One scoped exception (the G4 site surface,
// unchanged here): with a projection site attached, the site's own per-call
// checks (a null executor, null gather/scatter buffers or undersized ones)
// report kInvalidBuffer from the first window batch — after the scratch's
// window token batches have been written — while `wide_dst` and `acc_dst`
// stay untouched. The buffers must not overlap each other or the input maps
// (caller discipline, never rejected).
//
// The reference chunks windows only for scheduling; windows are arithmetically
// independent, so this implementation processes them in batches — one window
// at a time on the CPU paths, or one kSwinBatchWindows-window chunk per
// swin_mlp/swin_attention call on the G8 batched site path — with no
// observable difference.
inline Status swin_block(const FeatureBatchView& x, const FeatureBatchView* x_wide,
                         int shift_rows, int shift_cols,
                         const SwinBlockParams& params,
                         float* wide_dst, std::size_t wide_count,
                         float* acc_dst, std::size_t acc_count,
                         float* scratch, std::size_t scratch_capacity,
                         const SwinProjectionSite* site = nullptr) {
    if (x.data == nullptr || wide_dst == nullptr || scratch == nullptr) {
        return Status::kNullPointer;
    }
    if (params.expand.data == nullptr || params.contract.data == nullptr
        || params.mlp_scale.data == nullptr || params.qkv.data == nullptr
        || params.head_scale.data == nullptr || params.exp_bias.data == nullptr
        || params.attn_scale.data == nullptr || params.out_proj.data == nullptr) {
        return Status::kNullPointer;
    }
    if (params.mid != nullptr && params.mid->data == nullptr) {
        return Status::kNullPointer;
    }
    if (x.channels < 1 || x.channels > kMaxTokenChannels
        || x.rows < 1 || x.rows > kMaxRows || x.cols < 1 || x.cols > kMaxCols) {
        return Status::kInvalidDimensions;
    }
    if (x.row_stride < static_cast<std::ptrdiff_t>(x.cols)
        || x.row_stride > static_cast<std::ptrdiff_t>(kMaxCols)
        || x.plane_stride < static_cast<std::ptrdiff_t>(x.rows) * x.row_stride
        || x.plane_stride > static_cast<std::ptrdiff_t>(kMaxRows) * kMaxCols) {
        return Status::kInvalidStride;
    }
    // front_h/front_w (and the swin_to_windows grid outputs below) are
    // write-only here: the staged entries require the out-params, while the
    // window count is derived from window_rows * window_cols and each pass
    // re-derives its own grid internally.
    int window_rows = 0;
    int front_h = 0;
    int window_cols = 0;
    int front_w = 0;
    {
        const Status status = swin_window_grid(x.rows, shift_rows,
                                               &window_rows, &front_h);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = swin_window_grid(x.cols, shift_cols,
                                               &window_cols, &front_w);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (x_wide != nullptr) {
        if (x_wide->data == nullptr) {
            return Status::kNullPointer;
        }
        if (x_wide->channels != x.channels || x_wide->rows != x.rows
            || x_wide->cols != x.cols) {
            return Status::kInvalidDimensions;
        }
        if (x_wide->row_stride < static_cast<std::ptrdiff_t>(x_wide->cols)
            || x_wide->row_stride > static_cast<std::ptrdiff_t>(kMaxCols)
            || x_wide->plane_stride
                < static_cast<std::ptrdiff_t>(x_wide->rows) * x_wide->row_stride
            || x_wide->plane_stride
                > static_cast<std::ptrdiff_t>(kMaxRows) * kMaxCols) {
            return Status::kInvalidStride;
        }
    }
    {
        const Status status = swin_block_pairings(x, params);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(params.expand);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(params.contract);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (params.mid != nullptr) {
        const Status status = validate_linear_operand(*params.mid);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(params.qkv);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(params.out_proj);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_f16_matrix(params.mlp_scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_f16_matrix(params.attn_scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_head_scale(params.head_scale);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_bias_table_finite(params.exp_bias);
        if (status != Status::kOk) {
            return status;
        }
    }
    // Value classes of the input maps (the staged entries never scan them).
    for (int c = 0; c < x.channels; ++c) {
        const float* plane = x.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(x.plane_stride);
        for (int r = 0; r < x.rows; ++r) {
            const float* row = plane + static_cast<std::size_t>(r)
                * static_cast<std::size_t>(x.row_stride);
            for (int col = 0; col < x.cols; ++col) {
                if (!std::isfinite(row[col])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    if (x_wide != nullptr) {
        for (int c = 0; c < x_wide->channels; ++c) {
            const float* plane = x_wide->data + static_cast<std::size_t>(c)
                * static_cast<std::size_t>(x_wide->plane_stride);
            for (int r = 0; r < x_wide->rows; ++r) {
                const float* row = plane + static_cast<std::size_t>(r)
                    * static_cast<std::size_t>(x_wide->row_stride);
                for (int col = 0; col < x_wide->cols; ++col) {
                    if (!std::isfinite(row[col]) || !is_f16(row[col])) {
                        return Status::kInvalidValues;
                    }
                }
            }
        }
    }
    const std::size_t map_count = static_cast<std::size_t>(x.channels)
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    if (wide_count < map_count) {
        return Status::kInvalidBuffer;
    }
    if (acc_dst != nullptr && acc_count < map_count) {
        return Status::kInvalidBuffer;
    }
    // The scratch layout (all tightly packed).
    const int channels = x.channels;
    const long long windows = static_cast<long long>(window_rows) * window_cols;
    const std::size_t token_count = static_cast<std::size_t>(windows)
        * kWindowTokens * static_cast<std::size_t>(channels);
    float* map_quantised = scratch;
    float* tokens_q = map_quantised + map_count;
    float* tokens_r = tokens_q + token_count;
    const std::size_t residual_tokens = x_wide != nullptr ? token_count : 0;
    float* tokens_out = tokens_r + residual_tokens;
    float* y_buffer = tokens_out + token_count;
    float* y_q_buffer = y_buffer
        + static_cast<std::size_t>(kWindowTokens) * channels;
    float* mlp_scratch = y_q_buffer
        + static_cast<std::size_t>(kWindowTokens) * channels;
    float* attn_scratch = mlp_scratch
        + static_cast<std::size_t>(kWindowTokens)
            * static_cast<std::size_t>(params.expand.rows);
    const std::size_t attn_needed = static_cast<std::size_t>(kWindowTokens)
            * (static_cast<std::size_t>(3) * channels)
        + 4 * static_cast<std::size_t>(kWindowTokens)
            * static_cast<std::size_t>(channels)
        + 2 * static_cast<std::size_t>(params.exp_bias.rows) * kBiasTable;
    float* acc_map = acc_dst != nullptr ? nullptr : attn_scratch + attn_needed;
    const std::size_t acc_extra = acc_dst != nullptr ? 0 : map_count;
    const std::size_t needed = map_count + token_count + residual_tokens
        + token_count
        + 2 * static_cast<std::size_t>(kWindowTokens) * channels
        + static_cast<std::size_t>(kWindowTokens)
            * static_cast<std::size_t>(params.expand.rows)
        + attn_needed + acc_extra;
    if (scratch_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    // All validation is complete: every staged call below is guaranteed kOk on
    // the CPU paths (and wide_dst, acc_dst and scratch are written). With a
    // site attached the site's per-call rejection class above can still report
    // kInvalidBuffer from the first window batch, as documented.
    for (int c = 0; c < channels; ++c) {
        const float* plane = x.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(x.plane_stride);
        float* out_plane = map_quantised
            + static_cast<std::size_t>(c) * static_cast<std::size_t>(x.rows)
                * static_cast<std::size_t>(x.cols);
        for (int r = 0; r < x.rows; ++r) {
            const float* row = plane + static_cast<std::size_t>(r)
                * static_cast<std::size_t>(x.row_stride);
            float* out_row = out_plane
                + static_cast<std::size_t>(r) * static_cast<std::size_t>(x.cols);
            for (int col = 0; col < x.cols; ++col) {
                out_row[col] = e4m3(row[col]);
            }
        }
    }
    FeatureBatchView quantised_map;
    quantised_map.data = map_quantised;
    quantised_map.channels = channels;
    quantised_map.rows = x.rows;
    quantised_map.cols = x.cols;
    quantised_map.plane_stride =
        static_cast<std::ptrdiff_t>(x.rows) * static_cast<std::ptrdiff_t>(x.cols);
    quantised_map.row_stride = static_cast<std::ptrdiff_t>(x.cols);
    WindowTokenView q_tokens;
    q_tokens.data = tokens_q;
    q_tokens.channels = channels;
    q_tokens.window_stride =
        static_cast<std::ptrdiff_t>(kWindowTokens) * channels;
    q_tokens.token_stride = channels;
    WindowGrid grid;
    {
        const Status status = swin_to_windows(quantised_map, shift_rows,
                                              shift_cols, &grid, &q_tokens,
                                              token_count);
        if (status != Status::kOk) {
            return status;
        }
    }
    WindowTokenView r_tokens = q_tokens;
    if (x_wide != nullptr) {
        r_tokens.data = tokens_r;
        WindowGrid grid2;  // write-only: the reference keeps only the tokens ([0])
        const Status status = swin_to_windows(*x_wide, shift_rows, shift_cols,
                                              &grid2, &r_tokens, token_count);
        if (status != Status::kOk) {
            return status;
        }
    }
    const long long window_total = windows;
    // One window batch: the MLP -> stored-result -> attention chain over
    // `rows` tokens starting at window `w0`. Shared verbatim by the per-window
    // loop and the G8 chunked loop below, so the two paths differ only in the
    // batch extent (every stage is row/window independent, hence
    // bit-identical across extents).
    const auto run_window_batch = [&](long long w0, int windows_in_batch,
                                      float* y_dst, float* y_q_dst,
                                      float* mlp_dst, float* attn_dst,
                                      std::size_t attn_capacity) -> Status {
        const int rows = windows_in_batch * kWindowTokens;
        MatrixView x_batch;
        x_batch.data = tokens_q + static_cast<std::size_t>(w0)
            * static_cast<std::size_t>(q_tokens.window_stride);
        x_batch.rows = rows;
        x_batch.cols = channels;
        x_batch.row_stride = q_tokens.token_stride;
        MatrixView residual_batch = x_batch;
        residual_batch.data = r_tokens.data + static_cast<std::size_t>(w0)
            * static_cast<std::size_t>(r_tokens.window_stride);
        residual_batch.row_stride = r_tokens.token_stride;
        MutableMatrixView y_view;
        y_view.data = y_dst;
        y_view.rows = rows;
        y_view.cols = channels;
        y_view.row_stride = channels;
        MutableMatrixView mlp_scratch_view;
        mlp_scratch_view.data = mlp_dst;
        mlp_scratch_view.rows = rows;
        mlp_scratch_view.cols = params.expand.rows;
        mlp_scratch_view.row_stride = params.expand.rows;
        {
            const Status status = swin_mlp(x_batch, params.expand, params.mid,
                                           params.heads, params.contract,
                                           residual_batch, params.mlp_scale,
                                           &y_view,
                                           static_cast<std::size_t>(rows)
                                               * channels,
                                           &mlp_scratch_view,
                                           static_cast<std::size_t>(rows)
                                               * params.expand.rows,
                                           params.site, site);
            if (status != Status::kOk) {
                return status;
            }
        }
        const std::size_t batch_floats =
            static_cast<std::size_t>(rows) * channels;
        for (std::size_t i = 0; i < batch_floats; ++i) {
            y_q_dst[i] = swin_store(y_dst[i]);
        }
        MatrixView y_q_view;
        y_q_view.data = y_q_dst;
        y_q_view.rows = rows;
        y_q_view.cols = channels;
        y_q_view.row_stride = channels;
        MatrixView skip_view;
        skip_view.data = params.heads == 1 ? y_dst : y_q_dst;
        skip_view.rows = rows;
        skip_view.cols = channels;
        skip_view.row_stride = channels;
        MutableMatrixView out_view;
        out_view.data = tokens_out + static_cast<std::size_t>(w0)
            * static_cast<std::size_t>(kWindowTokens)
            * static_cast<std::size_t>(channels);
        out_view.rows = rows;
        out_view.cols = channels;
        out_view.row_stride = channels;
        return swin_attention(y_q_view, params.qkv, params.head_scale,
                              params.exp_bias, skip_view, params.attn_scale,
                              params.out_proj, &out_view, batch_floats,
                              attn_dst, attn_capacity,
                              params.site != nullptr ? params.site->int4_qkv
                                                     : nullptr,
                              site);
    };
    // The G8 chunked path: one batch per kSwinBatchWindows windows when the
    // site provides chunk-sized buffers (executor, dense input, products and
    // batch work). Any CAPACITY shortfall (or a null matmul_f32) keeps the
    // per-window path below exactly as before (its per-call site checks still
    // apply). A site with a null executor or null gather/scatter buffers is
    // the G4 rejection class — kInvalidBuffer from the first swin_linear call
    // in BOTH paths (a caller invariant, pinned by the tests), not a fallback.
    const long long chunk_windows = window_total < kSwinBatchWindows
        ? window_total : kSwinBatchWindows;
    const int chunk_rows = static_cast<int>(chunk_windows * kWindowTokens);
    const std::size_t chunk_products =
        static_cast<std::size_t>(chunk_rows)
        * static_cast<std::size_t>(params.expand.rows);
    const std::size_t chunk_qkv_products =
        static_cast<std::size_t>(chunk_rows)
        * static_cast<std::size_t>(3) * channels;
    const std::size_t chunk_product_capacity =
        chunk_products > chunk_qkv_products ? chunk_products
                                            : chunk_qkv_products;
    const bool batched = site != nullptr && site->ex != nullptr
        && site->ex->matmul_f32 != nullptr && site->batch_work != nullptr
        && site->input_capacity
               >= static_cast<std::size_t>(chunk_rows) * channels
        && site->products_capacity >= chunk_product_capacity
        && site->batch_capacity >= swin_batch_work_needed(
               chunk_rows, channels, params.expand.rows, params.heads);
    if (batched) {
        float* const work = site->batch_work;
        float* const work_y = work;
        float* const work_y_q = work_y
            + static_cast<std::size_t>(chunk_rows) * channels;
        float* const work_mlp = work_y_q
            + static_cast<std::size_t>(chunk_rows) * channels;
        float* const work_attn = work_mlp
            + static_cast<std::size_t>(chunk_rows) * params.expand.rows;
        for (long long w0 = 0; w0 < window_total; w0 += kSwinBatchWindows) {
            const long long left = window_total - w0;
            const int windows_in_batch = static_cast<int>(
                left < kSwinBatchWindows ? left : kSwinBatchWindows);
            const int rows = windows_in_batch * kWindowTokens;
            const std::size_t attn_capacity =
                static_cast<std::size_t>(rows)
                    * (static_cast<std::size_t>(3) * channels)
                + 4 * static_cast<std::size_t>(rows) * channels
                + 2 * static_cast<std::size_t>(params.exp_bias.rows) * kBiasTable;
            const Status status = run_window_batch(w0, windows_in_batch,
                                                   work_y, work_y_q, work_mlp,
                                                   work_attn, attn_capacity);
            if (status != Status::kOk) {
                return status;
            }
        }
    } else {
        for (long long w = 0; w < window_total; ++w) {
            const std::size_t attn_capacity =
                static_cast<std::size_t>(kWindowTokens)
                    * (static_cast<std::size_t>(3) * channels)
                + 4 * static_cast<std::size_t>(kWindowTokens) * channels
                + 2 * static_cast<std::size_t>(params.exp_bias.rows) * kBiasTable;
            const Status status = run_window_batch(w, 1, y_buffer, y_q_buffer,
                                                   mlp_scratch, attn_scratch,
                                                   attn_capacity);
            if (status != Status::kOk) {
                return status;
            }
        }
    }
    {
        WindowTokenView out_tokens;
        out_tokens.data = tokens_out;
        out_tokens.windows = static_cast<int>(window_total);
        out_tokens.channels = channels;
        out_tokens.window_stride =
            static_cast<std::ptrdiff_t>(kWindowTokens) * channels;
        out_tokens.token_stride = channels;
        float* acc_target = acc_dst != nullptr ? acc_dst : acc_map;
        const std::size_t acc_target_count = map_count;
        const Status status = swin_from_windows(out_tokens, grid, x.rows, x.cols,
                                                acc_target, acc_target_count);
        if (status != Status::kOk) {
            return status;
        }
        const float* acc_read = acc_target;
        for (std::size_t i = 0; i < map_count; ++i) {
            wide_dst[i] = f16(acc_read[i]);
        }
    }
    return Status::kOk;
}

// `swin_core`: `swin_block` on f16-exact data — `x` doubles as the MLP residual,
// exactly as the reference's `swin_core(params, x, shift_x, shift_y)`.
inline Status swin_core(const FeatureBatchView& x,
                        int shift_rows, int shift_cols,
                        const SwinBlockParams& params,
                        float* wide_dst, std::size_t wide_count,
                        float* acc_dst, std::size_t acc_count,
                        float* scratch, std::size_t scratch_capacity,
                        const SwinProjectionSite* site = nullptr) {
    return swin_block(x, &x, shift_rows, shift_cols, params,
                      wide_dst, wide_count, acc_dst, acc_count,
                      scratch, scratch_capacity, site);
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_BLOCK_HPP
