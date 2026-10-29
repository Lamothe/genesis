// Native Swin resampling: the `ds` and `upsample` layer variants around the
// block body — the pooled learned `C -> 2C` projection (`downsample`), the
// learned `2C -> C` projected nearest-2x gated blend (`upsample_blend`), and the
// thin `SwinLayer` compositions (`swin_layer_plain`, `swin_layer_upsample`).
// CPU-only, typed, allocation-free; no GPU, headset, proprietary payload,
// Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "downsample"/"upsample_blend"/"_project"/"_gated_sum"/
// "SwinLayer"). The full MIT licence notice is retained in
// vr/NATIVE-SWIN-RESAMPLE.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_RESAMPLE_HPP
#define DLSS5_NR_NATIVE_SWIN_RESAMPLE_HPP

#include "native-blend.hpp"
#include "native-pre-post.hpp"
#include "native-swin-block.hpp"

namespace dlss5_nr {

// ---- the resampling weight sets ------------------------------------------------------------------
// `resample` (the `ds` variant's learned `C -> 2C` projection of the pooled map)
// is [2C, C]; `project` (the `upsample` variant's learned `2C -> C` projection of
// the coarser level) is [C, 2C]; both e4m3-grid matrices. `gain` is the upsample
// blend's per-channel gain row [1, C] of finite, exactly binary16-representable
// values (the reference's `upsample_gain` f16 row). All operands are passed
// explicitly (original synthetic values in tests; no learned weights are used or
// implied).

// ---- the elementwise gated sum (`_gated_sum`) ----------------------------------------------------
// `f16(skip * gain + other)`: the product and the sum are computed in float64
// and the whole expression is rounded to binary16 ONCE (ties to even,
// `f16_from_double`'s documented zones). A float32 form of the sum is NOT
// rounding-equivalent: it can round onto a binary16 midpoint and then break
// that tie the wrong way (the tests pin a discriminator). Extreme operand sets
// can round to ±infinity exactly as the reference's float16 conversion does
// (the (65504, 65520) band rounds to 65504, |sum| >= 65520 to infinity); the
// result is never NaN for finite f16 operands.
inline float gated_sum_element(float skip, float gain, float other) {
    const double fused = static_cast<double>(skip) * static_cast<double>(gain)
        + static_cast<double>(other);
    return f16_from_double(fused);
}

// ---- the 1x1 projection (`_project`) -------------------------------------------------------------
// `sum_k weight[k] * map[k, y, x]` in float32: the products are exact (e4m3-grid
// operands) and are accumulated in the explicit k-ascending sequential float32
// accumulator this campaign's slices fix as a convention (the reference's
// plain-product path states no summation-order contract; no vendor
// correspondence is claimed). `swin_project_narrow` rounds the sum to binary16
// once (`_project`'s `narrow`); `swin_project_store` applies the stored-result
// double rounding f16 -> e4m3 (`downsample`'s projected pooled map).
inline float swin_project_dot(const float* weight, int k_count,
                              const float* base, std::ptrdiff_t plane_stride,
                              std::size_t offset) {
    float sum = 0.0f;
    for (int k = 0; k < k_count; ++k) {
        sum += weight[k]
            * base[static_cast<std::size_t>(k)
                * static_cast<std::size_t>(plane_stride) + offset];
    }
    return sum;
}

inline float swin_project_narrow(const float* weight, int k_count,
                                 const float* base, std::ptrdiff_t plane_stride,
                                 std::size_t offset) {
    return f16(swin_project_dot(weight, k_count, base, plane_stride, offset));
}

inline float swin_project_store(const float* weight, int k_count,
                                const float* base, std::ptrdiff_t plane_stride,
                                std::size_t offset) {
    return swin_store(swin_project_dot(weight, k_count, base, plane_stride,
                                       offset));
}

// ---- shared stride/extent validation for the maps -----------------------------------------------

inline Status validate_resample_map(const FeatureBatchView& m,
                                    int max_channels) {
    if (m.data == nullptr) {
        return Status::kNullPointer;
    }
    if (m.channels < 1 || m.channels > max_channels
        || m.rows < 1 || m.rows > kMaxRows || m.cols < 1 || m.cols > kMaxCols
        || static_cast<long long>(m.rows) * m.cols > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    if (m.row_stride < static_cast<std::ptrdiff_t>(m.cols)
        || m.row_stride > static_cast<std::ptrdiff_t>(kMaxCols)
        || m.plane_stride < static_cast<std::ptrdiff_t>(m.rows) * m.row_stride
        || m.plane_stride > static_cast<std::ptrdiff_t>(kMaxRows) * kMaxCols) {
        return Status::kInvalidStride;
    }
    return Status::kOk;
}

// ---- the block scratch mirror (`swin_block`'s `needed` layout) -----------------------------------
// Mirrors swin_block's internal scratch layout exactly (its `needed`), so a
// capacity boundary tested here pins the same arithmetic swin_block validates.
// `acc_staged` mirrors `acc_dst == nullptr` in the block call: the entries
// below either pass their own acc region (false) or let the block stage its
// acc map inside the scratch (true, `acc_extra` = map_count).
// Assumes validated inputs (the shift pair and the pairings); returns 0 for an
// unsupported shift pair, which the callers surface as kInvalidControls first.
inline std::size_t swin_block_scratch_needed(const FeatureBatchView& x,
                                             bool wide_residual,
                                             bool acc_staged,
                                             int shift_rows, int shift_cols,
                                             const SwinBlockParams& p) {
    int window_rows = 0;
    int front_h = 0;
    int window_cols = 0;
    int front_w = 0;
    if (swin_window_grid(x.rows, shift_rows, &window_rows, &front_h)
            != Status::kOk
        || swin_window_grid(x.cols, shift_cols, &window_cols, &front_w)
            != Status::kOk) {
        return 0;
    }
    const std::size_t map_count = static_cast<std::size_t>(x.channels)
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    const std::size_t acc_extra = acc_staged ? map_count : 0;
    const std::size_t token_count = static_cast<std::size_t>(
        static_cast<long long>(window_rows) * window_cols)
        * kWindowTokens * static_cast<std::size_t>(x.channels);
    const std::size_t residual_tokens = wide_residual ? token_count : 0;
    const std::size_t attn_needed = static_cast<std::size_t>(kWindowTokens)
            * (static_cast<std::size_t>(3) * x.channels)
        + 4 * static_cast<std::size_t>(kWindowTokens)
            * static_cast<std::size_t>(x.channels)
        + 2 * static_cast<std::size_t>(p.exp_bias.rows) * kBiasTable;
    return map_count + token_count + residual_tokens + token_count
        + 2 * static_cast<std::size_t>(kWindowTokens) * x.channels
        + static_cast<std::size_t>(kWindowTokens)
            * static_cast<std::size_t>(p.expand.rows)
        + attn_needed + acc_extra;
}

// ---- downsample (the `ds` variant) ---------------------------------------------------------------
// The reference's `downsample(params, resample, x, shift_x, shift_y, rows, columns)`:
//   wide, acc = swin_block(params, x, shift_rows, shift_cols, accumulator=True)
//   pooled[c] = pool2x2(acc plane c)                    [C, rows/2, cols/2], e4m3
//   next[n, oy, ox] = store(sum_k resample[n, k] * pooled[k, oy, ox])
//   body = round8(wide)
// `next` receives the projected pooled map resized to out_rows x out_cols by
// keeping its top-left: zero filled where the pooled extent is smaller
// (`_pad_to`'s zero padding), cropped bottom/right where it is larger (the
// reference's negative `_pad_to` padding crops there). `body` receives the
// block's own e4m3 output (`round8(wide)`).
//
// The map `x` is the block's input class (finite float32, C = heads * 32
// channels) with EVEN spatial extents (pool2x2's contract; the pipeline pads to
// even working extents first). `resample` is the [2C, C] e4m3-grid matrix above.
// `next_dst` is tight 2C * out_rows * out_cols floats, `body_dst` tight
// C * rows * cols. The scratch buffer is caller-owned and holds, in order:
//   the acc map (C*rows*cols), the pooled map (C*(rows/2)*(cols/2)) and
//   swin_block's scratch (swin_block_scratch_needed(x, false, false, shifts,
//   params); the block's acc map is the `acc_map` region above).
// Validation (nulls, dimensions, strides, pairings, operand classes,
// capacities) completes before the first write: the block call is the first
// write and is atomic on its own remaining classes (the input-map value scan),
// so on any error `next_dst`, `body_dst` and `scratch` are all untouched. The
// buffers must not overlap each other or the input map (caller discipline).
//
// Disclosed non-reproduction: the reference's SwinLayer.forward passes
// `spec.out_w, spec.out_h` into this function's (rows, columns) slots; the plan
// metadata convention that makes that naming correct is out of scope — this
// entry's output extents are explicit (out_rows, out_cols).
inline Status swin_downsample(const FeatureBatchView& x,
                              int shift_rows, int shift_cols,
                              const SwinBlockParams& params,
                              const MatrixView& resample,
                              int out_rows, int out_cols,
                              float* next_dst, std::size_t next_count,
                              float* body_dst, std::size_t body_count,
                              float* scratch, std::size_t scratch_capacity,
                              const SwinProjectionSite* site = nullptr) {
    if (x.data == nullptr || next_dst == nullptr || body_dst == nullptr
        || scratch == nullptr || resample.data == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_resample_map(x, kMaxTokenChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    if ((x.rows & 1) != 0 || (x.cols & 1) != 0) {
        return Status::kInvalidDimensions;
    }
    if (params.expand.data == nullptr || params.contract.data == nullptr
        || params.mlp_scale.data == nullptr || params.qkv.data == nullptr
        || params.head_scale.data == nullptr || params.exp_bias.data == nullptr
        || params.attn_scale.data == nullptr || params.out_proj.data == nullptr
        || (params.mid != nullptr && params.mid->data == nullptr)) {
        return Status::kNullPointer;
    }
    {
        const Status status = swin_block_pairings(x, params);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (resample.rows != 2 * x.channels || resample.cols != x.channels) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_operand(resample);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (out_rows < 1 || out_rows > kMaxRows || out_cols < 1
        || out_cols > kMaxCols
        || static_cast<long long>(out_rows) * out_cols > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    // The shift pair must be one the window grid accepts; surfaced here so an
    // unsupported pair reports kInvalidControls before any other class.
    int probe_windows = 0;
    int probe_front = 0;
    if (swin_window_grid(x.rows, shift_rows, &probe_windows, &probe_front)
            != Status::kOk
        || swin_window_grid(x.cols, shift_cols, &probe_windows, &probe_front)
            != Status::kOk) {
        return Status::kInvalidControls;
    }
    const std::size_t channels = static_cast<std::size_t>(x.channels);
    const std::size_t map_count = channels * static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t pooled_count = channels
        * static_cast<std::size_t>(x.rows / 2)
        * static_cast<std::size_t>(x.cols / 2);
    const std::size_t next_needed = 2 * channels
        * static_cast<std::size_t>(out_rows) * static_cast<std::size_t>(out_cols);
    if (next_count < next_needed || body_count < map_count) {
        return Status::kInvalidBuffer;
    }
    const std::size_t block_needed =
        swin_block_scratch_needed(x, false, false, shift_rows, shift_cols,
                                  params);
    const std::size_t needed = map_count + pooled_count + block_needed;
    if (block_needed == 0 || scratch_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    // All validation is complete: the staged block call below is the first
    // write and is atomic on its own remaining classes.
    float* acc_map = scratch;
    float* pooled_map = acc_map + map_count;
    float* block_scratch = pooled_map + pooled_count;
    {
        const Status status = swin_block(x, nullptr, shift_rows, shift_cols,
                                         params, body_dst, map_count,
                                         acc_map, map_count,
                                         block_scratch, block_needed, site);
        if (status != Status::kOk) {
            return status;
        }
    }
    // `round8(wide)` in place: the wide buffer is f16-exact, so the in-place
    // e4m3 rounding is the reference's round8 of its own output.
    for (std::size_t i = 0; i < map_count; ++i) {
        body_dst[i] = e4m3(body_dst[i]);
    }
    // The pooled map: pool2x2 per plane (its output is on the e4m3 grid).
    const std::size_t plane_pixels = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t pooled_pixels = static_cast<std::size_t>(x.rows / 2)
        * static_cast<std::size_t>(x.cols / 2);
    for (std::size_t c = 0; c < channels; ++c) {
        PlaneView plane;
        plane.data = acc_map + c * plane_pixels;
        plane.rows = x.rows;
        plane.cols = x.cols;
        plane.row_stride = x.cols;
        float* pooled_plane = pooled_map + c * pooled_pixels;
        {
            const Status status = pool2x2(plane, pooled_plane, pooled_pixels);
            if (status != Status::kOk) {
                return status;
            }
        }
    }
    // The projected pooled map, placed top-left in `next` with zero fill/crop.
    for (int n = 0; n < 2 * x.channels; ++n) {
        const float* weight = resample.data
            + static_cast<std::size_t>(n)
                * static_cast<std::size_t>(resample.row_stride);
        float* out_plane = next_dst
            + static_cast<std::size_t>(n) * static_cast<std::size_t>(out_rows)
                * static_cast<std::size_t>(out_cols);
        for (int oy = 0; oy < out_rows; ++oy) {
            float* out_row = out_plane
                + static_cast<std::size_t>(oy) * static_cast<std::size_t>(out_cols);
            for (int ox = 0; ox < out_cols; ++ox) {
                out_row[ox] = 0.0f;
            }
        }
        const int kept_rows = x.rows / 2 < out_rows ? x.rows / 2 : out_rows;
        const int kept_cols = x.cols / 2 < out_cols ? x.cols / 2 : out_cols;
        for (int oy = 0; oy < kept_rows; ++oy) {
            float* out_row = out_plane
                + static_cast<std::size_t>(oy) * static_cast<std::size_t>(out_cols);
            for (int ox = 0; ox < kept_cols; ++ox) {
                const std::size_t offset = static_cast<std::size_t>(oy)
                    * static_cast<std::size_t>(x.cols / 2)
                    + static_cast<std::size_t>(ox);
                out_row[ox] = swin_project_store(weight, x.channels,
                                                 pooled_map,
                                                 static_cast<std::ptrdiff_t>(
                                                     pooled_pixels),
                                                 offset);
            }
        }
    }
    return Status::kOk;
}

// ---- upsample_blend (the `upsample` variant's blend) ---------------------------------------------
// The reference's `upsample_blend(project, gain, x, skip)`:
//   projected[n, y, x0] = narrow(sum_k project[n, k] * x[k, y, x0])   [C, hx, wx]
//   blend[c, i, j] = f16(skip[c, i, j] * gain[c]
//                        + projected[c, min(i / 2, hx - 1), min(j / 2, wx - 1)])
// The blend is on the skip's grid [C, skip_rows, skip_cols]: the projected map
// is replicated 2x (each value repeated over a 2x2 block) and the last projected
// row/column is repeated where the skip is larger than the replicated map (the
// clamp in the reference's index arithmetic).
//
// Operand classes: `project` is the [C, 2C] e4m3-grid matrix and `gain` the
// [1, C] f16 row above; `x` (the coarser level) holds e4m3-grid values — the
// network feeds `round8` outputs here, matching the e4m3-linear slice's operand
// convention; `skip` (the matching `ds` layer's body) holds finite,
// exactly-binary16 values. `blend_dst` receives C * skip_rows * skip_cols tight
// floats; `scratch` holds the projected map (C * x.rows * x.cols floats).
// The projection sums are float32; the blend rounds to binary16 once per
// element, and extreme operand sets round to ±infinity exactly as the
// reference's float16 conversion (disclosed; the result is never NaN).
// Validation (nulls, dimensions, strides, pairings, value classes, capacities)
// completes before the first write: on any error `blend_dst` and `scratch` are
// both untouched. The buffers must not overlap the input maps (caller
// discipline).
inline Status swin_upsample_blend(const MatrixView& project,
                                  const MatrixView& gain,
                                  const FeatureBatchView& x,
                                  const FeatureBatchView& skip,
                                  float* blend_dst, std::size_t blend_count,
                                  float* scratch,
                                  std::size_t scratch_capacity) {
    if (project.data == nullptr || gain.data == nullptr || x.data == nullptr
        || skip.data == nullptr || blend_dst == nullptr || scratch == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_resample_map(x, 2 * kMaxTokenChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_resample_map(skip, kMaxTokenChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    const std::size_t channels = static_cast<std::size_t>(skip.channels);
    if (x.channels != 2 * skip.channels || skip.channels < 1) {
        return Status::kInvalidDimensions;
    }
    if (project.rows != skip.channels || project.cols != 2 * skip.channels
        || gain.rows != 1 || gain.cols != skip.channels) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_operand(project);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_f16_matrix(gain);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int c = 0; c < x.channels; ++c) {
        const float* plane = x.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(x.plane_stride);
        for (int r = 0; r < x.rows; ++r) {
            const float* row = plane + static_cast<std::size_t>(r)
                * static_cast<std::size_t>(x.row_stride);
            for (int col = 0; col < x.cols; ++col) {
                if (!is_e4m3(row[col])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    for (int c = 0; c < skip.channels; ++c) {
        const float* plane = skip.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(skip.plane_stride);
        for (int r = 0; r < skip.rows; ++r) {
            const float* row = plane + static_cast<std::size_t>(r)
                * static_cast<std::size_t>(skip.row_stride);
            for (int col = 0; col < skip.cols; ++col) {
                if (!std::isfinite(row[col]) || !is_f16(row[col])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    const std::size_t blend_needed = channels
        * static_cast<std::size_t>(skip.rows)
        * static_cast<std::size_t>(skip.cols);
    const std::size_t projected_count = channels
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    if (blend_count < blend_needed) {
        return Status::kInvalidBuffer;
    }
    if (scratch_capacity < projected_count) {
        return Status::kInvalidBuffer;
    }
    // All validation is complete; one write pass over the projected map and the
    // blend follows, all operands already guaranteed accepted.
    float* projected = scratch;
    for (int c = 0; c < skip.channels; ++c) {
        const float* weight = project.data
            + static_cast<std::size_t>(c)
                * static_cast<std::size_t>(project.row_stride);
        float* out_plane = projected + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
        for (int r = 0; r < x.rows; ++r) {
            float* out_row = out_plane + static_cast<std::size_t>(r)
                * static_cast<std::size_t>(x.cols);
            for (int col = 0; col < x.cols; ++col) {
                // The source read honours x's row stride: element (r, col)
                // of plane k sits at k * plane_stride + r * row_stride + col.
                const std::size_t offset = static_cast<std::size_t>(r)
                    * static_cast<std::size_t>(x.row_stride)
                    + static_cast<std::size_t>(col);
                out_row[col] = swin_project_narrow(weight, x.channels, x.data,
                                                   x.plane_stride, offset);
            }
        }
    }
    for (int c = 0; c < skip.channels; ++c) {
        const float* skip_plane = skip.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(skip.plane_stride);
        const float* proj_plane = projected + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
        // `gain` is the [1, C] row: channel c is row 0, column c.
        const float gain_c = gain.data[static_cast<std::size_t>(c)];
        float* out_plane = blend_dst + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(skip.rows)
            * static_cast<std::size_t>(skip.cols);
        for (int i = 0; i < skip.rows; ++i) {
            const int src_r = i / 2 < x.rows ? i / 2 : x.rows - 1;
            const float* skip_row = skip_plane + static_cast<std::size_t>(i)
                * static_cast<std::size_t>(skip.row_stride);
            float* out_row = out_plane + static_cast<std::size_t>(i)
                * static_cast<std::size_t>(skip.cols);
            for (int j = 0; j < skip.cols; ++j) {
                const int src_c = j / 2 < x.cols ? j / 2 : x.cols - 1;
                const float other = proj_plane[static_cast<std::size_t>(src_r)
                    * static_cast<std::size_t>(x.cols)
                    + static_cast<std::size_t>(src_c)];
                out_row[j] = gated_sum_element(skip_row[j], gain_c, other);
            }
        }
    }
    return Status::kOk;
}

// ---- the `SwinLayer` plain body (plain / inpview / outview) --------------------------------------
// `round8(swin_block(params, x, shift_rows, shift_cols))`: the block's f16
// `wide` output rounded to the e4m3 grid. `inpview`/`outview` compute exactly
// this (the reference's variant names only describe the neighbouring layer's
// buffer layout). Scratch is swin_block's (swin_block_scratch_needed(x, false,
// true, shifts, params); no acc destination is passed, so the block stages its
// acc map inside the scratch). The block's validation and write contract
// surface unchanged (on any error `out_dst` and `scratch` are untouched). The
// optional site threads swin_block's projection seam (and its G8 batched
// chunk path) exactly as the other layer entries; the default keeps the CPU
// kernels.
inline Status swin_layer_plain(const FeatureBatchView& x,
                               int shift_rows, int shift_cols,
                               const SwinBlockParams& params,
                               float* out_dst, std::size_t out_count,
                               float* scratch, std::size_t scratch_capacity,
                               const SwinProjectionSite* site = nullptr) {
    const std::size_t map_count = static_cast<std::size_t>(x.channels)
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    {
        const Status status = swin_block(x, nullptr, shift_rows, shift_cols,
                                         params, out_dst, out_count,
                                         nullptr, 0, scratch, scratch_capacity,
                                         site);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (std::size_t i = 0; i < map_count; ++i) {
        out_dst[i] = e4m3(out_dst[i]);
    }
    return Status::kOk;
}

// ---- the `SwinLayer` upsample variant ------------------------------------------------------------
// The reference's `SwinLayer.forward` for `variant == "upsample"`:
//   blend = upsample_blend(project, gain, x, skip)     [C, skip_rows, skip_cols]
//   C == 32: wide = swin_block(params, blend, shifts, x_wide = blend)
//   C != 32: wide = swin_block(params, round8(blend), shifts)
//   out = round8(wide)
// The `C == 32` path runs the block on the f16 blend with the blend doubling as
// the MLP residual (the `swin_core` rule); every other width rounds the blend
// first and takes the quantised residual path. `x` is the coarser level [2C, ...]
// and `skip` the matching `ds` body [C, ...]; the output is on the skip's grid.
//
// Scratch holds, in order: the blend map (C*skip.rows*skip.cols), the projected
// map (C*x.rows*x.cols) and the block's scratch
// (swin_block_scratch_needed(blend map, C == 32, true, shifts, params)).
// Validation completes before the first write and the outputs are untouched on
// every error: the block's weight classes are pre-validated in the entry, in
// the block's own order, so the block call cannot reject on them after the
// blend stage has written. One disclosed exception: if an extreme operand set
// rounds any blend element outside the finite binary16 range (the reference's
// float16 conversion returns infinity there), the entry reports kInvalidValues
// after the blend stage has written the blend/scratch regions — outputs
// untouched — since the block's finite-map class would otherwise reject there.
inline Status swin_layer_upsample(const FeatureBatchView& x,
                                  const FeatureBatchView& skip,
                                  int shift_rows, int shift_cols,
                                  const SwinBlockParams& params,
                                  const MatrixView& project,
                                  const MatrixView& gain,
                                  float* out_dst, std::size_t out_count,
                                  float* scratch,
                                  std::size_t scratch_capacity,
                                  const SwinProjectionSite* site = nullptr) {
    if (x.data == nullptr || skip.data == nullptr || out_dst == nullptr
        || scratch == nullptr || project.data == nullptr
        || gain.data == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_resample_map(x, 2 * kMaxTokenChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_resample_map(skip, kMaxTokenChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (x.channels != 2 * skip.channels || skip.channels < 1) {
        return Status::kInvalidDimensions;
    }
    if (params.expand.data == nullptr || params.contract.data == nullptr
        || params.mlp_scale.data == nullptr || params.qkv.data == nullptr
        || params.head_scale.data == nullptr || params.exp_bias.data == nullptr
        || params.attn_scale.data == nullptr || params.out_proj.data == nullptr
        || (params.mid != nullptr && params.mid->data == nullptr)) {
        return Status::kNullPointer;
    }
    FeatureBatchView blend_map;
    blend_map.data = scratch;
    blend_map.channels = skip.channels;
    blend_map.rows = skip.rows;
    blend_map.cols = skip.cols;
    blend_map.plane_stride =
        static_cast<std::ptrdiff_t>(skip.rows) * skip.cols;
    blend_map.row_stride = static_cast<std::ptrdiff_t>(skip.cols);
    {
        const Status status = swin_block_pairings(blend_map, params);
        if (status != Status::kOk) {
            return status;
        }
    }
    // The block's weight classes, pre-validated here in the block's own order
    // (pairings above, then the operand grids, the scale rows, the head scale
    // and the bias table) so the block call below cannot reject on them after
    // the blend stage has written the scratch regions.
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
    if (project.rows != skip.channels || project.cols != 2 * skip.channels
        || gain.rows != 1 || gain.cols != skip.channels) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_operand(project);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_f16_matrix(gain);
        if (status != Status::kOk) {
            return status;
        }
    }
    int probe_windows = 0;
    int probe_front = 0;
    if (swin_window_grid(skip.rows, shift_rows, &probe_windows, &probe_front)
            != Status::kOk
        || swin_window_grid(skip.cols, shift_cols, &probe_windows, &probe_front)
            != Status::kOk) {
        return Status::kInvalidControls;
    }
    const std::size_t channels = static_cast<std::size_t>(skip.channels);
    const std::size_t blend_count = channels
        * static_cast<std::size_t>(skip.rows)
        * static_cast<std::size_t>(skip.cols);
    const std::size_t projected_count = channels
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    const std::size_t block_needed = swin_block_scratch_needed(
        blend_map, skip.channels == 32, true, shift_rows, shift_cols, params);
    const std::size_t needed = blend_count + projected_count + block_needed;
    if (out_count < blend_count) {
        return Status::kInvalidBuffer;
    }
    if (block_needed == 0 || scratch_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    float* blend_region = scratch;
    float* projected_region = blend_region + blend_count;
    float* block_scratch = projected_region + projected_count;
    {
        const Status status = swin_upsample_blend(project, gain, x, skip,
                                                  blend_region, blend_count,
                                                  projected_region,
                                                  projected_count);
        if (status != Status::kOk) {
            return status;
        }
    }
    // The disclosed extreme-overflow class: a blend element rounded outside the
    // finite binary16 range (the reference returns infinity there) is rejected
    // before the block call; the outputs stay untouched.
    for (std::size_t i = 0; i < blend_count; ++i) {
        if (!std::isfinite(blend_region[i])) {
            return Status::kInvalidValues;
        }
    }
    if (skip.channels == 32) {
        // The f16 path: the blend doubles as the MLP residual (`swin_core`).
        const Status status = swin_block(blend_map, &blend_map, shift_rows,
                                         shift_cols, params,
                                         out_dst, out_count, nullptr, 0,
                                         block_scratch, block_needed, site);
        if (status != Status::kOk) {
            return status;
        }
    } else {
        for (std::size_t i = 0; i < blend_count; ++i) {
            blend_region[i] = e4m3(blend_region[i]);
        }
        const Status status = swin_block(blend_map, nullptr, shift_rows,
                                         shift_cols, params,
                                         out_dst, out_count, nullptr, 0,
                                         block_scratch, block_needed, site);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (std::size_t i = 0; i < blend_count; ++i) {
        out_dst[i] = e4m3(out_dst[i]);
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_RESAMPLE_HPP
