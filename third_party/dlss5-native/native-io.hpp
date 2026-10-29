// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Native Linux DLSS-style neural rendering — host-image boundary and the
// image-level native evaluation.
//
// The frame boundary follows the pinned reference's `dlssnr/io.py`
// (`frame_to_tensor`/`tensor_to_frame` at the same MIT pin as the rest of the
// campaign): 8-bit RGB host rows map to the planar float colour map through
// the `levels` table (k/255), padded to the working extent by repeating the
// frame mirrored about its last row/column (`_mirror`: reflect an
// out-of-range index once about the last pixel, then clamp what is still
// outside); the result clamps to [0,1] and rounds to 8 bits half-up
// (floor(255 x + 0.5)) after cropping to the frame extent.
//
// The chain's spatial convention (native-graph-plan.hpp, pinned plan.py):
// the tensor is [C, rows, columns] with rows = the image height axis;
// `LayerSpec.out_w` is the row count and `out_h` the column count. The
// planar layout here is `map[c * rows * cols + row * cols + col]`.
//
// `image_evaluation` ties the boundary to the typed run: host frame in,
// planar image map -> `network_run` (the pre block lifts the map; the post
// block's planar output is cropped and rounded), with the temporal carrier's
// history colour fed to the input lift (the single-frame stub when no carrier
// is given). The host stays the caller's boundary: the weight-file load and
// the split feed-forward's `ffwd` boundary decode remain host duties.
//
// Allocation-free; validation completes before any output write. No GPU,
// headset, proprietary payload, Python or PyTorch runtime required. No vendor
// code and no vendor correspondence.
#ifndef DLSS5_NR_NATIVE_IO_HPP
#define DLSS5_NR_NATIVE_IO_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "native-eval.hpp"
#include "native-history.hpp"
#include "native-input-lift.hpp"
#include "native-network.hpp"

namespace dlss5_nr {

namespace io_detail {

// The pinned `_mirror`: reflect an out-of-range index once about the last
// pixel, then clamp what is still outside. In range the index is unchanged.
inline int mirror_index(int index, int extent) {
    const int reflected = index < extent ? index : 2 * extent - index - 2;
    return reflected < 0 ? 0 : (reflected > extent - 1 ? extent - 1 : reflected);
}

// The pinned `levels` table: k/255 as float32.
inline float level(std::uint8_t code) {
    return static_cast<float>(static_cast<int>(code)) / 255.0f;
}

inline Status validate_host(const std::uint8_t* host, int host_row_stride,
                            int width, int height) {
    if (host == nullptr) {
        return Status::kNullPointer;
    }
    if (width < 1 || height < 1 || width > kMaxCols || height > kMaxRows) {
        return Status::kInvalidDimensions;
    }
    if (host_row_stride < 3 * width) {
        return Status::kInvalidStride;
    }
    return Status::kOk;
}

}  // namespace io_detail

// Host frame in: `host` is uint8 [height][width][3] rows of `host_row_stride`
// bytes. `out` is the planar float colour map of the working extent in the
// chain's orientation: `out[c * work_rows * work_cols + r * work_cols + col]`
// with `work_rows >= height` (the row axis) and `work_cols >= width`, values
// k/255 with the mirrored edge padding (`_mirror` on each axis). Validation
// completes before any output write.
inline Status frame_from_u8(const std::uint8_t* host, int host_row_stride,
                            int width, int height,
                            float* out, std::size_t out_count,
                            int work_rows, int work_cols) {
    Status status = io_detail::validate_host(host, host_row_stride, width,
                                             height);
    if (status != Status::kOk) {
        return status;
    }
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    if (work_rows < height || work_cols < width || work_rows > kMaxRows
        || work_cols > kMaxCols
        || static_cast<long long>(work_rows) * work_cols > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    const std::size_t needed = 3u * static_cast<std::size_t>(work_rows)
        * static_cast<std::size_t>(work_cols);
    if (out_count < needed) {
        return Status::kInvalidBuffer;
    }
    for (int r = 0; r < work_rows; ++r) {
        const int sy = io_detail::mirror_index(r, height);
        for (int col = 0; col < work_cols; ++col) {
            const int sx = io_detail::mirror_index(col, width);
            const std::uint8_t* pixel = host
                + static_cast<std::size_t>(sy)
                    * static_cast<std::size_t>(host_row_stride)
                + 3 * static_cast<std::ptrdiff_t>(sx);
            const std::size_t at = static_cast<std::size_t>(r)
                * static_cast<std::size_t>(work_cols)
                + static_cast<std::size_t>(col);
            out[at] = io_detail::level(pixel[0]);
            out[static_cast<std::size_t>(work_rows) * work_cols + at]
                = io_detail::level(pixel[1]);
            out[2u * static_cast<std::size_t>(work_rows) * work_cols + at]
                = io_detail::level(pixel[2]);
        }
    }
    return Status::kOk;
}

// Host frame out: `in` is the planar float colour map of the working extent
// (the post block's output) in the chain's orientation, cropped to
// [0, height) x [0, width) and written as uint8 [height][width][3] rows of
// `host_row_stride` bytes, clamped to [0, 1] and rounded to 8 bits half-up
// exactly as the pinned `tensor_to_frame` (whole = floor(scaled); code =
// whole + (scaled - whole >= 0.5)); NaN inputs map to code 0 (an explicit
// policy choice: the pinned cast leaves NaN's code implementation-defined).
// Validation completes before any output write.
inline Status frame_to_u8(const float* in, int work_rows, int work_cols,
                          std::uint8_t* host, int host_row_stride,
                          int width, int height) {
    if (in == nullptr) {
        return Status::kNullPointer;
    }
    Status status = io_detail::validate_host(host, host_row_stride, width,
                                             height);
    if (status != Status::kOk) {
        return status;
    }
    if (work_rows < height || work_cols < width || work_rows > kMaxRows
        || work_cols > kMaxCols
        || static_cast<long long>(work_rows) * work_cols > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    for (int y = 0; y < height; ++y) {
        std::uint8_t* row = host
            + static_cast<std::size_t>(y)
                * static_cast<std::size_t>(host_row_stride);
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float value = in[static_cast<std::size_t>(c)
                    * static_cast<std::size_t>(work_rows) * work_cols
                    + static_cast<std::size_t>(y) * work_cols
                    + static_cast<std::size_t>(x)];
                float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
                if (!(clamped == clamped)) {
                    clamped = 0.0f;
                }
                // The pinned rule: floor + (frac >= 0.5). The subtraction is
                // exact; floor(scaled + 0.5f) would diverge at 0.5 - 2^-25.
                const float scaled = clamped * 255.0f;
                const float whole = std::floor(scaled);
                row[3 * x + c] = static_cast<std::uint8_t>(
                    whole + ((scaled - whole) >= 0.5f ? 1.0f : 0.0f));
            }
        }
    }
    return Status::kOk;
}

// The image-level native evaluation, map level: host frame in, the run's
// padded planar output map out (run.outputs[count - 1].main, the post
// block's [3, work_rows, work_cols] result — cropped to the host frame by
// frame_to_u8). `run` supplies the layer/release wiring and the caller's
// `LayerValue` outputs; the caller's `EvalContext` supplies the records and
// controls. `image_storage` holds the planar image map (>= 3 * work_w *
// work_h floats, outside the arena scratch). `history` may be null (the
// single-frame stub: the lift's slots 7..9 copy the current colour — the
// reference's no-history stand-in) or the temporal carrier of
// native-history.hpp: its history colour is fed to the input lift's slots
// 7..9 as-is, including the zero map at start-up and after a reset (the
// capture contract; the stub is never a temporal-mode feed). The caller
// commits the produced frame into the carrier and applies the reset
// protocol (see temporal_image_evaluation, which performs the whole step);
// this function performs one evaluation only. Validation completes before
// any output write.
inline Status image_map_evaluation(const std::uint8_t* host,
                               int host_row_stride, int width, int height,
                               float* image_storage, std::size_t image_count,
                               const LayerSpec* specs, int count,
                               NetworkRun run, EvalContext& context,
                               const TemporalCarrier* history) {
    if (specs == nullptr || image_storage == nullptr) {
        return Status::kNullPointer;
    }
    if (count < 1) {
        return Status::kInvalidDimensions;
    }
    if (run.outputs == nullptr || run.layer == nullptr) {
        return Status::kNullPointer;
    }
    // The run must execute against this context (the history feed and the
    // records live there); the by-value copy is ours to wire.
    run.context = &context;
    // The image map's extent is the pre block's input (the plan's padded
    // frame extent, its pooled output doubled back); rows = height axis.
    const LayerSpec& pre = specs[0];
    const LayerSpec& post = specs[count - 1];
    if (pre.kind != LayerKind::kPreBlock || pre.c_in != 3
        || pre.out_w < 1 || pre.out_h < 1
        || pre.out_w > kMaxRows / 2 || pre.out_h > kMaxCols / 2) {
        // The last pair bounds the doubled extent below (no signed overflow
        // in work_rows/work_cols, both within the map limits).
        return Status::kInvalidValues;
    }
    if (post.kind != LayerKind::kPostBlock) {
        // The final row must be the post block: anything else leaves a
        // smaller map whose read-back would pass the row's end.
        return Status::kInvalidValues;
    }
    const int work_rows = pre.out_w * 2;
    const int work_cols = pre.out_h * 2;
    // The temporal history (when given) must match the padded extent in the
    // chain's orientation and hold [0, 1] colour (the lift's requirement).
    const FrameView history_view = history == nullptr
        ? FrameView{}
        : temporal_history_view(*history);
    if (history != nullptr && history_view.rgb == nullptr) {
        // A carrier whose map never initialised (temporal_init failed) has
        // rows/cols but no bytes; reject here rather than mid-run.
        return Status::kNullPointer;
    }
    if (history != nullptr
        && (history_view.rows != work_rows || history_view.cols != work_cols)) {
        return Status::kInvalidDimensions;
    }
    Status status = frame_from_u8(host, host_row_stride, width, height,
                                  image_storage, image_count, work_rows,
                                  work_cols);
    if (status != Status::kOk) {
        return status;
    }
    FeatureBatchView image = FeatureBatchView{
        image_storage, 3, work_rows, work_cols,
        static_cast<std::ptrdiff_t>(work_rows) * work_cols,
        static_cast<std::ptrdiff_t>(work_cols)};
    run.image = &image;
    context.history = history == nullptr ? nullptr : &history_view;
    status = network_run(specs, count, run);
    context.history = nullptr;
    if (status != Status::kOk) {
        return status;
    }
    const FeatureBatchView& result = run.outputs[count - 1].main;
    if (result.channels != 3 || result.rows != work_rows
        || result.cols != work_cols
        || result.plane_stride
            != static_cast<std::ptrdiff_t>(work_rows) * work_cols
        || result.row_stride != static_cast<std::ptrdiff_t>(work_cols)) {
        // The extent and the planar-map layout convention (tight
        // [c*rows*cols + r*cols + col]): frame_to_u8's read-back and
        // frame_to_history's store both assume it, so a padded result is
        // rejected here rather than misread.
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// The image-level native evaluation: host frame in, enhanced host frame out,
// through the typed run. Same boundary as image_map_evaluation with the
// host read-back (frame_to_u8) appended. Validation completes before any
// output write.
inline Status image_evaluation(const std::uint8_t* host, int host_row_stride,
                               int width, int height, std::uint8_t* out_host,
                               int out_host_row_stride,
                               float* image_storage, std::size_t image_count,
                               const LayerSpec* specs, int count,
                               NetworkRun run, EvalContext& context,
                               const TemporalCarrier* history) {
    const Status status = image_map_evaluation(host, host_row_stride, width,
        height, image_storage, image_count, specs, count, run, context,
        history);
    if (status != Status::kOk) {
        return status;
    }
    const FeatureBatchView& result = run.outputs[count - 1].main;
    return frame_to_u8(result.data, result.rows, result.cols, out_host,
                       out_host_row_stride, width, height);
}

// The FP16 output-texture store: the run's padded planar [3, rows, cols]
// output map becomes the interleaved RGB pixel frame in `store` (pixel p,
// channel c: store[3p + c] = map[c * rows * cols + p]), each value rounded
// to the binary16 grid — the output-texture store of the hardware route
// (the temporal carrier holds FP16 texture content bit-exactly). For values
// already on the binary16 grid the rounding is the identity. Validation is
// a full first pass: a non-finite value or one beyond the max finite
// binary16 magnitude (65504) rejects the whole store before any write
// (kInvalidValues). `store_count >= 3 * rows * cols` floats.
inline Status frame_to_history(const FeatureBatchView& map, float* store,
                               std::size_t store_count) {
    if (store == nullptr) {
        return Status::kNullPointer;
    }
    if (map.data == nullptr || map.channels != 3 || map.rows < 1
        || map.cols < 1 || map.rows > kMaxRows || map.cols > kMaxCols
        || static_cast<long long>(map.rows) * map.cols > kMaxPixels) {
        return Status::kInvalidValues;
    }
    if (map.plane_stride < 0 || map.row_stride < map.cols
        || static_cast<long long>(map.rows) * map.row_stride
            > map.plane_stride) {
        return Status::kInvalidStride;
    }
    const std::size_t plane =
        static_cast<std::size_t>(map.rows) * static_cast<std::size_t>(map.cols);
    if (store_count < 3 * plane) {
        return Status::kInvalidBuffer;
    }
    const std::size_t plane_stride = static_cast<std::size_t>(map.plane_stride);
    const std::size_t row_stride = static_cast<std::size_t>(map.row_stride);
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < map.rows; ++r) {
            for (int col = 0; col < map.cols; ++col) {
                const float v = map.data[c * plane_stride
                    + static_cast<std::size_t>(r) * row_stride
                    + static_cast<std::size_t>(col)];
                if (!(v == v) || v > 65504.0f || v < -65504.0f) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    for (int r = 0; r < map.rows; ++r) {
        for (int col = 0; col < map.cols; ++col) {
            const std::size_t p = static_cast<std::size_t>(r) * map.cols
                + static_cast<std::size_t>(col);
            for (int c = 0; c < 3; ++c) {
                store[3 * p + static_cast<std::size_t>(c)] = f16(map.data[
                    c * plane_stride + static_cast<std::size_t>(r) * row_stride
                    + static_cast<std::size_t>(col)]);
            }
        }
    }
    return Status::kOk;
}

// The temporal history consumer: one frame step of the temporal protocol
// (native-history.hpp's capture contract):
//
//   1. `reset_before` applies DLSSNR.Reset first — the carrier drops to its
//      zero map and counter 0 (the reset frame behaves as if history = 0).
//   2. The history colour fed to the input lift is ALWAYS the carrier's
//      view — the zero map at start-up and after a reset, the previous
//      frame's stored output otherwise. The single-frame stub (the lift's
//      current-colour copy) is never a temporal-mode feed.
//   3. The run's padded planar output is stored through the FP16
//      output-texture rounding (frame_to_history, into the caller's
//      `store` scratch) and committed bit-exactly as the next frame's
//      history (temporal_commit advances the frames-since-reset counter).
//
// Validation completes before any output write: on any error `out_host`,
// `store` and the carrier are untouched. The requested reset, when asked,
// applies immediately and first — a failed step leaves the carrier as the
// reset left it (zero map, counter 0), which is the caller's requested
// state, not an output. Allocation-free.
inline Status temporal_image_evaluation(const std::uint8_t* host,
                               int host_row_stride, int width, int height,
                               std::uint8_t* out_host, int out_host_row_stride,
                               float* image_storage, std::size_t image_count,
                               const LayerSpec* specs, int count,
                               NetworkRun run, EvalContext& context,
                               TemporalCarrier& history, bool reset_before,
                               float* store, std::size_t store_count) {
    if (reset_before) {
        const Status reset_status = temporal_reset(history);
        if (reset_status != Status::kOk) {
            return reset_status;
        }
    }
    if (temporal_counter(history) == 0xFFFFFFFFu) {
        // temporal_commit's exhaustion guard, hoisted to the step's entry:
        // the frame cannot be committed, so nothing may be written (the
        // store and out_host would otherwise hold an uncommittable frame).
        return Status::kInvalidValues;
    }
    // The host read-back's failure classes are validated here so the commit
    // below is the step's only state change after the run (frame_to_u8
    // cannot fail once this passes: its only remaining classes are the work
    // extent, pre-validated by frame_from_u8 and the result-geometry check
    // inside the map evaluation).
    const Status host_status = io_detail::validate_host(out_host,
        out_host_row_stride, width, height);
    if (host_status != Status::kOk) {
        return host_status;
    }
    const Status status = image_map_evaluation(host, host_row_stride, width,
        height, image_storage, image_count, specs, count, run, context,
        &history);
    if (status != Status::kOk) {
        return status;
    }
    const FeatureBatchView& result = run.outputs[count - 1].main;
    const Status store_status = frame_to_history(result, store, store_count);
    if (store_status != Status::kOk) {
        return store_status;
    }
    const Status read_status = frame_to_u8(result.data, result.rows,
        result.cols, out_host, out_host_row_stride, width, height);
    if (read_status != Status::kOk) {
        return read_status;  // unreachable: pre-validated above and inside
    }
    return temporal_commit(history, FrameView{store, result.rows, result.cols,
        static_cast<std::ptrdiff_t>(3) * result.cols});
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_IO_HPP
