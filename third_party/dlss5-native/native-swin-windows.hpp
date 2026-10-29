// Native Swin window partition: the window-grid layout and the planar <-> window-token
// transforms of the reference's windowed-attention blocks. CPU-only, typed,
// allocation-free; no GPU, headset, proprietary payload, Python or PyTorch runtime
// required. Pure layout arithmetic: values pass through bit-exactly (including
// non-finite ones), padded regions read as +0.0 and dropped tokens are never written
// back.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "_window_grid", "_to_windows" and "_from_windows"). The full
// MIT licence notice is retained in vr/NATIVE-SWIN-WINDOWS.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented layout semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_WINDOWS_HPP
#define DLSS5_NR_NATIVE_SWIN_WINDOWS_HPP

#include "native-blend.hpp"

namespace dlss5_nr {

// ---- documented constants ----------------------------------------------------------------------
inline constexpr int kWindow = 8;               // pixels along each side of an attention window
inline constexpr int kTile = 4;                 // pixels along each side of a tile; a window is 2x2 tiles
inline constexpr int kTokens = kWindow * kWindow;   // 64 tokens per window
inline constexpr int kMaxTokenChannels = 256;   // bound on the channel count of a token batch
inline constexpr int kMaxWindows = 2049 * 2049; // ceil((kMaxRows + kTile) / kWindow) squared

// The window grid for one axis: window_rows/window_cols are windows along each axis of
// the padded plane, front_h/front_w the zero pixels placed in front (the reference's
// (rows, cols, front_h, front_w) grid tuple).
struct WindowGrid {
    int window_rows = 0;
    int window_cols = 0;
    int front_h = 0;
    int front_w = 0;
};

// A batch of window tokens: `windows` groups of kTokens tokens of `channels` floats
// each. Element (w, t, c) lives at
//   data + w * window_stride + t * token_stride + c
// with strides in floats: channels <= token_stride <= kMaxTokenChannels and
// kTokens * token_stride <= window_stride <= kTokens * kMaxTokenChannels.
// Token index within a window is tile order (tile row, tile column, y, x):
//   t = (py / kTile) * 32 + (px / kTile) * 16 + (py % kTile) * kTile + (px % kTile)
// for window-local pixel (py, px).
struct WindowTokenView {
    float* data = nullptr;
    int windows = 0;
    int channels = 0;
    std::ptrdiff_t window_stride = 0;
    std::ptrdiff_t token_stride = 0;
};

// (windows along one axis, zero pixels in front) for a grid moved by `shift` tiles:
//   front = -kTile * shift
//   windows = ceil((extent + front) / kWindow)
// Only shift = 0 and shift = -1 are supported (the reference rejects anything else).
inline Status swin_window_grid(int extent, int shift, int* windows_out, int* front_out) {
    if (windows_out == nullptr || front_out == nullptr) {
        return Status::kNullPointer;
    }
    if (shift != -1 && shift != 0) {
        return Status::kInvalidControls;
    }
    if (extent < 1 || extent > kMaxRows) {
        return Status::kInvalidDimensions;
    }
    const int front = -kTile * shift;
    const int windows = (extent + front + kWindow - 1) / kWindow;
    *windows_out = windows;
    *front_out = front;
    return Status::kOk;
}

// Planar feature batch -> window tokens, exactly as `_to_windows`: the batch is zero
// padded to the grid (front_h/front_w zeros in front, the remainder behind), split
// into kWindow x kWindow windows over the padded plane, and each window's pixels are
// written in tile order (tile row, tile column, y, x). `x` is a channels-first
// `FeatureBatchView`; `tokens` supplies data/strides/channels (channels must equal
// x.channels, windows is derived and written). Pixels the grid hangs over the frame
// edge become +0.0 token entries.
//
// Validation (dimensions, strides, controls, capacity) completes before the first
// output write, so on any error both outputs are untouched. `token_capacity` is the
// number of floats accessible at `tokens->data`; it must be at least
// (windows - 1) * window_stride + 63 * token_stride + channels (the last written
// element's offset plus one). Only
// elements (w, t, c) for c < channels are written: the token stride padding and the
// window stride padding are never touched. The token buffer must not overlap the
// input batch; overlapping writes can silently corrupt the copy (they are never
// rejected: all access stays in bounds). Values pass through bit-exactly, including
// non-finite ones (the reference's pad/copy is a pure layout operation).
inline Status swin_to_windows(const FeatureBatchView& x, int shift_rows, int shift_cols,
                              WindowGrid* grid_out, WindowTokenView* tokens,
                              std::size_t token_capacity) {
    if (x.data == nullptr || grid_out == nullptr || tokens == nullptr
        || tokens->data == nullptr) {
        return Status::kNullPointer;
    }
    if (x.channels < 1 || x.channels > kMaxTokenChannels) {
        return Status::kInvalidDimensions;
    }
    if (x.rows < 1 || x.rows > kMaxRows || x.cols < 1 || x.cols > kMaxCols) {
        return Status::kInvalidDimensions;
    }
    if (x.row_stride < static_cast<std::ptrdiff_t>(x.cols)
        || x.row_stride > static_cast<std::ptrdiff_t>(kMaxCols)
        || x.plane_stride < static_cast<std::ptrdiff_t>(x.rows) * x.row_stride
        || x.plane_stride > static_cast<std::ptrdiff_t>(kMaxRows) * kMaxCols) {
        return Status::kInvalidStride;
    }
    int window_rows = 0;
    int window_cols = 0;
    int front_h = 0;
    int front_w = 0;
    {
        const Status status = swin_window_grid(x.rows, shift_rows, &window_rows, &front_h);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = swin_window_grid(x.cols, shift_cols, &window_cols, &front_w);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (tokens->channels != x.channels) {
        return Status::kInvalidDimensions;
    }
    if (tokens->token_stride < static_cast<std::ptrdiff_t>(tokens->channels)
        || tokens->token_stride > static_cast<std::ptrdiff_t>(kMaxTokenChannels)
        || tokens->window_stride
            < static_cast<std::ptrdiff_t>(kTokens) * tokens->token_stride
        || tokens->window_stride
            > static_cast<std::ptrdiff_t>(kTokens) * kMaxTokenChannels) {
        return Status::kInvalidStride;
    }
    const long long windows = static_cast<long long>(window_rows) * window_cols;
    const std::size_t required
        = static_cast<std::size_t>(windows - 1)
            * static_cast<std::size_t>(tokens->window_stride)
        + static_cast<std::size_t>(kTokens - 1)
            * static_cast<std::size_t>(tokens->token_stride)
        + static_cast<std::size_t>(tokens->channels);
    if (token_capacity < required) {
        return Status::kInvalidBuffer;
    }
    grid_out->window_rows = window_rows;
    grid_out->window_cols = window_cols;
    grid_out->front_h = front_h;
    grid_out->front_w = front_w;
    tokens->windows = static_cast<int>(windows);
    for (int wy = 0; wy < window_rows; ++wy) {
        for (int wx = 0; wx < window_cols; ++wx) {
            const std::size_t w = static_cast<std::size_t>(wy) * window_cols + wx;
            float* window = tokens->data
                + w * static_cast<std::size_t>(tokens->window_stride);
            for (int t = 0; t < kTokens; ++t) {
                float* token = window
                    + static_cast<std::size_t>(t) * tokens->token_stride;
                const int qy = (t / 32) & 1;
                const int qx = (t / 16) & 1;
                const int y = (t / kTile) % kTile;
                const int px_in = t % kTile;
                const int padded_y = wy * kWindow + qy * kTile + y;
                const int padded_x = wx * kWindow + qx * kTile + px_in;
                const int src_y = padded_y - front_h;
                const int src_x = padded_x - front_w;
                const bool inside = src_y >= 0 && src_y < x.rows
                    && src_x >= 0 && src_x < x.cols;
                for (int c = 0; c < x.channels; ++c) {
                    if (inside) {
                        const float* plane = x.data
                            + static_cast<std::size_t>(c)
                                * static_cast<std::size_t>(x.plane_stride);
                        token[c] = plane[static_cast<std::size_t>(src_y)
                            * static_cast<std::size_t>(x.row_stride) + src_x];
                    } else {
                        token[c] = 0.0f;
                    }
                }
            }
        }
    }
    return Status::kOk;
}

// Window tokens -> planar output, exactly as `_from_windows`: the inverse mapping of
// swin_to_windows for a grid it produced. Pixels the grid hangs over the frame edge
// are dropped ("not written back"): dst receives exactly height x width pixels per
// channel, planes in channel order, row-major within a plane (channels * height *
// width floats). Tokens whose window-local pixels fall outside the crop are simply
// not read.
//
// Validation completes before the first output write. Values pass through
// bit-exactly. Like aliasing, the token buffer's extent is caller discipline: there
// is no capacity parameter, so the buffer must be readable over
// (windows - 1) * window_stride + 63 * token_stride + channels floats. The token
// buffer must not overlap dst (as above, never rejected).
inline Status swin_from_windows(const WindowTokenView& tokens, const WindowGrid& grid,
                                int height, int width, float* dst,
                                std::size_t dst_count) {
    if (tokens.data == nullptr || dst == nullptr) {
        return Status::kNullPointer;
    }
    if (height < 1 || height > kMaxRows || width < 1 || width > kMaxCols) {
        return Status::kInvalidDimensions;
    }
    if (grid.window_rows < 1 || grid.window_rows > kMaxRows / kWindow + 1
        || grid.window_cols < 1 || grid.window_cols > kMaxCols / kWindow + 1) {
        return Status::kInvalidDimensions;
    }
    if ((grid.front_h != 0 && grid.front_h != kTile)
        || (grid.front_w != 0 && grid.front_w != kTile)) {
        return Status::kInvalidDimensions;
    }
    if (grid.front_h + height > grid.window_rows * kWindow
        || grid.front_w + width > grid.window_cols * kWindow) {
        return Status::kInvalidDimensions;
    }
    const long long windows
        = static_cast<long long>(grid.window_rows) * grid.window_cols;
    if (tokens.windows < 1 || tokens.windows != windows) {
        return Status::kInvalidDimensions;
    }
    if (tokens.channels < 1 || tokens.channels > kMaxTokenChannels) {
        return Status::kInvalidDimensions;
    }
    if (tokens.token_stride < static_cast<std::ptrdiff_t>(tokens.channels)
        || tokens.token_stride > static_cast<std::ptrdiff_t>(kMaxTokenChannels)
        || tokens.window_stride
            < static_cast<std::ptrdiff_t>(kTokens) * tokens.token_stride
        || tokens.window_stride
            > static_cast<std::ptrdiff_t>(kTokens) * kMaxTokenChannels) {
        return Status::kInvalidStride;
    }
    const std::size_t needed = static_cast<std::size_t>(tokens.channels)
        * static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (dst_count < needed) {
        return Status::kInvalidBuffer;
    }
    for (int c = 0; c < tokens.channels; ++c) {
        for (int py = 0; py < height; ++py) {
            for (int px = 0; px < width; ++px) {
                const int padded_y = py + grid.front_h;
                const int padded_x = px + grid.front_w;
                const int wy = padded_y / kWindow;
                const int wx = padded_x / kWindow;
                const int local_y = padded_y % kWindow;
                const int local_x = padded_x % kWindow;
                const int t = (local_y / kTile) * 32 + (local_x / kTile) * 16
                    + (local_y % kTile) * kTile + (local_x % kTile);
                const std::size_t w
                    = static_cast<std::size_t>(wy) * grid.window_cols + wx;
                const float* token = tokens.data
                    + w * static_cast<std::size_t>(tokens.window_stride)
                    + static_cast<std::size_t>(t) * tokens.token_stride;
                dst[static_cast<std::size_t>(c) * static_cast<std::size_t>(height)
                        * static_cast<std::size_t>(width)
                    + static_cast<std::size_t>(py) * width + px] = token[c];
            }
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_WINDOWS_HPP
