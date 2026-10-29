// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Native Linux DLSS-style neural rendering — temporal history carrier.
//
// The temporal contract established for the tested regime by the capture
// evidence in vr/TEMPORAL-HISTORY.md (gate stage 13; Proton-side evidence,
// stated in behavioural form):
//
//     output_N = M(inputs, history_N, counter_N)   history_N = output_{N-1}
//     counter_N = N since the last reset           (bit-exact)
//     Reset -> behave as if history = 0, counter = 0
//
// The carrier is one FP16 colour map per feature (the history texture; the
// capture's reset value is all zeros), written after the network output is
// complete. The reset frame ignores the history content (the capture's
// injectreset control): the evidence cannot distinguish "reset zeroes the
// texture" from "the model ignores the history when its counter is 0", so
// this stage implements the strongest form consistent with both readings -
// the behavioural form, with the carrier bytes zeroed at reset and start-up.
//
// The frame counter ("slot 1 word 25" in the captured parameter dump) is
// part of the contract and is tracked here; which stage consumes it is an
// open boundary of the proprietary route (TEMPORAL-HISTORY.md's "Next"), and
// the pinned single-frame reference exposes no counter input. The input
// lift's feature slots 7..9 are where the previous frame's colour goes in
// temporal mode; the lift's current-colour copy is the documented
// single-frame stub and is deliberately left untouched by this stage.
//
// Reference-derived naming and the Controls.history/Controls.reset field
// semantics ("accepted ... and are ignored" in the pinned single-frame path)
// are from the MIT reference mauri870/dlssnr-pytorch
// 3beef7935a8ef976b47ab78b906e75600c52d2de (dlssnr/io.py); the behavioural
// contract above is this repository's own capture evidence, not reference
// code. No network weights, captures, raw payloads or binaries are used or
// reproduced here.
//
// Allocation-free; validation completes before any state write, so on any
// error the carrier bytes, counter and history flag are untouched (the
// carrier's map fields alone take the documented failure form: `map` null
// with zero capacity). The frames-since-reset counter refuses to advance
// past UINT32_MAX rather than wrapping into the reset-frame value 0.
#ifndef DLSS5_NR_NATIVE_HISTORY_HPP
#define DLSS5_NR_NATIVE_HISTORY_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "native-input-lift.hpp"

namespace dlss5_nr {

// The per-feature history carrier and frame counter around one network run.
// `map` is the interleaved FP16-grid colour history (`rows * cols` pixels of
// three floats), the form the input lift's temporal history feed takes. The
// lift requires [0,1] colour at the plan's padded extent (rows = the height
// axis, per native-graph-plan.hpp); commit values outside [0,1] are legal
// here (the FP16 texture contract) but will be rejected when fed to the lift.
struct TemporalCarrier {
    float* map;
    std::size_t map_capacity;  // capacity of `map` in float elements (>= 3*rows*cols)
    int rows;
    int cols;
    std::uint32_t counter;
    bool has_history;
};

namespace temporal_detail {

// The reset value (all zeros) written over the carrier. The explicit null
// guard keeps the memset's non-null contract provable at every call site.
inline Status zero_map(const TemporalCarrier& state) {
    float* dest = state.map;
    if (dest == nullptr) {
        return Status::kNullPointer;
    }
    std::memset(dest, 0, 3u * static_cast<std::size_t>(state.rows)
        * static_cast<std::size_t>(state.cols) * sizeof(float));
    return Status::kOk;
}

inline Status validate_geometry(const TemporalCarrier& state) {
    if (state.map == nullptr) {
        return Status::kNullPointer;
    }
    // The input-lift bounds keep every size and index computation in range.
    if (state.rows < 1 || state.cols < 1 || state.rows > kMaxRows
        || state.cols > kMaxCols
        || static_cast<long long>(state.rows) * state.cols > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    const std::size_t needed = 3u * static_cast<std::size_t>(state.rows)
        * static_cast<std::size_t>(state.cols);
    if (state.map_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// A commit source: finite binary16-grid interleaved colour in one row-major
// map. The padding of a strided source is never read.
inline Status validate_source(const FrameView& frame, const TemporalCarrier& state) {
    if (frame.rgb == nullptr) {
        return Status::kNullPointer;
    }
    if (frame.rows != state.rows || frame.cols != state.cols) {
        return Status::kInvalidDimensions;
    }
    if (frame.row_stride < static_cast<std::ptrdiff_t>(3) * frame.cols
        || frame.row_stride > static_cast<std::ptrdiff_t>(3) * kMaxCols) {
        return Status::kInvalidStride;
    }
    for (int y = 0; y < frame.rows; ++y) {
        const float* row = frame.rgb
            + static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.row_stride);
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(3) * frame.cols; ++i) {
            const float value = row[i];
            // finite and binary16-representable: the carrier stores the value
            // bit-exactly (history_N = output_{N-1}), so anything the FP16
            // texture could not hold is rejected rather than rounded.
            if (!is_f16(value) || !std::isfinite(value)) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

}  // namespace temporal_detail

// Initialises the carrier over caller-provided storage: the reset value is all
// zeros (the capture's history0-init.bin), counter 0, no history yet. The
// storage belongs to the caller and must outlive the carrier.
inline Status temporal_init(TemporalCarrier& state, float* map,
                            std::size_t map_capacity, int rows, int cols) {
    const TemporalCarrier probe{map, map_capacity, rows, cols, 0u, false};
    const Status status = temporal_detail::validate_geometry(probe);
    if (status != Status::kOk) {
        state.map = nullptr;
        state.map_capacity = 0;
        return status;
    }
    state.map = map;
    state.map_capacity = map_capacity;
    state.rows = rows;
    state.cols = cols;
    state.counter = 0;
    state.has_history = false;
    return temporal_detail::zero_map(state);
}

// "Behave as if history = 0, counter = 0": the carrier bytes go to the reset
// value (all zeros) and the history flag drops, so any history content staged
// before the reset is ignored (the capture's injectreset control). Idempotent.
inline Status temporal_reset(TemporalCarrier& state) {
    const Status status = temporal_detail::validate_geometry(state);
    if (status != Status::kOk) {
        return status;
    }
    const Status zeroed = temporal_detail::zero_map(state);
    if (zeroed != Status::kOk) {
        return zeroed;
    }
    state.counter = 0;
    state.has_history = false;
    return Status::kOk;
}

// The frames-since-reset counter of the frame about to be evaluated.
inline std::uint32_t temporal_counter(const TemporalCarrier& state) {
    return state.counter;
}

// Whether the carrier holds a committed frame (false at start-up and after a
// reset: those frames run on the zero history).
inline bool temporal_has_history(const TemporalCarrier& state) {
    return state.has_history;
}

// The history colour for the frame about to be evaluated, as the interleaved
// view the input lift's temporal feed takes. At start-up and after a reset the
// view is the all-zero map (the behavioural "history = 0" form).
inline FrameView temporal_history_view(const TemporalCarrier& state) {
    return FrameView{state.map, state.rows, state.cols,
                     static_cast<std::ptrdiff_t>(3) * state.cols};
}

// Commits the evaluated frame as the next frame's history
// (history_{N+1} = output_N, bit-exactly) and advances the counter. The
// source is read through its own row stride; padding is never read.
inline Status temporal_commit(TemporalCarrier& state, const FrameView& output) {
    Status status = temporal_detail::validate_geometry(state);
    if (status != Status::kOk) {
        return status;
    }
    status = temporal_detail::validate_source(output, state);
    if (status != Status::kOk) {
        return status;
    }
    if (state.counter == 0xFFFFFFFFu) {
        // Refuse rather than wrap into the reset-frame value 0.
        return Status::kInvalidValues;
    }
    for (int y = 0; y < state.rows; ++y) {
        const float* src = output.rgb
            + static_cast<std::size_t>(y) * static_cast<std::size_t>(output.row_stride);
        float* dst = state.map
            + 3u * static_cast<std::size_t>(y) * static_cast<std::size_t>(state.cols);
        // memmove: a self-commit through temporal_history_view is legal.
        std::memmove(dst, src,
                     3u * static_cast<std::size_t>(state.cols) * sizeof(float));
    }
    state.counter += 1u;
    state.has_history = true;
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_HISTORY_HPP
