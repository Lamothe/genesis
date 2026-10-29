// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// The stable host boundary for desktop renderers (the Centre Stage
// experiment's seam): a C ABI over the native evaluator. The host renders
// its scene at a LOW resolution, hands the RGB bytes to the boundary and
// presents the upscaled result. Synthetic weights: the transformation is
// ARBITRARY — `dlss5_nr_label()` returns the string a host must display
// alongside any presented output ("synthetic weights — arbitrary
// transform"). No vendor code, no proprietary payloads, no runtime or
// capture probes; CPU-only (the evaluator's kernels are the same ones the
// demo and the GPU seam use).
#ifndef DLSS5_NR_NATIVE_HOST_H
#define DLSS5_NR_NATIVE_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The Status enum's exact values (native-pre-post.hpp): the ABI is stable
// and must not track internal renumbering silently.
typedef enum dlss5_nr_status {
    DLSS5_NR_OK = 0,
    DLSS5_NR_NULL_POINTER = 1,
    DLSS5_NR_INVALID_DIMENSIONS = 2,
    DLSS5_NR_INVALID_STRIDE = 3,
    DLSS5_NR_INVALID_VALUES = 4,
    DLSS5_NR_INVALID_CONTROLS = 5,
    DLSS5_NR_INVALID_BUFFER = 6
} dlss5_nr_status;

typedef struct dlss5_nr_handle dlss5_nr_handle;

// The label a host must display alongside any presented output (UTF-8).
const char* dlss5_nr_label(void);
// The status code's name (a stable string for logs/overlays).
const char* dlss5_nr_status_name(dlss5_nr_status status);

// Create a pipeline for width x height RGB frames (8-bit, 3 bytes per
// pixel). `threads` selects the evaluator's parallelism (1..64; 0 = 8).
// The internal working extent comes from the plan's padding; frames whose
// padded extent exceeds the demo geometry's floor (320x320) are refused
// with DLSS5_NR_INVALID_DIMENSIONS (the documented bound of this
// boundary's fixed arena). On success `*out_handle` owns the pipeline; on
// any failure it is null and nothing was allocated.
dlss5_nr_status dlss5_nr_create(int width, int height, int threads,
                                dlss5_nr_handle** out_handle);

// One frame: `rgb` (row_stride bytes per row, at least 3*width) in,
// `out_rgb` (out_stride bytes per row, at least 3*width) out at the same
// extent (the host presents the upscaled result itself). A non-zero
// `reset` runs DLSSNR.Reset first (history = 0, counter = 0). The output
// buffer is untouched unless the call returns DLSS5_NR_OK.
dlss5_nr_status dlss5_nr_evaluate(dlss5_nr_handle* handle,
                                  const unsigned char* rgb, int row_stride,
                                  unsigned char* out_rgb, int out_stride,
                                  int reset);

// The frames-since-reset counter (the capture's schedule).
unsigned int dlss5_nr_counter(const dlss5_nr_handle* handle);
// The evaluator arena's high-water mark in floats (the host's memory
// accounting; constant across frames).
dlss5_nr_status dlss5_nr_arena_high_water(const dlss5_nr_handle* handle,
                                          size_t* out_floats);

void dlss5_nr_destroy(dlss5_nr_handle* handle);

#ifdef __cplusplus
}
#endif

#endif  // DLSS5_NR_NATIVE_HOST_H
