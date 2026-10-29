// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// The stable host boundary's implementation (vr/native-host.h): one
// synthetic-weight pipeline per handle over the temporal frame step
// (temporal_image_evaluation — reset, history feed, run, store, read-back,
// commit, counter). The weights are the deterministic synthetic DemoStore
// records: the transformation is arbitrary and the label says so.
#include "native-host.h"

#include <cstdlib>
#include <new>

#include "native-demo-support.hpp"
#include "native-eval.hpp"
#include "native-history.hpp"
#include "native-io.hpp"
#include "native-network.hpp"

// The opaque handle (the C header's forward declaration).
struct dlss5_nr_handle {
    int width = 0;
    int height = 0;
    dlss5_nr::LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    dlss5_nr::LayerValue outputs[dlss5_nr::kPlanMaxLayers];
    int skip_scratch[dlss5_nr::kPlanMaxLayers];
    int last_scratch[dlss5_nr::kPlanMaxLayers];
    int row_count = 0;
    dlss5_demo::DemoStore store;
    dlss5_nr::EvalWeights weights;
    dlss5_nr::EvalContext context;
    dlss5_nr::NetworkRun run;
    dlss5_nr::TemporalCarrier carrier;
    float* arena = nullptr;
    float* image_storage = nullptr;
    float* store_scratch = nullptr;
    float* carrier_map = nullptr;
    std::size_t image_floats = 0;
    std::size_t arena_floats = 0;
    std::size_t arena_high_water = 0;
};

namespace {

constexpr char kLabel[] = "synthetic weights \xe2\x80\x94 arbitrary transform";
// The demo geometry's floor: the small plan's padded extent at 64x72 (the
// boundary's fixed arena covers it; larger frames are refused).
constexpr int kMaxPaddedSide = 320;

dlss5_nr_status to_abi(dlss5_nr::Status status) {
    return static_cast<dlss5_nr_status>(static_cast<int>(status));
}

}  // namespace

extern "C" {

const char* dlss5_nr_label(void) { return kLabel; }

const char* dlss5_nr_status_name(dlss5_nr_status status) {
    switch (status) {
        case DLSS5_NR_OK: return "ok";
        case DLSS5_NR_NULL_POINTER: return "null pointer";
        case DLSS5_NR_INVALID_DIMENSIONS: return "invalid dimensions";
        case DLSS5_NR_INVALID_STRIDE: return "invalid stride";
        case DLSS5_NR_INVALID_VALUES: return "invalid values";
        case DLSS5_NR_INVALID_CONTROLS: return "invalid controls";
        case DLSS5_NR_INVALID_BUFFER: return "invalid buffer";
    }
    return "unknown";
}

dlss5_nr_status dlss5_nr_create(int width, int height, int threads,
                                dlss5_nr_handle** out_handle) {
    using namespace dlss5_nr;
    if (out_handle == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    *out_handle = nullptr;
    if (width < 1 || height < 1 || width > kMaxPaddedSide
        || height > kMaxPaddedSide) {
        return DLSS5_NR_INVALID_DIMENSIONS;
    }
    if (threads < 0 || threads > 64) {
        return DLSS5_NR_INVALID_VALUES;
    }
    PlanPadding padding{};
    if (plan_padding(width, height, NetworkKind::kSmall, padding)
        != Status::kOk) {
        return DLSS5_NR_INVALID_DIMENSIONS;
    }
    if (padding.width > kMaxPaddedSide || padding.height > kMaxPaddedSide) {
        return DLSS5_NR_INVALID_DIMENSIONS;
    }
    void* raw = std::calloc(1, sizeof(dlss5_nr_handle));
    if (raw == nullptr) {
        return DLSS5_NR_INVALID_BUFFER;
    }
    dlss5_nr_handle* handle = new (raw) dlss5_nr_handle();
    handle->width = width;
    handle->height = height;
    if (build_plan(width, height, NetworkKind::kSmall, handle->specs,
                   kPlanMaxLayers, handle->row_count)
        != Status::kOk) {
        handle->~dlss5_nr_handle();
        std::free(raw);
        return DLSS5_NR_INVALID_VALUES;
    }
    handle->image_floats = 3u
        * static_cast<std::size_t>(padding.width)
        * static_cast<std::size_t>(padding.height);
    handle->arena_floats = std::size_t{1} << 27;  // 512 MiB (the demo's bound)
    handle->arena = static_cast<float*>(
        std::malloc(handle->arena_floats * sizeof(float)));
    handle->image_storage = static_cast<float*>(
        std::malloc(handle->image_floats * sizeof(float)));
    handle->store_scratch = static_cast<float*>(
        std::malloc(handle->image_floats * sizeof(float)));
    handle->carrier_map = static_cast<float*>(
        std::malloc(handle->image_floats * sizeof(float)));
    if (handle->arena == nullptr || handle->image_storage == nullptr
        || handle->store_scratch == nullptr || handle->carrier_map == nullptr) {
        dlss5_nr_destroy(handle);
        return DLSS5_NR_INVALID_BUFFER;
    }
    handle->store.preset = dlss5_demo::DemoStore::Preset::kSeeded;
    handle->store.seed = 20261028u;
    dlss5_demo::demo_bind_weights(handle->weights, handle->store);
    handle->context.weights = &handle->weights;
    handle->context.scratch = handle->arena;
    handle->context.scratch_capacity = handle->arena_floats;
    if (eval_reset(handle->context) != Status::kOk
        || temporal_init(handle->carrier, handle->carrier_map,
                         handle->image_floats, padding.height, padding.width)
            != Status::kOk) {
        dlss5_nr_destroy(handle);
        return DLSS5_NR_INVALID_BUFFER;
    }
    handle->run.outputs = handle->outputs;
    handle->run.skip_sources = handle->skip_scratch;
    handle->run.last_uses = handle->last_scratch;
    handle->run.layer = eval_layer;
    handle->run.release = eval_release;
    set_parallel_threads(threads == 0 ? 8 : threads);
    *out_handle = handle;
    return DLSS5_NR_OK;
}

dlss5_nr_status dlss5_nr_evaluate(dlss5_nr_handle* handle,
                                  const unsigned char* rgb, int row_stride,
                                  unsigned char* out_rgb, int out_stride,
                                  int reset) {
    using namespace dlss5_nr;
    if (handle == nullptr || rgb == nullptr || out_rgb == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    const int minimum = 3 * handle->width;
    if (row_stride < minimum || out_stride < minimum) {
        return DLSS5_NR_INVALID_STRIDE;
    }
    const Status status = temporal_image_evaluation(
        rgb, row_stride, handle->width, handle->height, out_rgb, out_stride,
        handle->image_storage, handle->image_floats, handle->specs,
        handle->row_count, handle->run, handle->context, handle->carrier,
        reset != 0, handle->store_scratch, handle->image_floats);
    if (status == Status::kOk) {
        const std::size_t used = eval_arena_used(handle->context);
        if (used > handle->arena_high_water) {
            handle->arena_high_water = used;
        }
    }
    return to_abi(status);
}

unsigned int dlss5_nr_counter(const dlss5_nr_handle* handle) {
    return handle == nullptr ? 0u : temporal_counter(handle->carrier);
}

dlss5_nr_status dlss5_nr_arena_high_water(const dlss5_nr_handle* handle,
                                          size_t* out_floats) {
    if (handle == nullptr || out_floats == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    *out_floats = handle->arena_high_water;
    return DLSS5_NR_OK;
}

void dlss5_nr_destroy(dlss5_nr_handle* handle) {
    if (handle == nullptr) {
        return;
    }
    std::free(handle->arena);
    std::free(handle->image_storage);
    std::free(handle->store_scratch);
    std::free(handle->carrier_map);
    handle->~dlss5_nr_handle();
    std::free(handle);
}

}  // extern "C"
