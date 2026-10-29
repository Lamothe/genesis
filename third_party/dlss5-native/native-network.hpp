// SPDX-License-Identifier: MIT
// Native CPU network dispatch: the plan-row run loop of the whole network
// (`NRNet.run`/`_skip_sources`/`_last_uses`) — which tensor every layer reads,
// the tuple semantics of the multi-output layers and the last-use release of
// finished tensors — over the typed plan rows of native-graph-plan.hpp. The
// layer kernels stay in their own families (native-pre-post/native-blend/
// native-compose, native-swin-*, native-split-swin, native-vit,
// native-decoder); this header wires them, it does not reimplement them.
// CPU-only, allocation-free; no GPU, headset, proprietary payload, Python or
// PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/model.py "NRNet.run"/"NRNet._skip_sources"/"NRNet._last_uses" and
// the layer modules' forward contracts). The full MIT licence notice is
// retained in vr/NATIVE-NETWORK.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic
// correctness only — never correspondence to NVIDIA's implementation. No
// proprietary weights, captures or binaries are used or implied.
//
// Semantics carried over exactly (`NRNet.run`):
//   - every layer reads the main output of the layer before it (the pre block
//     reads the padded frame instead); a layer's skip input comes from the row
//     `plan_skip_source` names (the pinned `NRNet._skip_sources`).
//   - a layer that produces more than one tensor keeps them as (main, aside):
//     element 0 is what the next layer reads. The producers are the pre block
//     (the pooled map and the block's full-resolution output before pooling),
//     the `ds` Swin layers (the projected pooled map and the block's own
//     output), `SplitFfwd` (its result and the unchanged block input) and
//     `SplitProjPool` (the projection and its half-resolution copy).
//   - the reader contracts follow the pinned layer modules' forwards:
//     `SwinLayer` "upsample" and `SplitFinalHead` read the aside of their skip
//     source (`skip[1]`; the upsample falls back to the whole skip, the final
//     head to its own main, exactly as the pinned fallbacks); `PostBlock` reads
//     `skip[1]` and the raw padded frame; `SplitFfwdProj` reads the previous
//     layer's raw pair (main, aside); `DecInputUpsample`, `SplitProj`,
//     `SplitProjPool`, `VitFfnContract` and `VitProjection` read the main of
//     their skip source (`skip[0]`/`_main`); the plain bodies read only their
//     main input.
//   - the last-use release (the pinned `_last_uses` + the tuple lifetimes):
//     each output is kept until the last layer that reads it — its sequential
//     reader and every layer that names it as a skip source — has run, then
//     released exactly once, ascending; the final layer's output is never
//     released inside the run. A `SplitFfwd` row's second element is its
//     input's own tensor (the pinned `(out, x)` tuple), so the input row's
//     release is deferred to the `SplitFfwd` row's own release: the pinned
//     keeps that tensor alive past its dict entry. A null release callback
//     keeps every output (the pinned `capture=True`).
//   - the reference's ablation hook (`NRNet.skip`) and its int4/int8 experiment
//     hooks are not modelled: the real network leaves them empty.

#ifndef DLSS5_NR_NATIVE_NETWORK_HPP
#define DLSS5_NR_NATIVE_NETWORK_HPP

#include "native-graph-plan.hpp"  // LayerSpec/LayerKind/LayerVariant, plan_skip_source, kPlanMaxLayers
#include "native-blend.hpp"       // FeatureBatchView

#include <cstddef>

namespace dlss5_nr {

// ---- the layer outputs (the pinned tuple semantics) ----------------------------------------------

// One layer's produced tensors. `main` is what the next layer reads (the
// pinned tuple element 0); `aside` is the second tensor of the multi-output
// layers (the pinned tuple element 1). `SplitFfwd`'s aside is its input's own
// tensor (the unchanged block input), so it aliases the previous row's main;
// every other aside is a distinct tensor. Single-output layers leave `aside`
// untouched and `has_aside` false.
// The binary16-side stash (the pinned `int4.side`): named producer->consumer
// copies of a tensor at its pre-store precision, cleared per frame (the
// pinned `side.clear()`). The keys are the pinned's: "hidden" (the vit ffn
// expand's UNROUNDED activation -> its contraction's int4 channels), "c" (a
// vit ffn contract's binary16 value -> this block's QKV), "stream" (that
// same value -> the NEXT block's expansion), "attn" (the split projection's
// value -> its attention's QKV). Every buffer holds [rows * cols, channels]
// token-major floats; the writers' exact contents are in the producers'
// contracts below.
struct StashView {
    const float* hidden = nullptr;
    const float* c = nullptr;
    const float* stream = nullptr;
    const float* attn = nullptr;
};

// The caller-owned stash output buffers (the same keys); a row writes only
// its own key(s) per its kind's contract. Null buffers keep no stash (the
// consumer falls back to the run's stored map — the documented seam
// divergence from the pinned's side channel).
struct StashBuffers {
    float* hidden = nullptr;
    float* c = nullptr;
    float* stream = nullptr;
    float* attn = nullptr;
};

struct LayerValue {
    FeatureBatchView main;
    FeatureBatchView aside;
    bool has_aside = false;
    StashBuffers stash;
};

// ---- the call contract (what one layer reads) ----------------------------------------------------

// The exact inputs of one layer call: `x` is the main input (the previous
// layer's main, or nothing for the pre block), `aux` the kind's second input
// (a skip main, a skip aside, or the previous layer's aside for
// `SplitFfwdProj`), and `image` the padded frame (the pre and post blocks
// only). Every pointer is either null or a view the run produced.
struct LayerCall {
    const FeatureBatchView* x = nullptr;
    const FeatureBatchView* aux = nullptr;
    const FeatureBatchView* image = nullptr;
    StashView stash;
};

// One layer's kernel entry. Writes `out` per the kind's contract (see
// `layer_produces_pair`); returns non-ok to reject (the run stops and keeps
// the outputs written so far).
using LayerFn = Status (*)(void* context, const LayerSpec& spec,
                           const LayerCall& call, LayerValue& out);

// The last-use release of one finished tensor (the pinned `del outputs[j]`).
// A null `ReleaseFn` keeps every output (capture mode).
using ReleaseFn = void (*)(void* context, int index);

// ---- the layer contracts -------------------------------------------------------------------------

// Whether the kind produces (main, aside) rather than a single tensor.
inline bool layer_produces_pair(LayerKind kind, LayerVariant variant) {
    if (kind == LayerKind::kPreBlock
        || kind == LayerKind::kSplitFfwd
        || kind == LayerKind::kSplitProjPool) {
        return true;
    }
    return kind >= LayerKind::kSwin1H && kind <= LayerKind::kSwin8H
        && variant == LayerVariant::kDs;
}

// Whether the kind reads the padded frame (the pre and post blocks).
inline bool layer_reads_image(LayerKind kind) {
    return kind == LayerKind::kPreBlock || kind == LayerKind::kPostBlock;
}

// ---- the last-use schedule (the pinned `NRNet._last_uses`) ---------------------------------------

// For every row, the index of the last layer that reads its output: its
// sequential reader and every layer naming it as a skip source. `skip_sources`
// is the caller's `plan_skip_source` row (length `count`; -1: no skip);
// `out`/`out_count` receive one entry per row. The final row's entry is
// `count - 1` (its output leaves through the caller).
inline Status network_last_uses(const int* skip_sources, int count,
                                int* out, int out_count) {
    if (skip_sources == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (count < 1 || count > kPlanMaxLayers || out_count < count) {
        return Status::kInvalidBuffer;
    }
    for (int i = 0; i < count; ++i) {
        if (skip_sources[i] < -1 || skip_sources[i] >= count) {
            return Status::kInvalidValues;
        }
    }
    for (int i = 0; i < count; ++i) {
        out[i] = i;
    }
    for (int i = 0; i < count; ++i) {
        const int source = skip_sources[i];
        if (source >= 0 && out[source] < i) {
            out[source] = i;
        }
    }
    for (int i = 1; i < count; ++i) {
        if (out[i - 1] < i) {
            out[i - 1] = i;
        }
    }
    return Status::kOk;
}

// ---- the run loop (the pinned `NRNet.run`) --------------------------------------------------------

// One dispatch over the plan rows. `outputs` holds one `LayerValue` per row
// (the layer functions fill it; the run routes it); `skip_sources` and
// `last_uses` are `count`-sized scratch the run fills before the first call.
struct NetworkRun {
    const FeatureBatchView* image = nullptr;
    LayerValue* outputs = nullptr;
    int* skip_sources = nullptr;
    int* last_uses = nullptr;
    LayerFn layer = nullptr;
    ReleaseFn release = nullptr;
    void* context = nullptr;
    // The caller-owned side-channel buffers (one per key, sized for the
    // largest [rows * cols, channels] map; see StashBuffers).
    StashBuffers stash;
};

// Run the `count` plan rows in order. Upfront validation (nulls, row indices,
// view geometry, kind/variant ranges) completes before the first write: on an
// upfront error every output stays untouched (the scratch is working memory
// and may already hold the computed schedule). Mid-run structural rejections
// (a layer violating its output contract, or a required second input the plan
// does not supply) fail fast: the completed rows keep their outputs, the
// failing row's entry is whatever its layer last wrote (the empty value it was
// handed if the layer wrote nothing) or untouched when the row is rejected
// before its layer runs, and rows not reached stay untouched.
inline Status network_run(const LayerSpec* specs, int count,
                          const NetworkRun& run) {
    if (specs == nullptr) {
        return Status::kNullPointer;
    }
    if (run.image == nullptr || run.outputs == nullptr
        || run.skip_sources == nullptr || run.last_uses == nullptr
        || run.layer == nullptr) {
        return Status::kNullPointer;
    }
    if (count < 1 || count > kPlanMaxLayers) {
        return Status::kInvalidDimensions;
    }
    if (run.image->data == nullptr || run.image->channels < 1
        || run.image->rows < 1 || run.image->cols < 1) {
        return Status::kInvalidValues;
    }
    for (int i = 0; i < count; ++i) {
        const LayerSpec& spec = specs[i];
        if (spec.index != i) {
            return Status::kInvalidValues;  // rows run in plan order by index
        }
        if (static_cast<int>(spec.kind) < 0
            || static_cast<int>(spec.kind) > static_cast<int>(LayerKind::kPostBlock)
            || static_cast<int>(spec.variant) < 0
            || static_cast<int>(spec.variant) > static_cast<int>(LayerVariant::kFinalHead)) {
            return Status::kInvalidValues;
        }
        if (spec.out_w < 1 || spec.out_h < 1) {
            return Status::kInvalidDimensions;
        }
    }
    // The skip sources of every row (the pinned `_skip_sources`), then the
    // release schedule derived from them (the pinned `_last_uses`).
    for (int i = 0; i < count; ++i) {
        run.skip_sources[i] = plan_skip_source(specs, count, i);
    }
    for (int i = 0; i < count; ++i) {
        // The pinned reads `outputs[source]` before it writes `outputs[i]`, so
        // a skip source must be an earlier row (a self or future source would
        // read a not-yet-produced tensor).
        if (run.skip_sources[i] >= i) {
            return Status::kInvalidValues;
        }
    }
    {
        const Status status = network_last_uses(run.skip_sources, count,
                                                run.last_uses, count);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int i = 0; i < count; ++i) {
        const LayerSpec& spec = specs[i];
        const LayerValue* previous = i > 0 ? &run.outputs[i - 1] : nullptr;
        const int source = run.skip_sources[i];
        const LayerValue* skip = source >= 0 ? &run.outputs[source] : nullptr;
        LayerCall call;
        // The binary16 side channel: every row may read the caller's
        // per-key stash buffers (the pinned's `int4.side` dict; each key's
        // write is consumed before the next write in the plan order).
        call.stash.hidden = run.stash.hidden;
        call.stash.c = run.stash.c;
        call.stash.stream = run.stash.stream;
        call.stash.attn = run.stash.attn;
        switch (spec.kind) {
            case LayerKind::kPreBlock:
                call.image = run.image;
                break;
            case LayerKind::kPostBlock:
                // `layer(_main(previous), skip_value[1], image)`.
                if (previous == nullptr || skip == nullptr || !skip->has_aside) {
                    return Status::kInvalidValues;
                }
                call.x = &previous->main;
                call.aux = &skip->aside;
                call.image = run.image;
                break;
            case LayerKind::kSplitFfwdProj:
                // `layer(previous)` with the raw pair: `x, skip = x`.
                if (previous == nullptr) {
                    return Status::kInvalidValues;
                }
                call.x = &previous->main;
                call.aux = previous->has_aside ? &previous->aside : nullptr;
                break;
            case LayerKind::kSplitProj:
            case LayerKind::kSplitProjPool:
            case LayerKind::kVitFfnContract:
            case LayerKind::kVitProjection:
            case LayerKind::kDecInputUpsample:
                // `layer(_main(previous), skip)` / `_main(skip_value)`.
                if (previous == nullptr || skip == nullptr) {
                    return Status::kInvalidValues;
                }
                call.x = &previous->main;
                call.aux = &skip->main;
                break;
            case LayerKind::kSplitFinalHead:
                // `pooled = skip[1] if isinstance(skip, tuple) else x`.
                if (previous == nullptr) {
                    return Status::kInvalidValues;
                }
                call.x = &previous->main;
                call.aux = (skip != nullptr && skip->has_aside)
                    ? &skip->aside : &previous->main;
                break;
            case LayerKind::kSwin1H:
            case LayerKind::kSwin2H:
            case LayerKind::kSwin4H:
            case LayerKind::kSwin8H:
                if (previous == nullptr) {
                    return Status::kInvalidValues;
                }
                call.x = &previous->main;
                if (spec.variant == LayerVariant::kUpsample) {
                    // `skip = skip[1] if isinstance(skip, tuple) else skip`.
                    if (skip == nullptr) {
                        return Status::kInvalidValues;
                    }
                    call.aux = skip->has_aside ? &skip->aside : &skip->main;
                }
                break;
            default:
                // The plain bodies (SplitFfwd, SplitQKVAttn, the vit
                // expand/QKV/attention) read only their main input.
                if (previous == nullptr) {
                    return Status::kInvalidValues;
                }
                call.x = &previous->main;
                break;
        }
        LayerValue& out = run.outputs[i];
        out = LayerValue();
        out.stash = run.stash;
        const Status status = run.layer(run.context, spec, call, out);
        if (status != Status::kOk) {
            return status;
        }
        // The layer's structural contract before the value is routed on.
        const bool pair = layer_produces_pair(spec.kind, spec.variant);
        if (out.main.data == nullptr
            || out.main.rows != spec.out_w || out.main.cols != spec.out_h) {
            return Status::kInvalidValues;
        }
        if (pair != out.has_aside || (pair && out.aside.data == nullptr)) {
            return Status::kInvalidValues;
        }
        // The pinned release pass: every finished tensor whose last reader has
        // now run is released once, in ascending order. A row j is finished
        // exactly at max(last_use[j], j + 1) (the pinned deletes it at the
        // first step that is its last reader and is not j itself). A
        // `SplitFfwd` row (j + 1) also holds j's tensor as its second
        // element, so j waits for that row's entry to be dropped as well.
        if (run.release != nullptr) {
            for (int j = 0; j < i; ++j) {
                int release_at = run.last_uses[j];
                if (release_at < j + 1) {
                    release_at = j + 1;
                }
                if (j + 1 < count && specs[j + 1].kind == LayerKind::kSplitFfwd) {
                    int alias_at = run.last_uses[j + 1];
                    if (alias_at < j + 2) {
                        alias_at = j + 2;
                    }
                    if (release_at < alias_at) {
                        release_at = alias_at;
                    }
                }
                if (release_at == i) {
                    run.release(run.context, j);
                }
            }
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_NETWORK_HPP
