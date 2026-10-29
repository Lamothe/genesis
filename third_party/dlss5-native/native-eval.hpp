// SPDX-License-Identifier: MIT
// Native CPU evaluation binding: `eval_layer`, a `LayerFn` for
// native-network.hpp's `network_run`, binding all 18 `LayerKind` kinds to
// the existing family kernels (native-pre-post/native-input-lift/
// native-blend/native-compose, native-swin-*, native-split-swin,
// native-vit, native-decoder) so a synthetic frame runs through the whole
// network. Weights are caller-supplied synthetic values fetched through a
// record interface mirroring the pinned `Weights` keying
// (`block{B}.layer{L}.{name}` with the typed accessors' shape assertions);
// the split feed-forward's packed `ffwd_a`/`ffwd_b` scatter is the declared
// weight-file boundary and stays on the record side of the `ffwd` callback.
// CPU-only, allocation-free (the caller's scratch holds the feature-map
// arena and every working buffer); no GPU, headset, proprietary payload,
// Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/model.py + dlssnr/layers/*). The full MIT licence notice is
// retained in vr/NATIVE-EVAL.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic
// correctness only — never correspondence to NVIDIA's implementation. No
// proprietary weights, captures or binaries are used or implied.
//
// The weight records (pinned constructors; every shape below is verified
// against the pinned layer modules):
//   Swin body (Swin1H..Swin8H any variant; also the pre/post block bodies,
//   which the pinned PreBlock/PostBlock build at channels 32, heads 1):
//     e4m3 "mlp_expand" [4C, C], "mlp_mid" [C, 4C/heads] only heads > 1
//     (else the mid projection is absent), "mlp_contract" [C, C] heads > 1 /
//     [C, 4C] heads == 1, "qkv" [3C, C], "attn_out_proj" [C, C];
//     f16 "residual_scale" (the pinned `_residual_table`: mlp_scale is the
//     entries [8, 8+C) when the record's first 8 entries are all zero, else
//     [0, C)), f16 "attn_residual_scale" (attn_scale = the first C entries),
//     f32 "scalars_b" (head_scale = the first `heads` entries), f16
//     "attn_pos_bias" [heads*4096]; the exp_bias [heads, 4096] table the
//     kernels consume is derived from that record per head by deswizzling
//     the 64x64 fragment order (swin_deswizzle_index) and carrying every
//     entry through the exponent ramp — one float32-from-double rounding of
//     bias*0.044921875 + 1.30078125 (swin_exp_bias).
//   Swin ds variant adds e4m3 "resample" [2C, C]; the upsample variant adds
//   e4m3 "resample" [C, 2C] and f16 "upsample_gain" with the pinned wide
//   fallback: when the record has fewer than C entries, the last (C - m)
//   entries of the "residual_scale" table are prepended to its m entries and
//   the first C of the concatenation used.
//   VitFfnContract: the e4m3 "weight" + f16 "skip_weight", OR the int4
//     site ("vit_ffn" through the `int4` callback) with its "tail" e4m3
//     record and "skip_weight" — the site's presence selects the mixed path
//     (the pinned's table-presence gating); the mixed path never fetches
//     "weight". kVitProjection never takes the site.
//   VitQKV: the e4m3 "q"/"k"/"v" + f32 "header" temperatures, OR the int4
//     site ("vit_qkv" through the `int4` callback; one [3C, C] matrix, no
//     permutation, non-half) with the shared "header" — the site's presence
//     selects the mixed path and never fetches q/k/v.
//   SplitQKVAttn: the e4m3 "qkv" + "attn_pos_bias" + f32 "tail", OR the int4
//     site ("attn_qkv" through the `int4` callback; one [1536, 512] matrix,
//     no permutation, non-half) with the shared bias/gain records — the
//     site's presence selects the mixed path and never fetches "qkv". Both
//     mixed sites quantise the map the run hands them; in a full chain that
//     map is the producer's e4m3 STORE, where the pinned stashes the
//     producer's f16 value for the quantiser (int4.side) — a seam-wide
//     divergence from the pinned's side-channel plumbing, documented here
//     and in NATIVE-INT4.md rather than hidden.
//   SplitFfwd: the ffwd callback, which 0 = expand_in [512, 512], 1 =
//   expand_hidden [2048, 64], 2 = contract [512, 256] (raw e4m3 codes).
//   SplitFfwdProj/SplitProj/SplitProjPool: e4m3 "weight" [c_out, c_in] +
//   f16 "skip_weight" [c_out]. SplitFinalHead: e4m3 "weight" only.
//   SplitQKVAttn: e4m3 "qkv" [1536, 512], f16 "attn_pos_bias" [16*4096]
//   deswizzled with the same fragment-order map but WITHOUT the exp ramp
//   (the kernel takes the raw unpacked bias), f32 "tail" [16] head gains.
//   VitFfnExpand: e4m3 "weight" [c_out, c_in]. VitFfnContract/VitProjection:
//   e4m3 "weight" + f16 "skip_weight" [c_out]. VitQKV: e4m3 "q", "k", "v"
//   each [C, C] + f32 "header" [C/32] temperature. VitAttention: no records.
//   DecInputUpsample: e4m3 "weight" [c_out, c_in] + f16 "skip_weight"
//   [c_out]. PreBlock: f16 "input_lift" [512] (decoded with decode_lift) +
//   the body's Swin names. PostBlock: f16 "main_gain" [32], f16 "skip_gain"
//   [32], f16 "out_project" [16*32] (the first 3 rows, [3, 32], are used) +
//   the body's Swin names.
//
// Record interface contract (the pinned `Weights` accessors):
//   - `e4m3`/`ffwd` return the record's rows*cols raw e4m3 codes laid out
//     row-major, or null when the record is missing or its code count is not
//     exactly rows*cols (the pinned `reshape(*shape)`).
//   - `f16`/`f32` with count >= 0 return the record's entries when the
//     record holds exactly `count` entries (the pinned `reshape(count)`),
//     null otherwise. With count == -1 they return the record whatever its
//     length (the pinned `reshape(-1)`); a wildcard record's stored length
//     is then established with exact-count probes bounded by
//     kEvalRecordEntriesMax (a longer record is rejected as kInvalidBuffer).
//   - The pre/post bodies use channels 32 and heads 1 (the pinned
//     `SwinParams.from_weights(..., CHANNELS, 1)`), independent of the
//     plan row's `channels`/`heads` metadata.
//
// Memory contract: `EvalContext::scratch` holds, from the front, the arena
// state header (kEvalStateWords floats, initialised by eval_reset) and the
// feature-map arena; one layer's working block is one more arena block,
// released before the call returns. Outputs are arena blocks (the pinned
// tuple semantics: SplitFfwd's aside aliases its input's own tensor and
// allocates nothing). Pass eval_release as NetworkRun::release to honour the
// last-use releases (the released blocks are recycled; the SplitFfwd alias
// is already handled by network_run's deferred schedule); a null release
// keeps every output (capture mode). Call eval_reset before each run.
// Validation completes before the first write to `out`: on any error `out`
// is untouched and the row's arena blocks are released.

#ifndef DLSS5_NR_NATIVE_EVAL_HPP
#define DLSS5_NR_NATIVE_EVAL_HPP

#include "native-compose.hpp"
#include "native-decoder.hpp"
#include "native-int4.hpp"
#include "native-network.hpp"
#include "native-split-swin.hpp"
#include "native-swin-resample.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dlss5_nr {

// ---- the record interface (the pinned Weights keying) -------------------------------------------

// The caller's weight records, keyed like the pinned `Weights`
// (`block{B}.layer{L}.{name}`). Every callback returns the record's storage
// or null (missing record, or a shape/count the record refuses); see the
// header contract above for the exact count/shape semantics. All callbacks
// are pure lookups: the binding probes wildcard record lengths with
// exact-count requests.
// The SwinSite record (the pinned int4.py SwinSite): the folded scale and
// the optional int4 matrices of one Swin block's MLP expansion ("mlp_expand"
// [4C, C]) and QKV ("qkv" [3C, C]) — both block 1 binary16 form. The pinned
// builds both or neither (both aq tables required) and only for heads > 1;
// the run enforces the same. `fold` drives the hard-swish activation swap
// exactly as the pinned (hs set iff fold != 0).
struct SwinSiteRecord {
    float fold = 0.0f;
    Int4MatrixView ex;
    Int4MatrixView qkv;
    bool has_ex = false;
    bool has_qkv = false;
};

struct EvalWeights {
    void* context = nullptr;
    const std::uint8_t* (*e4m3)(void* context, int block, int layer,
                                const char* name, int rows, int cols) = nullptr;
    const float* (*f16)(void* context, int block, int layer,
                        const char* name, int count) = nullptr;
    const float* (*f32)(void* context, int block, int layer,
                        const char* name, int count) = nullptr;
    const std::uint8_t* (*ffwd)(void* context, int block, int layer, int which,
                                int rows, int cols) = nullptr;
    // The int4 mixed-network boundary (native-int4.hpp's Int4Site): the named
    // site's decoded record ("vit_ffn", "vit_qkv"), or null to run the site
    // in e4m3 — the pinned's table-presence gating. The ViT feed-forward's
    // e4m3 tail arrives through `e4m3` as "tail".
    // Site names: "vit_ffn", "vit_qkv", "attn_qkv" (see the record contract
    // above); unknown names return null and the row runs e4m3. `swin`
    // returns the per-block SwinSiteRecord (or null = the e4m3 dataflow).
    const SwinSiteRecord* (*swin)(void* context, int block, int layer)
        = nullptr;
    const Int4Site* (*int4)(void* context, int block, int layer,
                            const char* name) = nullptr;
};

// One evaluation context: the records, the caller-provided scratch (the
// feature-map arena plus every working buffer), the lift controls and the
// output intensity.
struct EvalContext {
    const EvalWeights* weights = nullptr;
    float* scratch = nullptr;
    std::size_t scratch_capacity = 0;
    Controls controls;
    double intensity = 1.0;
    // The temporal history colour for the pre block's lift (slots 7..9);
    // null keeps the documented single-frame stub (current-colour copy).
    const FrameView* history = nullptr;
    // The binary16 side channel's ("stream", block) state: the plan block
    // whose value the caller's stash.stream buffer currently holds, or -1
    // when no producer has written it. The pinned's `int4.side` is a dict
    // whose key may be ABSENT — the first vit block's expansion has no
    // producer (the split block before it writes no stream) and the pinned
    // falls back to the e4m3 form there — and this stamp reproduces that
    // gate exactly while keeping a never-written buffer from being read
    // (the stale-content hazard). eval_reset clears it; the vit projection
    // rows stamp the block they write for.
    int stash_stream_block = -1;
    // Optional raw-projection executor for the vit projection rows (the
    // G2 GPU backend in native-vulkan.hpp). Null keeps the CPU kernel —
    // the deterministic reference. The kernels' post chains always run on
    // the CPU, so executor results flow through the reference arithmetic.
    const ProjectionExecutor* projection = nullptr;
    // The swin rows' batched projection seam (the G8 chunked path: one
    // executor call per kSwinBatchWindows windows instead of per 64-token
    // window). Measured faster than the CPU rows on the dev host (6,850 vs
    // 8,301 ms/frame end-to-end; NATIVE-PERFORMANCE.md G8) with byte-identical
    // artefacts, so the swin rows default ON whenever a projection executor
    // is attached. `false` (the demo's --cpu-swin) keeps the G6/G7 measured
    // configuration (swin rows on the CPU kernels) for A/B measurement. The
    // batched scratch raises the caller's arena requirement (about 345 MiB
    // high-water for the 64x72 small plan; see NATIVE-PERFORMANCE.md G8).
    bool swin_gpu = true;
};

// ---- vocabulary ---------------------------------------------------------------------------------

inline constexpr int kEvalRecordEntriesMax = 4096;  // wildcard record probe bound

// The arena state header occupies the first kEvalStateWords floats of the
// scratch (raw uint32 words, moved with memcpy). The block table bounds the
// live and recycled arena blocks (two per plan row plus split remainders).
inline constexpr int kEvalMaxBlocks = 2048;
inline constexpr int kEvalStateWords = 4 + 4 * kEvalMaxBlocks;

// ---- the arena ----------------------------------------------------------------------------------

namespace eval_detail {

// The pinned's hard-swish (A, B) by the block's channel count
// (HARD_SWISH = {32: (0.187367, 0.500139)}; the others take the wide pair).
inline constexpr float kHardSwish32[2] = {0.187367f, 0.500139f};
inline constexpr float kHardSwishWide[2] = {0.182412f, 0.500465f};

// The int4 SwinSite overlay (the pinned's gating): the hard-swish swap when
// the record's fold is non-zero, and the ex/qkv matrices only for heads > 1
// with both present (the pinned builds both or neither).
inline void build_swin_overlay(const EvalWeights* weights, int block,
                               int layer, int channels, int heads,
                               SwinSiteOverlay& overlay) {
    overlay = SwinSiteOverlay{};
    if (weights == nullptr || weights->swin == nullptr) {
        return;
    }
    const SwinSiteRecord* site = weights->swin(weights->context, block, layer);
    if (site == nullptr) {
        return;
    }
    if (site->fold != 0.0f) {
        overlay.hs = channels == 32 ? kHardSwish32 : kHardSwishWide;
    }
    if (heads > 1 && site->has_ex && site->has_qkv) {
        overlay.int4_expand = &site->ex;
        overlay.int4_qkv = &site->qkv;
    }
}

inline constexpr std::uint32_t kStateMagic = 0x45564c31u;
inline constexpr std::uint32_t kTransientOwner = 0xffffffffu;  // a call's working block
inline constexpr std::uint32_t kBlockFreed = 1u;
inline constexpr std::uint32_t kMinSplitWords = 64;  // smallest split remainder kept

// The state header words: 0 magic, 1 arena cursor (floats from scratch[0]),
// 2 block count, 3 reserved; then one (offset, words, owner, flags) tuple
// per block entry. Accessed through memcpy'd uint32 words (no aliasing).
struct StateView {
    float* base = nullptr;
    std::uint32_t get(int word) const {
        std::uint32_t value = 0;
        std::memcpy(&value, base + word, sizeof value);
        return value;
    }
    void put(int word, std::uint32_t value) {
        std::memcpy(base + word, &value, sizeof value);
    }
};

inline int entry_word(int entry, int field) {
    return 4 + 4 * entry + field;
}

// First-fit-smallest allocation over the freed blocks (splitting a larger
// block when the table has room), else a bump at the arena cursor. The
// cursor is the arena's high-water mark; released blocks stay reusable.
inline Status arena_alloc(StateView state, std::size_t capacity,
                          std::size_t need, std::uint32_t owner,
                          std::size_t& offset) {
    if (need < 1 || need > 0xffffffffu || capacity > 0xffffffffu) {
        return Status::kInvalidBuffer;
    }
    const std::uint32_t count = state.get(2);
    if (count > kEvalMaxBlocks) {
        return Status::kInvalidValues;
    }
    int best = -1;
    std::uint32_t best_words = 0;
    for (std::uint32_t k = 0; k < count; ++k) {
        if ((state.get(entry_word(static_cast<int>(k), 3)) & kBlockFreed) == 0u) {
            continue;
        }
        const std::uint32_t words = state.get(entry_word(static_cast<int>(k), 1));
        if (static_cast<std::size_t>(words) < need) {
            continue;
        }
        if (best < 0 || words < best_words) {
            best = static_cast<int>(k);
            best_words = words;
        }
    }
    if (best >= 0) {
        const std::uint32_t base = state.get(entry_word(best, 0));
        std::uint32_t words = state.get(entry_word(best, 1));
        if (static_cast<std::size_t>(words) > need + kMinSplitWords
            && count < kEvalMaxBlocks) {
            state.put(entry_word(static_cast<int>(count), 0),
                      base + static_cast<std::uint32_t>(need));
            state.put(entry_word(static_cast<int>(count), 1),
                      words - static_cast<std::uint32_t>(need));
            state.put(entry_word(static_cast<int>(count), 2), 0u);
            state.put(entry_word(static_cast<int>(count), 3), kBlockFreed);
            state.put(2, count + 1u);
            words = static_cast<std::uint32_t>(need);
            state.put(entry_word(best, 1), words);
        }
        state.put(entry_word(best, 2), owner);
        state.put(entry_word(best, 3), 0u);
        offset = base;
        return Status::kOk;
    }
    const std::size_t cursor = state.get(1);
    if (cursor + need > capacity || count >= kEvalMaxBlocks) {
        return Status::kInvalidBuffer;
    }
    state.put(entry_word(static_cast<int>(count), 0),
              static_cast<std::uint32_t>(cursor));
    state.put(entry_word(static_cast<int>(count), 1),
              static_cast<std::uint32_t>(need));
    state.put(entry_word(static_cast<int>(count), 2), owner);
    state.put(entry_word(static_cast<int>(count), 3), 0u);
    state.put(1, static_cast<std::uint32_t>(cursor + need));
    state.put(2, count + 1u);
    offset = cursor;
    return Status::kOk;
}

// Release every live block owned by `owner` (idempotent).
inline void arena_free(StateView state, std::uint32_t owner) {
    const std::uint32_t count = state.get(2);
    for (std::uint32_t k = 0; k < count; ++k) {
        if (state.get(entry_word(static_cast<int>(k), 2)) == owner
            && (state.get(entry_word(static_cast<int>(k), 3)) & kBlockFreed) == 0u) {
            state.put(entry_word(static_cast<int>(k), 3), kBlockFreed);
        }
    }
}

}  // namespace eval_detail

// Initialise the arena state in `ctx.scratch`; call before every run.
inline Status eval_reset(EvalContext& ctx) {
    if (ctx.scratch == nullptr) {
        return Status::kNullPointer;
    }
    if (ctx.scratch_capacity < static_cast<std::size_t>(kEvalStateWords)) {
        return Status::kInvalidBuffer;
    }
    eval_detail::StateView state{ctx.scratch};
    state.put(0, eval_detail::kStateMagic);
    state.put(1, static_cast<std::uint32_t>(kEvalStateWords));
    state.put(2, 0u);
    state.put(3, 0u);
    ctx.stash_stream_block = -1;
    return Status::kOk;
}

// The last-use release entry (a `ReleaseFn`): recycle the released row's
// arena blocks. network_run calls it in ascending row order; a call for an
// unknown row or without an initialised state is ignored.
inline void eval_release(void* context, int index) {
    EvalContext* ctx = static_cast<EvalContext*>(context);
    if (ctx == nullptr || ctx->scratch == nullptr
        || ctx->scratch_capacity < static_cast<std::size_t>(kEvalStateWords)
        || index < 0) {
        return;
    }
    eval_detail::StateView state{ctx->scratch};
    if (state.get(0) != eval_detail::kStateMagic) {
        return;
    }
    eval_detail::arena_free(state, static_cast<std::uint32_t>(index));
}

// The arena's high-water mark in floats (diagnostics/sizing).
inline std::size_t eval_arena_used(const EvalContext& ctx) {
    if (ctx.scratch == nullptr
        || ctx.scratch_capacity < static_cast<std::size_t>(kEvalStateWords)) {
        return 0;
    }
    eval_detail::StateView state{ctx.scratch};
    if (state.get(0) != eval_detail::kStateMagic) {
        return 0;
    }
    return state.get(1);
}

// ---- token/planar transposes --------------------------------------------------------------------
// The split and vit families run on token-major [rows * cols][channels]
// tensors (token = r * cols + c); the layer values are channels-first maps.
// Both helpers work on tight token buffers and arbitrary valid map strides.

// src [channels, rows, cols] -> dst [rows * cols][channels] (row-major tokens).
inline Status eval_planar_to_tokens(const FeatureBatchView& src, float* dst,
                                    std::size_t dst_count) {
    if (src.data == nullptr || dst == nullptr) {
        return Status::kNullPointer;
    }
    if (src.channels < 1 || src.rows < 1 || src.cols < 1) {
        return Status::kInvalidDimensions;
    }
    if (src.row_stride < static_cast<std::ptrdiff_t>(src.cols)
        || src.row_stride > static_cast<std::ptrdiff_t>(kMaxCols)
        || src.plane_stride
            < static_cast<std::ptrdiff_t>(src.rows) * src.row_stride
        || src.plane_stride
            > static_cast<std::ptrdiff_t>(kMaxRows) * kMaxCols) {
        return Status::kInvalidStride;
    }
    const std::size_t tokens = static_cast<std::size_t>(src.rows)
        * static_cast<std::size_t>(src.cols);
    if (dst_count < tokens * static_cast<std::size_t>(src.channels)) {
        return Status::kInvalidBuffer;
    }
    for (int c = 0; c < src.channels; ++c) {
        const float* plane = src.data
            + static_cast<std::size_t>(c) * static_cast<std::size_t>(src.plane_stride);
        for (int r = 0; r < src.rows; ++r) {
            const float* row = plane
                + static_cast<std::size_t>(r) * static_cast<std::size_t>(src.row_stride);
            for (int col = 0; col < src.cols; ++col) {
                dst[(static_cast<std::size_t>(r) * src.cols + col)
                        * static_cast<std::size_t>(src.channels)
                    + static_cast<std::size_t>(c)] = row[col];
            }
        }
    }
    return Status::kOk;
}

// src [rows * cols][channels] -> dst [channels, rows, cols] (tight planes).
inline Status eval_tokens_to_planar(const float* src, std::size_t src_count,
                                    int channels, int rows, int cols,
                                    float* dst, std::size_t dst_count) {
    if (src == nullptr || dst == nullptr) {
        return Status::kNullPointer;
    }
    if (channels < 1 || rows < 1 || cols < 1) {
        return Status::kInvalidDimensions;
    }
    const std::size_t tokens = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(cols);
    const std::size_t needed = tokens * static_cast<std::size_t>(channels);
    if (src_count < needed || dst_count < needed) {
        return Status::kInvalidBuffer;
    }
    for (int c = 0; c < channels; ++c) {
        float* plane = dst + static_cast<std::size_t>(c) * tokens;
        for (std::size_t t = 0; t < tokens; ++t) {
            plane[t] = src[t * static_cast<std::size_t>(channels)
                          + static_cast<std::size_t>(c)];
        }
    }
    return Status::kOk;
}

// ---- record plumbing -----------------------------------------------------------------------------

namespace eval_detail {

inline void decode_e4m3(const std::uint8_t* codes, std::size_t count, float* dst) {
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = e4m3_from_bits(codes[i]);
    }
}

inline Status fetch_e4m3(const EvalWeights* w, int block, int layer,
                         const char* name, int rows, int cols,
                         const std::uint8_t*& out) {
    if (w == nullptr || w->e4m3 == nullptr) {
        return Status::kNullPointer;
    }
    out = w->e4m3(w->context, block, layer, name, rows, cols);
    return out != nullptr ? Status::kOk : Status::kNullPointer;
}

inline Status fetch_ffwd(const EvalWeights* w, int block, int layer, int which,
                         int rows, int cols, const std::uint8_t*& out) {
    if (w == nullptr || w->ffwd == nullptr) {
        return Status::kNullPointer;
    }
    out = w->ffwd(w->context, block, layer, which, rows, cols);
    return out != nullptr ? Status::kOk : Status::kNullPointer;
}

inline Status fetch_f16(const EvalWeights* w, int block, int layer,
                        const char* name, int count, const float*& out) {
    if (w == nullptr || w->f16 == nullptr) {
        return Status::kNullPointer;
    }
    out = w->f16(w->context, block, layer, name, count);
    return out != nullptr ? Status::kOk : Status::kNullPointer;
}

inline Status fetch_f32(const EvalWeights* w, int block, int layer,
                        const char* name, int count, const float*& out) {
    if (w == nullptr || w->f32 == nullptr) {
        return Status::kNullPointer;
    }
    out = w->f32(w->context, block, layer, name, count);
    return out != nullptr ? Status::kOk : Status::kNullPointer;
}

// The stored entry count of a wildcard record (the pinned reshape(-1)
// length), established with exact-count probes: the unique n whose exact
// request succeeds is the record's length. -1 when the record is missing or
// longer than kEvalRecordEntriesMax.
inline int record_length_f16(const EvalWeights* w, int block, int layer,
                             const char* name) {
    if (w == nullptr || w->f16 == nullptr) {
        return -1;
    }
    for (int n = 0; n <= kEvalRecordEntriesMax; ++n) {
        if (w->f16(w->context, block, layer, name, n) != nullptr) {
            return n;
        }
    }
    return -1;
}

inline int record_length_f32(const EvalWeights* w, int block, int layer,
                             const char* name) {
    if (w == nullptr || w->f32 == nullptr) {
        return -1;
    }
    for (int n = 0; n <= kEvalRecordEntriesMax; ++n) {
        if (w->f32(w->context, block, layer, name, n) != nullptr) {
            return n;
        }
    }
    return -1;
}

// A wildcard f16 record and its stored length (the pinned reshape(-1)).
inline Status fetch_f16_any(const EvalWeights* w, int block, int layer,
                            const char* name, const float*& out, int& length) {
    if (w == nullptr || w->f16 == nullptr) {
        return Status::kNullPointer;
    }
    out = w->f16(w->context, block, layer, name, -1);
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    length = record_length_f16(w, block, layer, name);
    return length >= 0 ? Status::kOk : Status::kInvalidBuffer;
}

inline Status fetch_f32_any(const EvalWeights* w, int block, int layer,
                            const char* name, const float*& out, int& length) {
    if (w == nullptr || w->f32 == nullptr) {
        return Status::kNullPointer;
    }
    out = w->f32(w->context, block, layer, name, -1);
    if (out == nullptr) {
        return Status::kNullPointer;
    }
    length = record_length_f32(w, block, layer, name);
    return length >= 0 ? Status::kOk : Status::kInvalidBuffer;
}

// The pinned `_residual_table` content rule: the table is zero padded in
// front (eight entries) except on the wide upsample layers, where it is the
// C scales followed by the skip-gain entries. mlp_scale starts at 8 when the
// first 8 entries are all zero, else at 0.
inline Status mlp_scale_slice(const float* table, int length, int channels,
                              const float*& out) {
    if (table == nullptr) {
        return Status::kNullPointer;
    }
    if (channels < 1) {
        return Status::kInvalidDimensions;
    }
    if (length < channels) {
        return Status::kInvalidBuffer;
    }
    bool padded = length >= 8;
    if (padded) {
        for (int i = 0; i < 8; ++i) {
            if (table[i] != 0.0f) {
                padded = false;
                break;
            }
        }
    }
    const int start = padded ? 8 : 0;
    if (length < start + channels) {
        return Status::kInvalidBuffer;
    }
    out = table + start;
    return Status::kOk;
}

// The upsample blend's gain row: the record's entries when it has at least
// C of them, else the pinned wide fallback — the last (C - m) entries of the
// residual_scale table prepended to the record's m entries, first C taken.
// `work` receives the C entries of the concatenated form.
inline Status upsample_gain_row(const float* gain, int gain_length,
                                const float* table, int table_length,
                                int channels, float* work,
                                const float*& out) {
    if (gain == nullptr || channels < 1) {
        return Status::kNullPointer;
    }
    if (gain_length < 0 || table_length < 0) {
        return Status::kInvalidBuffer;
    }
    if (gain_length >= channels) {
        out = gain;
        return Status::kOk;
    }
    const int needed = channels - gain_length;
    if (table == nullptr || table_length < needed) {
        return Status::kInvalidBuffer;
    }
    for (int i = 0; i < needed; ++i) {
        work[i] = table[table_length - needed + i];
    }
    for (int i = 0; i < gain_length; ++i) {
        work[needed + i] = gain[i];
    }
    out = work;
    return Status::kOk;
}

// ---- the Swin body's records ---------------------------------------------------------------------

struct SwinFetch {
    const std::uint8_t* expand = nullptr;    // [4C, C]
    const std::uint8_t* mid = nullptr;       // [C, 4C/heads], heads > 1 only
    const std::uint8_t* contract = nullptr;  // [C, C] / [C, 4C]
    const std::uint8_t* qkv = nullptr;       // [3C, C]
    const std::uint8_t* out_proj = nullptr;  // [C, C]
    const float* mlp_scale = nullptr;        // [C], inside the residual table
    const float* attn_scale = nullptr;       // [C], the first C entries
    const float* head_scale = nullptr;       // [heads], the first heads entries
    const float* exp_bias_raw = nullptr;     // [heads*4096] f16 fragment order
    int mid_cols = 0;
    int contract_cols = 0;
};

inline std::size_t swin_decoded_floats(int channels, int heads,
                                       const SwinFetch& f) {
    const std::size_t c = static_cast<std::size_t>(channels);
    std::size_t total = 4 * c * c + 3 * c * c + c * c;  // expand + qkv + out_proj
    if (heads > 1) {
        total += c * static_cast<std::size_t>(f.mid_cols);
    }
    total += c * static_cast<std::size_t>(f.contract_cols);
    total += static_cast<std::size_t>(heads) * kBiasTable;  // exp_bias
    return total;
}

inline Status swin_fetch(const EvalWeights* w, int block, int layer,
                         int channels, int heads, SwinFetch& f) {
    if (channels < 1 || heads < 1 || channels != heads * kHeadDim) {
        return Status::kInvalidDimensions;
    }
    Status status = fetch_e4m3(w, block, layer, "mlp_expand", 4 * channels,
                               channels, f.expand);
    if (status != Status::kOk) {
        return status;
    }
    if (heads > 1) {
        f.mid_cols = 4 * channels / heads;
        status = fetch_e4m3(w, block, layer, "mlp_mid", channels, f.mid_cols,
                            f.mid);
        if (status != Status::kOk) {
            return status;
        }
        f.contract_cols = channels;
    } else {
        f.contract_cols = 4 * channels;
    }
    status = fetch_e4m3(w, block, layer, "mlp_contract", channels,
                        f.contract_cols, f.contract);
    if (status != Status::kOk) {
        return status;
    }
    status = fetch_e4m3(w, block, layer, "qkv", 3 * channels, channels, f.qkv);
    if (status != Status::kOk) {
        return status;
    }
    status = fetch_e4m3(w, block, layer, "attn_out_proj", channels, channels,
                        f.out_proj);
    if (status != Status::kOk) {
        return status;
    }
    const float* table = nullptr;
    int table_length = 0;
    status = fetch_f16_any(w, block, layer, "residual_scale", table,
                           table_length);
    if (status != Status::kOk) {
        return status;
    }
    status = mlp_scale_slice(table, table_length, channels, f.mlp_scale);
    if (status != Status::kOk) {
        return status;
    }
    const float* attn = nullptr;
    int attn_length = 0;
    status = fetch_f16_any(w, block, layer, "attn_residual_scale", attn,
                           attn_length);
    if (status != Status::kOk) {
        return status;
    }
    if (attn_length < channels) {
        return Status::kInvalidBuffer;
    }
    f.attn_scale = attn;
    const float* scalars = nullptr;
    int scalars_length = 0;
    status = fetch_f32_any(w, block, layer, "scalars_b", scalars,
                           scalars_length);
    if (status != Status::kOk) {
        return status;
    }
    if (scalars_length < heads) {
        return Status::kInvalidBuffer;
    }
    f.head_scale = scalars;
    return fetch_f16(w, block, layer, "attn_pos_bias", heads * kBiasTable,
                     f.exp_bias_raw);
}

// Decode the body's e4m3 records to the e4m3 grid in float32 and derive the
// exp_bias table (the deswizzle + exponent ramp). `mid_storage` holds the
// `SwinBlockParams::mid` view for the call's lifetime.
inline Status swin_decode(const SwinFetch& f, int channels, int heads,
                          float* work, std::size_t work_count,
                          MatrixView& mid_storage,
                          SwinBlockParams& params) {
    const std::size_t c = static_cast<std::size_t>(channels);
    const std::size_t needed = swin_decoded_floats(channels, heads, f);
    if (work_count < needed) {
        return Status::kInvalidBuffer;
    }
    float* cursor = work;
    decode_e4m3(f.expand, 4 * c * c, cursor);
    params.expand = MatrixView{cursor, 4 * channels, channels, channels};
    cursor += 4 * c * c;
    if (heads > 1) {
        decode_e4m3(f.mid, c * static_cast<std::size_t>(f.mid_cols), cursor);
        mid_storage = MatrixView{cursor, channels, f.mid_cols, f.mid_cols};
        params.mid = &mid_storage;
        cursor += c * static_cast<std::size_t>(f.mid_cols);
    } else {
        params.mid = nullptr;
    }
    decode_e4m3(f.contract, c * static_cast<std::size_t>(f.contract_cols),
                cursor);
    params.contract = MatrixView{cursor, channels, f.contract_cols,
                                 f.contract_cols};
    cursor += c * static_cast<std::size_t>(f.contract_cols);
    decode_e4m3(f.qkv, 3 * c * c, cursor);
    params.qkv = MatrixView{cursor, 3 * channels, channels, channels};
    cursor += 3 * c * c;
    decode_e4m3(f.out_proj, c * c, cursor);
    params.out_proj = MatrixView{cursor, channels, channels, channels};
    cursor += c * c;
    params.heads = heads;
    params.mlp_scale = MatrixView{f.mlp_scale, 1, channels, channels};
    params.attn_scale = MatrixView{f.attn_scale, 1, channels, channels};
    params.head_scale = MatrixView{f.head_scale, 1, heads, heads};
    const MatrixView raw{f.exp_bias_raw, heads, kBiasTable, kBiasTable};
    MutableMatrixView derived{cursor, heads, kBiasTable, kBiasTable};
    params.exp_bias = MatrixView{cursor, heads, kBiasTable, kBiasTable};
    return swin_exp_bias(raw, &derived,
                         static_cast<std::size_t>(heads) * kBiasTable);
}

// A dimensions-only body params (for the scratch-size helpers, which read
// only `expand.rows` and `exp_bias.rows`).
inline SwinBlockParams swin_shape_params(int channels, int heads) {
    SwinBlockParams params;
    params.expand = MatrixView{nullptr, 4 * channels, channels, channels};
    params.heads = heads;
    params.contract = MatrixView{nullptr, channels, channels, channels};
    params.mlp_scale = MatrixView{nullptr, 1, channels, channels};
    params.qkv = MatrixView{nullptr, 3 * channels, channels, channels};
    params.head_scale = MatrixView{nullptr, 1, heads, heads};
    params.exp_bias = MatrixView{nullptr, heads, kBiasTable, kBiasTable};
    params.attn_scale = MatrixView{nullptr, 1, channels, channels};
    params.out_proj = MatrixView{nullptr, channels, channels, channels};
    return params;
}

// ---- per-call arena blocks -----------------------------------------------------------------------

struct RowBlocks {
    float* main = nullptr;
    float* aside = nullptr;
    float* work = nullptr;
    std::size_t work_count = 0;
};

inline Status row_begin(EvalContext& ctx, const LayerSpec& spec,
                        std::size_t main_count, std::size_t aside_count,
                        std::size_t work_count, RowBlocks& blocks) {
    if (ctx.scratch == nullptr
        || ctx.scratch_capacity < static_cast<std::size_t>(kEvalStateWords)) {
        return Status::kInvalidBuffer;
    }
    StateView state{ctx.scratch};
    if (state.get(0) != kStateMagic) {
        return Status::kInvalidValues;  // eval_reset was not called
    }
    const std::uint32_t owner = static_cast<std::uint32_t>(spec.index);
    std::size_t offset = 0;
    Status status = arena_alloc(state, ctx.scratch_capacity, main_count, owner,
                                offset);
    if (status != Status::kOk) {
        return status;
    }
    blocks.main = ctx.scratch + offset;
    if (aside_count > 0) {
        status = arena_alloc(state, ctx.scratch_capacity, aside_count, owner,
                             offset);
        if (status != Status::kOk) {
            arena_free(state, owner);
            return status;
        }
        blocks.aside = ctx.scratch + offset;
    }
    status = arena_alloc(state, ctx.scratch_capacity, work_count,
                         kTransientOwner, offset);
    if (status != Status::kOk) {
        arena_free(state, owner);
        return status;
    }
    blocks.work = ctx.scratch + offset;
    blocks.work_count = work_count;
    return Status::kOk;
}

inline void row_abort(EvalContext& ctx, const LayerSpec& spec) {
    StateView state{ctx.scratch};
    arena_free(state, static_cast<std::uint32_t>(spec.index));
    arena_free(state, kTransientOwner);
}

inline void row_end(EvalContext& ctx) {
    StateView state{ctx.scratch};
    arena_free(state, kTransientOwner);
}

// The image view carries the interleaved RGB frame (FrameView semantics).
// The frame boundary: the lift and compose read interleaved RGB rows
// (`FrameView`), while the layer values and the run's image are planar
// `[3, rows, cols]` maps. `frame_from_planar` copies one into a tight
// interleaved frame.
inline Status frame_from_planar(const FeatureBatchView& src, float* dst,
                                std::size_t dst_count) {
    if (src.data == nullptr || dst == nullptr) {
        return Status::kNullPointer;
    }
    if (src.channels != kOutputChannels || src.rows < 1 || src.cols < 1) {
        return Status::kInvalidDimensions;
    }
    if (src.row_stride < static_cast<std::ptrdiff_t>(src.cols)
        || src.plane_stride < static_cast<std::ptrdiff_t>(src.rows) * src.row_stride) {
        return Status::kInvalidStride;
    }
    const std::size_t pixels = static_cast<std::size_t>(src.rows)
        * static_cast<std::size_t>(src.cols);
    if (dst_count < 3u * pixels) {
        return Status::kInvalidBuffer;
    }
    for (int r = 0; r < src.rows; ++r) {
        for (int col = 0; col < src.cols; ++col) {
            float* px = dst + 3u * (static_cast<std::size_t>(r) * src.cols
                                    + static_cast<std::size_t>(col));
            for (int c = 0; c < 3; ++c) {
                const float* plane = src.data + static_cast<std::size_t>(c)
                    * static_cast<std::size_t>(src.plane_stride);
                px[c] = plane[static_cast<std::size_t>(r)
                              * static_cast<std::size_t>(src.row_stride)
                              + static_cast<std::size_t>(col)];
            }
        }
    }
    return Status::kOk;
}

// The inverse: a tight interleaved frame -> a tight `[3, rows, cols]` map.
inline Status frame_to_planar(const float* src, std::size_t src_count,
                              int rows, int cols, float* dst,
                              std::size_t dst_count) {
    if (src == nullptr || dst == nullptr) {
        return Status::kNullPointer;
    }
    if (rows < 1 || cols < 1) {
        return Status::kInvalidDimensions;
    }
    const std::size_t pixels = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(cols);
    if (src_count < 3u * pixels || dst_count < 3u * pixels) {
        return Status::kInvalidBuffer;
    }
    for (int c = 0; c < 3; ++c) {
        float* plane = dst + static_cast<std::size_t>(c) * pixels;
        for (std::size_t p = 0; p < pixels; ++p) {
            plane[p] = src[3u * p + static_cast<std::size_t>(c)];
        }
    }
    return Status::kOk;
}

// The shifted window grid in the swin entries' (rows, columns) order: the
// plan's shift_x steps along the columns and shift_y along the rows, so the
// pair is swapped for the native entries (only split_qkv_attn keeps the pin
// order).
inline void swin_shifts(const LayerSpec& spec, int& shift_rows,
                        int& shift_cols) {
    shift_rows = spec.shift_y;
    shift_cols = spec.shift_x;
}

// The swin rows' projection-site scratch (the G8 batched seam): swin_linear's
// dense input and products for one batched chunk plus swin_block's chunk
// work, carved from one caller-provided buffer in the order
// swin_site_scratch_needed declares. Sized for the largest chunk
// (kSwinBatchWindows windows = kSwinBatchWindows * kWindowTokens rows); the
// eval's expand matrix is always [4C, C] (swin_fetch/swin_shape_params), so
// the products region covers 4 * channels rows per token.
inline std::size_t swin_site_count(int channels, int heads) {
    return swin_site_scratch_needed(kSwinBatchWindows * kWindowTokens,
                                    channels, 4 * channels, heads);
}

inline void swin_site_carve(float* base, int channels, int heads,
                            SwinProjectionSite& site) {
    const std::size_t rows = static_cast<std::size_t>(kSwinBatchWindows)
        * kWindowTokens;
    site.input_dense = base;
    site.input_capacity = rows * static_cast<std::size_t>(channels);
    site.products = base + site.input_capacity;
    site.products_capacity = rows * static_cast<std::size_t>(4 * channels);
    site.batch_work = site.products + site.products_capacity;
    site.batch_capacity = swin_batch_work_needed(
        kSwinBatchWindows * kWindowTokens, channels, 4 * channels, heads);
}

}  // namespace eval_detail

// ---- the layer bindings --------------------------------------------------------------------------

namespace eval_detail {

// ---- PreBlock: image -> lifted features -> the 32-channel body -> (pooled, body) ------------------

inline Status eval_pre_block(EvalContext& ctx, const LayerSpec& spec,
                             const LayerCall& call, LayerValue& out) {
    const int channels = kLiftedChannels;
    const int heads = 1;
    if (call.image == nullptr || call.image->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& image = *call.image;
    if (image.channels != 3 || image.rows < 1 || image.cols < 1
        || (image.rows & 1) != 0 || (image.cols & 1) != 0) {
        return Status::kInvalidDimensions;
    }
    if (image.rows / 2 != spec.out_w || image.cols / 2 != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    SwinFetch fetch;
    Status status = swin_fetch(ctx.weights, spec.block, spec.layer, channels,
                               heads, fetch);
    if (status != Status::kOk) {
        return status;
    }
    const float* lift_record = nullptr;
    status = fetch_f16(ctx.weights, spec.block, spec.layer, "input_lift",
                       kStoredLiftValues, lift_record);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t pixels = static_cast<std::size_t>(image.rows)
        * static_cast<std::size_t>(image.cols);
    const std::size_t map = static_cast<std::size_t>(channels) * pixels;
    int shift_rows = 0;
    int shift_cols = 0;
    swin_shifts(spec, shift_rows, shift_cols);
    const FeatureBatchView lifted_view{nullptr, channels, image.rows, image.cols,
                                       static_cast<std::ptrdiff_t>(image.rows)
                                           * image.cols,
                                       static_cast<std::ptrdiff_t>(image.cols)};
    const SwinBlockParams shape = swin_shape_params(channels, heads);
    const std::size_t block_needed = swin_block_scratch_needed(
        lifted_view, true, false, shift_rows, shift_cols, shape);
    const std::size_t pooled_count = static_cast<std::size_t>(channels)
        * pixels / 4;
    const std::size_t frame_count = 3u * pixels;
    const bool gpu_projection =
        ctx.projection != nullptr && ctx.projection->matmul_f32 != nullptr
        && ctx.swin_gpu;
    // The G8 batched seam's site scratch: one chunk's swin_linear buffers
    // plus swin_block's chunk work (swin_site_count).
    const std::size_t site_scratch =
        gpu_projection ? swin_site_count(channels, heads) : 0;
    const std::size_t work_count = static_cast<std::size_t>(kStoredLiftValues)
        + map + map + frame_count + swin_decoded_floats(channels, heads, fetch)
        + block_needed + site_scratch;
    RowBlocks blocks;
    status = row_begin(ctx, spec, pooled_count, map, work_count, blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* work = blocks.work;
    float* lift = work;
    float* lifted = lift + kStoredLiftValues;
    float* acc = lifted + map;
    float* frame = acc + map;
    float* decoded = frame + frame_count;
    float* block_scratch = decoded + swin_decoded_floats(channels, heads, fetch);
    SwinProjectionSite swsite;
    if (gpu_projection) {
        swsite.ex = ctx.projection;
        swin_site_carve(block_scratch + block_needed, channels, heads, swsite);
    }
    SwinSiteOverlay overlay;
    build_swin_overlay(ctx.weights, spec.block, spec.layer, channels, 1,
                       overlay);
    MatrixView mid_storage;
    SwinBlockParams params;
    status = swin_decode(fetch, channels, heads, decoded,
                         swin_decoded_floats(channels, heads, fetch),
                         mid_storage, params);
    params.site = &overlay;
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    // The stored 512-value record carries the hardware permutation; the lift
    // matrix is its decode (the pinned `_lift_matrix`).
    status = decode_lift(lift_record, static_cast<std::size_t>(kStoredLiftValues),
                         lift);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    status = frame_from_planar(*call.image, frame, frame_count);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    const FrameView frame_view{frame, image.rows, image.cols,
                               static_cast<std::ptrdiff_t>(3) * image.cols};
    status = input_lift(frame_view,
                        ctx.history != nullptr ? *ctx.history : frame_view,
                        ctx.controls, lift,
                        static_cast<std::size_t>(kStoredLiftValues), lifted,
                        map);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    FeatureBatchView lifted_map;
    lifted_map.data = lifted;
    lifted_map.channels = channels;
    lifted_map.rows = image.rows;
    lifted_map.cols = image.cols;
    lifted_map.plane_stride =
        static_cast<std::ptrdiff_t>(image.rows) * image.cols;
    lifted_map.row_stride = static_cast<std::ptrdiff_t>(image.cols);
    status = swin_core(lifted_map, shift_rows, shift_cols, params, blocks.aside,
                       map, acc, map, block_scratch, block_needed,
                       gpu_projection ? &swsite : nullptr);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    // The aside is the body's f16 output stored on the e4m3 grid (round8).
    for (std::size_t i = 0; i < map; ++i) {
        blocks.aside[i] = e4m3(blocks.aside[i]);
    }
    // The main output is the pooled map (pool2x2 per plane of the acc map).
    const std::size_t pooled_plane = pixels / 4;
    for (int c = 0; c < channels; ++c) {
        PlaneView plane;
        plane.data = acc + static_cast<std::size_t>(c) * pixels;
        plane.rows = image.rows;
        plane.cols = image.cols;
        plane.row_stride = static_cast<std::ptrdiff_t>(image.cols);
        status = pool2x2(plane, blocks.main + static_cast<std::size_t>(c)
                             * pooled_plane,
                         pooled_plane);
        if (status != Status::kOk) {
            row_abort(ctx, spec);
            return status;
        }
    }
    out.main = FeatureBatchView{blocks.main, channels, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.aside = FeatureBatchView{blocks.aside, channels, image.rows, image.cols,
                                 static_cast<std::ptrdiff_t>(image.rows)
                                     * image.cols,
                                 static_cast<std::ptrdiff_t>(image.cols)};
    out.has_aside = true;
    row_end(ctx);
    return Status::kOk;
}

// ---- the Swin1H..Swin8H variants -----------------------------------------------------------------

inline Status eval_swin_layer(EvalContext& ctx, const LayerSpec& spec,
                              const LayerCall& call, LayerValue& out) {
    const int channels = spec.channels;
    const int heads = spec.heads;
    if (call.x == nullptr || call.x->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    // A wide upsample row's body is `channels` wide but reads its input at
    // twice that (`swin_upsample_blend`'s [C, 2C] projection); every other
    // variant reads its own width.
    const bool upsample = spec.variant == LayerVariant::kUpsample;
    if (x.channels != (upsample ? 2 * channels : channels)) {
        return Status::kInvalidDimensions;
    }
    SwinFetch fetch;
    Status status = swin_fetch(ctx.weights, spec.block, spec.layer, channels,
                               heads, fetch);
    if (status != Status::kOk) {
        return status;
    }
    // The int4 SwinSite overlay; it outlives every call below.
    SwinSiteOverlay overlay;
    build_swin_overlay(ctx.weights, spec.block, spec.layer, channels, heads,
                       overlay);
    int shift_rows = 0;
    int shift_cols = 0;
    swin_shifts(spec, shift_rows, shift_cols);
    const SwinBlockParams shape = swin_shape_params(channels, heads);
    MatrixView mid_storage;
    SwinBlockParams params;
    const std::size_t decoded_count = swin_decoded_floats(channels, heads,
                                                          fetch);
    RowBlocks blocks;
    if (spec.variant == LayerVariant::kDs) {
        // The body's pool2x2 contract: even input extents.
        if ((x.rows & 1) != 0 || (x.cols & 1) != 0) {
            return Status::kInvalidDimensions;
        }
        const std::uint8_t* resample_codes = nullptr;
        status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "resample",
                            2 * channels, channels, resample_codes);
        if (status != Status::kOk) {
            return status;
        }
        const std::size_t map = static_cast<std::size_t>(channels)
            * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
        const std::size_t next_count = static_cast<std::size_t>(2 * channels)
            * static_cast<std::size_t>(spec.out_w)
            * static_cast<std::size_t>(spec.out_h);
        const std::size_t pooled = static_cast<std::size_t>(channels)
            * static_cast<std::size_t>(x.rows / 2)
            * static_cast<std::size_t>(x.cols / 2);
        const std::size_t block_needed = swin_block_scratch_needed(
            x, false, false, shift_rows, shift_cols, shape);
        const bool gpu_projection =
            ctx.projection != nullptr && ctx.projection->matmul_f32 != nullptr
            && ctx.swin_gpu;
        // The G8 batched seam's site scratch (swin_site_count).
        const std::size_t site_scratch =
            gpu_projection ? swin_site_count(channels, heads) : 0;
        const std::size_t work_count = decoded_count
            + static_cast<std::size_t>(2 * channels) * channels  // resample
            + map + pooled + block_needed + site_scratch;
        status = row_begin(ctx, spec, next_count, map, work_count, blocks);
        if (status != Status::kOk) {
            return status;
        }
        float* decoded = blocks.work;
        float* resample_dst = decoded + decoded_count;
        float* ds_scratch = resample_dst
            + static_cast<std::size_t>(2 * channels) * channels;
        SwinProjectionSite swsite;
        if (gpu_projection) {
            swsite.ex = ctx.projection;
            swin_site_carve(ds_scratch + map + pooled + block_needed,
                            channels, heads, swsite);
        }
        status = swin_decode(fetch, channels, heads, decoded, decoded_count,
                             mid_storage, params);
        params.site = &overlay;
        if (status != Status::kOk) {
            row_abort(ctx, spec);
            return status;
        }
        decode_e4m3(resample_codes,
                    static_cast<std::size_t>(2 * channels) * channels,
                    resample_dst);
        const MatrixView resample{resample_dst, 2 * channels, channels,
                                  channels};
        status = swin_downsample(x, shift_rows, shift_cols, params, resample,
                                 spec.out_w, spec.out_h, blocks.main,
                                 next_count, blocks.aside, map, ds_scratch,
                                 map + pooled + block_needed,
                                 gpu_projection ? &swsite : nullptr);
        if (status != Status::kOk) {
            row_abort(ctx, spec);
            return status;
        }
        out.main = FeatureBatchView{blocks.main, 2 * channels, spec.out_w,
                                    spec.out_h,
                                    static_cast<std::ptrdiff_t>(spec.out_w)
                                        * spec.out_h,
                                    static_cast<std::ptrdiff_t>(spec.out_h)};
        out.aside = FeatureBatchView{blocks.aside, channels, x.rows, x.cols,
                                     static_cast<std::ptrdiff_t>(x.rows) * x.cols,
                                     static_cast<std::ptrdiff_t>(x.cols)};
        out.has_aside = true;
        row_end(ctx);
        return Status::kOk;
    }
    if (spec.variant == LayerVariant::kUpsample) {
        if (call.aux == nullptr || call.aux->data == nullptr) {
            return Status::kNullPointer;
        }
        const FeatureBatchView& skip = *call.aux;
        if (skip.channels != channels
            || skip.rows != spec.out_w || skip.cols != spec.out_h) {
            return Status::kInvalidDimensions;
        }
        const std::uint8_t* resample_codes = nullptr;
        status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "resample",
                            channels, 2 * channels, resample_codes);
        if (status != Status::kOk) {
            return status;
        }
        const float* gain_record = nullptr;
        int gain_length = 0;
        status = fetch_f16_any(ctx.weights, spec.block, spec.layer,
                               "upsample_gain", gain_record, gain_length);
        if (status != Status::kOk) {
            return status;
        }
        const float* table = nullptr;
        int table_length = 0;
        const float* gain_row = nullptr;
        if (gain_length < channels) {
            // The pinned wide fallback prepends the tail of the residual table.
            status = fetch_f16_any(ctx.weights, spec.block, spec.layer,
                                   "residual_scale", table, table_length);
            if (status != Status::kOk) {
                return status;
            }
        }
        const std::size_t blend_count = static_cast<std::size_t>(channels)
            * static_cast<std::size_t>(skip.rows) * static_cast<std::size_t>(skip.cols);
        const std::size_t projected_count = static_cast<std::size_t>(channels)
            * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
        FeatureBatchView blend_view;
        blend_view.data = nullptr;
        blend_view.channels = channels;
        blend_view.rows = skip.rows;
        blend_view.cols = skip.cols;
        blend_view.plane_stride =
            static_cast<std::ptrdiff_t>(skip.rows) * skip.cols;
        blend_view.row_stride = static_cast<std::ptrdiff_t>(skip.cols);
        const std::size_t block_needed = swin_block_scratch_needed(
            blend_view, channels == 32, true, shift_rows, shift_cols, shape);
        const bool gpu_projection =
            ctx.projection != nullptr && ctx.projection->matmul_f32 != nullptr
            && ctx.swin_gpu;
        // The G8 batched seam's site scratch (swin_site_count).
        const std::size_t site_scratch =
            gpu_projection ? swin_site_count(channels, heads) : 0;
        const std::size_t work_count = decoded_count
            + static_cast<std::size_t>(channels) * 2 * channels  // resample
            + static_cast<std::size_t>(channels)                 // gain copy
            + blend_count + projected_count + block_needed + site_scratch;
        status = row_begin(ctx, spec, blend_count, 0, work_count, blocks);
        if (status != Status::kOk) {
            return status;
        }
        float* decoded = blocks.work;
        float* resample_dst = decoded + decoded_count;
        float* gain_dst = resample_dst
            + static_cast<std::size_t>(channels) * 2 * channels;
        float* up_scratch = gain_dst + static_cast<std::size_t>(channels);
        SwinProjectionSite swsite;
        if (gpu_projection) {
            swsite.ex = ctx.projection;
            swin_site_carve(up_scratch + blend_count + projected_count
                                + block_needed,
                            channels, heads, swsite);
        }
        status = swin_decode(fetch, channels, heads, decoded, decoded_count,
                             mid_storage, params);
        params.site = &overlay;
        if (status != Status::kOk) {
            row_abort(ctx, spec);
            return status;
        }
        decode_e4m3(resample_codes,
                    static_cast<std::size_t>(channels) * 2 * channels,
                    resample_dst);
        const MatrixView resample{resample_dst, channels, 2 * channels,
                                  2 * channels};
        status = upsample_gain_row(gain_record, gain_length, table, table_length,
                                   channels, gain_dst, gain_row);
        if (status != Status::kOk) {
            row_abort(ctx, spec);
            return status;
        }
        const MatrixView gain{gain_row, 1, channels, channels};
        status = swin_layer_upsample(x, skip, shift_rows, shift_cols, params,
                                     resample, gain, blocks.main, blend_count,
                                     up_scratch,
                                     blend_count + projected_count
                                         + block_needed,
                                     gpu_projection ? &swsite : nullptr);
        if (status != Status::kOk) {
            row_abort(ctx, spec);
            return status;
        }
        out.main = FeatureBatchView{blocks.main, channels, spec.out_w,
                                    spec.out_h,
                                    static_cast<std::ptrdiff_t>(spec.out_w)
                                        * spec.out_h,
                                    static_cast<std::ptrdiff_t>(spec.out_h)};
        out.has_aside = false;
        row_end(ctx);
        return Status::kOk;
    }
    // Plain body (plain / inpview / outview / no variant).
    if (x.rows != spec.out_w || x.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    const std::size_t map = static_cast<std::size_t>(channels)
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    const std::size_t block_needed = swin_block_scratch_needed(
        x, false, true, shift_rows, shift_cols, shape);
    const bool gpu_projection =
        ctx.projection != nullptr && ctx.projection->matmul_f32 != nullptr
        && ctx.swin_gpu;
    // The G8 batched seam's site scratch (swin_site_count).
    const std::size_t site_scratch =
        gpu_projection ? swin_site_count(channels, heads) : 0;
    const std::size_t work_count = decoded_count + block_needed + site_scratch;
    status = row_begin(ctx, spec, map, 0, work_count, blocks);
    if (status != Status::kOk) {
        return status;
    }
    SwinProjectionSite swsite;
    if (gpu_projection) {
        swsite.ex = ctx.projection;
        swin_site_carve(blocks.work + decoded_count + block_needed, channels,
                        heads, swsite);
    }
    status = swin_decode(fetch, channels, heads, blocks.work, decoded_count,
                         mid_storage, params);
    params.site = &overlay;
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    status = swin_layer_plain(x, shift_rows, shift_cols, params, blocks.main,
                              map, blocks.work + decoded_count, block_needed,
                              gpu_projection ? &swsite : nullptr);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, channels, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

// ---- the split family -----------------------------------------------------------------------------

// The projection records shared by the split projections.
inline Status split_proj_fetch(const EvalWeights* w, const LayerSpec& spec,
                               bool with_gain, const std::uint8_t*& codes,
                               const float*& gain) {
    Status status = fetch_e4m3(w, spec.block, spec.layer, "weight", spec.c_out,
                               spec.c_in, codes);
    if (status != Status::kOk) {
        return status;
    }
    gain = nullptr;
    if (with_gain) {
        return fetch_f16(w, spec.block, spec.layer, "skip_weight", spec.c_out,
                         gain);
    }
    return Status::kOk;
}

inline Status eval_split_ffwd(EvalContext& ctx, const LayerSpec& spec,
                              const LayerCall& call, LayerValue& out) {
    if (call.x == nullptr || call.x->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    if (x.channels != kSplitChannels || x.rows != spec.out_w
        || x.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    const std::uint8_t* expand_in = nullptr;
    const std::uint8_t* expand_hidden = nullptr;
    const std::uint8_t* contract = nullptr;
    Status status = fetch_ffwd(ctx.weights, spec.block, spec.layer, 0,
                               kSplitChannels, kSplitChannels, expand_in);
    if (status != Status::kOk) {
        return status;
    }
    status = fetch_ffwd(ctx.weights, spec.block, spec.layer, 1,
                        kSplitGroups * kSplitGroupHidden, kSplitGroupIn,
                        expand_hidden);
    if (status != Status::kOk) {
        return status;
    }
    status = fetch_ffwd(ctx.weights, spec.block, spec.layer, 2,
                        kSplitGroups * kSplitGroupIn, kSplitGroupHidden,
                        contract);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t tokens = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t map = tokens * kSplitChannels;
    RowBlocks blocks;
    status = row_begin(ctx, spec, map, 0, 2 * map, blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* out_tokens = in_tokens + map;
    status = eval_planar_to_tokens(x, in_tokens, map);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    const SplitFfwdWeight weight{
        E4m3Matrix{expand_in, kSplitChannels, kSplitChannels},
        E4m3Matrix{expand_hidden, kSplitGroups * kSplitGroupHidden,
                   kSplitGroupIn},
        E4m3Matrix{contract, kSplitGroups * kSplitGroupIn, kSplitGroupHidden}};
    status = split_ffwd(in_tokens, x.rows, x.cols, weight, out_tokens);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    status = eval_tokens_to_planar(out_tokens, map, kSplitChannels, x.rows,
                                   x.cols, blocks.main, map);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, kSplitChannels, spec.out_w,
                                spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.aside = *call.x;  // the pinned (out, x) tuple: the unchanged input
    out.has_aside = true;
    row_end(ctx);
    return Status::kOk;
}

// SplitFfwdProj / SplitProj (the same projection, different residual).
inline Status eval_split_projection(EvalContext& ctx, const LayerSpec& spec,
                                    const LayerCall& call, LayerValue& out) {
    if (call.x == nullptr || call.x->data == nullptr
        || call.aux == nullptr || call.aux->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    const FeatureBatchView& residual = *call.aux;
    if (x.channels != spec.c_in || residual.channels != spec.c_out
        || x.rows != spec.out_w || x.cols != spec.out_h
        || residual.rows != x.rows || residual.cols != x.cols) {
        return Status::kInvalidDimensions;
    }
    const std::uint8_t* codes = nullptr;
    const float* gain = nullptr;
    Status status = split_proj_fetch(ctx.weights, spec, true, codes, gain);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t tokens = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t in_count = tokens * static_cast<std::size_t>(spec.c_in);
    const std::size_t out_count = tokens * static_cast<std::size_t>(spec.c_out);
    const bool gpu_projection =
        ctx.projection != nullptr && ctx.projection->matmul != nullptr;
    RowBlocks blocks;
    status = row_begin(ctx, spec, out_count, 0,
                       in_count + out_count + out_count
                           + (gpu_projection ? out_count : 0),
                       blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* residual_tokens = in_tokens + in_count;
    float* out_tokens = residual_tokens + out_count;
    SplitProjectionSite site;
    if (gpu_projection) {
        site.ex = ctx.projection;
        site.products = out_tokens + out_count;
        site.products_capacity = out_count;
    }
    status = eval_planar_to_tokens(x, in_tokens, in_count);
    if (status == Status::kOk) {
        status = eval_planar_to_tokens(residual, residual_tokens, out_count);
    }
    if (status == Status::kOk) {
        const SplitProjectionWeight weight{E4m3Matrix{codes, spec.c_out,
                                                      spec.c_in}, gain};
        status = spec.kind == LayerKind::kSplitFfwdProj
            ? split_ffwd_proj(in_tokens, residual_tokens, x.rows, x.cols, weight,
                              out_tokens,
                              out.stash.attn,   // the pinned's ("attn", block)
                              gpu_projection ? &site : nullptr)
            : split_proj(in_tokens, residual_tokens, x.rows, x.cols, weight,
                         out_tokens, gpu_projection ? &site : nullptr);
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(out_tokens, out_count, spec.c_out, x.rows,
                                       x.cols, blocks.main, out_count);
    }
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, spec.c_out, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

inline Status eval_split_qkv_attn(EvalContext& ctx, const LayerSpec& spec,
                                  const LayerCall& call, LayerValue& out) {
    if (call.x == nullptr || call.x->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    if (x.channels != kSplitChannels || x.rows != spec.out_w
        || x.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    const float* bias_raw = nullptr;
    const float* gain = nullptr;
    Status status = Status::kOk;
    // The int4 site's record presence selects the mixed path (the pinned's
    // table-presence gating); it never fetches the e4m3 "qkv".
    const Int4Site* site = nullptr;
    if (ctx.weights->int4 != nullptr) {
        site = ctx.weights->int4(ctx.weights->context, spec.block, spec.layer,
                                 "attn_qkv");
    }
    if (site != nullptr) {
        // The pinned attention_matrix is [1536, 512] block 16 non-half with
        // no permutation; a mis-delivered record rejects rather than
        // silently evaluating with the wrong quantiser.
        if (site->matrix.rows != kSplitQkvOut
            || site->matrix.cols != kSplitChannels
            || site->matrix.block < 1
            || site->matrix.block > kSplitChannels
            || kSplitChannels % site->matrix.block != 0
            || site->perm != nullptr || site->matrix.half) {
            return Status::kInvalidValues;
        }
    }
    const std::uint8_t* qkv_codes = nullptr;
    if (site == nullptr) {
        status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "qkv",
                            kSplitQkvOut, kSplitChannels, qkv_codes);
    }
    if (status != Status::kOk) {
        return status;
    }
    status = fetch_f16(ctx.weights, spec.block, spec.layer, "attn_pos_bias",
                       kSplitHeads * kBiasTable, bias_raw);
    if (status != Status::kOk) {
        return status;
    }
    status = fetch_f32(ctx.weights, spec.block, spec.layer, "tail", kSplitHeads,
                       gain);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t tokens = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t map = tokens * kSplitChannels;
    RowBlocks blocks;
    const std::size_t qkv_map = tokens
        * static_cast<std::size_t>(kSplitQkvOut);
    // The hoisted QKV projection (below, non-int4 only) needs products + a
    // zero padded projection; the int4 mode has its own codes/scratch.
    const std::size_t work = 2 * map
        + static_cast<std::size_t>(kSplitHeads) * kBiasTable
        + (site == nullptr ? qkv_map + kSplitQkvOut : 0)
        + (site != nullptr ? map + qkv_map + kSplitQkvOut : 0);
    status = row_begin(ctx, spec, map, 0, work, blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* out_tokens = in_tokens + map;
    float* bias = out_tokens + map;
    float* hoist_products = nullptr;
    float* hoist_padded = nullptr;
    float* codes = nullptr;
    float* products = nullptr;
    float* padded = nullptr;
    if (site != nullptr) {
        codes = bias + static_cast<std::size_t>(kSplitHeads) * kBiasTable;
        products = codes + map;
        padded = products + qkv_map;
    } else {
        hoist_products = bias
            + static_cast<std::size_t>(kSplitHeads) * kBiasTable;
        hoist_padded = hoist_products + qkv_map;
    }
    status = eval_planar_to_tokens(x, in_tokens, map);
    if (status == Status::kOk) {
        // The record holds the accumulator-fragment order; the kernel takes
        // the raw unpacked [head][query, key] table (no exponent ramp).
        const MatrixView raw{bias_raw, kSplitHeads, kBiasTable, kBiasTable};
        MutableMatrixView unpacked{bias, kSplitHeads, kBiasTable, kBiasTable};
        status = swin_deswizzle_bias(raw, &unpacked,
                                     static_cast<std::size_t>(kSplitHeads)
                                         * kBiasTable);
    }
    if (status == Status::kOk && site != nullptr) {
        // The mixed projection: the map quantised and accumulated into the
        // site's [tokens, kSplitQkvOut] products, plus the zero-input
        // projection the padded tokens take. Note the seam-wide divergence
        // (record contract above): the run's map is the producer's e4m3
        // store, where the pinned quantises the producer's f16 value.
        // The quantiser reads the projection's stash when the run provides
        // it (the pinned `int4.side[("attn", block)]`); the stored-map
        // fallback is the documented seam divergence.
        const float* attn_src = call.stash.attn != nullptr
            ? call.stash.attn : in_tokens;
        status = int4_quantise(site->matrix, attn_src,
                               static_cast<int>(tokens), codes);
        if (status == Status::kOk) {
            status = int4_accumulate(site->matrix, codes,
                                     static_cast<int>(tokens), products);
        }
        if (status == Status::kOk) {
            // The zero-input projection for the padded tokens; the zero
            // row's codes borrow out_tokens (written later by the kernel).
            float zero_row[kSplitChannels] = {};
            status = int4_quantise(site->matrix, zero_row, 1, out_tokens);
            if (status == Status::kOk) {
                status = int4_accumulate(site->matrix, out_tokens, 1, padded);
            }
        }
    }
    if (status == Status::kOk && site == nullptr) {
        // Hoisted QKV projection: one batched [tokens, 512] x [1536, 512]
        // matmul replaces the kernel's per-token projections (the same row
        // dots in the same key order; the kernel's products branch applies
        // the same f16 round to each product). The padded tokens' zero-input
        // projection through the e4m3 matrix is exactly +0. Bit-exact with
        // the CPU fallback; the executor path takes the Vulkan backend's
        // documented normal-range tolerance (the same convention as the vit
        // rows' hoisted projections). The fallback uses split_matmul (the
        // split family's 2^29 token envelope, wider than matmul_e4m3).
        const E4m3Matrix qkv{qkv_codes, kSplitQkvOut, kSplitChannels};
        if (ctx.projection != nullptr && ctx.projection->matmul != nullptr) {
            status = ctx.projection->matmul(
                ctx.projection->ctx, in_tokens,
                static_cast<int>(tokens), kSplitChannels, qkv, hoist_products);
        } else {
            status = split_matmul(in_tokens,
                                  static_cast<long long>(tokens),
                                  kSplitChannels, qkv, hoist_products);
        }
        if (status == Status::kOk) {
            for (std::size_t i = 0; i < kSplitQkvOut; ++i) {
                hoist_padded[i] = 0.0f;
            }
        }
    }
    if (status == Status::kOk) {
        const SplitQkvWeight weight{E4m3Matrix{qkv_codes, kSplitQkvOut,
                                               kSplitChannels}, bias, gain,
                                    site != nullptr ? products : hoist_products,
                                    site != nullptr ? padded : hoist_padded};
        status = split_qkv_attn(in_tokens, x.rows, x.cols, spec.shift_x,
                                spec.shift_y, weight, out_tokens);
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(out_tokens, map, kSplitChannels, x.rows,
                                       x.cols, blocks.main, map);
    }
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, kSplitChannels, spec.out_w,
                                spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

inline Status eval_split_proj_pool(EvalContext& ctx, const LayerSpec& spec,
                                   const LayerCall& call, LayerValue& out) {
    if (call.x == nullptr || call.x->data == nullptr
        || call.aux == nullptr || call.aux->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    const FeatureBatchView& residual = *call.aux;
    if (x.channels != spec.c_in || residual.channels != spec.c_out
        || x.rows != spec.out_w || x.cols != spec.out_h
        || residual.rows != x.rows || residual.cols != x.cols) {
        return Status::kInvalidDimensions;
    }
    const std::uint8_t* codes = nullptr;
    const float* gain = nullptr;
    Status status = split_proj_fetch(ctx.weights, spec, true, codes, gain);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t tokens = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t in_count = tokens * static_cast<std::size_t>(spec.c_in);
    const std::size_t out_count = tokens * static_cast<std::size_t>(spec.c_out);
    const int pad_rows = (((x.rows + 1) / 2) + 3) & ~3;
    const int pad_cols = (((x.cols + 1) / 2) + 3) & ~3;
    const std::size_t pooled_count = static_cast<std::size_t>(pad_rows)
        * static_cast<std::size_t>(pad_cols)
        * static_cast<std::size_t>(spec.c_out);
    const bool gpu_projection =
        ctx.projection != nullptr && ctx.projection->matmul != nullptr;
    RowBlocks blocks;
    status = row_begin(ctx, spec, out_count, pooled_count,
                       in_count + out_count + out_count + pooled_count
                           + (gpu_projection ? out_count : 0),
                       blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* residual_tokens = in_tokens + in_count;
    float* out_tokens = residual_tokens + out_count;
    float* pooled_tokens = out_tokens + out_count;
    SplitProjectionSite site;
    if (gpu_projection) {
        site.ex = ctx.projection;
        site.products = pooled_tokens + pooled_count;
        site.products_capacity = out_count;
    }
    status = eval_planar_to_tokens(x, in_tokens, in_count);
    if (status == Status::kOk) {
        status = eval_planar_to_tokens(residual, residual_tokens, out_count);
    }
    int pooled_rows = 0;
    int pooled_cols = 0;
    if (status == Status::kOk) {
        const SplitProjectionWeight weight{E4m3Matrix{codes, spec.c_out,
                                                      spec.c_in}, gain};
        status = split_proj_pool(in_tokens, residual_tokens, x.rows, x.cols,
                                 weight, out_tokens, pooled_tokens, &pooled_rows,
                                 &pooled_cols,
                                 gpu_projection ? &site : nullptr);
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(out_tokens, out_count, spec.c_out, x.rows,
                                       x.cols, blocks.main, out_count);
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(pooled_tokens, pooled_count, spec.c_out,
                                       pooled_rows, pooled_cols, blocks.aside,
                                       pooled_count);
    }
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, spec.c_out, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.aside = FeatureBatchView{blocks.aside, spec.c_out, pooled_rows,
                                 pooled_cols,
                                 static_cast<std::ptrdiff_t>(pooled_rows)
                                     * pooled_cols,
                                 static_cast<std::ptrdiff_t>(pooled_cols)};
    out.has_aside = true;
    row_end(ctx);
    return Status::kOk;
}

inline Status eval_split_final_head(EvalContext& ctx, const LayerSpec& spec,
                                    const LayerCall& call, LayerValue& out) {
    if (call.aux == nullptr || call.aux->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& pooled = *call.aux;
    if (pooled.channels != spec.c_in || pooled.rows != spec.out_w
        || pooled.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    const std::uint8_t* codes = nullptr;
    const float* gain = nullptr;
    Status status = split_proj_fetch(ctx.weights, spec, false, codes, gain);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t tokens = static_cast<std::size_t>(pooled.rows)
        * static_cast<std::size_t>(pooled.cols);
    const std::size_t in_count = tokens * static_cast<std::size_t>(spec.c_in);
    const std::size_t out_count = tokens * static_cast<std::size_t>(spec.c_out);
    const bool gpu_projection =
        ctx.projection != nullptr && ctx.projection->matmul != nullptr;
    RowBlocks blocks;
    status = row_begin(ctx, spec, out_count, 0,
                       in_count + out_count + (gpu_projection ? out_count : 0),
                       blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* out_tokens = in_tokens + in_count;
    SplitProjectionSite site;
    if (gpu_projection) {
        site.ex = ctx.projection;
        site.products = out_tokens + out_count;
        site.products_capacity = out_count;
    }
    status = eval_planar_to_tokens(pooled, in_tokens, in_count);
    if (status == Status::kOk) {
        const SplitProjectionWeight weight{E4m3Matrix{codes, spec.c_out,
                                                      spec.c_in}, nullptr};
        status = split_final_head(in_tokens, pooled.rows, pooled.cols, weight,
                                  out_tokens, gpu_projection ? &site : nullptr);
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(out_tokens, out_count, spec.c_out,
                                       pooled.rows, pooled.cols, blocks.main,
                                       out_count);
    }
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, spec.c_out, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

// ---- the vit family ------------------------------------------------------------------------------

// The int4 mixed contraction's residual form: `product` is the mixed int4 +
// e4m3-tail total per token (float32); the residual arithmetic is
// vit_residual_linear's exactly.
inline Status vit_residual_mixed(const float* product, const float* skip,
                                 int tokens, int c_out, const float* gain,
                                 float* out, float* stash = nullptr) {
    if (product == nullptr || skip == nullptr || gain == nullptr
        || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens <= 0 || c_out <= 0) {
        return Status::kInvalidDimensions;
    }
    for (int t = 0; t < tokens; ++t) {
        const float* row = product + static_cast<std::size_t>(t) * c_out;
        const float* skip_row = skip + static_cast<std::size_t>(t) * c_out;
        float* dst = out + static_cast<std::size_t>(t) * c_out;
        for (int n = 0; n < c_out; ++n) {
            // The binary16 value is the pinned's stash (`int4.side`); the
            // store rounds it to e4m3.
            const float value = vit_f16(row[n] + vit_f16(gain[n]) * skip_row[n]);
            if (stash != nullptr) {
                stash[static_cast<std::size_t>(t) * c_out + n] = value;
            }
            dst[n] = e4m3(value);
        }
    }
    return Status::kOk;
}

// The contract rows' stash destination (the pinned's `self.stash`): ("c",
// block) for layer 1 (this block's QKV), else ("stream", block + 1) when the
// next block has an expansion table. The writer value is the row's binary16
// `value` (product + gain * skip before the e4m3 store).
inline float* contract_stash_dest(const EvalWeights* weights, int block,
                                  int layer, StashBuffers& stash) {
    // The pinned sets `self.stash` only when `int4.active()`.
    if (weights == nullptr || weights->int4 == nullptr) {
        return nullptr;
    }
    if (layer == 1) {
        return stash.c;
    }
    {
        const Int4Site* next = weights->int4(weights->context, block + 1, 0,
                                             "vit_ffn");
        if (next != nullptr && next->has_ex) {
            return stash.stream;
        }
    }
    return nullptr;
}

// The ViT feed-forward's mixed contraction (VitFfn.contraction): the int4
// channels from the unrounded activation through the site's matrix, the rest
// from the e4m3 copy through the tail matrix, summed in float32, then the
// residual arithmetic of vit_residual_linear. The host's records: the site
// (matrix + hidden permutation) and the tail through `e4m3` as "tail".
inline Status eval_vit_ffn_mixed(EvalContext& ctx, const LayerSpec& spec,
                                 const FeatureBatchView& x,
                                 const FeatureBatchView& skip,
                                 const Int4Site& site, std::size_t tokens,
                                 RowBlocks& blocks,
                                 const float* hidden_stash = nullptr,
                                 StashBuffers* stash_out = nullptr) {
    const int c_in = spec.c_in;
    const int c_out = spec.c_out;
    const int k4 = site.matrix.cols;
    const int tail_cols = c_in - k4;
    if (site.matrix.rows != c_out || tail_cols < 1
        || site.matrix.block < 1 || site.matrix.block > k4
        || k4 % site.matrix.block != 0) {
        return Status::kInvalidValues;
    }
    if (site.perm == nullptr || site.perm_count != c_in) {
        return Status::kInvalidValues;
    }
    // The permutation must cover [0, c_in): every index is range-checked
    // below (which keeps the gathers in bounds), and the sum/xor pair is a
    // strong heuristic for completeness — it is NOT a proof (a duplicate can
    // preserve both), so a corrupt record may mis-evaluate but cannot read
    // out of bounds. Allocation-free; c_in <= 4096 keeps the sums exact.
    {
        long long sum = 0;
        long long xors = 0;
        for (int i = 0; i < c_in; ++i) {
            const int p = site.perm[i];
            if (p < 0 || p >= c_in) {
                return Status::kInvalidValues;
            }
            sum += p;
            xors ^= p;
        }
        long long want_sum = 0;
        long long want_xor = 0;
        for (int i = 0; i < c_in; ++i) {
            want_sum += i;
            want_xor ^= i;
        }
        if (sum != want_sum || xors != want_xor) {
            return Status::kInvalidValues;
        }
    }
    const std::uint8_t* tail_codes = nullptr;
    Status status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "tail",
                               c_out, tail_cols, tail_codes);
    if (status != Status::kOk) {
        return status;
    }
    const float* gain = nullptr;
    status = fetch_f16(ctx.weights, spec.block, spec.layer, "skip_weight",
                       c_out, gain);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t in_count = tokens * static_cast<std::size_t>(c_in);
    const std::size_t out_count = tokens * static_cast<std::size_t>(c_out);
    const std::size_t gather_count = tokens * static_cast<std::size_t>(k4);
    const std::size_t tail_size = static_cast<std::size_t>(c_out)
        * static_cast<std::size_t>(tail_cols);
    status = row_begin(ctx, spec, out_count, 0,
                       in_count + out_count + out_count + in_count
                           + gather_count + gather_count + out_count
                           + tail_size,
                       blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* skip_tokens = in_tokens + in_count;
    float* out_tokens = skip_tokens + out_count;
    float* hidden_e4m3 = out_tokens + out_count;
    float* gathered = hidden_e4m3 + in_count;
    float* codes = gathered + gather_count;
    float* product = codes + gather_count;
    float* tail_values = product + out_count;
    status = eval_planar_to_tokens(x, in_tokens, in_count);
    if (status == Status::kOk) {
        status = eval_planar_to_tokens(skip, skip_tokens, out_count);
    }
    if (status != Status::kOk) {
        return status;
    }
    for (std::size_t i = 0; i < in_count; ++i) {
        hidden_e4m3[i] = e4m3(in_tokens[i]);
    }
    // The int4 channels quantise the producer's stash (the pinned's
    // unrounded activation) when the run provides it; the stored-map copy is
    // the documented fallback (the seam divergence).
    const float* int4_src = hidden_stash != nullptr ? hidden_stash : in_tokens;
    for (std::size_t t = 0; t < tokens; ++t) {
        for (int j = 0; j < k4; ++j) {
            gathered[t * static_cast<std::size_t>(k4) + j] =
                int4_src[t * static_cast<std::size_t>(c_in)
                         + static_cast<std::size_t>(site.perm[j])];
        }
    }
    // The tail is exact e4m3 values delivered as raw codes. The pinned's
    // tail decode is the SOLVER's table, where the two NaN codes (0x7f, 0xff)
    // read as 0 — not the matmul table's canonical NaN.
    for (std::size_t i = 0; i < tail_size; ++i) {
        tail_values[i] = (tail_codes[i] & 0x7fu) == 0x7fu
            ? 0.0f : e4m3_from_bits(tail_codes[i]);
    }
    status = int4_quantise(site.matrix, gathered,
                           static_cast<int>(tokens), codes);
    if (status == Status::kOk) {
        status = int4_mixed_contraction(site.matrix, codes, tail_values,
                                        tail_cols, hidden_e4m3, site.perm,
                                        static_cast<int>(tokens), product);
    }
    if (status != Status::kOk) {
        return status;
    }
    float* stash_dest = stash_out != nullptr
        ? contract_stash_dest(ctx.weights, spec.block, spec.layer, *stash_out)
        : nullptr;
    status = vit_residual_mixed(product, skip_tokens,
                                static_cast<int>(tokens), c_out, gain,
                                out_tokens, stash_dest);
    if (status == Status::kOk && stash_dest != nullptr
        && stash_dest == stash_out->stream) {
        // The symmetric stamp to the e4m3 path's: the ("stream", block + 1)
        // key exists only after the write succeeded.
        ctx.stash_stream_block = spec.block + 1;
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(out_tokens, out_count, spec.c_out,
                                       x.rows, x.cols, blocks.main, out_count);
    }
    return status;
}

// The ViT QKV's mixed projection (the pinned `vit_matrix`): one int4 matrix
// [3 * channels, channels] through the quantiser + accumulator, then
// vit_qkv_post — the same post chain as the e4m3 path. The host's records:
// the "vit_qkv" site and the f32 "header" temperatures (shared with the e4m3
// path).
inline Status eval_vit_qkv_int4(EvalContext& ctx, const LayerSpec& spec,
                                const FeatureBatchView& x,
                                const Int4Site& site, int channels,
                                std::size_t tokens, RowBlocks& blocks,
                                const float* c_stash = nullptr) {
    // The pinned vit_matrix is block 16 non-half with no permutation: a
    // mis-delivered half-form or FFN-style record rejects rather than
    // silently evaluating with the wrong quantiser.
    if (site.matrix.rows != 3 * channels || site.matrix.cols != channels
        || site.matrix.block < 1 || site.matrix.block > channels
        || channels % site.matrix.block != 0 || site.perm != nullptr
        || site.matrix.half) {
        return Status::kInvalidValues;
    }
    const float* temperature = nullptr;
    Status status = fetch_f32(ctx.weights, spec.block, spec.layer, "header",
                              channels / kVitHeadDim, temperature);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t in_count = tokens
        * static_cast<std::size_t>(channels);
    const std::size_t out_count = tokens * 3u
        * static_cast<std::size_t>(channels);
    status = row_begin(ctx, spec, out_count, 0,
                       in_count + in_count + out_count, blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* in_tokens = blocks.work;
    float* codes = in_tokens + in_count;
    float* out_tokens = codes + in_count;
    status = eval_planar_to_tokens(x, in_tokens, in_count);
    if (status == Status::kOk) {
        // The quantiser reads the contract's stash (the pinned
        // `int4.side[("c", block)]`) when the run provides it; the
        // stored-map fallback is the documented seam divergence.
        const float* qkv_src = c_stash != nullptr ? c_stash : in_tokens;
        status = int4_quantise(site.matrix, qkv_src,
                               static_cast<int>(tokens), codes);
    }
    if (status == Status::kOk) {
        status = int4_accumulate(site.matrix, codes,
                                 static_cast<int>(tokens), out_tokens);
    }
    if (status == Status::kOk) {
        status = vit_qkv_post(out_tokens, static_cast<int>(tokens), channels,
                              temperature, out_tokens);
    }
    if (status == Status::kOk) {
        status = eval_tokens_to_planar(out_tokens, out_count, 3 * channels,
                                       x.rows, x.cols, blocks.main, out_count);
    }
    return status;
}

inline Status eval_vit(EvalContext& ctx, const LayerSpec& spec,
                       const LayerCall& call, LayerValue& out) {
    if (call.x == nullptr || call.x->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    if (x.rows != spec.out_w || x.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    const std::size_t tokens = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    RowBlocks blocks;
    Status status = Status::kOk;
    switch (spec.kind) {
        case LayerKind::kVitFfnExpand: {
            if (x.channels != spec.c_in) {
                return Status::kInvalidDimensions;
            }
            // The int4 expansion (the pinned `expand_preactivation`): with
            // the site's ex matrix AND the ("stream", block) stash present
            // the expansion quantises the STASH (never the layer input) and
            // scatters the permuted rows into the original channel order;
            // otherwise the e4m3 matmul reads the layer input (the pinned's
            // `exact(tokens)`). The stamp is the pinned's dict-key presence:
            // no producer wrote ("stream", block) (the first vit block) or a
            // later block already consumed/overwrote it means e4m3.
            const Int4Site* site = nullptr;
            if (ctx.weights->int4 != nullptr) {
                site = ctx.weights->int4(ctx.weights->context, spec.block,
                                         spec.layer, "vit_ffn");
            }
            if (site != nullptr && site->has_ex
                && call.stash.stream != nullptr
                && ctx.stash_stream_block == spec.block) {
                if (site->perm == nullptr || site->perm_count != spec.c_out
                    || site->ex.rows != spec.c_out
                    || site->ex.cols != spec.c_in
                    || site->ex.block < 1
                    || site->ex.block > site->ex.cols
                    || site->ex.cols % site->ex.block != 0
                    || site->ex.q == nullptr || site->ex.scale == nullptr
                    || site->ex.constant == nullptr
                    || site->ex.aq == nullptr) {
                    return Status::kInvalidValues;
                }
                if (site->ex.half) {
                    // The pinned builds the ViT sites non-half.
                    return Status::kInvalidValues;
                }
                // The permutation is validated before any allocation (the
                // range check plus the contraction's sum/xor completeness
                // heuristic — a corrupt record may mis-evaluate but cannot
                // write out of bounds; out_tokens is zeroed so a duplicate
                // cannot leak indeterminate arena values).
                {
                    long long sum = 0;
                    long long xors = 0;
                    for (int i = 0; i < spec.c_out; ++i) {
                        const int p = site->perm[i];
                        if (p < 0 || p >= spec.c_out) {
                            return Status::kInvalidValues;
                        }
                        sum += p;
                        xors ^= p;
                    }
                    long long want_sum = 0;
                    long long want_xor = 0;
                    for (int i = 0; i < spec.c_out; ++i) {
                        want_sum += i;
                        want_xor ^= i;
                    }
                    if (sum != want_sum || xors != want_xor) {
                        return Status::kInvalidValues;
                    }
                }
                const std::size_t in_count = tokens
                    * static_cast<std::size_t>(spec.c_in);
                const std::size_t out_count = tokens
                    * static_cast<std::size_t>(spec.c_out);
                status = row_begin(ctx, spec, out_count, 0,
                                   in_count + out_count + out_count, blocks);
                if (status != Status::kOk) {
                    return status;
                }
                float* permuted = blocks.work;
                float* out_tokens = permuted + out_count;
                float* codes = out_tokens + out_count;
                for (std::size_t i = 0; i < out_count; ++i) {
                    out_tokens[i] = 0.0f;
                }
                status = int4_quantise(site->ex, call.stash.stream,
                                       static_cast<int>(tokens), codes);
                if (status == Status::kOk) {
                    status = int4_accumulate(site->ex, codes,
                                             static_cast<int>(tokens),
                                             permuted);
                }
                if (status == Status::kOk) {
                    for (std::size_t t = 0; t < tokens; ++t) {
                        for (int n = 0; n < spec.c_out; ++n) {
                            // `out[:, perm] = permuted` (the pinned).
                            const float hidden = ffn_activation(
                                permuted[t * static_cast<std::size_t>(
                                                spec.c_out)
                                        + static_cast<std::size_t>(n)]);
                            const int target = site->perm[n];
                            if (out.stash.hidden != nullptr) {
                                out.stash.hidden[t * static_cast<std::size_t>(
                                                       spec.c_out)
                                                 + static_cast<std::size_t>(
                                                       target)] = hidden;
                            }
                            out_tokens[t * static_cast<std::size_t>(
                                               spec.c_out)
                                       + static_cast<std::size_t>(target)]
                                = e4m3(vit_f16(hidden));
                        }
                    }
                }
                if (status == Status::kOk) {
                    status = eval_tokens_to_planar(out_tokens, out_count,
                                                   spec.c_out, x.rows, x.cols,
                                                   blocks.main, out_count);
                }
                break;
            }
            const std::uint8_t* codes = nullptr;
            status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "weight",
                                spec.c_out, spec.c_in, codes);
            if (status != Status::kOk) {
                return status;
            }
            const std::size_t in_count = tokens
                * static_cast<std::size_t>(spec.c_in);
            const std::size_t out_count = tokens
                * static_cast<std::size_t>(spec.c_out);
            const bool gpu_projection =
                ctx.projection != nullptr && ctx.projection->matmul != nullptr;
            status = row_begin(ctx, spec, out_count, 0,
                               in_count + out_count
                                   + (gpu_projection ? out_count : 0),
                               blocks);
            if (status != Status::kOk) {
                return status;
            }
            float* in_tokens = blocks.work;
            float* out_tokens = in_tokens + in_count;
            status = eval_planar_to_tokens(x, in_tokens, in_count);
            if (status == Status::kOk && gpu_projection) {
                // G2: the raw projection runs through the executor; the
                // post chain (vit_ffn_expand_post) is the CPU reference's.
                float* products = out_tokens + out_count;
                status = ctx.projection->matmul(
                    ctx.projection->ctx, in_tokens, static_cast<int>(tokens),
                    spec.c_in, E4m3Matrix{codes, spec.c_out, spec.c_in},
                    products);
                if (status == Status::kOk) {
                    for (std::size_t i = 0; i < out_count; ++i) {
                        out_tokens[i] = vit_ffn_expand_post(
                            products[i],
                            out.stash.hidden != nullptr
                                ? &out.stash.hidden[i]
                                : nullptr);
                    }
                }
            } else if (status == Status::kOk) {
                status = vit_ffn_expand(in_tokens, static_cast<int>(tokens),
                                        spec.c_in, spec.c_out,
                                        E4m3Matrix{codes, spec.c_out, spec.c_in},
                                        out_tokens, out.stash.hidden);
            }
            if (status == Status::kOk) {
                status = eval_tokens_to_planar(out_tokens, out_count,
                                               spec.c_out, x.rows, x.cols,
                                               blocks.main, out_count);
            }
            break;
        }
        case LayerKind::kVitFfnContract:
        case LayerKind::kVitProjection: {
            if (call.aux == nullptr || call.aux->data == nullptr) {
                return Status::kNullPointer;
            }
            const FeatureBatchView& skip = *call.aux;
            if (x.channels != spec.c_in || skip.channels != spec.c_out
                || skip.rows != x.rows || skip.cols != x.cols) {
                return Status::kInvalidDimensions;
            }
            // The int4 site's record presence selects the mixed path (the
            // pinned's table-presence gating); the mixed site carries its own
            // matrix and never fetches the e4m3 "weight". kVitProjection
            // always runs e4m3.
            const Int4Site* site = nullptr;
            if (spec.kind == LayerKind::kVitFfnContract
                && ctx.weights->int4 != nullptr) {
                site = ctx.weights->int4(ctx.weights->context, spec.block,
                                         spec.layer, "vit_ffn");
            }
            if (site != nullptr) {
                status = eval_vit_ffn_mixed(ctx, spec, x, skip, *site, tokens,
                                            blocks, call.stash.hidden,
                                            &out.stash);
                break;
            }
            const std::uint8_t* codes = nullptr;
            status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "weight",
                                spec.c_out, spec.c_in, codes);
            if (status != Status::kOk) {
                return status;
            }
            const float* gain = nullptr;
            status = fetch_f16(ctx.weights, spec.block, spec.layer,
                               "skip_weight", spec.c_out, gain);
            if (status != Status::kOk) {
                return status;
            }
            const std::size_t in_count = tokens
                * static_cast<std::size_t>(spec.c_in);
            const std::size_t out_count = tokens
                * static_cast<std::size_t>(spec.c_out);
            const bool gpu_projection =
                ctx.projection != nullptr && ctx.projection->matmul != nullptr;
            status = row_begin(ctx, spec, out_count, 0,
                               in_count + out_count + out_count
                                   + (gpu_projection ? out_count : 0),
                               blocks);
            if (status != Status::kOk) {
                return status;
            }
            float* in_tokens = blocks.work;
            float* skip_tokens = in_tokens + in_count;
            float* out_tokens = skip_tokens + out_count;
            float* stash_dest = contract_stash_dest(
                ctx.weights, spec.block, spec.layer, out.stash);
            status = eval_planar_to_tokens(x, in_tokens, in_count);
            if (status == Status::kOk) {
                status = eval_planar_to_tokens(skip, skip_tokens, out_count);
            }
            if (status == Status::kOk && gpu_projection) {
                float* products = out_tokens + out_count;
                status = ctx.projection->matmul(
                    ctx.projection->ctx, in_tokens, static_cast<int>(tokens),
                    spec.c_in, E4m3Matrix{codes, spec.c_out, spec.c_in},
                    products);
                if (status == Status::kOk) {
                    for (std::size_t t = 0; t < tokens; ++t) {
                        for (int n = 0; n < spec.c_out; ++n) {
                            const std::size_t i =
                                t * static_cast<std::size_t>(spec.c_out)
                                + static_cast<std::size_t>(n);
                            out_tokens[i] = vit_residual_post(
                                products[i], skip_tokens[i], gain[n],
                                stash_dest != nullptr ? &stash_dest[i]
                                                      : nullptr);
                        }
                    }
                }
            } else if (status == Status::kOk) {
                status = vit_residual_linear(in_tokens, skip_tokens,
                                             static_cast<int>(tokens), spec.c_in,
                                             spec.c_out,
                                             E4m3Matrix{codes, spec.c_out,
                                                        spec.c_in},
                                             gain, out_tokens, stash_dest);
            }
            if (status == Status::kOk && stash_dest != nullptr
                && stash_dest == out.stash.stream) {
                // The ("stream", block + 1) key comes into existence HERE
                // (the pinned's dict entry appears only when this row writes
                // it) — stamped only after the write succeeded, so a failed
                // row never leaves the stamp claiming an unwritten buffer.
                ctx.stash_stream_block = spec.block + 1;
            }
            if (status == Status::kOk) {
                status = eval_tokens_to_planar(out_tokens, out_count,
                                               spec.c_out, x.rows, x.cols,
                                               blocks.main, out_count);
            }
            break;
        }
        case LayerKind::kVitQKV: {
            const int channels = spec.channels;
            if (x.channels != channels || channels % kVitHeadDim != 0) {
                return Status::kInvalidDimensions;
            }
            // The int4 site's record presence selects the mixed path (the
            // pinned's table-presence gating); it never fetches "q"/"k"/"v".
            const Int4Site* qkv_site = nullptr;
            if (ctx.weights->int4 != nullptr) {
                qkv_site = ctx.weights->int4(ctx.weights->context, spec.block,
                                             spec.layer, "vit_qkv");
            }
            if (qkv_site != nullptr) {
                status = eval_vit_qkv_int4(ctx, spec, x, *qkv_site, channels,
                                           tokens, blocks, call.stash.c);
                break;
            }
            const std::uint8_t* q_codes = nullptr;
            const std::uint8_t* k_codes = nullptr;
            const std::uint8_t* v_codes = nullptr;
            const float* temperature = nullptr;
            status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "q",
                                channels, channels, q_codes);
            if (status == Status::kOk) {
                status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "k",
                                    channels, channels, k_codes);
            }
            if (status == Status::kOk) {
                status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "v",
                                    channels, channels, v_codes);
            }
            if (status == Status::kOk) {
                status = fetch_f32(ctx.weights, spec.block, spec.layer,
                                   "header", channels / kVitHeadDim,
                                   temperature);
            }
            if (status != Status::kOk) {
                return status;
            }
            const std::size_t in_count = tokens
                * static_cast<std::size_t>(channels);
            const std::size_t out_count = tokens * 3u
                * static_cast<std::size_t>(channels);
            const bool gpu_projection =
                ctx.projection != nullptr && ctx.projection->matmul != nullptr;
            status = row_begin(ctx, spec, out_count, 0,
                               in_count + out_count
                                   + (gpu_projection ? out_count : 0),
                               blocks);
            if (status != Status::kOk) {
                return status;
            }
            float* in_tokens = blocks.work;
            float* out_tokens = in_tokens + in_count;
            status = eval_planar_to_tokens(x, in_tokens, in_count);
            if (status == Status::kOk && gpu_projection) {
                // G2: the three raw projections run through the executor
                // into contiguous [tokens, channels] buffers; the assembly
                // into the reference's interleaved [t*3C+r] layout then
                // vit_qkv_post are the CPU reference's operations.
                float* raw_q = out_tokens + out_count;
                float* raw_k = raw_q + tokens
                    * static_cast<std::size_t>(channels);
                float* raw_v = raw_k + tokens
                    * static_cast<std::size_t>(channels);
                status = ctx.projection->matmul(
                    ctx.projection->ctx, in_tokens, static_cast<int>(tokens),
                    channels, E4m3Matrix{q_codes, channels, channels}, raw_q);
                if (status == Status::kOk) {
                    status = ctx.projection->matmul(
                        ctx.projection->ctx, in_tokens,
                        static_cast<int>(tokens), channels,
                        E4m3Matrix{k_codes, channels, channels}, raw_k);
                }
                if (status == Status::kOk) {
                    status = ctx.projection->matmul(
                        ctx.projection->ctx, in_tokens,
                        static_cast<int>(tokens), channels,
                        E4m3Matrix{v_codes, channels, channels}, raw_v);
                }
                if (status == Status::kOk) {
                    for (std::size_t t = 0; t < tokens; ++t) {
                        for (int r = 0; r < 3 * channels; ++r) {
                            const float* src = r < channels ? raw_q
                                            : r < 2 * channels ? raw_k
                                                               : raw_v;
                            out_tokens[t * 3u
                                           * static_cast<std::size_t>(channels)
                                       + static_cast<std::size_t>(r)]
                                = src[t * static_cast<std::size_t>(channels)
                                      + static_cast<std::size_t>(
                                            r % channels)];
                        }
                    }
                    status = vit_qkv_post(
                        out_tokens, static_cast<int>(tokens), channels,
                        temperature, out_tokens);
                }
            } else if (status == Status::kOk) {
                status = vit_qkv(in_tokens, static_cast<int>(tokens), channels,
                                 E4m3Matrix{q_codes, channels, channels},
                                 E4m3Matrix{k_codes, channels, channels},
                                 E4m3Matrix{v_codes, channels, channels},
                                 temperature, out_tokens);
            }
            if (status == Status::kOk) {
                status = eval_tokens_to_planar(out_tokens, out_count,
                                               3 * channels, x.rows, x.cols,
                                               blocks.main, out_count);
            }
            break;
        }
        case LayerKind::kVitAttention: {
            const int channels = spec.channels;
            if (x.channels != 3 * channels || channels % kVitHeadDim != 0) {
                return Status::kInvalidDimensions;
            }
            const std::size_t in_count = tokens * 3u
                * static_cast<std::size_t>(channels);
            const std::size_t out_count = tokens
                * static_cast<std::size_t>(channels);
            status = row_begin(ctx, spec, out_count, 0, in_count + out_count,
                               blocks);
            if (status != Status::kOk) {
                return status;
            }
            float* in_tokens = blocks.work;
            float* out_tokens = in_tokens + in_count;
            status = eval_planar_to_tokens(x, in_tokens, in_count);
            if (status == Status::kOk) {
                status = vit_attention(in_tokens, static_cast<int>(tokens),
                                       channels, out_tokens);
            }
            if (status == Status::kOk) {
                status = eval_tokens_to_planar(out_tokens, out_count, channels,
                                               x.rows, x.cols, blocks.main,
                                               out_count);
            }
            break;
        }
        default:
            return Status::kInvalidValues;
    }
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, spec.kind == LayerKind::kVitQKV
                                    ? 3 * spec.channels
                                    : (spec.kind == LayerKind::kVitAttention
                                           ? spec.channels : spec.c_out),
                                spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

// ---- DecInputUpsample ----------------------------------------------------------------------------

inline Status eval_dec_input_upsample(EvalContext& ctx, const LayerSpec& spec,
                                      const LayerCall& call, LayerValue& out) {
    if (call.x == nullptr || call.x->data == nullptr
        || call.aux == nullptr || call.aux->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    const FeatureBatchView& skip = *call.aux;
    if (x.channels != spec.c_in || skip.channels != spec.c_out
        || skip.rows != spec.out_w || skip.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    const std::uint8_t* codes = nullptr;
    const float* gain = nullptr;
    Status status = fetch_e4m3(ctx.weights, spec.block, spec.layer, "weight",
                               spec.c_out, spec.c_in, codes);
    if (status == Status::kOk) {
        status = fetch_f16(ctx.weights, spec.block, spec.layer, "skip_weight",
                           spec.c_out, gain);
    }
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t out_count = static_cast<std::size_t>(spec.c_out)
        * static_cast<std::size_t>(skip.rows) * static_cast<std::size_t>(skip.cols);
    const std::size_t projected = static_cast<std::size_t>(spec.c_out)
        * static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols);
    RowBlocks blocks;
    status = row_begin(ctx, spec, out_count, 0, projected, blocks);
    if (status != Status::kOk) {
        return status;
    }
    const DecInputWeight weight{E4m3Matrix{codes, spec.c_out, spec.c_in}, gain,
                                static_cast<std::size_t>(spec.c_out)};
    status = dec_input_upsample(x, skip, weight, blocks.main, out_count,
                                blocks.work, projected);
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    out.main = FeatureBatchView{blocks.main, spec.c_out, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

// ---- PostBlock: blend -> body -> residual projection -> compose -----------------------------------

inline Status eval_post_block(EvalContext& ctx, const LayerSpec& spec,
                              const LayerCall& call, LayerValue& out) {
    const int channels = kLiftedChannels;
    const int heads = 1;
    if (call.x == nullptr || call.x->data == nullptr
        || call.aux == nullptr || call.aux->data == nullptr
        || call.image == nullptr || call.image->data == nullptr) {
        return Status::kNullPointer;
    }
    const FeatureBatchView& x = *call.x;
    const FeatureBatchView& skip = *call.aux;
    const FeatureBatchView& image = *call.image;
    if (x.channels != channels || skip.channels != channels
        || image.channels != 3 || skip.rows != spec.out_w
        || skip.cols != spec.out_h || image.rows != spec.out_w
        || image.cols != spec.out_h) {
        return Status::kInvalidDimensions;
    }
    SwinFetch fetch;
    Status status = swin_fetch(ctx.weights, spec.block, spec.layer, channels,
                               heads, fetch);
    if (status != Status::kOk) {
        return status;
    }
    const float* main_gain = nullptr;
    const float* skip_gain = nullptr;
    const float* project = nullptr;
    status = fetch_f16(ctx.weights, spec.block, spec.layer, "main_gain",
                       channels, main_gain);
    if (status == Status::kOk) {
        status = fetch_f16(ctx.weights, spec.block, spec.layer, "skip_gain",
                           channels, skip_gain);
    }
    if (status == Status::kOk) {
        status = fetch_f16(ctx.weights, spec.block, spec.layer, "out_project",
                           16 * channels, project);
    }
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t pixels = static_cast<std::size_t>(spec.out_w)
        * static_cast<std::size_t>(spec.out_h);
    const std::size_t map = static_cast<std::size_t>(channels) * pixels;
    const std::size_t frame_count = 3u * pixels;
    int shift_rows = 0;
    int shift_cols = 0;
    swin_shifts(spec, shift_rows, shift_cols);
    FeatureBatchView blend_view;
    blend_view.data = nullptr;
    blend_view.channels = channels;
    blend_view.rows = spec.out_w;
    blend_view.cols = spec.out_h;
    blend_view.plane_stride = static_cast<std::ptrdiff_t>(pixels);
    blend_view.row_stride = static_cast<std::ptrdiff_t>(spec.out_h);
    const SwinBlockParams shape = swin_shape_params(channels, heads);
    const std::size_t block_needed = swin_block_scratch_needed(
        blend_view, true, true, shift_rows, shift_cols, shape);
    const bool gpu_projection =
        ctx.projection != nullptr && ctx.projection->matmul_f32 != nullptr
        && ctx.swin_gpu;
    // The G8 batched seam's site scratch (swin_site_count).
    const std::size_t site_scratch =
        gpu_projection ? swin_site_count(channels, heads) : 0;
    const std::size_t work_count = map + map + frame_count + frame_count
        + frame_count + swin_decoded_floats(channels, heads, fetch)
        + block_needed + site_scratch;
    RowBlocks blocks;
    status = row_begin(ctx, spec, frame_count, 0, work_count, blocks);
    if (status != Status::kOk) {
        return status;
    }
    float* blend_map = blocks.work;
    float* wide_map = blend_map + map;
    float* residual_map = wide_map + map;
    float* frame = residual_map + frame_count;
    float* staged = frame + frame_count;
    float* decoded = staged + frame_count;
    float* block_scratch = decoded + swin_decoded_floats(channels, heads, fetch);
    SwinProjectionSite swsite;
    if (gpu_projection) {
        swsite.ex = ctx.projection;
        swin_site_carve(block_scratch + block_needed, channels, heads, swsite);
    }
    SwinSiteOverlay overlay;
    build_swin_overlay(ctx.weights, spec.block, spec.layer, channels, 1,
                       overlay);
    MatrixView mid_storage;
    SwinBlockParams params;
    status = swin_decode(fetch, channels, heads, decoded,
                         swin_decoded_floats(channels, heads, fetch),
                         mid_storage, params);
    params.site = &overlay;
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    FeatureBatchView blend_batch;
    blend_batch.data = blend_map;
    blend_batch.channels = channels;
    blend_batch.rows = spec.out_w;
    blend_batch.cols = spec.out_h;
    blend_batch.plane_stride = static_cast<std::ptrdiff_t>(pixels);
    blend_batch.row_stride = static_cast<std::ptrdiff_t>(spec.out_h);
    status = blend(x, skip, main_gain, skip_gain,
                   static_cast<std::size_t>(channels), blend_map, map);
    if (status == Status::kOk) {
        status = swin_core(blend_batch, shift_rows, shift_cols, params,
                           wide_map, map, nullptr, 0, block_scratch,
                           block_needed, gpu_projection ? &swsite : nullptr);
    }
    if (status == Status::kOk) {
        FeatureBatchView wide;
        wide.data = wide_map;
        wide.channels = channels;
        wide.rows = spec.out_w;
        wide.cols = spec.out_h;
        wide.plane_stride = static_cast<std::ptrdiff_t>(pixels);
        wide.row_stride = static_cast<std::ptrdiff_t>(spec.out_h);
        status = project_residual(wide, project,
                                  static_cast<std::size_t>(kProjectionWeights),
                                  residual_map, frame_count);
    }
    if (status == Status::kOk) {
        FeatureBatchView residual;
        residual.data = residual_map;
        residual.channels = kOutputChannels;
        residual.rows = spec.out_w;
        residual.cols = spec.out_h;
        residual.plane_stride = static_cast<std::ptrdiff_t>(pixels);
        residual.row_stride = static_cast<std::ptrdiff_t>(spec.out_h);
        status = frame_from_planar(*call.image, frame, frame_count);
        if (status == Status::kOk) {
            status = compose(FrameView{frame, spec.out_w, spec.out_h,
                                       static_cast<std::ptrdiff_t>(3) * spec.out_h},
                             residual, ctx.intensity, staged, frame_count);
        }
    }
    if (status == Status::kOk) {
        status = frame_to_planar(staged, frame_count, spec.out_w, spec.out_h,
                                 blocks.main, frame_count);
    }
    if (status != Status::kOk) {
        row_abort(ctx, spec);
        return status;
    }
    // The output frame is a planar [3, rows, cols] map like every other main
    // (the host-side frame conversion stays the caller's boundary).
    out.main = FeatureBatchView{blocks.main, kOutputChannels, spec.out_w,
                                spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_w)
                                    * spec.out_h,
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    out.has_aside = false;
    row_end(ctx);
    return Status::kOk;
}

}  // namespace eval_detail

// ---- the layer entry point -----------------------------------------------------------------------

// One plan row over the family kernels (the LayerFn of network_run). The
// records are fetched and every extent/shape checked before the first write;
// on success `out` carries the row's tensors (arena blocks), on any error
// `out` is untouched and the row's blocks are released. Kernel Status codes
// propagate unchanged.
inline Status eval_layer(void* context, const LayerSpec& spec,
                         const LayerCall& call, LayerValue& out) {
    EvalContext* ctx = static_cast<EvalContext*>(context);
    if (ctx == nullptr || ctx->weights == nullptr || ctx->scratch == nullptr) {
        return Status::kNullPointer;
    }
    if (spec.out_w < 1 || spec.out_h < 1) {
        return Status::kInvalidDimensions;
    }
    switch (spec.kind) {
        case LayerKind::kPreBlock:
            return eval_detail::eval_pre_block(*ctx, spec, call, out);
        case LayerKind::kSwin1H:
        case LayerKind::kSwin2H:
        case LayerKind::kSwin4H:
        case LayerKind::kSwin8H:
            return eval_detail::eval_swin_layer(*ctx, spec, call, out);
        case LayerKind::kSplitFfwd:
            return eval_detail::eval_split_ffwd(*ctx, spec, call, out);
        case LayerKind::kSplitFfwdProj:
        case LayerKind::kSplitProj:
            return eval_detail::eval_split_projection(*ctx, spec, call, out);
        case LayerKind::kSplitQKVAttn:
            return eval_detail::eval_split_qkv_attn(*ctx, spec, call, out);
        case LayerKind::kSplitProjPool:
            return eval_detail::eval_split_proj_pool(*ctx, spec, call, out);
        case LayerKind::kSplitFinalHead:
            return eval_detail::eval_split_final_head(*ctx, spec, call, out);
        case LayerKind::kVitFfnExpand:
        case LayerKind::kVitFfnContract:
        case LayerKind::kVitQKV:
        case LayerKind::kVitAttention:
        case LayerKind::kVitProjection:
            return eval_detail::eval_vit(*ctx, spec, call, out);
        case LayerKind::kDecInputUpsample:
            return eval_detail::eval_dec_input_upsample(*ctx, spec, call, out);
        case LayerKind::kPostBlock:
            return eval_detail::eval_post_block(*ctx, spec, call, out);
    }
    return Status::kInvalidValues;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_EVAL_HPP
