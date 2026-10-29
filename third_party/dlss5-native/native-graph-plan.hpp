// Native network graph plan: the typed construction of the enhancement
// network's layer plan — the layer table (one entry per layer family member),
// the extent walk (how every layer's grid and output extent follow from the
// padded frame), the frame padding/working extent and the skip wiring (which
// layer's output each layer reads as its skip input). Two networks are
// described: "ht" (152 layers, 71 blocks) and "small" (76 layers, 31 blocks;
// the pinned reference docstring's "77" counts weight-file records, not
// layers). CPU-only, typed, allocation-free; no GPU, headset, proprietary
// payload, Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/plan.py `LayerSpec`/`_rows`/`_walk`/`padding`/`working_extent`/
// `build_plan` and dlssnr/model.py `NRNet._skip_sources`). The full MIT
// licence notice is retained in vr/NATIVE-GRAPH-PLAN.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented construction
// semantics. It is NOT vendor code, and passing synthetic tests establishes
// synthetic correctness only — never correspondence to NVIDIA's
// implementation. No proprietary weights, captures or binaries are used or
// implied.

#ifndef DLSS5_NR_NATIVE_GRAPH_PLAN_HPP
#define DLSS5_NR_NATIVE_GRAPH_PLAN_HPP

#include <cstddef>
#include <cstring>

#include "native-input-lift.hpp"  // Status, shared validation vocabulary

namespace dlss5_nr {

// ---- plan vocabulary -----------------------------------------------------------------------------
// Frame sizes: 0 < width,height <= 16384 (the pinned `padding` bounds).
inline constexpr int kPlanFrameDimMax = 16384;
inline constexpr int kPlanFloorExtent = 320;  // every padded axis is at least 320

// The plan is frame-size independent in structure: "ht" always has 152 layers
// in 71 blocks and "small" always has 76 layers in 31 blocks; only the
// extents (w/h/tokens/out_*/grid_*) depend on the frame size.
inline constexpr int kPlanMaxLayers = 152;
inline constexpr int kPlanMaxBlocks = 71;

enum class NetworkKind : int {
    kHt = 0,    // the network of 310.8.0 and later (152 layers, 71 blocks)
    kSmall = 1, // the lighter network 310.9.16.0 adds (76 layers, 31 blocks)
};

// The layer families, in the pinned `KINDS` spelling (`layer_kind_name`).
enum class LayerKind : int {
    kPreBlock = 0,
    kSwin1H,
    kSwin2H,
    kSwin4H,
    kSwin8H,
    kSplitFfwd,
    kSplitFfwdProj,
    kSplitQKVAttn,
    kSplitProj,
    kSplitProjPool,
    kSplitFinalHead,
    kVitFfnExpand,
    kVitFfnContract,
    kVitQKV,
    kVitAttention,
    kVitProjection,
    kDecInputUpsample,
    kPostBlock,
};

// The name-tag variant (the pinned `_variant`: the first `_<tag>_` substring of
// the layer name, in this order; `kNone` when no tag matches).
enum class LayerVariant : int {
    kNone = 0,
    kInpview,
    kDs,
    kUpsample,
    kOutview,
    kProjPool,
    kFinalHead,
};

// How a layer changes the extent (the pinned `_Row.role`, walk semantics in
// `build_plan`): plain layers keep the extent; pre/down/final-head halve both
// axes (rounded up to a multiple of 4); a Swin upsample takes the skip's
// extent before its downsample, the decoder entry the extent after the pool
// layer it names; the post block returns to the padded frame extent.
enum class LayerRole : int {
    kPlain = 0,
    kPre,
    kPost,
    kDown,
    kSwinUp,
    kDecoderUp,
    kFinalHead,
};

// One layer of the plan (the pinned `LayerSpec`, field for field). Axis
// convention (historical names kept for traceability): `w` is the number of
// image rows and `h` the number of image columns of this layer's grid; feature
// maps are `[1, C, w, h] = [1, C, rows, columns]`.
struct LayerSpec {
    int index = 0;              // position in the plan
    int block = 0;              // block number (0..70 for "ht", 0..30 for "small")
    int layer = 0;              // layer number inside the block
    LayerKind kind = LayerKind::kPreBlock;
    LayerVariant variant = LayerVariant::kNone;
    int w = 0;                  // rows of this layer's grid (padded)
    int h = 0;                  // columns of this layer's grid (padded)
    int tokens = 0;             // tokens the layer processes (padded)
    int channels = 0;           // channel count of the layer's body
    int heads = 0;              // attention heads
    int c_in = 0;               // input channels
    int c_out = 0;              // output channels
    bool shifted = false;       // the window grid of this layer is shifted
    int shift_x = 0;            // window shift, -1/0/1 steps along x
    int shift_y = 0;            // window shift, -1/0/1 steps along y
    int input_block = 0;        // block whose output feeds this layer
    int skip_block = 0;         // block whose output is the skip input (-1: none)
    const char* name = nullptr; // the full layer name (static storage)
    int out_w = 0;              // rows of the tensor this layer produces
    int out_h = 0;              // columns of the tensor this layer produces
    int grid_x = 0;             // windows across the columns (shift included)
    int grid_y = 0;             // windows down the rows (shift included)
};

// The extent the frame is padded to before entering the network, and the
// padding added (the pinned `Padding`; width/height in frame pixels).
struct PlanPadding {
    int width = 0;
    int height = 0;
    int pad_x = 0;
    int pad_y = 0;
};

// ---- names --------------------------------------------------------------------------------------
inline const char* layer_kind_name(LayerKind kind) {
    switch (kind) {
        case LayerKind::kPreBlock: return "PreBlock";
        case LayerKind::kSwin1H: return "Swin1H";
        case LayerKind::kSwin2H: return "Swin2H";
        case LayerKind::kSwin4H: return "Swin4H";
        case LayerKind::kSwin8H: return "Swin8H";
        case LayerKind::kSplitFfwd: return "SplitFfwd";
        case LayerKind::kSplitFfwdProj: return "SplitFfwdProj";
        case LayerKind::kSplitQKVAttn: return "SplitQKVAttn";
        case LayerKind::kSplitProj: return "SplitProj";
        case LayerKind::kSplitProjPool: return "SplitProjPool";
        case LayerKind::kSplitFinalHead: return "SplitFinalHead";
        case LayerKind::kVitFfnExpand: return "VitFfnExpand";
        case LayerKind::kVitFfnContract: return "VitFfnContract";
        case LayerKind::kVitQKV: return "VitQKV";
        case LayerKind::kVitAttention: return "VitAttention";
        case LayerKind::kVitProjection: return "VitProjection";
        case LayerKind::kDecInputUpsample: return "DecInputUpsample";
        case LayerKind::kPostBlock: return "PostBlock";
    }
    return "?";
}

inline const char* layer_variant_name(LayerVariant variant) {
    switch (variant) {
        case LayerVariant::kNone: return "";
        case LayerVariant::kInpview: return "inpview";
        case LayerVariant::kDs: return "ds";
        case LayerVariant::kUpsample: return "upsample";
        case LayerVariant::kOutview: return "outview";
        case LayerVariant::kProjPool: return "proj_pool";
        case LayerVariant::kFinalHead: return "final_head";
    }
    return "?";
}

// ---- structural counts --------------------------------------------------------------------------
inline int plan_layer_count(NetworkKind network) {
    return network == NetworkKind::kHt ? 152 : 76;
}

inline int plan_block_count(NetworkKind network) {
    return network == NetworkKind::kHt ? 71 : 31;
}

// ---- internal construction -----------------------------------------------------------------------
// The pieces below implement the pinned plan semantics (`_rows`, `_walk`,
// `_variant`, `padding`, `build_plan`) exactly; they are exposed because the
// tests pin them directly, and the assembler needs nothing beyond them.
namespace plan_detail {

// Extents as the pinned `_Shape`: w = image rows, h = image columns.
struct Shape {
    int w = 0;
    int h = 0;
};

// Round n (>= 0) up to a multiple of k: the pinned `up(n, k)`.
inline int up(int n, int k) {
    return (n + k - 1) / k * k;
}

// The window-shift cycle `_SHIFT_CYCLE = ((0,0), (-1,-1), (-1,0), (0,-1))`.
inline void shift_cycle(int position, int& shift_x, int& shift_y) {
    static constexpr int kCycle[4][2] = {{0, 0}, {-1, -1}, {-1, 0}, {0, -1}};
    const int* entry = kCycle[position & 3];
    shift_x = entry[0];
    shift_y = entry[1];
}

// One row of the pinned layer table (`_Row`): the frame-size independent part
// of a layer's identity. `skip_block`/`skip_layer` name the row whose extent
// this row's output takes over (`_Row.skip`); -1/-1 when there is none.
struct PlanRow {
    int block = 0;
    int layer = 0;
    LayerKind kind = LayerKind::kPreBlock;
    const char* name = nullptr;
    int channels = 0;
    int heads = 0;
    int c_in = 0;
    int c_out = 0;
    int shift_x = 0;
    int shift_y = 0;
    LayerRole role = LayerRole::kPlain;
    int skip_block = -1;
    int skip_layer = -1;
};

// How deep one network is (the pinned `_Shape_`).
struct NetworkShape {
    int encoder[4];       // layers per encoder level (C = 32, 64, 128, 256)
    int split;            // blocks in each of the two split sections
    int decoder_start[4]; // shift-cycle position each decoder level starts at
    bool wide_alignment;  // an axis aligns to twice as much where the pre
                          // block's output is 4 mod 8 (the "small" network)
};

inline NetworkShape network_shape(NetworkKind network) {
    if (network == NetworkKind::kHt) {
        return NetworkShape{{4, 4, 6, 8}, 8, {0, 2, 0, 0}, false};
    }
    return NetworkShape{{2, 2, 2, 2}, 2, {2, 2, 2, 2}, true};
}

// The `_variant` name scan: the first "_<tag>_" substring of the layer name in
// the pinned `_VARIANTS` order, `kNone` when none matches.
inline LayerVariant variant_from_name(const char* name) {
    static constexpr const char* kTags[6] = {
        "_inpview_", "_ds_", "_upsample_", "_outview_", "_proj_pool_", "_final_head_"};
    static constexpr LayerVariant kVariants[6] = {
        LayerVariant::kInpview, LayerVariant::kDs, LayerVariant::kUpsample,
        LayerVariant::kOutview, LayerVariant::kProjPool, LayerVariant::kFinalHead};
    for (int i = 0; i < 6; ++i) {
        if (std::strstr(name, kTags[i]) != nullptr) {
            return kVariants[i];
        }
    }
    return LayerVariant::kNone;
}

// Swin body name for encoder/decoder level 0..3 ((1,32)..(8,256)) and tag
// 0 = plain, 1 = inpview, 2 = ds, 3 = upsample, 4 = outview: the pinned
// `stem = cc_tinlayout_fused_swin_{heads}h_{channels}_{heads}` + `_{tag}fp8`.
inline const char* swin_name(int level, int tag) {
    static constexpr const char* kNames[4][5] = {
        {"cc_tinlayout_fused_swin_1h_32_1_fp8",
         "cc_tinlayout_fused_swin_1h_32_1_inpview_fp8",
         "cc_tinlayout_fused_swin_1h_32_1_ds_fp8",
         "cc_tinlayout_fused_swin_1h_32_1_upsample_fp8",
         "cc_tinlayout_fused_swin_1h_32_1_outview_fp8"},
        {"cc_tinlayout_fused_swin_2h_64_2_fp8",
         "cc_tinlayout_fused_swin_2h_64_2_inpview_fp8",
         "cc_tinlayout_fused_swin_2h_64_2_ds_fp8",
         "cc_tinlayout_fused_swin_2h_64_2_upsample_fp8",
         "cc_tinlayout_fused_swin_2h_64_2_outview_fp8"},
        {"cc_tinlayout_fused_swin_4h_128_4_fp8",
         "cc_tinlayout_fused_swin_4h_128_4_inpview_fp8",
         "cc_tinlayout_fused_swin_4h_128_4_ds_fp8",
         "cc_tinlayout_fused_swin_4h_128_4_upsample_fp8",
         "cc_tinlayout_fused_swin_4h_128_4_outview_fp8"},
        {"cc_tinlayout_fused_swin_8h_256_8_fp8",
         "cc_tinlayout_fused_swin_8h_256_8_inpview_fp8",
         "cc_tinlayout_fused_swin_8h_256_8_ds_fp8",
         "cc_tinlayout_fused_swin_8h_256_8_upsample_fp8",
         "cc_tinlayout_fused_swin_8h_256_8_outview_fp8"},
    };
    return kNames[level][tag];
}

inline LayerKind swin_kind(int level) {
    static constexpr LayerKind kKinds[4] = {
        LayerKind::kSwin1H, LayerKind::kSwin2H, LayerKind::kSwin4H, LayerKind::kSwin8H};
    return kKinds[level];
}

// Build the pinned layer table (`_rows`) for one network into `rows`, which
// must have room for plan_layer_count(network) entries; returns the row count.
inline int build_rows(NetworkKind network, PlanRow* rows) {
    const NetworkShape shape = network_shape(network);
    int count = 0;
    auto push = [&](int block, int layer, LayerKind kind, const char* name, int channels,
                    int heads, int c_in, int c_out, int shift_x, int shift_y, LayerRole role,
                    int skip_block, int skip_layer) {
        PlanRow& row = rows[count++];
        row.block = block;
        row.layer = layer;
        row.kind = kind;
        row.name = name;
        row.channels = channels;
        row.heads = heads;
        row.c_in = c_in;
        row.c_out = c_out;
        row.shift_x = shift_x;
        row.shift_y = shift_y;
        row.role = role;
        row.skip_block = skip_block;
        row.skip_layer = skip_layer;
    };

    // Pre block (block 0).
    push(0, 0, LayerKind::kPreBlock, "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8",
         3, 0, 3, 32, 0, 0, LayerRole::kPre, -1, -1);

    // Encoder: four levels of Swin blocks, each ending in a downsample that
    // doubles the channels (`swin`, `role=DOWN` on the last block).
    static constexpr int kLevelHeads[4] = {1, 2, 4, 8};
    static constexpr int kLevelChannels[4] = {32, 64, 128, 256};
    int block = 1;
    int downsample_blocks[4] = {0, 0, 0, 0};
    for (int level = 0; level < 4; ++level) {
        for (int i = 0; i < shape.encoder[level]; ++i) {
            const bool first = i == 0;
            const bool last = i == shape.encoder[level] - 1;
            const int tag = first ? 1 : (last ? 2 : 0);
            int shift_x = 0;
            int shift_y = 0;
            shift_cycle(i, shift_x, shift_y);
            const int channels = kLevelChannels[level];
            push(block, 0, swin_kind(level), swin_name(level, tag), channels,
                 kLevelHeads[level], channels,
                 last ? 2 * channels : channels, shift_x, shift_y,
                 last ? LayerRole::kDown : LayerRole::kPlain, -1, -1);
            if (last) {
                downsample_blocks[level] = block;
            }
            ++block;
        }
    }

    // Split blocks: feed-forward, its projection, the shifted QKV attention and
    // the block's projection; the last encoder-side block adds the pooled
    // projection and the final head (`split_block`).
    auto split_block = [&](int split_shift_x, int split_shift_y, bool first, bool last,
                           bool pool) {
        static constexpr const char* kFfwd[2] = {
            "cc_split_swin_16h_ffwd_512_fp8", "cc_split_swin_16h_ffwd_inpview_512_fp8"};
        static constexpr const char* kFfwdProj[2] = {
            "cc_split_swin_16h_ffwd_proj_512_fp8",
            "cc_split_swin_16h_ffwd_proj_inpview_512_fp8"};
        const int v = first ? 1 : 0;
        push(block, 0, LayerKind::kSplitFfwd, kFfwd[v], 512, 4, 512, 512, 0, 0,
             LayerRole::kPlain, -1, -1);
        push(block, 1, LayerKind::kSplitFfwdProj, kFfwdProj[v], 512, 4, 512, 512, 0, 0,
             LayerRole::kPlain, -1, -1);
        push(block, 2, LayerKind::kSplitQKVAttn, "cc_split_swin_16h_qkv_512_fp8", 512, 4,
             512, 512, split_shift_x, split_shift_y, LayerRole::kPlain, -1, -1);
        if (pool) {
            push(block, 3, LayerKind::kSplitProjPool, "cc_split_swin_16h_proj_pool_512_fp8",
                 512, 4, 512, 512, 0, 0, LayerRole::kPlain, -1, -1);
            push(block, 4, LayerKind::kSplitFinalHead,
                 "cc_split_swin_16h_final_head_512_fp8", 512, 8, 512, 1024, 0, 0,
                 LayerRole::kFinalHead, -1, -1);
        } else if (last) {
            push(block, 3, LayerKind::kSplitProj, "cc_split_swin_16h_proj_512_outview_fp8",
                 512, 4, 512, 512, 0, 0, LayerRole::kPlain, -1, -1);
        } else {
            push(block, 3, LayerKind::kSplitProj, "cc_split_swin_16h_proj_512_fp8", 512, 8,
                 512, 512, 0, 0, LayerRole::kPlain, -1, -1);
        }
        ++block;
    };

    for (int i = 0; i < shape.split; ++i) {
        int shift_x = 0;
        int shift_y = 0;
        shift_cycle(i, shift_x, shift_y);
        split_block(shift_x, shift_y, i == 0, false, i == shape.split - 1);
    }
    const int pool_block = block - 1;

    // Bottleneck: eight transformer blocks over the flattened tokens.
    static constexpr LayerKind kVitKinds[5] = {
        LayerKind::kVitFfnExpand, LayerKind::kVitFfnContract, LayerKind::kVitQKV,
        LayerKind::kVitAttention, LayerKind::kVitProjection};
    static constexpr const char* kVitNames[5] = {
        "cc_vit_1d_ffn_expand_fp8", "cc_vit_1d_ffn_contract_fp8", "cc_vit_1d_qkv_fp8",
        "cc_vit_1d_attention_fp8", "cc_vit_1d_projection_fp8"};
    static constexpr int kVitIn[5] = {1024, 4096, 1024, 1024, 1024};
    static constexpr int kVitOut[5] = {4096, 1024, 1024, 1024, 1024};
    for (int i = 0; i < 8; ++i) {
        for (int layer = 0; layer < 5; ++layer) {
            push(block, layer, kVitKinds[layer], kVitNames[layer], 1024, 4,
                 kVitIn[layer], kVitOut[layer], 0, 0, LayerRole::kPlain, -1, -1);
        }
        ++block;
    }

    // Decoder: the bottleneck output is upsampled onto the final-head extent
    // (the `DecInputUpsample` reads the pool block's projection), then split
    // blocks continuing the shift cycle, then one Swin level per encoder level,
    // each opened by an upsample that takes the matching encoder skip.
    push(block, 0, LayerKind::kDecInputUpsample, "cc_dec_input_upsample_1024_512_fp8",
         512, 1, 1024, 512, 0, 0, LayerRole::kDecoderUp, pool_block, 3);
    ++block;

    for (int i = 0; i < shape.split; ++i) {
        int shift_x = 0;
        int shift_y = 0;
        shift_cycle(shape.split + i, shift_x, shift_y);
        split_block(shift_x, shift_y, false, i == shape.split - 1, false);
    }

    static constexpr int kDecoderHeads[4] = {8, 4, 2, 1};
    static constexpr int kDecoderChannels[4] = {256, 128, 64, 32};
    for (int level = 0; level < 4; ++level) {
        const int count_at_level = shape.encoder[3 - level];
        const int start = shape.decoder_start[level];
        const int skip_block = downsample_blocks[3 - level];
        for (int i = 0; i < count_at_level; ++i) {
            const bool first = i == 0;
            const bool last = i == count_at_level - 1;
            const int tag = first ? 3 : (last ? 4 : 0);
            int shift_x = 0;
            int shift_y = 0;
            shift_cycle(start + i, shift_x, shift_y);
            const int channels = kDecoderChannels[level];
            // Decoder level 0 is the deepest ((8, 256)), the name table's level 3.
            push(block, 0, swin_kind(3 - level), swin_name(3 - level, tag), channels,
                 kDecoderHeads[level], first ? 2 * channels : channels, channels,
                 shift_x, shift_y,
                 first ? LayerRole::kSwinUp : LayerRole::kPlain,
                 first ? skip_block : -1, first ? 0 : -1);
            ++block;
        }
    }

    // Post block: blends back onto the padded frame extent.
    push(block, 0, LayerKind::kPostBlock, "cc_tinlayout_fused_post_block_swin_1h_32_fp8",
         3, 0, 32, 3, -1, -1, LayerRole::kPost, -1, -1);
    return count;
}

// The row whose (block, layer) position is named, or -1.
inline int row_index(const PlanRow* rows, int count, int block, int layer) {
    for (int i = 0; i < count; ++i) {
        if (rows[i].block == block && rows[i].layer == layer) {
            return i;
        }
    }
    return -1;
}

// The extent walk (`_walk`) over an already padded extent (`frame_w` x
// `frame_h` in frame pixels): `shapes[i]` is (extent in, extent out) for row i,
// and the halving counters count the blocks whose extent actually shrank on
// each axis. `shapes` receives `count` entries. Rows must come from
// `build_rows` (or satisfy its skip preconditions): a skip that does not
// resolve is a caller error and leaves that row's extent unchanged.
inline void walk(int frame_w, int frame_h, const PlanRow* rows, int count,
                 Shape shapes[][2], int& halvings_x, int& halvings_y) {
    Shape current{frame_h, frame_w};
    Shape previous{frame_h, frame_w};
    halvings_x = 0;
    halvings_y = 0;
    int last_block = -1;
    for (int i = 0; i < count; ++i) {
        const PlanRow& row = rows[i];
        if (row.block != last_block) {
            if (current.h < previous.h) {
                ++halvings_x;
            }
            if (current.w < previous.w) {
                ++halvings_y;
            }
            previous = current;
            last_block = row.block;
        }
        Shape out = current;
        switch (row.role) {
            case LayerRole::kPre:
            case LayerRole::kDown:
            case LayerRole::kFinalHead:
                out = Shape{up((current.w + 1) / 2, 4), up((current.h + 1) / 2, 4)};
                break;
            case LayerRole::kSwinUp:
            case LayerRole::kDecoderUp: {
                const int skip = row_index(rows, count, row.skip_block, row.skip_layer);
                if (skip >= 0) {
                    const Shape before = shapes[skip][0];
                    const Shape after = shapes[skip][1];
                    out = row.role == LayerRole::kSwinUp ? before : after;
                    if (row.role == LayerRole::kSwinUp && row.c_out == 32) {
                        out = Shape{up(out.w, 8), up(out.h, 8)};
                    }
                }
                break;
            }
            case LayerRole::kPost:
                out = Shape{frame_h, frame_w};
                break;
            case LayerRole::kPlain:
                break;
        }
        shapes[i][0] = current;
        shapes[i][1] = out;
        current = out;
    }
}

}  // namespace plan_detail

// ---- padding and plan ---------------------------------------------------------------------------
// The extent the frame is padded to before entering the network, and the
// padding added (the pinned `padding`). Every halving of an axis needs its
// extent divisible by two, so each axis is rounded up to a multiple of
// 2**halvings, with a floor of 320; a frame already aligned to full 4-pixel
// cells at every level gets one extra unit on the width so that the shifted
// windows have room. The "small" network's library code aligns an axis to
// twice as much where the pre block's output along it, up((n + 1) / 2, 4), is
// 4 mod 8. Frame sizes outside 0 < width,height <= 16384 are rejected with
// Status::kInvalidDimensions and leave `out` untouched.
inline Status plan_padding(int frame_w, int frame_h, NetworkKind network, PlanPadding& out) {
    if (frame_w <= 0 || frame_h <= 0
        || frame_w > kPlanFrameDimMax || frame_h > kPlanFrameDimMax) {
        return Status::kInvalidDimensions;
    }
    plan_detail::PlanRow rows[kPlanMaxLayers];
    const int count = plan_detail::build_rows(network, rows);
    plan_detail::Shape shapes[kPlanMaxLayers][2];
    int halvings_x = 0;
    int halvings_y = 0;
    plan_detail::walk(frame_w, frame_h, rows, count, shapes, halvings_x, halvings_y);
    int align_x = 1 << halvings_x;
    int align_y = 1 << halvings_y;
    if (plan_detail::network_shape(network).wide_alignment) {
        align_x *= plan_detail::up((frame_w + 1) / 2, 4) % 8 == 4 ? 2 : 1;
        align_y *= plan_detail::up((frame_h + 1) / 2, 4) % 8 == 4 ? 2 : 1;
    }
    int padded_w = plan_detail::up(frame_w, align_x);
    if (padded_w < kPlanFloorExtent) {
        padded_w = kPlanFloorExtent;
    }
    int padded_h = plan_detail::up(frame_h, align_y);
    if (padded_h < kPlanFloorExtent) {
        padded_h = kPlanFloorExtent;
    }
    if (padded_h % (align_y * 4) == 0 && padded_w % (align_x * 4) == 0) {
        padded_w += align_x;
    }
    out.width = padded_w;
    out.height = padded_h;
    out.pad_x = padded_w - frame_w;
    out.pad_y = padded_h - frame_h;
    return Status::kOk;
}

// (width, height) in frame pixels of the padded frame the network processes
// (the pinned `working_extent`).
inline Status plan_working_extent(int frame_w, int frame_h, NetworkKind network,
                                  int& out_w, int& out_h) {
    PlanPadding padded;
    const Status status = plan_padding(frame_w, frame_h, network, padded);
    if (status != Status::kOk) {
        return status;
    }
    out_w = padded.width;
    out_h = padded.height;
    return Status::kOk;
}

// The number of times each axis is halved along the way (the pinned `_walk`
// counters `padding` derives its alignment from), for a frame extent
// (0 < extent <= kPlanFrameDimMax). Padded extents beyond the frame bound are
// walked by `build_plan` internally and are outside this entry's domain.
inline Status plan_halvings(int extent_w, int extent_h, NetworkKind network,
                            int& halvings_x, int& halvings_y) {
    if (extent_w <= 0 || extent_h <= 0
        || extent_w > kPlanFrameDimMax || extent_h > kPlanFrameDimMax) {
        return Status::kInvalidDimensions;
    }
    plan_detail::PlanRow rows[kPlanMaxLayers];
    const int count = plan_detail::build_rows(network, rows);
    plan_detail::Shape shapes[kPlanMaxLayers][2];
    plan_detail::walk(extent_w, extent_h, rows, count, shapes, halvings_x, halvings_y);
    return Status::kOk;
}

// The layer specs of `network` (152 for "ht", 76 for "small") for a
// `frame_w` x `frame_h` frame (the pinned `build_plan`). `specs` receives
// exactly plan_layer_count(network) entries and must have room for them
// (Status::kInvalidBuffer otherwise); `out_count` receives the entry count.
// The specs are pure metadata (names point at static storage), so the call is
// deterministic and allocation-free.
inline Status build_plan(int frame_w, int frame_h, NetworkKind network,
                         LayerSpec* specs, int capacity, int& out_count) {
    if (specs == nullptr) {
        return Status::kNullPointer;
    }
    if (frame_w <= 0 || frame_h <= 0
        || frame_w > kPlanFrameDimMax || frame_h > kPlanFrameDimMax) {
        return Status::kInvalidDimensions;
    }
    const int expected = plan_layer_count(network);
    if (capacity < expected) {
        return Status::kInvalidBuffer;
    }
    plan_detail::PlanRow rows[kPlanMaxLayers];
    const int count = plan_detail::build_rows(network, rows);

    PlanPadding padded;
    plan_padding(frame_w, frame_h, network, padded);  // dimensions validated above

    plan_detail::Shape shapes[kPlanMaxLayers][2];
    int halvings_x = 0;
    int halvings_y = 0;
    plan_detail::walk(padded.width, padded.height, rows, count, shapes, halvings_x, halvings_y);

    for (int i = 0; i < count; ++i) {
        const plan_detail::PlanRow& row = rows[i];
        plan_detail::Shape plan = shapes[i][0];
        const plan_detail::Shape out = shapes[i][1];
        if (row.role == LayerRole::kFinalHead) {
            plan = out;
        } else if (row.role == LayerRole::kDecoderUp) {
            plan = plan_detail::Shape{
                plan_detail::up(out.w, 4), plan_detail::up(out.h, 4)};
        } else if (row.role == LayerRole::kPost) {
            plan = plan_detail::Shape{padded.height / 2, padded.width / 2};
        }
        LayerSpec& spec = specs[i];
        spec.index = i;
        spec.block = row.block;
        spec.layer = row.layer;
        spec.kind = row.kind;
        spec.variant = plan_detail::variant_from_name(row.name);
        spec.w = plan.w;
        spec.h = plan.h;
        spec.tokens = row.kind == LayerKind::kVitFfnExpand
                || row.kind == LayerKind::kVitFfnContract
                || row.kind == LayerKind::kVitQKV
                || row.kind == LayerKind::kVitAttention
                || row.kind == LayerKind::kVitProjection
            ? plan.w * plan.h
            : plan_detail::up(plan.w, 8) * plan_detail::up(plan.h, 8);
        spec.channels = row.channels;
        spec.heads = row.heads;
        spec.c_in = row.c_in;
        spec.c_out = row.c_out;
        spec.shifted = (row.shift_x != 0 || row.shift_y != 0) && row.role != LayerRole::kPost;
        spec.shift_x = row.shift_x;
        spec.shift_y = row.shift_y;
        spec.input_block = row.layer == 0 ? row.block - 1 : row.block;
        spec.skip_block = row.skip_block >= 0 ? row.skip_block
            : (row.role == LayerRole::kPost ? 0 : -1);
        spec.name = row.name;
        spec.out_w = out.w;
        spec.out_h = out.h;
        const plan_detail::Shape grid =
            row.role == LayerRole::kPost || row.role == LayerRole::kSwinUp ? out : plan;
        spec.grid_x = (grid.h - 4 * row.shift_x + 7) / 8;
        spec.grid_y = (grid.w - 4 * row.shift_y + 7) / 8;
    }
    out_count = count;
    return Status::kOk;
}

// The index of the layer whose output `specs[index]` reads as its skip input
// (the pinned `NRNet._skip_sources`), or -1 when the layer has no skip input:
// a split block's projections add back its feed-forward projection (the final
// head reads the pooled copy), a vit block's contract reads the previous
// block's projection and its projection reads its own contract, and otherwise
// the named skip block's downsample (or, for the decoder entry, its pool
// layer; for the post block, the pre block) is the source.
inline int plan_skip_source(const LayerSpec* specs, int count, int index) {
    if (specs == nullptr || index < 0 || index >= count) {
        return -1;
    }
    const LayerSpec& spec = specs[index];
    int want_block = 0;
    int want_layer = 0;
    switch (spec.kind) {
        case LayerKind::kSplitProj:
        case LayerKind::kSplitProjPool:
            want_block = spec.block;
            want_layer = 1;
            break;
        case LayerKind::kSplitFinalHead:
            want_block = spec.block;
            want_layer = 3;
            break;
        case LayerKind::kVitFfnContract:
            want_block = spec.block - 1;
            want_layer = 4;
            break;
        case LayerKind::kVitProjection:
            want_block = spec.block;
            want_layer = 1;
            break;
        default:
            if (spec.skip_block < 0) {
                return -1;
            }
            want_block = spec.skip_block;
            want_layer = spec.kind == LayerKind::kDecInputUpsample ? 3 : 0;
            break;
    }
    for (int i = 0; i < count; ++i) {
        if (specs[i].block == want_block && specs[i].layer == want_layer) {
            return i;
        }
    }
    return -1;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_GRAPH_PLAN_HPP
