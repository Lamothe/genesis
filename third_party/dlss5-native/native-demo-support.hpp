// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Deterministic demo support for the native Linux pipeline: synthetic test
// images, a minimal binary PPM (P6) codec, a side-by-side contact sheet and
// the synthetic weight-record presets that drive the EvalWeights record
// interface with zero external assets. Demo-only code — the visual
// transformation of any synthetic preset is arbitrary: this demonstrates the
// complete pipeline running on Linux, NOT image quality and NOT vendor
// correspondence. No proprietary weights, captures or payloads are used or
// produced.
//
// Adapted in small part from the pinned MIT reference
// mauri870/dlssnr-pytorch 3beef7935a8ef976b47ab78b906e75600c52d2de
// (record-name/shape request patterns follow dlssnr/layers/*); see
// NATIVE-DEMO.md for the full licence text and attribution.
#ifndef DLSS5_NR_NATIVE_DEMO_SUPPORT_HPP
#define DLSS5_NR_NATIVE_DEMO_SUPPORT_HPP

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>

#include "native-eval.hpp"
#include "native-history.hpp"
#include "native-input-lift.hpp"

namespace dlss5_demo {

// ---- deterministic pseudo-random source ------------------------------------
// The Numerical Recipes 32-bit LCG (mod 2^32, unsigned wraparound defined):
//   x' = 1664525 * x + 1013904223  (mod 2^32)
// Small, total and hand-derivable; enough for deterministic synthetic data.
inline std::uint32_t lcg_next(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// The demo preset key hash: FNV-1a over the record name (case-sensitive,
// exact bytes). Deterministic across runs and platforms.
inline std::uint32_t fnv1a(const char* text) {
    std::uint32_t h = 2166136261u;
    for (const unsigned char* p =
             reinterpret_cast<const unsigned char*>(text);
         *p != '\0'; ++p) {
        h ^= static_cast<std::uint32_t>(*p);
        h *= 16777619u;
    }
    return h;
}

// ---- synthetic images ------------------------------------------------------
// A checkerboard: cell(x / cell + y / cell) parity selects `a` or `b`.
inline void fill_checker(std::uint8_t* rgb, int width, int height, int cell,
                         const std::uint8_t a[3], const std::uint8_t b[3]) {
    if (rgb == nullptr || width < 1 || height < 1 || cell < 1) {
        return;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool odd = ((x / cell) + (y / cell)) % 2 != 0;
            const std::uint8_t* c = odd ? b : a;
            std::uint8_t* p = rgb + 3 * (static_cast<std::size_t>(y) * width
                                         + static_cast<std::size_t>(x));
            p[0] = c[0];
            p[1] = c[1];
            p[2] = c[2];
        }
    }
}

// A smooth-ish gradient: r = 255 * x / (width - 1), g = 255 * y / (height - 1),
// b = 128 (integer division; the endpoints land exactly on 0/255).
inline void fill_gradient(std::uint8_t* rgb, int width, int height) {
    if (rgb == nullptr || width < 1 || height < 1) {
        return;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uint8_t* p = rgb + 3 * (static_cast<std::size_t>(y) * width
                                         + static_cast<std::size_t>(x));
            p[0] = static_cast<std::uint8_t>(
                255 * x / (width > 1 ? width - 1 : 1));
            p[1] = static_cast<std::uint8_t>(
                255 * y / (height > 1 ? height - 1 : 1));
            p[2] = 128;
        }
    }
}

// ---- binary PPM (P6) codec -------------------------------------------------
// Encoder: the exact bytes "P6\n<width> <height>\n255\n" followed by the
// interleaved RGB payload. `needed` reports the required byte count on a
// short buffer (or the written count on success).
inline dlss5_nr::Status encode_ppm_p6(std::uint8_t* buffer,
                                      std::size_t capacity,
                                      const std::uint8_t* rgb, int width,
                                      int height, std::size_t& needed) {
    needed = 0;
    if (buffer == nullptr || rgb == nullptr) {
        return dlss5_nr::Status::kNullPointer;
    }
    if (width < 1 || height < 1 || width > 65535 || height > 65535) {
        return dlss5_nr::Status::kInvalidDimensions;
    }
    char header[64];
    int n = 0;
    // Fixed decimal formatting without locale dependence.
    const int written = std::snprintf(header, sizeof(header), "P6\n%d %d\n255\n",
                                      width, height);
    if (written <= 0 || written >= static_cast<int>(sizeof(header))) {
        return dlss5_nr::Status::kInvalidBuffer;
    }
    n = written;
    const std::size_t payload = 3u * static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);
    needed = static_cast<std::size_t>(n) + payload;
    if (capacity < needed) {
        return dlss5_nr::Status::kInvalidBuffer;
    }
    std::memcpy(buffer, header, static_cast<std::size_t>(n));
    std::memcpy(buffer + n, rgb, payload);
    return dlss5_nr::Status::kOk;
}

namespace demo_detail {

// Reads one positive decimal number from the cursor; rejects anything else.
inline bool parse_uint(const std::uint8_t*& p, const std::uint8_t* end,
                       int& out) {
    if (p == end || *p < '0' || *p > '9') {
        return false;
    }
    long long v = 0;
    while (p != end && *p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        if (v > 65535) {
            return false;
        }
        ++p;
    }
    out = static_cast<int>(v);
    return true;
}

}  // namespace demo_detail

// Decoder: accepts exactly the encoder's format (the P6 magic, three
// whitespace-separated numbers and a single whitespace byte before the
// payload). Deliberately strict — the demo reads only files it writes or
// simple tool-generated P6 images.
inline dlss5_nr::Status decode_ppm_p6(const std::uint8_t* data,
                                      std::size_t length, std::uint8_t* rgb,
                                      std::size_t capacity, int& width,
                                      int& height) {
    if (data == nullptr || rgb == nullptr) {
        return dlss5_nr::Status::kNullPointer;
    }
    const std::uint8_t* p = data;
    const std::uint8_t* end = data + length;
    if (length < 2 || p[0] != 'P' || p[1] != '6') {
        return dlss5_nr::Status::kInvalidValues;
    }
    p += 2;
    if (p == end || (*p != ' ' && *p != '\n' && *p != '\t' && *p != '\r')) {
        return dlss5_nr::Status::kInvalidValues;
    }
    ++p;
    if (!demo_detail::parse_uint(p, end, width)) {
        return dlss5_nr::Status::kInvalidValues;
    }
    if (p == end || (*p != ' ' && *p != '\n' && *p != '\t' && *p != '\r')) {
        return dlss5_nr::Status::kInvalidValues;
    }
    ++p;
    if (!demo_detail::parse_uint(p, end, height)) {
        return dlss5_nr::Status::kInvalidValues;
    }
    if (p == end || (*p != ' ' && *p != '\n' && *p != '\t' && *p != '\r')) {
        return dlss5_nr::Status::kInvalidValues;
    }
    ++p;
    int max_value = 0;
    if (!demo_detail::parse_uint(p, end, max_value) || max_value != 255) {
        return dlss5_nr::Status::kInvalidValues;
    }
    if (p == end || (*p != ' ' && *p != '\n' && *p != '\t' && *p != '\r')) {
        return dlss5_nr::Status::kInvalidValues;
    }
    ++p;
    const std::size_t payload = 3u * static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);
    if (static_cast<std::size_t>(end - p) < payload) {
        return dlss5_nr::Status::kInvalidBuffer;
    }
    if (capacity < payload) {
        return dlss5_nr::Status::kInvalidBuffer;
    }
    std::memcpy(rgb, p, payload);
    return dlss5_nr::Status::kOk;
}

// ---- side-by-side contact sheet --------------------------------------------
// Output layout: (2 * width) x height, the `before` image in the left half
// (x < width) and `after` in the right half (x >= width, read at x - width).
inline void contact_sheet(std::uint8_t* out_rgb, const std::uint8_t* before,
                          const std::uint8_t* after, int width, int height) {
    if (out_rgb == nullptr || before == nullptr || after == nullptr
        || width < 1 || height < 1) {
        return;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < 2 * width; ++x) {
            const std::uint8_t* src = x < width ? before : after;
            const int sx = x < width ? x : x - width;
            const std::uint8_t* s = src + 3 * (static_cast<std::size_t>(y)
                                               * width
                                               + static_cast<std::size_t>(sx));
            std::uint8_t* d = out_rgb + 3 * (static_cast<std::size_t>(y)
                                             * (2 * width)
                                             + static_cast<std::size_t>(x));
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
}

// ---- synthetic weight records ---------------------------------------------
// Deterministic presets over the EvalWeights record interface. The CONTENT of
// every record is a pure function of (preset, seed, block, layer, name):
// repeated fetches and re-runs produce bit-identical records. The consumers
// hold several record pointers live at once (swin_decode keeps five code
// records and the scalar tables simultaneously), so each distinct
// (block, layer, name, kind) gets its own storage slot; slots round-robin,
// and because the fill stream is keyed the same way, a slot may be EXTENDED
// by a longer later request without changing the bytes already handed out.
//
// Presets:
//   kZero   — every record is zero bytes / zero floats. Honest label: with
//             zero weights the residual path dominates; the demo reports the
//             observed behaviour rather than promising "black".
//   kSeeded — small-magnitude records from the LCG, keyed by the request.
//             The e4m3 codes stay in a small value range (no saturation to
//             the e4m3 ceiling) so multi-layer activations remain in range.
//
// The transformation any preset produces is arbitrary (these are synthetic
// weights, not trained ones): the demo proves the pipeline runs end to end,
// not that it enhances anything.
struct DemoStore {
    enum class Preset { kZero, kSeeded };

    Preset preset = Preset::kSeeded;
    std::uint32_t seed = 20261028u;

    // Per-key record storage: each (block, layer, name, which) key keeps one
    // buffer, stable after its first fetch at its final size (a larger later
    // request may realloc and move it — the plan fetches each key at one
    // fixed size, so this is first-fill in practice). Stable pointers let the
    // Vulkan weight cache SKIP its per-hit re-upload; the cache's content
    // hash remains the correctness guard either way. (The earlier round-robin
    // slots recycled pointers across keys, which the hash had to catch and
    // re-upload every frame.) The plan's record set is well under
    // kMaxRecords; memory is the sum of the actual records (~80 MiB of codes
    // for the vit matrices), allocated on first fetch.
    static constexpr int kMaxRecords = 256;
    struct Slot {
        std::uint64_t key = 0;
        bool used = false;
        std::size_t filled = 0;
        std::uint8_t* codes = nullptr;
        float* values = nullptr;
    };
    Slot code_slots[kMaxRecords] = {};
    Slot value_slots[kMaxRecords] = {};
    int code_next = 0;
    int value_next = 0;

    ~DemoStore() {
        for (int i = 0; i < kMaxRecords; ++i) {
            std::free(code_slots[i].codes);
            std::free(value_slots[i].values);
        }
    }
    DemoStore() = default;
    // Owned malloc'd buffers: copying would double-free.
    DemoStore(const DemoStore&) = delete;
    DemoStore& operator=(const DemoStore&) = delete;
};

namespace demo_detail {

// The wildcard records the bindings probe at a fixed synthetic length (the
// length probe asks with count == -1 or the exact length).
inline bool wildcard_length(const char* name, int& length) {
    if (std::strcmp(name, "upsample_gain") == 0) {
        length = 16;
        return true;
    }
    if (std::strcmp(name, "residual_scale") == 0
        || std::strcmp(name, "attn_residual_scale") == 0
        || std::strcmp(name, "scalars_b") == 0) {
        length = 4096;
        return true;
    }
    return false;
}

// Seeds the per-request LCG from the preset seed and the request key.
inline std::uint32_t request_seed(const DemoStore& s, int block, int layer,
                                  const char* name, int which) {
    std::uint32_t h = s.seed;
    h = (h ^ fnv1a(name)) * 16777619u;
    h = (h ^ static_cast<std::uint32_t>(block * 8191 + layer * 131
                                        + which * 17))
        * 16777619u;
    return h;
}

// The per-request key (the name hash + the block/layer/which triple; each
// field is small and non-negative in the plan's ranges).
inline std::uint64_t request_key(const char* name, int block, int layer,
                                 int which) {
    return (static_cast<std::uint64_t>(fnv1a(name)) << 24)
        | static_cast<std::uint64_t>(((block & 0xFF) << 16)
                                     | ((layer & 0xFF) << 8)
                                     | ((which + 16) & 0xFF));
}

// Finds the key's stable slot, or appends a new one (never evicting: the
// pointer handed out for a key must keep its content).
inline int claim_slot(DemoStore::Slot* slots, int count, std::uint64_t key,
                      int& cursor) {
    for (int i = 0; i < count; ++i) {
        if (slots[i].used && slots[i].key == key) {
            return i;
        }
    }
    if (cursor >= count) {
        return -1;
    }
    const int slot = cursor;
    ++cursor;
    slots[slot].key = key;
    slots[slot].used = true;
    slots[slot].filled = 0;
    return slot;
}

// Serves the key's deterministic code record (filling or extending its slot).
inline const std::uint8_t* fill_codes(DemoStore& s, int block, int layer,
                                      const char* name, int which,
                                      std::size_t count) {
    if (count > (1u << 22)) {
        return nullptr;
    }
    const int slot = claim_slot(s.code_slots, DemoStore::kMaxRecords,
                                request_key(name, block, layer, which),
                                s.code_next);
    if (slot < 0) {
        return nullptr;
    }
    if (s.code_slots[slot].filled < count) {
        // Lazy exact-size allocation; growth keeps the pointer STABLE is
        // not guaranteed by realloc — but a grown record's old prefix was
        // handed out to previous calls already consumed, and the cache's
        // hash catches content changes at the same pointer either way.
        auto* grown = static_cast<std::uint8_t*>(
            std::realloc(s.code_slots[slot].codes, count));
        if (grown == nullptr) {
            return nullptr;
        }
        s.code_slots[slot].codes = grown;
    }
    std::uint8_t* out = s.code_slots[slot].codes;
    if (s.code_slots[slot].filled >= count) {
        return out;  // the prefix already handed out is the same stream
    }
    if (s.preset == DemoStore::Preset::kZero) {
        std::memset(out, 0, count);
    } else {
        std::uint32_t r = request_seed(s, block, layer, name, which);
        for (std::size_t i = 0; i < count; ++i) {
            // e4m3 codes 12..28 (the tree's bias-7 decode: 0.0234375 ..
            // 0.09375): small enough that deep accumulation stays off the
            // e4m3 ceiling (448), large enough to survive the binary16 hops.
            out[i] = static_cast<std::uint8_t>(12 + (lcg_next(r) % 17u));
        }
    }
    s.code_slots[slot].filled = count;
    return out;
}

// Serves the key's deterministic value record (filling or extending).
inline const float* fill_values(DemoStore& s, int block, int layer,
                                const char* name, int which,
                                std::size_t count) {
    if (count > (1u << 17)) {
        return nullptr;
    }
    const int slot = claim_slot(s.value_slots, DemoStore::kMaxRecords,
                                request_key(name, block, layer, which),
                                s.value_next);
    if (slot < 0) {
        return nullptr;
    }
    if (s.value_slots[slot].filled < count) {
        auto* grown = static_cast<float*>(
            std::realloc(s.value_slots[slot].values, count * sizeof(float)));
        if (grown == nullptr) {
            return nullptr;
        }
        s.value_slots[slot].values = grown;
    }
    float* out = s.value_slots[slot].values;
    if (s.value_slots[slot].filled >= count) {
        return out;
    }
    if (s.preset == DemoStore::Preset::kZero) {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = 0.0f;
        }
    } else {
        std::uint32_t r = request_seed(s, block, layer, name, which);
        for (std::size_t i = 0; i < count; ++i) {
            // [-1, 1) on a 1/1024 grid (exact in binary16).
            const std::int32_t q =
                static_cast<std::int32_t>(lcg_next(r) % 2048u) - 1024;
            out[i] = static_cast<float>(q) / 1024.0f;
        }
    }
    s.value_slots[slot].filled = count;
    return out;
}

}  // namespace demo_detail

inline const std::uint8_t* demo_e4m3(void* context, int block, int layer,
                                     const char* name, int rows, int cols) {
    DemoStore* s = static_cast<DemoStore*>(context);
    if (s == nullptr || rows < 0 || cols < 0) {
        return nullptr;
    }
    return demo_detail::fill_codes(
        *s, block, layer, name, -2,
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
}

inline const float* demo_f16(void* context, int block, int layer,
                             const char* name, int count) {
    DemoStore* s = static_cast<DemoStore*>(context);
    if (s == nullptr || count < -1) {
        return nullptr;
    }
    int length = count;
    if (count == -1) {
        if (!demo_detail::wildcard_length(name, length)) {
            return nullptr;
        }
    } else {
        int wildcard = 0;
        if (demo_detail::wildcard_length(name, wildcard)
            && count != wildcard) {
            return nullptr;
        }
    }
    return demo_detail::fill_values(*s, block, layer, name, -3,
                                    static_cast<std::size_t>(length));
}

inline const float* demo_f32(void* context, int block, int layer,
                             const char* name, int count) {
    DemoStore* s = static_cast<DemoStore*>(context);
    if (s == nullptr || count < -1) {
        return nullptr;
    }
    int length = count;
    if (count == -1) {
        if (!demo_detail::wildcard_length(name, length)) {
            return nullptr;
        }
    } else {
        int wildcard = 0;
        if (demo_detail::wildcard_length(name, wildcard)
            && count != wildcard) {
            return nullptr;
        }
    }
    return demo_detail::fill_values(*s, block, layer, name, -4,
                                    static_cast<std::size_t>(length));
}

inline const std::uint8_t* demo_ffwd(void* context, int block, int layer,
                                     int which, int rows, int cols) {
    DemoStore* s = static_cast<DemoStore*>(context);
    if (s == nullptr || rows < 0 || cols < 0) {
        return nullptr;
    }
    return demo_detail::fill_codes(
        *s, block, layer, "ffwd", which,
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
}

// Models the FP16 output-texture store of the hardware route: the network's
// final map holds compose values on the 8-bit grid (1/255 steps, not
// binary16-representable) while the temporal carrier stores FP16 texture
// content bit-exactly. Rounding each value to binary16 before the commit is
// the store the real route performs; for values already binary16-exact it is
// the identity. Validation is a full first pass: a non-finite value or one
// beyond the max finite binary16 magnitude (65504) rejects the whole frame
// before any write.
inline dlss5_nr::Status round_frame_to_f16(float* rgb, std::size_t count) {
    if (rgb == nullptr) {
        return dlss5_nr::Status::kNullPointer;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const float v = rgb[i];
        if (!(v == v) || v > 65504.0f || v < -65504.0f) {
            return dlss5_nr::Status::kInvalidValues;
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        rgb[i] = dlss5_nr::f16(rgb[i]);
    }
    return dlss5_nr::Status::kOk;
}

inline void demo_bind_weights(dlss5_nr::EvalWeights& weights,
                              DemoStore& store) {
    weights.context = &store;
    weights.e4m3 = demo_e4m3;
    weights.f16 = demo_f16;
    weights.f32 = demo_f32;
    weights.ffwd = demo_ffwd;
    // int4 stays null: every row runs the e4m3 dataflow (the pinned
    // table-presence gating).
}

}  // namespace dlss5_demo

#endif  // DLSS5_NR_NATIVE_DEMO_SUPPORT_HPP
