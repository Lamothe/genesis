// SPDX-License-Identifier: MIT
// Native CPU decoder input stage (block 39): `DecInputUpsample` — the
// bottleneck's output meets the encoder's full-resolution skip. The e4m3
// projection kept in float16, the nearest-2x replication cropped to the skip's
// extent, the per-channel gated skip and the q8 store — typed, deterministic
// CPU kernels. CPU-only, allocation-free; no GPU, headset, proprietary
// payload, Python or PyTorch runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/decoder.py "DecInputUpsample", with its imported
// dlssnr/layers/vit.py helpers `_Matmul`/`to_tokens`/`f16` and
// dlssnr/quant.py's `q8`). The full MIT licence notice is retained in
// vr/NATIVE-DECODER-INPUT.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.
//
// Semantics carried over exactly (`DecInputUpsample.forward`):
//   - the projection `f16(matmul(to_tokens(x)))` is the vit family's matmul
//     (`_Matmul` with the reference's "decoder" site tag; the int8 experiment
//     hook is off, so it is a plain float32 product): the map is read as
//     `[rows * cols][c_in]` tokens in row-major order, `out[t][n] =
//     sum_k in[t][k] * W[n][k]` with e4m3 operands exact in float32 and
//     float32 accumulation in key order, and every value is rounded to
//     binary16 once. The projected map `[c_out][x.rows][x.cols]` is kept in
//     binary16 (the caller's scratch mirror of the reference's `projected`),
//     computed in full exactly as the reference computes it.
//   - `upsampled` repeats each projected value over a 2x2 block (nearest
//     neighbour) and crops to the skip's extent: output (y, x) reads
//     projected (y >> 1, x >> 1). This is a crop, NOT the resample family's
//     clamp-replicate padding: a skip larger than the replicated map is a
//     shape error in the reference (its broadcast add cannot match) and is
//     rejected here with Status::kInvalidDimensions.
//   - `gated = f16(f16(skip) * gain)`: the skip value is rounded to binary16
//     FIRST (the reference's explicit `f16(skip)`; in-pipeline skip values are
//     e4m3-grid, whose binary16 rounding is exact, so the rounding is live
//     semantics only for synthetic inputs), then the product with the layer's
//     binary16 `skip_weight` gain is computed in float32 — two binary16
//     significands make the product exact, at most 22 significant bits — and
//     rounded to binary16 once, ties to even. Unlike the resample family's
//     `gated_sum_element` (one fused binary64 rounding of `skip*gain +
//     other`), the decoder gate rounds twice and the final add is separate.
//   - the store is `q8(f16(upsampled + gated))`: the two binary16 values are
//     added in float32 and rounded to binary16, then to the nearest e4m3
//     value, saturating at +-448 (ties to even) — the network's double
//     rounding (the binary16 hop matters at e4m3 half-steps). For binary16
//     addends the float32 add is either exact or its rounding cannot change
//     the binary16 result; the kernel still performs the literal float32 add.
//   - overflow/NaN edges are total and follow the reference: binary16
//     roundings overflow to +-infinity exactly as the reference's float16
//     conversion; the e4m3 store saturates infinities to +-448 exactly as
//     q8's clamp; `inf + (-inf)` or `inf * 0` produces NaN, which the
//     reference's q8 canonicalises to the e4m3 NaN code ("NaN is not
//     preserved") and the kernels reproduce (a canonical quiet NaN).

#ifndef DLSS5_NR_NATIVE_DECODER_HPP
#define DLSS5_NR_NATIVE_DECODER_HPP

#include "native-vit.hpp"  // E4m3Matrix, Status; f16/e4m3/is_f16/is_e4m3 via its includes

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace dlss5_nr {

// ---- kernel envelopes ------------------------------------------------------------------------------
// Checked with plain comparisons and widened products before any offset
// maths. 16448 is the padded-frame bound (16384 frame plus the 64-pixel
// alignment of the plan's working extent), matching the split family; the
// channel envelope follows the vit family's matmul (kVitMaxChannels).
inline constexpr int kDecMaxExtent = 16448;
inline constexpr long long kDecMaxTokens = 1LL << 29;    // rows * columns

// ---- the layer's weights --------------------------------------------------------------------------
// `weight` is the e4m3 projection [c_out, c_in] in the `_Matmul`/weight-file
// representation: row-major raw e4m3 codes. The codes 0x7f/0xff are the e4m3
// NaN encodings and are rejected (every other byte decodes to a finite value).
// `gain` is the layer's f16 `skip_weight` row: exactly gain_count finite,
// binary16-representable values (the weight loader's contract). All operands
// are passed explicitly (original synthetic values in tests; no learned
// weights are used or implied; weight-record decode stays the declared
// weight-file boundary outside this slice).
struct DecInputWeight {
    E4m3Matrix weight;
    const float* gain = nullptr;
    std::size_t gain_count = 0;
};

// ---- the gated skip term (`f16(f16(skip) * gain)`) -------------------------------------------------
// The skip value rounds to binary16 first, then the product with the binary16
// gain is computed in float32 (exact: at most 22 significant bits, and finite
// binary16 factors never overflow float32) and rounded to binary16 once. The
// caller's gain is the validated binary16 row inside the kernel; this element
// form is total for any float32 inputs (binary16 overflow of the pre-rounding
// yields +-infinity; infinity times zero yields NaN, canonicalised by f16).
inline float dec_gated_element(float skip, float gain) {
    return f16(f16(skip) * gain);
}

// ---- the network store (`q8(f16(value))`) ----------------------------------------------------------
// The double rounding of every stored result: float32 -> float16 -> e4m3,
// saturating at +-448 with ties to even, NaN canonicalised to the e4m3 NaN
// code 0x7f exactly as the reference's q8 ("NaN is not preserved"). Direct
// e4m3 rounding differs at binary16 half-steps (e.g. 68.00390625 stores as 64
// through the binary16 hop but rounds to 72 directly); the tests pin that.
inline float dec_store(float value) { return e4m3(f16(value)); }

// ---- shared map validation -------------------------------------------------------------------------

inline Status validate_decoder_map(const FeatureBatchView& m, int max_channels) {
    if (m.data == nullptr) {
        return Status::kNullPointer;
    }
    if (m.channels < 1 || m.channels > max_channels
        || m.rows < 1 || m.rows > kDecMaxExtent
        || m.cols < 1 || m.cols > kDecMaxExtent
        || static_cast<long long>(m.rows) * m.cols > kDecMaxTokens) {
        return Status::kInvalidDimensions;
    }
    const std::ptrdiff_t plane_span =
        static_cast<std::ptrdiff_t>(kDecMaxExtent) * kDecMaxExtent;
    if (m.row_stride < static_cast<std::ptrdiff_t>(m.cols)
        || m.row_stride > static_cast<std::ptrdiff_t>(kDecMaxExtent)
        || m.plane_stride < static_cast<std::ptrdiff_t>(m.rows) * m.row_stride
        || m.plane_stride > plane_span) {
        return Status::kInvalidStride;
    }
    return Status::kOk;
}

// ---- DecInputUpsample -----------------------------------------------------------------------------
// The reference's `DecInputUpsample.forward(x, skip)`: `x` is the bottleneck
// output map [c_in, x.rows, x.cols] on the FP8 boundary (e4m3-grid values),
// `skip` the encoder's full-resolution skip [c_out, skip.rows, skip.cols] (any
// finite values; the layer rounds them to binary16 itself); the output is the
// skip's extent [c_out, skip.rows, skip.cols], tight (channel planes in order,
// row-major within a plane), e4m3-grid values. `scratch` mirrors the
// reference's projected map: c_out * x.rows * x.cols binary16 values (tight),
// written in full. `weight`/`gain` are the layer's projection and skip gain
// (see DecInputWeight).
//
// Validation (nulls, map dimensions/strides, the crop extent, weight and gain
// shapes, value classes, output and scratch capacities) completes before the
// first write: on any error `out` and `scratch` are both untouched. The
// buffers must not overlap the input maps or each other (caller discipline).
inline Status dec_input_upsample(const FeatureBatchView& x,
                                 const FeatureBatchView& skip,
                                 const DecInputWeight& w,
                                 float* out, std::size_t out_count,
                                 float* scratch,
                                 std::size_t scratch_capacity) {
    if (x.data == nullptr || skip.data == nullptr || w.weight.bits == nullptr
        || w.gain == nullptr || out == nullptr || scratch == nullptr) {
        return Status::kNullPointer;
    }
    {
        const Status status = validate_decoder_map(x, kVitMaxChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_decoder_map(skip, kVitMaxChannels);
        if (status != Status::kOk) {
            return status;
        }
    }
    // The 2x replication is cropped to the skip's extent; a skip beyond the
    // replicated map cannot be produced by the reference's crop (its add then
    // shape-errors), so it is rejected rather than clamp-replicated.
    if (skip.rows > 2 * x.rows || skip.cols > 2 * x.cols) {
        return Status::kInvalidDimensions;
    }
    const int c_in = x.channels;
    const int c_out = skip.channels;
    if (w.weight.rows != c_out || w.weight.cols != c_in) {
        return Status::kInvalidDimensions;
    }
    if (w.gain_count != static_cast<std::size_t>(c_out)) {
        return Status::kInvalidBuffer;
    }
    // Value classes before any write: the projection reads the FP8 boundary
    // (e4m3-grid x, rejecting the e4m3 NaN codes in the weight), the gate
    // takes any finite skip, and the gain row is the layer's binary16 buffer.
    for (int c = 0; c < c_out; ++c) {
        const std::uint8_t* wrow = w.weight.bits
            + static_cast<std::size_t>(c) * static_cast<std::size_t>(c_in);
        for (int k = 0; k < c_in; ++k) {
            if (wrow[k] == 0x7fu || wrow[k] == 0xffu) {
                return Status::kInvalidValues;  // the e4m3 NaN encodings
            }
        }
    }
    for (int c = 0; c < c_in; ++c) {
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
    for (int c = 0; c < c_out; ++c) {
        const float* plane = skip.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(skip.plane_stride);
        for (int r = 0; r < skip.rows; ++r) {
            const float* row = plane + static_cast<std::size_t>(r)
                * static_cast<std::size_t>(skip.row_stride);
            for (int col = 0; col < skip.cols; ++col) {
                if (!std::isfinite(row[col])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    for (int c = 0; c < c_out; ++c) {
        if (!std::isfinite(w.gain[c]) || !is_f16(w.gain[c])) {
            return Status::kInvalidValues;
        }
    }
    const std::size_t map_plane = static_cast<std::size_t>(x.rows)
        * static_cast<std::size_t>(x.cols);
    const std::size_t projected_count =
        static_cast<std::size_t>(c_out) * map_plane;
    const std::size_t out_plane = static_cast<std::size_t>(skip.rows)
        * static_cast<std::size_t>(skip.cols);
    const std::size_t out_needed = static_cast<std::size_t>(c_out) * out_plane;
    if (out_count < out_needed) {
        return Status::kInvalidBuffer;
    }
    if (scratch_capacity < projected_count) {
        return Status::kInvalidBuffer;
    }
    // All validation is complete; one write pass over the projected map and
    // one over the output follow, all operands already guaranteed accepted.
    // The projection is the vit matmul over a strided map read: element (c, r,
    // col) of plane k sits at k * plane_stride + r * row_stride + col, and the
    // float32 sums run in key order exactly like matmul_e4m3.
    for (int c = 0; c < c_out; ++c) {
        const std::uint8_t* wrow = w.weight.bits
            + static_cast<std::size_t>(c) * static_cast<std::size_t>(c_in);
        float* dst_plane = scratch + static_cast<std::size_t>(c) * map_plane;
        for (int r = 0; r < x.rows; ++r) {
            float* dst_row = dst_plane
                + static_cast<std::size_t>(r) * static_cast<std::size_t>(x.cols);
            for (int col = 0; col < x.cols; ++col) {
                const std::size_t offset = static_cast<std::size_t>(r)
                    * static_cast<std::size_t>(x.row_stride)
                    + static_cast<std::size_t>(col);
                float acc = 0.0f;
                for (int k = 0; k < c_in; ++k) {
                    acc += x.data[static_cast<std::size_t>(k)
                        * static_cast<std::size_t>(x.plane_stride) + offset]
                        * e4m3_from_bits(wrow[k]);
                }
                dst_row[col] = f16(acc);
            }
        }
    }
    for (int c = 0; c < c_out; ++c) {
        const float* skip_plane = skip.data + static_cast<std::size_t>(c)
            * static_cast<std::size_t>(skip.plane_stride);
        const float* proj_plane = scratch
            + static_cast<std::size_t>(c) * map_plane;
        float* dst_plane = out + static_cast<std::size_t>(c) * out_plane;
        const float gain_c = w.gain[c];
        for (int y = 0; y < skip.rows; ++y) {
            const float* skip_row = skip_plane + static_cast<std::size_t>(y)
                * static_cast<std::size_t>(skip.row_stride);
            const float* proj_row = proj_plane
                + static_cast<std::size_t>(y >> 1) * static_cast<std::size_t>(x.cols);
            float* dst_row = dst_plane
                + static_cast<std::size_t>(y) * static_cast<std::size_t>(skip.cols);
            for (int col = 0; col < skip.cols; ++col) {
                const float up = proj_row[col >> 1];
                const float gated = dec_gated_element(skip_row[col], gain_c);
                dst_row[col] = dec_store(up + gated);
            }
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_DECODER_HPP
