// Native output-conversion stage: the learned 32-channel -> 3-colour residual
// projection and the output compose (residual add with weight 1/4, clamp, intensity
// blend, and the intensity > 1 luminance-ratio amplification). CPU-only, typed,
// allocation-free; no GPU, headset, proprietary payload, Python or PyTorch runtime
// required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/pre_post.py "PostBlock.forward" projection and the module-level
// "compose"). The full MIT licence notice is retained in vr/NATIVE-COMPOSE.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_COMPOSE_HPP
#define DLSS5_NR_NATIVE_COMPOSE_HPP

#include <cfloat>
#include <cmath>
#include <cstddef>

#include "native-blend.hpp"

namespace dlss5_nr {

// ---- documented constants ------------------------------------------------------------------
inline constexpr int kOutputChannels = 3;                          // colour residuals
inline constexpr int kProjectionWeights = kOutputChannels * kLiftedChannels;  // 96
inline constexpr float kOutputScale = 0.25f;                       // weight of residual
inline constexpr float kIntensityFloor = 0x1p-9f;                  // luminance floor 1/512
inline constexpr float kAmplificationGuard = 2.0f;
// Rec. 709 luminance weights, as the reference builds them (float32).
inline constexpr float kLumaR = 0.2126f;
inline constexpr float kLumaG = 0.7152f;
inline constexpr float kLumaB = 0.0722f;

// ---- the residual projection ----------------------------------------------------------------
// The output projection of the post-block: three colour residuals from the body's 32
// f16 output channels, with the learned out_project rows (96 f16-representable finite
// values widened to float32, rows in colour order). Per output channel r:
//   dst_r(y, x) = sum over c = 0..31 of weights[32 * r + c] * wide(c, y, x)
// Each product is a float32 multiply and the 32 terms accumulate sequentially in
// ascending channel order in float32. The reference's einsum does not specify a
// contraction order; ascending sequential accumulation is this implementation's
// explicit semantics and the tests pin it (a different order can round differently).
// These roundings are the specified semantics, so the build contract is
// -ffp-contract=off (FMA contraction would change the product-plus-add roundings);
// the regression runner compiles with that flag.
//
// wide is a channels-first FeatureBatchView with exactly kLiftedChannels planes; dst
// receives kOutputChannels * rows * cols floats, planes in channel order, row-major
// within a plane. Allocation-free; all validation (including a finite-value scan of
// wide) completes before the first output write.
inline Status project_residual(const FeatureBatchView& wide,
                               const float* weights, std::size_t weight_count,
                               float* dst, std::size_t dst_count) {
    if (wide.data == nullptr || weights == nullptr) {
        return Status::kNullPointer;
    }
    if (wide.channels != kLiftedChannels || wide.rows <= 0 || wide.cols <= 0
        || wide.rows > kMaxRows || wide.cols > kMaxCols) {
        return Status::kInvalidDimensions;
    }
    const long long pixels = static_cast<long long>(wide.rows) * wide.cols;
    if (pixels > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    const long long plane_span = static_cast<long long>(kMaxRows) * kMaxCols;
    if (wide.row_stride < wide.cols || wide.row_stride > kMaxCols
        || wide.plane_stride < static_cast<long long>(wide.rows) * wide.row_stride
        || wide.plane_stride > plane_span) {
        return Status::kInvalidStride;
    }
    if (weight_count != static_cast<std::size_t>(kProjectionWeights)) {
        return Status::kInvalidBuffer;
    }
    for (int i = 0; i < kProjectionWeights; ++i) {
        // f16-representable and finite weights, matching the lift/blend record contract.
        if (!is_f16(weights[i]) || !std::isfinite(weights[i])) {
            return Status::kInvalidValues;
        }
    }
    if (dst == nullptr) {
        return Status::kNullPointer;
    }
    const std::size_t out_plane = static_cast<std::size_t>(wide.rows) * wide.cols;
    if (dst_count < static_cast<std::size_t>(kOutputChannels) * out_plane) {
        return Status::kInvalidBuffer;
    }
    // Finite-value scan completes all validation before the first output write.
    for (int c = 0; c < wide.channels; ++c) {
        for (int y = 0; y < wide.rows; ++y) {
            const float* row = wide.data
                + static_cast<std::size_t>(c) * static_cast<std::size_t>(wide.plane_stride)
                + static_cast<std::size_t>(y) * static_cast<std::size_t>(wide.row_stride);
            for (int x = 0; x < wide.cols; ++x) {
                if (!std::isfinite(row[x])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    for (int y = 0; y < wide.rows; ++y) {
        for (int x = 0; x < wide.cols; ++x) {
            for (int r = 0; r < kOutputChannels; ++r) {
                const float* weight_row = weights + r * kLiftedChannels;
                float sum = 0.0f;
                for (int c = 0; c < kLiftedChannels; ++c) {
                    const float* plane = wide.data
                        + static_cast<std::size_t>(c) * static_cast<std::size_t>(wide.plane_stride);
                    const float value = plane[static_cast<std::size_t>(y)
                        * static_cast<std::size_t>(wide.row_stride) + x];
                    sum += weight_row[c] * value;
                }
                dst[r * out_plane + static_cast<std::size_t>(y) * wide.cols + x] = sum;
            }
        }
    }
    return Status::kOk;
}

// ---- the output compose ----------------------------------------------------------------------
// The output colour of one pixel from the input frame colour and the projected
// residual, in this exact order (the reference's `compose`):
//   network = clamp01(frame + residual * 0.25)              (float32)
//   out     = clamp01(frame + min(intensity, 1) * (network - frame))   (float32)
//   for intensity <= 1 the result is `out`.
//   for intensity > 1: luminance l = out.r * 0.2126 + out.g * 0.7152 + out.b * 0.0722
//   (left-to-right float32) and likewise lf for the frame; ratio = (l + 2^-9) /
//   (lf + 2^-9) (float32); amplification = clamp(pow(ratio, intensity), 0.5, 2) with
//   the pow evaluated in double precision and narrowed once to float32; the result is
//   clamp01((out * amplification) / ratio).
//
// Explicit choices where the reference is silent: `min(intensity, 1)` is taken in
// double on the double intensity input and narrowed once to float32 (with an explicit
// range guard — narrowing an out-of-float32-range double is undefined in C++); the
// three-term luminance sums accumulate left-to-right in float32. The pow is evaluated
// in double (platform libm) and narrowed once — a disclosed deviation from the
// reference, whose torch pow computes in float32; like the lift's gaussian stage the
// result is NOT claimed bit-exact across platforms or against any torch backend. The
// reference's ratio.clamp_min(1e-6) cannot bind here:
// with frame and out in [0, 1] the ratio lies in [2^-9 / (1 + 2^-9), (1 + 2^-9) / 2^-9]
// (luminance is at most the weight sum ~1), far from 1e-6, so its comparison semantics
// are unobservable through this stage. The float32 roundings above are the specified
// semantics: the build contract is -ffp-contract=off (as for the projection).
//
// Precondition: frame values are in [0, 1] (as `compose` validates), residual values
// are finite, and intensity is finite (the batch entry point validates these; this
// per-pixel routine assumes them and defines the min/narrowing for every finite
// intensity).
inline void compose_pixel(const float frame[3], const float residual[3],
                          double intensity, float out[3]) {
    float network[3];
    for (int c = 0; c < 3; ++c) {
        const float scaled = residual[c] * kOutputScale;
        const float sum = frame[c] + scaled;
        network[c] = (sum < 0.0f) ? 0.0f : ((sum > 1.0f) ? 1.0f : sum);
    }
    double capped = (intensity < 1.0) ? intensity : 1.0;
    // Range-guarded narrowing: a finite intensity outside the float32 range would
    // otherwise be an out-of-range conversion (undefined in C++).
    capped = (capped < -static_cast<double>(FLT_MAX)) ? -static_cast<double>(FLT_MAX)
        : ((capped > static_cast<double>(FLT_MAX)) ? static_cast<double>(FLT_MAX)
            : capped);
    const float t = static_cast<float>(capped);
    float blended[3];
    for (int c = 0; c < 3; ++c) {
        const float delta = network[c] - frame[c];
        const float sum = frame[c] + t * delta;
        blended[c] = (sum < 0.0f) ? 0.0f : ((sum > 1.0f) ? 1.0f : sum);
    }
    if (!(intensity > 1.0)) {
        out[0] = blended[0];
        out[1] = blended[1];
        out[2] = blended[2];
        return;
    }
    const float luma_out = (blended[0] * kLumaR + blended[1] * kLumaG)
        + blended[2] * kLumaB;
    const float luma_frame = (frame[0] * kLumaR + frame[1] * kLumaG)
        + frame[2] * kLumaB;
    const float ratio = (luma_out + kIntensityFloor) / (luma_frame + kIntensityFloor);
    // The pow is evaluated in double; its clamp is applied in double before the
    // narrowing, so a double result beyond the float32 range is never converted
    // (out-of-range narrowing is undefined in C++). 0.5 and 2.0 are exact in float32,
    // so the narrowing after the clamp is exact.
    double amplification_d = std::pow(static_cast<double>(ratio), intensity);
    amplification_d = (amplification_d < 0.5) ? 0.5
        : ((amplification_d > static_cast<double>(kAmplificationGuard))
            ? static_cast<double>(kAmplificationGuard) : amplification_d);
    const float amplification = static_cast<float>(amplification_d);
    for (int c = 0; c < 3; ++c) {
        const float product = blended[c] * amplification;
        const float scaled = product / ratio;
        out[c] = (scaled < 0.0f) ? 0.0f : ((scaled > 1.0f) ? 1.0f : scaled);
    }
}

// The compose of a whole frame: frame is interleaved RGB row-major (FrameView, values
// in [0, 1]); residual is a kOutputChannels-plane FeatureBatchView of the same rows and
// cols; dst receives rows * cols * 3 interleaved RGB floats. Allocation-free; all
// validation (including frame range checks and finite-value scans) completes before
// the first output write. dst must not overlap frame.rgb or residual.data (overlapping
// writes can silently corrupt the result; they are never rejected). intensity must be
// finite; any finite intensity has defined semantics (negative values extrapolate away
// from the network colour).
inline Status compose(const FrameView& frame, const FeatureBatchView& residual,
                      double intensity, float* dst, std::size_t dst_count) {
    if (frame.rgb == nullptr || residual.data == nullptr) {
        return Status::kNullPointer;
    }
    if (!std::isfinite(intensity)) {
        return Status::kInvalidControls;
    }
    if (residual.channels != kOutputChannels || frame.rows <= 0 || frame.cols <= 0
        || frame.rows > kMaxRows || frame.cols > kMaxCols
        || residual.rows != frame.rows || residual.cols != frame.cols) {
        return Status::kInvalidDimensions;
    }
    const long long pixels = static_cast<long long>(frame.rows) * frame.cols;
    if (pixels > kMaxPixels) {
        return Status::kInvalidDimensions;
    }
    const long long plane_span = static_cast<long long>(kMaxRows) * kMaxCols;
    if (frame.row_stride < static_cast<std::ptrdiff_t>(3 * frame.cols)
        || frame.row_stride > static_cast<std::ptrdiff_t>(3 * kMaxCols)) {
        return Status::kInvalidStride;
    }
    if (residual.row_stride < residual.cols || residual.row_stride > kMaxCols
        || residual.plane_stride < static_cast<long long>(residual.rows) * residual.row_stride
        || residual.plane_stride > plane_span) {
        return Status::kInvalidStride;
    }
    if (dst == nullptr) {
        return Status::kNullPointer;
    }
    const std::size_t frame_stride = static_cast<std::size_t>(frame.row_stride);
    if (dst_count < static_cast<std::size_t>(pixels) * 3) {
        return Status::kInvalidBuffer;
    }
    // Range and finite scans complete all validation before the first output write.
    // The [0, 1] frame range is a documented input contract: the intensity > 1 ratio
    // bounds above rely on it.
    for (int y = 0; y < frame.rows; ++y) {
        const float* row = frame.rgb + static_cast<std::size_t>(y) * frame_stride;
        for (int x = 0; x < 3 * frame.cols; ++x) {
            if (!std::isfinite(row[x]) || row[x] < 0.0f || row[x] > 1.0f) {
                return Status::kInvalidValues;
            }
        }
    }
    for (int c = 0; c < residual.channels; ++c) {
        for (int y = 0; y < residual.rows; ++y) {
            const float* row = residual.data
                + static_cast<std::size_t>(c) * static_cast<std::size_t>(residual.plane_stride)
                + static_cast<std::size_t>(y) * static_cast<std::size_t>(residual.row_stride);
            for (int x = 0; x < residual.cols; ++x) {
                if (!std::isfinite(row[x])) {
                    return Status::kInvalidValues;
                }
            }
        }
    }
    for (int y = 0; y < frame.rows; ++y) {
        const float* frame_row = frame.rgb + static_cast<std::size_t>(y) * frame_stride;
        for (int x = 0; x < frame.cols; ++x) {
            float pixel_frame[3];
            float pixel_residual[3];
            float pixel_out[3];
            for (int c = 0; c < 3; ++c) {
                pixel_frame[c] = frame_row[x * 3 + c];
                const float* plane = residual.data
                    + static_cast<std::size_t>(c) * static_cast<std::size_t>(residual.plane_stride);
                pixel_residual[c] = plane[static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(residual.row_stride) + x];
            }
            compose_pixel(pixel_frame, pixel_residual, intensity, pixel_out);
            float* out_pixel = dst + (static_cast<std::size_t>(y) * frame.cols + x) * 3;
            out_pixel[0] = pixel_out[0];
            out_pixel[1] = pixel_out[1];
            out_pixel[2] = pixel_out[2];
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_COMPOSE_HPP
