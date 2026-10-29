// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux authors.
//
// Bit-exact SIMD helpers for the e4m3 dot-product matmuls. The AVX2 variant
// computes eight output rows per key-loop pass: lane i accumulates output
// n+i's sum in the SAME key order as the scalar reference (mul, then add,
// one sequential accumulator per lane), so every result is bit-identical to
// the scalar form at any width. Short keys are gather-bound and slower than
// scalar; callers gate at >= 256 keys (measured: 1.9x at 512, 1.8x at 1024,
// ~1.2x at 128, slower below).
//
// The kernel is emitted with __attribute__((target("avx2"))) so the
// translation units compile without -march flags; the call sites pair it
// with a __builtin_cpu_supports("avx2") runtime check.
//
// Provenance: the e4m3 (OCP FP8 E4M3) construction follows the pinned
// reference mauri870/dlssnr-pytorch
// 3beef7935a8ef976b47ab78b906e75600c52d2de (MIT), decoded bit-exactly here.
// See vr/NATIVE-PERFORMANCE.md for the measured numbers and the honesty
// statement (synthetic weights: speed is real, image quality is not).

#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace dlss5_nr {

// True when the CPU supports AVX2 and the dot8 kernels below are usable.
inline bool matmul_simd_available() {
#if defined(__x86_64__) || defined(__i386__)
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
#else
    return false;
#endif
}

#if defined(__x86_64__) || defined(__i386__)

namespace simd_detail {

// Bit-exact e4m3 decode of eight raw codes to eight float32 lanes (the same
// construction as e4m3_from_bits in native-pre-post.hpp). The vit test
// sweeps all 256 codes through decode8 against e4m3_from_bits directly.
__attribute__((target("avx2")))
inline __m256 decode8(const std::uint8_t* w0, const std::uint8_t* w1,
                      const std::uint8_t* w2, const std::uint8_t* w3,
                      const std::uint8_t* w4, const std::uint8_t* w5,
                      const std::uint8_t* w6, const std::uint8_t* w7) {
    alignas(32) static const std::int32_t kSub[8] = {
        0x00000000, 0x3B000000, 0x3B800000, 0x3BC00000,
        0x3C000000, 0x3C200000, 0x3C400000, 0x3C600000};
    const __m256i raw = _mm256_setr_epi32(
        static_cast<int>(*w0), static_cast<int>(*w1),
        static_cast<int>(*w2), static_cast<int>(*w3),
        static_cast<int>(*w4), static_cast<int>(*w5),
        static_cast<int>(*w6), static_cast<int>(*w7));
    const __m256i s = _mm256_slli_epi32(
        _mm256_and_si256(raw, _mm256_set1_epi32(0x80)), 24);
    const __m256i e = _mm256_and_si256(
        _mm256_srli_epi32(raw, 3), _mm256_set1_epi32(0xF));
    const __m256i m = _mm256_and_si256(raw, _mm256_set1_epi32(0x7));
    const __m256i normal = _mm256_or_si256(
        _mm256_or_si256(s, _mm256_slli_epi32(
            _mm256_add_epi32(e, _mm256_set1_epi32(120)), 23)),
        _mm256_slli_epi32(m, 20));
    const __m256i sub = _mm256_or_si256(_mm256_i32gather_epi32(kSub, m, 4), s);
    const __m256i is_sub = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    __m256i bits = _mm256_blendv_epi8(normal, sub, is_sub);
    const __m256i is_nan = _mm256_and_si256(
        _mm256_cmpeq_epi32(e, _mm256_set1_epi32(0xF)),
        _mm256_cmpeq_epi32(m, _mm256_set1_epi32(0x7)));
    bits = _mm256_blendv_epi8(
        bits, _mm256_set1_epi32(static_cast<int>(0x7fc00000u)), is_nan);
    return _mm256_castsi256_ps(bits);
}

}  // namespace simd_detail

// Eight outputs (rows n..n+7) of `out = row @ W^T` with W e4m3 [rows, c_in]
// row-major: out[n+i] = sum_k row[k] * w[n+i][k], each lane's accumulation
// in the scalar reference's key order (bit-identical).
__attribute__((target("avx2")))
inline void e4m3_dot8(const float* row, int c_in,
                      const std::uint8_t* const* wrows, float* out8) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < c_in; ++k) {
        const __m256 w8 = simd_detail::decode8(
            wrows[0] + k, wrows[1] + k, wrows[2] + k, wrows[3] + k,
            wrows[4] + k, wrows[5] + k, wrows[6] + k, wrows[7] + k);
        const __m256 x8 = _mm256_set1_ps(row[k]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(w8, x8));
    }
    _mm256_storeu_ps(out8, acc);
}

#endif  // x86

}  // namespace dlss5_nr
