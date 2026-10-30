// Painter's-algorithm triangle ordering for alpha-blended primitives.

#include "triangle_sort.h"

#include <cstring>

namespace vredit {
namespace {

// A depth (the triangle's vertex-depth sum) and its original triangle index
// packed into one key: float bits made monotone (negative floats invert) in
// the high word, triangle index in the low word. UNSIGNED comparison is
// essential — depths on both sides of zero are normal when the camera sits
// inside the geometry, and a signed sort would invert those groups.
inline std::uint64_t packKey(float depth, std::uint32_t triangle) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &depth, sizeof(bits));
  std::uint32_t const sortable =
      (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
  return (static_cast<std::uint64_t>(sortable) << 32) | triangle;
}

// Stable LSD radix sort of the packed keys by their high word (the monotone
// depth word); the low word (the original triangle index) rides along as the
// tie-break, and stability preserves it. Radix rather than std::sort because
// this runs per blend mesh, per view, per frame — 90K hair triangles is the
// real case, and introsort's random-access compares cost milliseconds there.
// `keys` must hold 2 * triCount entries: the passes ping-pong through the
// second half. Two passes because the high word is what orders the keys.
void radixSortByHighWord(std::uint64_t* keys, std::uint32_t triCount) {
  std::uint64_t* src = keys;
  std::uint64_t* dst = keys + triCount;
  // thread_local keeps the 256 KB histogram off the caller's stack (this runs
  // inside the draw path).
  static thread_local std::uint32_t counts[65536];
  for (std::uint32_t pass = 0; pass < 2U; ++pass) {
    std::uint32_t const shift = pass == 0U ? 32U : 48U; // low half first: LSD order
    std::memset(counts, 0, sizeof(counts));
    for (std::uint32_t t = 0; t < triCount; ++t) {
      ++counts[(src[t] >> shift) & 0xFFFFU];
    }
    std::uint32_t total = 0;
    for (std::size_t i = 0; i < 65536U; ++i) {
      std::uint32_t const count = counts[i];
      counts[i] = total;
      total += count;
    }
    for (std::uint32_t t = 0; t < triCount; ++t) {
      dst[counts[(src[t] >> shift) & 0xFFFFU]++] = src[t];
    }
    std::uint64_t* const swap = src;
    src = dst;
    dst = swap;
  }
  // Two passes: the sorted result lands back in `keys`.
}

void sortRewrite(std::uint32_t const* indices, std::uint32_t triCount, std::uint32_t* sortedIndices,
                 std::uint64_t* keys) {
  radixSortByHighWord(keys, triCount);
  for (std::uint32_t t = 0; t < triCount; ++t) {
    std::uint32_t const src = 3U * static_cast<std::uint32_t>(keys[t] & 0xFFFFFFFFULL);
    sortedIndices[3U * t] = indices[src];
    sortedIndices[3U * t + 1U] = indices[src + 1U];
    sortedIndices[3U * t + 2U] = indices[src + 2U];
  }
}

} // namespace

void sortTrianglesBackToFront(float const* vertexDepth, std::uint32_t const* indices,
                              std::uint32_t triCount, std::uint32_t* sortedIndices,
                              std::uint64_t* keys) {
  for (std::uint32_t t = 0; t < triCount; ++t) {
    std::uint32_t const i0 = indices[3U * t];
    std::uint32_t const i1 = indices[3U * t + 1U];
    std::uint32_t const i2 = indices[3U * t + 2U];
    // Triangle depth is the SUM of its three vertex depths (equivalent to the
    // centroid mean for ordering); see the header for the view-space rule.
    float const depth = vertexDepth[i0] + vertexDepth[i1] + vertexDepth[i2];
    keys[t] = packKey(depth, t);
  }
  sortRewrite(indices, triCount, sortedIndices, keys);
}

void sortTrianglesBackToFrontSums(float const* triDepthSums, std::uint32_t const* indices,
                                  std::uint32_t triCount, std::uint32_t* sortedIndices,
                                  std::uint64_t* keys) {
  for (std::uint32_t t = 0; t < triCount; ++t) {
    keys[t] = packKey(triDepthSums[t], t);
  }
  sortRewrite(indices, triCount, sortedIndices, keys);
}

} // namespace vredit
