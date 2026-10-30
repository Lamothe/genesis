#pragma once

#include <cstdint>

namespace vredit {

// Orders the triangles of an alpha-blended primitive back-to-front. Straight
// alpha compositing is order-dependent: drawing in index order lets near
// triangles paint over far ones that should show through them (hair cards are
// the usual victims). Painter's algorithm on triangle depth — the same rule
// Centre Stage settled on.
//
// `vertexDepth` is per-vertex view-space Z through exactly the transform chain
// the draw call uses (view space looks down -Z, so the most negative depth is
// farthest and sorts first). Triangle depth is the SUM of its three vertex
// depths (equivalent to the centroid mean for ordering). Ties keep the
// original triangle order, so coplanar cards never flicker between frames,
// and each triangle's three indices keep their winding.
//
// `sortedIndices` receives 3 * triCount indices; `keys` is caller scratch of
// at least 2 * triCount elements (the radix sort ping-pongs through it).
void sortTrianglesBackToFront(float const* vertexDepth, std::uint32_t const* indices,
                              std::uint32_t triCount, std::uint32_t* sortedIndices,
                              std::uint64_t* keys);

// The same ordering from precomputed per-triangle depth sums (the sum of the
// triangle's three vertex depths). Callers that transform whole meshes every
// frame precompute per-triangle position sums once and dot them with the view
// row per frame — no per-vertex indirection.
void sortTrianglesBackToFrontSums(float const* triDepthSums, std::uint32_t const* indices,
                                  std::uint32_t triCount, std::uint32_t* sortedIndices,
                                  std::uint64_t* keys);

} // namespace vredit
