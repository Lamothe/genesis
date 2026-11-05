#include "picking.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

namespace vredit {

// Flat, immutable triangle hierarchy. Node bounds are local-space unions of
// exact triangle bounds; the per-candidate world transform is applied to the
// bounds during traversal. That keeps every accepted triangle on the original
// world-space path (same transformPoint + rayTriangle), so the tree can only
// reject work, never change which triangle a ray hits.
struct PickBvhNode {
  Aabb bounds{};
  std::uint32_t begin{0}; // leaf: first entry in PickBvh::triangles
  std::uint32_t count{0}; // leaf: entry count; 0 = internal node
  std::uint32_t left{0};  // internal: left child index
  std::uint32_t right{0}; // internal: right child index
};

struct PickBvh {
  std::vector<PickBvhNode> nodes;
  std::vector<std::uint32_t> triangles; // offsets into PickGeometry::indices
  float expansion{0.0F}; // conservative local padding against slab rounding
};

bool rayAabb(Ray const& ray, Aabb const& box, float& tOut) {
  float tMin = 0.0F;
  float tMax = std::numeric_limits<float>::max();
  float const origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
  float const direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
  float const lo[3] = {box.min.x, box.min.y, box.min.z};
  float const hi[3] = {box.max.x, box.max.y, box.max.z};
  for (int i = 0; i < 3; ++i) {
    if (std::fabs(direction[i]) < 1e-12F) {
      if (origin[i] < lo[i] || origin[i] > hi[i]) {
        return false;
      }
      continue;
    }
    float inv = 1.0F / direction[i];
    float t0 = (lo[i] - origin[i]) * inv;
    float t1 = (hi[i] - origin[i]) * inv;
    if (t0 > t1) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    if (t0 > tMin) {
      tMin = t0;
    }
    if (t1 < tMax) {
      tMax = t1;
    }
    if (tMin > tMax) {
      return false;
    }
  }
  tOut = tMin;
  return true;
}

bool rayTriangle(Ray const& ray, Vec3 const& a, Vec3 const& b, Vec3 const& c, float& tOut) {
  float const eps = 1e-7F;
  Vec3 e1 = b - a;
  Vec3 e2 = c - a;
  Vec3 p = cross(ray.direction, e2);
  float det = dot(e1, p);
  if (std::fabs(det) < eps) {
    return false;
  }
  float invDet = 1.0F / det;
  Vec3 s = ray.origin - a;
  float u = dot(s, p) * invDet;
  if (u < 0.0F || u > 1.0F) {
    return false;
  }
  Vec3 q = cross(s, e1);
  float v = dot(ray.direction, q) * invDet;
  if (v < 0.0F || u + v > 1.0F) {
    return false;
  }
  float t = dot(e2, q) * invDet;
  if (t < 0.0F) {
    return false;
  }
  tOut = t;
  return true;
}

namespace {

constexpr std::uint32_t kLeafTriangles = 1024U;

// One valid triangle during the build: its index offset and centroid. Sorting
// this array (not the geometry) is what makes the tree cheap to build: the
// centroid is computed once, and leaf bounds are the only triangle-bound work.
struct BuildTriangle {
  std::uint32_t offset{0};
  Vec3 centre{};
};

Aabb triangleBounds(PickGeometry const& geometry, std::uint32_t offset) {
  Vec3 const a = geometry.positions[geometry.indices[offset]];
  Vec3 const b = geometry.positions[geometry.indices[offset + 1U]];
  Vec3 const c = geometry.positions[geometry.indices[offset + 2U]];
  return {{std::fmin(a.x, std::fmin(b.x, c.x)), std::fmin(a.y, std::fmin(b.y, c.y)),
           std::fmin(a.z, std::fmin(b.z, c.z))},
          {std::fmax(a.x, std::fmax(b.x, c.x)), std::fmax(a.y, std::fmax(b.y, c.y)),
           std::fmax(a.z, std::fmax(b.z, c.z))}};
}

std::uint32_t buildNode(PickGeometry const& geometry, PickBvh& bvh,
                        std::vector<BuildTriangle>& work,
                        std::uint32_t begin, std::uint32_t end) {
  std::uint32_t const index = static_cast<std::uint32_t>(bvh.nodes.size());
  bvh.nodes.push_back(PickBvhNode{});
  if (end - begin <= kLeafTriangles) {
    Aabb bounds = triangleBounds(geometry, work[begin].offset);
    for (std::uint32_t i = begin + 1U; i < end; ++i) {
      bounds = unite(bounds, triangleBounds(geometry, work[i].offset));
    }
    PickBvhNode& node = bvh.nodes[index];
    node.bounds = bounds;
    node.begin = begin;
    node.count = end - begin;
    return index;
  }
  Aabb centreBounds{work[begin].centre, work[begin].centre};
  for (std::uint32_t i = begin + 1U; i < end; ++i) {
    Vec3 const& c = work[i].centre;
    centreBounds.min = {std::fmin(centreBounds.min.x, c.x),
                        std::fmin(centreBounds.min.y, c.y),
                        std::fmin(centreBounds.min.z, c.z)};
    centreBounds.max = {std::fmax(centreBounds.max.x, c.x),
                        std::fmax(centreBounds.max.y, c.y),
                        std::fmax(centreBounds.max.z, c.z)};
  }
  Vec3 const extent = centreBounds.max - centreBounds.min;
  int const axis = extent.x >= extent.y && extent.x >= extent.z ? 0
                 : extent.y >= extent.z ? 1 : 2;
  std::uint32_t const mid = begin + (end - begin) / 2U;
  std::nth_element(work.begin() + begin, work.begin() + mid, work.begin() + end,
                   [axis](BuildTriangle const& a, BuildTriangle const& b) {
                     float const av = axis == 0 ? a.centre.x : axis == 1 ? a.centre.y : a.centre.z;
                     float const bv = axis == 0 ? b.centre.x : axis == 1 ? b.centre.y : b.centre.z;
                     return av < bv;
                   });
  std::uint32_t const left = buildNode(geometry, bvh, work, begin, mid);
  std::uint32_t const right = buildNode(geometry, bvh, work, mid, end);
  PickBvhNode& node = bvh.nodes[index];
  node.bounds = unite(bvh.nodes[left].bounds, bvh.nodes[right].bounds);
  node.left = left;
  node.right = right;
  node.count = 0U;
  return index;
}

PickBvh const* pickBvhFor(PickGeometry const& geometry) {
  if (!geometry.bvh && !geometry.bvhUnavailable) {
    try {
      auto bvh = std::make_shared<PickBvh>();
      std::vector<BuildTriangle> work;
      work.reserve(geometry.indices.size() / 3U);
      for (std::size_t i = 0; i + 2U < geometry.indices.size(); i += 3U) {
        std::uint32_t const i0 = geometry.indices[i];
        std::uint32_t const i1 = geometry.indices[i + 1U];
        std::uint32_t const i2 = geometry.indices[i + 2U];
        if (i0 >= geometry.positions.size() || i1 >= geometry.positions.size() ||
            i2 >= geometry.positions.size()) {
          continue; // never trust indices blindly
        }
        Vec3 const& a = geometry.positions[i0];
        Vec3 const& b = geometry.positions[i1];
        Vec3 const& c = geometry.positions[i2];
        if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z) ||
            !std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.z) ||
            !std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z)) {
          continue; // a non-finite triangle cannot be picked reliably
        }
        work.push_back({static_cast<std::uint32_t>(i),
                        a * (1.0F / 3.0F) + b * (1.0F / 3.0F) + c * (1.0F / 3.0F)});
      }
      if (!work.empty()) {
        bvh->nodes.reserve(2U * (work.size() / (kLeafTriangles / 2U) + 1U));
        bvh->triangles.reserve(work.size());
        buildNode(geometry, *bvh, work, 0U, static_cast<std::uint32_t>(work.size()));
        for (BuildTriangle const& triangle : work) {
          bvh->triangles.push_back(triangle.offset);
        }
        Aabb const& root = bvh->nodes[0].bounds;
        Vec3 const extent = root.max - root.min;
        float const longest = std::fmax(extent.x, std::fmax(extent.y, extent.z));
        // Local padding: enough to absorb slab/transform rounding at any node,
        // small enough to stay a real rejection test. Never used for the exact
        // triangle test itself.
        bvh->expansion = std::fmax(1e-4F, longest * 1e-5F);
      }
      geometry.bvh = std::move(bvh);
    } catch (std::exception const& e) {
      // A hostile or enormous mesh must not terminate the render loop. Keep
      // the old exact full scan for this geometry; it is slower, not wrong.
      geometry.bvhUnavailable = true;
      geometry.bvh.reset();
      std::fprintf(stderr, "[pick] BVH build failed (%s); using a full scan\n", e.what());
    }
  }
  return geometry.bvh.get();
}

Aabb worldNodeBounds(PickBvh const& bvh, PickBvhNode const& node, Mat4 const& world) {
  Vec3 const pad{bvh.expansion, bvh.expansion, bvh.expansion};
  Aabb box = transformAabb(world, Aabb{node.bounds.min - pad, node.bounds.max + pad});
  // Also pad in WORLD units by a fraction of the coordinate magnitude: at a
  // large translation, float rounding in transformPoint can exceed the local
  // pad, and a silhouette-grazing ray must not be pruned at a node whose
  // triangles the full scan would hit. This is rejection padding only.
  float const magnitude = std::fmax(
      std::fmax(std::fabs(box.min.x), std::fabs(box.max.x)),
      std::fmax(std::fmax(std::fabs(box.min.y), std::fabs(box.max.y)),
                std::fmax(std::fabs(box.min.z), std::fabs(box.max.z))));
  float const worldPad = std::fmax(1e-5F, magnitude * 1e-6F);
  Vec3 const worldPadding{worldPad, worldPad, worldPad};
  return {box.min - worldPadding, box.max + worldPadding};
}

void pickBvhNode(PickGeometry const& geometry, PickBvh const& bvh, Ray const& ray,
                 Mat4 const& world, std::uint32_t index, float entry, float& bestT) {
  if (entry >= bestT) {
    return;
  }
  PickBvhNode const& node = bvh.nodes[index];
  if (node.count != 0U) {
    for (std::uint32_t i = 0; i < node.count; ++i) {
      std::uint32_t const offset = bvh.triangles[node.begin + i];
      Vec3 const a = transformPoint(world, geometry.positions[geometry.indices[offset]]);
      Vec3 const b = transformPoint(world, geometry.positions[geometry.indices[offset + 1U]]);
      Vec3 const c = transformPoint(world, geometry.positions[geometry.indices[offset + 2U]]);
      float t = 0.0F;
      if (rayTriangle(ray, a, b, c, t) && t < bestT) {
        bestT = t;
      }
    }
    return;
  }
  float leftEntry = 0.0F;
  float rightEntry = 0.0F;
  bool const leftHits = rayAabb(ray, worldNodeBounds(bvh, bvh.nodes[node.left], world), leftEntry);
  bool const rightHits = rayAabb(ray, worldNodeBounds(bvh, bvh.nodes[node.right], world), rightEntry);
  if (leftHits && rightHits) {
    // Visit the nearer child first so the farther one can be pruned by its
    // entry distance once the nearer subtree tightens bestT.
    if (leftEntry <= rightEntry) {
      pickBvhNode(geometry, bvh, ray, world, node.left, leftEntry, bestT);
      pickBvhNode(geometry, bvh, ray, world, node.right, rightEntry, bestT);
    } else {
      pickBvhNode(geometry, bvh, ray, world, node.right, rightEntry, bestT);
      pickBvhNode(geometry, bvh, ray, world, node.left, leftEntry, bestT);
    }
  } else if (leftHits) {
    pickBvhNode(geometry, bvh, ray, world, node.left, leftEntry, bestT);
  } else if (rightHits) {
    pickBvhNode(geometry, bvh, ray, world, node.right, rightEntry, bestT);
  }
}

void pickGeometryScan(PickGeometry const& geometry, Ray const& ray, Mat4 const& world,
                      float& bestT) {
  for (std::size_t i = 0; i + 2U < geometry.indices.size(); i += 3U) {
    std::uint32_t const i0 = geometry.indices[i];
    std::uint32_t const i1 = geometry.indices[i + 1U];
    std::uint32_t const i2 = geometry.indices[i + 2U];
    if (i0 >= geometry.positions.size() || i1 >= geometry.positions.size() ||
        i2 >= geometry.positions.size()) {
      continue;
    }
    Vec3 const a = transformPoint(world, geometry.positions[i0]);
    Vec3 const b = transformPoint(world, geometry.positions[i1]);
    Vec3 const c = transformPoint(world, geometry.positions[i2]);
    float t = 0.0F;
    if (rayTriangle(ray, a, b, c, t) && t < bestT) {
      bestT = t;
    }
  }
}

void pickGeometryBvh(PickGeometry const& geometry, PickBvh const& bvh, Ray const& ray,
                     Mat4 const& world, float& bestT) {
  if (bvh.nodes.empty()) {
    return;
  }
  float entry = 0.0F;
  if (!rayAabb(ray, worldNodeBounds(bvh, bvh.nodes[0], world), entry)) {
    return;
  }
  pickBvhNode(geometry, bvh, ray, world, 0U, entry, bestT);
}

} // namespace

std::optional<PickHit> pickClosest(Ray const& ray,
                                   std::vector<std::pair<NodeId, Aabb>> const& candidates) {
  std::optional<PickHit> best;
  for (auto const& candidate : candidates) {
    float t = 0.0F;
    if (!rayAabb(ray, candidate.second, t)) {
      continue;
    }
    if (!best.has_value() || t < best->t) {
      best = PickHit{candidate.first, t, ray.origin + ray.direction * t};
    }
  }
  return best;
}

std::optional<PickHit> pickClosestSurface(Ray const& ray,
                                          std::vector<PickCandidate> const& candidates) {
  std::optional<PickHit> best;
  for (auto const& candidate : candidates) {
    float boundT = 0.0F;
    if (!rayAabb(ray, candidate.worldBounds, boundT)) {
      continue;
    }
    // Bounds entry is a lower bound for any hit inside them, so nothing in
    // this candidate can beat a closer hit already found.
    if (best.has_value() && boundT >= best->t) {
      continue;
    }
    if (!candidate.geometry || candidate.geometry->positions.empty() ||
        candidate.geometry->indices.size() < 3U) {
      // Box-shaped contract: null geometry means "the bounds ARE the shape"
      // (demo cubes), and clicking a visible cube face must select it. The
      // empty-space fix is at the call site: no surface-miss bounds fallback.
      if (!best.has_value() || boundT < best->t) {
        best = PickHit{candidate.node, boundT, ray.origin + ray.direction * boundT};
      }
      continue;
    }
    // Triangle pass in world space: distance must be world-comparable across
    // candidates, and transforming vertices keeps that exact under any scale.
    // The BVH only rejects triangles whose node bounds the ray misses; every
    // accepted triangle takes exactly the old full-scan path below.
    PickGeometry const& geometry = *candidate.geometry;
    float const limit = best.has_value() ? best->t : std::numeric_limits<float>::max();
    float hitT = limit;
    PickBvh const* bvh = pickBvhFor(geometry);
    if (bvh != nullptr) {
      pickGeometryBvh(geometry, *bvh, ray, candidate.world, hitT);
    } else {
      pickGeometryScan(geometry, ray, candidate.world, hitT);
    }
    if (hitT < limit) {
      best = PickHit{candidate.node, hitT, ray.origin + ray.direction * hitT};
    }
  }
  return best;
}

std::vector<NodeId> expandSelection(std::vector<std::pair<NodeId, std::uint32_t>> const& groups,
                                    NodeId picked,
                                    bool partOnly) {
  std::vector<NodeId> out;
  if (partOnly) {
    out.push_back(picked);
    return out;
  }
  std::uint32_t group = 0;
  bool found = false;
  for (auto const& entry : groups) {
    if (entry.first == picked) {
      group = entry.second;
      found = true;
      break;
    }
  }
  if (!found || group == 0) {
    out.push_back(picked); // ungrouped nodes stay individually selectable
    return out;
  }
  for (auto const& entry : groups) {
    if (entry.second == group) {
      out.push_back(entry.first);
    }
  }
  return out;
}

} // namespace vredit
