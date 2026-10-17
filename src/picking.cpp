#include "picking.h"

#include <cmath>
#include <limits>

namespace vredit {

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
      if (!best.has_value() || boundT < best->t) {
        best = PickHit{candidate.node, boundT, ray.origin + ray.direction * boundT};
      }
      continue;
    }
    // Triangle pass in world space: distance must be world-comparable across
    // candidates, and transforming vertices keeps that exact under any scale.
    PickGeometry const& geometry = *candidate.geometry;
    float hitT = 0.0F;
    bool hit = false;
    for (std::size_t i = 0; i + 2U < geometry.indices.size(); i += 3U) {
      std::uint32_t i0 = geometry.indices[i];
      std::uint32_t i1 = geometry.indices[i + 1U];
      std::uint32_t i2 = geometry.indices[i + 2U];
      if (i0 >= geometry.positions.size() || i1 >= geometry.positions.size() ||
          i2 >= geometry.positions.size()) {
        continue; // never trust indices blindly
      }
      Vec3 a = transformPoint(candidate.world, geometry.positions[i0]);
      Vec3 b = transformPoint(candidate.world, geometry.positions[i1]);
      Vec3 c = transformPoint(candidate.world, geometry.positions[i2]);
      float t = 0.0F;
      if (!rayTriangle(ray, a, b, c, t)) {
        continue;
      }
      if (!hit || t < hitT) {
        hitT = t;
        hit = true;
      }
    }
    if (hit && (!best.has_value() || hitT < best->t)) {
      best = PickHit{candidate.node, hitT, ray.origin + ray.direction * hitT};
    }
  }
  return best;
}

} // namespace vredit
