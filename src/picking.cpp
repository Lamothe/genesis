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

} // namespace vredit
