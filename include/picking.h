#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "scene.h"

namespace vredit {

struct Ray {
  Vec3 origin{};
  Vec3 direction{}; // expected normalised
};

struct Aabb {
  Vec3 min{};
  Vec3 max{};
};

bool rayAabb(Ray const& ray, Aabb const& box, float& tOut);
bool rayTriangle(Ray const& ray, Vec3 const& a, Vec3 const& b, Vec3 const& c, float& tOut);

struct PickHit {
  NodeId node{kInvalidNode};
  float t{0.0F};
  Vec3 point{};
};

std::optional<PickHit> pickClosest(Ray const& ray,
                                   std::vector<std::pair<NodeId, Aabb>> const& candidates);

} // namespace vredit
