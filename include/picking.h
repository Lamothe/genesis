#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "scene.h"

namespace vredit {

bool rayAabb(Ray const& ray, Aabb const& box, float& tOut);
bool rayTriangle(Ray const& ray, Vec3 const& a, Vec3 const& b, Vec3 const& c, float& tOut);

struct PickHit {
  NodeId node{kInvalidNode};
  float t{0.0F};
  Vec3 point{};
};

std::optional<PickHit> pickClosest(Ray const& ray,
                                   std::vector<std::pair<NodeId, Aabb>> const& candidates);

// Exact picking geometry in node-local space (positions + triangle indices).
struct PickGeometry {
  std::vector<Vec3> positions;
  std::vector<std::uint32_t> indices;
};

// A pickable visual: world bounds for broadphase plus optional exact geometry.
struct PickCandidate {
  NodeId node{kInvalidNode};
  Aabb worldBounds{};
  Mat4 world{}; // local -> world
  std::shared_ptr<const PickGeometry> geometry; // null: the bounds are the shape
};

// Closest-*surface* pick: compares true distance along the ray, so a small
// object visible in front of a larger one behind it wins even when the larger
// one's bounding box is entered first (box entry distance is not surface
// distance — that mismatch is what made selection feel wrong). Candidates
// without geometry fall back to their bounds entry, which is exact for
// box-shaped meshes such as the built-in cube.
std::optional<PickHit> pickClosestSurface(Ray const& ray,
                                          std::vector<PickCandidate> const& candidates);

} // namespace vredit
