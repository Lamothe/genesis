#pragma once

#include <cstdint>
#include <vector>

#include "vecmath.h"

namespace vredit {

using NodeId = std::uint32_t;
inline constexpr NodeId kInvalidNode = 0;

struct Transform {
  Vec3 translation{};
  Quat rotation{};
  Vec3 scale{1.0F, 1.0F, 1.0F};
};

inline Mat4 toMatrix(Transform const& t) {
  return trs(t.translation, t.rotation, t.scale);
}

inline Transform toTransform(Mat4 const& m) {
  Transform t;
  decompose(m, t.translation, t.rotation, t.scale);
  return t;
}

// Transform hierarchy with cached world matrices. All transforms are TRS
// (translation, rotation, scale); nested non-uniform scale under rotation can
// introduce shear that TRS cannot represent, so the gizmo layer assumes
// shear-free hierarchies (see vr/editor/README.md).
class Scene {
public:
  Scene();

  NodeId create(NodeId parent = kInvalidNode);
  void destroy(NodeId id);                // destroys the whole subtree
  void setParent(NodeId id, NodeId parent); // keeps the local transform; rejects cycles

  bool valid(NodeId id) const;
  NodeId parent(NodeId id) const;

  Transform const& local(NodeId id) const;
  void setLocal(NodeId id, Transform const& t); // no-op when exactly unchanged

  Mat4 world(NodeId id);
  Vec3 worldPosition(NodeId id);

  // Monotonic counter, bumps on any structural or transform change. The neural
  // seam maps a change here to a temporal discontinuity (vr/NEURAL-SEAM.md).
  std::uint64_t epoch() const { return epoch_; }

private:
  struct Slot {
    bool alive{false};
    NodeId parent{kInvalidNode};
    std::vector<NodeId> children;
    Transform local{};
    Mat4 world{};
    bool worldDirty{true};
  };

  std::vector<Slot> slots_; // index 0 reserved for kInvalidNode
  std::uint64_t epoch_{0};

  void markSubtreeDirty(NodeId id);
  void ensureWorld(NodeId id);
};

} // namespace vredit
