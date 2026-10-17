#pragma once

#include <cstdint>
#include <utility>
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

inline bool sameTransform(Transform const& a, Transform const& b) {
  return a.translation.x == b.translation.x && a.translation.y == b.translation.y &&
         a.translation.z == b.translation.z && a.rotation.x == b.rotation.x &&
         a.rotation.y == b.rotation.y && a.rotation.z == b.rotation.z &&
         a.rotation.w == b.rotation.w && a.scale.x == b.scale.x && a.scale.y == b.scale.y &&
         a.scale.z == b.scale.z;
}

// Snapshot of every alive node's local transform, for undo/redo. Entries keep
// NodeId order (ascending) so snapshots compare deterministically.
struct SceneSnapshot {
  std::vector<std::pair<NodeId, Transform>> transforms;
};

inline bool sameSnapshot(SceneSnapshot const& a, SceneSnapshot const& b) {
  if (a.transforms.size() != b.transforms.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.transforms.size(); ++i) {
    if (a.transforms[i].first != b.transforms[i].first ||
        !sameTransform(a.transforms[i].second, b.transforms[i].second)) {
      return false;
    }
  }
  return true;
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

  // Undo support: capture/restore all local transforms. Restore skips nodes
  // destroyed since the snapshot was taken.
  SceneSnapshot snapshotTransforms() const;
  void restoreTransforms(SceneSnapshot const& snapshot);

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
