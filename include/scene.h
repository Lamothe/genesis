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

// One alive node's full record: identity, hierarchy link and local transform.
struct NodeRecord {
  NodeId id{kInvalidNode};
  NodeId parent{kInvalidNode};
  Transform local{};
};

inline bool sameRecord(NodeRecord const& a, NodeRecord const& b) {
  return a.id == b.id && a.parent == b.parent && sameTransform(a.local, b.local);
}

// Snapshot of every alive node, for undo/redo. Entries keep NodeId order
// (ascending) so snapshots compare deterministically.
struct SceneSnapshot {
  std::vector<NodeRecord> nodes;
};

inline bool sameSnapshot(SceneSnapshot const& a, SceneSnapshot const& b) {
  if (a.nodes.size() != b.nodes.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.nodes.size(); ++i) {
    if (!sameRecord(a.nodes[i], b.nodes[i])) {
      return false;
    }
  }
  return true;
}

// Transform hierarchy with cached world matrices. All transforms are TRS
// (translation, rotation, scale); nested non-uniform scale under rotation can
// introduce shear that TRS cannot represent, so the gizmo layer assumes
// shear-free hierarchies (see README.md).
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

  // Undo support: capture/restore every alive node (id, parent, local).
  // Restore is a faithful state restore: nodes destroyed since the snapshot
  // are RECREATED with their original ids (safe: ids are monotonic — create()
  // never reuses a tombstone), and nodes created since are destroyed. The
  // snapshot's parent links are rebuilt wholesale, so a restored hierarchy is
  // exactly the captured one. Restore trusts the snapshot to be acyclic (all
  // captured states are).
  SceneSnapshot snapshot() const;
  void restore(SceneSnapshot const& snapshot);

  // Monotonic counter, bumps on any structural or transform change. The neural
  // seam maps a change here to a temporal discontinuity (NEURAL-SEAM.md).
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
