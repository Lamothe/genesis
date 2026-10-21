#include "scene.h"

#include <algorithm>
#include <utility>

namespace vredit {

namespace {

Transform const& emptyTransform() {
  static Transform const empty{};
  return empty;
}

} // namespace

Scene::Scene() {
  slots_.emplace_back(); // index 0 = kInvalidNode
}

NodeId Scene::create(NodeId parent) {
  if (!valid(parent)) {
    parent = kInvalidNode;
  }
  Slot slot;
  slot.alive = true;
  slot.parent = parent;
  slots_.push_back(std::move(slot));
  NodeId id = static_cast<NodeId>(slots_.size() - 1U);
  if (parent != kInvalidNode) {
    slots_[parent].children.push_back(id);
  }
  ++epoch_;
  return id;
}

void Scene::destroy(NodeId id) {
  if (!valid(id)) {
    return;
  }
  Slot& slot = slots_[id];
  if (slot.parent != kInvalidNode && valid(slot.parent)) {
    auto& siblings = slots_[slot.parent].children;
    siblings.erase(std::remove(siblings.begin(), siblings.end(), id), siblings.end());
  }
  std::vector<NodeId> children = slot.children;
  slot.children.clear();
  for (NodeId child : children) {
    slots_[child].parent = kInvalidNode;
    destroy(child);
  }
  slot.alive = false;
  ++epoch_;
}

void Scene::setParent(NodeId id, NodeId parent) {
  if (!valid(id)) {
    return;
  }
  if (parent != kInvalidNode && !valid(parent)) {
    return;
  }
  if (parent == id) {
    return;
  }
  for (NodeId p = parent; p != kInvalidNode; p = slots_[p].parent) {
    if (p == id) {
      return; // would create a cycle
    }
  }
  Slot& slot = slots_[id];
  if (slot.parent == parent) {
    return;
  }
  if (slot.parent != kInvalidNode && valid(slot.parent)) {
    auto& siblings = slots_[slot.parent].children;
    siblings.erase(std::remove(siblings.begin(), siblings.end(), id), siblings.end());
  }
  slot.parent = parent;
  if (parent != kInvalidNode) {
    slots_[parent].children.push_back(id);
  }
  markSubtreeDirty(id);
  ++epoch_;
}

bool Scene::valid(NodeId id) const {
  return id != kInvalidNode && id < slots_.size() && slots_[id].alive;
}

NodeId Scene::parent(NodeId id) const {
  return valid(id) ? slots_[id].parent : kInvalidNode;
}

std::vector<NodeId> Scene::descendants(NodeId id) const {
  std::vector<NodeId> out;
  if (!valid(id)) {
    return out;
  }
  // Iterative depth-first over the private children links.
  std::vector<NodeId> stack = slots_[id].children;
  while (!stack.empty()) {
    NodeId const n = stack.back();
    stack.pop_back();
    if (!valid(n)) {
      continue;
    }
    out.push_back(n);
    for (NodeId const c : slots_[n].children) {
      stack.push_back(c);
    }
  }
  return out;
}

Transform const& Scene::local(NodeId id) const {
  if (!valid(id)) {
    return emptyTransform();
  }
  return slots_[id].local;
}

void Scene::setLocal(NodeId id, Transform const& t) {
  if (!valid(id)) {
    return;
  }
  Slot& slot = slots_[id];
  bool same = slot.local.translation.x == t.translation.x &&
              slot.local.translation.y == t.translation.y &&
              slot.local.translation.z == t.translation.z &&
              slot.local.rotation.x == t.rotation.x && slot.local.rotation.y == t.rotation.y &&
              slot.local.rotation.z == t.rotation.z && slot.local.rotation.w == t.rotation.w &&
              slot.local.scale.x == t.scale.x && slot.local.scale.y == t.scale.y &&
              slot.local.scale.z == t.scale.z;
  if (same) {
    return;
  }
  slot.local = t;
  markSubtreeDirty(id);
  ++epoch_;
}

Mat4 Scene::world(NodeId id) {
  if (!valid(id)) {
    return Mat4{};
  }
  ensureWorld(id);
  return slots_[id].world;
}

Vec3 Scene::worldPosition(NodeId id) {
  return transformPoint(world(id), Vec3{});
}

SceneSnapshot Scene::snapshot() const {
  SceneSnapshot snapshot;
  for (NodeId id = 1; id < static_cast<NodeId>(slots_.size()); ++id) {
    if (slots_[id].alive) {
      snapshot.nodes.push_back(NodeRecord{id, slots_[id].parent, slots_[id].local});
    }
  }
  return snapshot;
}

void Scene::restore(SceneSnapshot const& snapshot) {
  // Dedupe by id (first wins) and find the largest id to restore.
  std::vector<NodeRecord> records = snapshot.nodes;
  std::sort(records.begin(), records.end(),
            [](NodeRecord const& a, NodeRecord const& b) { return a.id < b.id; });
  records.erase(std::unique(records.begin(), records.end(),
                            [](NodeRecord const& a, NodeRecord const& b) {
                              return a.id == b.id;
                            }),
                records.end());
  NodeId maxId = 0;
  for (NodeRecord const& record : records) {
    if (record.id != kInvalidNode && record.id > maxId) {
      maxId = record.id;
    }
  }
  if (static_cast<std::size_t>(maxId) + 1U > slots_.size()) {
    slots_.resize(static_cast<std::size_t>(maxId) + 1U);
  }
  std::vector<bool> keep(slots_.size(), false);
  for (NodeRecord const& record : records) {
    if (record.id != kInvalidNode) {
      keep[record.id] = true;
    }
  }
  // Nodes created since the snapshot go away.
  for (NodeId id = 1; id < static_cast<NodeId>(slots_.size()); ++id) {
    if (slots_[id].alive && !keep[id]) {
      slots_[id].alive = false;
    }
  }
  // Revive or create the recorded nodes in place: tombstones are reused at
  // their original ids (safe — create() never recycles an id), and world
  // caches are dropped wholesale below.
  for (NodeRecord const& record : records) {
    if (record.id == kInvalidNode) {
      continue;
    }
    Slot& slot = slots_[record.id];
    slot.alive = true;
    slot.local = record.local;
    slot.worldDirty = true;
    slot.children.clear();
    slot.parent = kInvalidNode;
  }
  // Rebuild parent links and children lists wholesale from the records; a
  // record whose parent is not part of the snapshot is detached to root.
  for (NodeRecord const& record : records) {
    if (record.id == kInvalidNode) {
      continue;
    }
    NodeId parent = record.parent;
    if (parent != kInvalidNode &&
        (parent >= static_cast<NodeId>(slots_.size()) || !slots_[parent].alive)) {
      parent = kInvalidNode;
    }
    slots_[record.id].parent = parent;
    if (parent != kInvalidNode) {
      slots_[parent].children.push_back(record.id);
    }
  }
  ++epoch_;
}

void Scene::markSubtreeDirty(NodeId id) {
  std::vector<NodeId> pending{id};
  while (!pending.empty()) {
    NodeId current = pending.back();
    pending.pop_back();
    Slot& slot = slots_[current];
    slot.worldDirty = true;
    for (NodeId child : slot.children) {
      pending.push_back(child);
    }
  }
}

void Scene::ensureWorld(NodeId id) {
  Slot& slot = slots_[id];
  if (!slot.worldDirty) {
    return;
  }
  Mat4 localMatrix = toMatrix(slot.local);
  if (slot.parent != kInvalidNode && valid(slot.parent)) {
    ensureWorld(slot.parent);
    slot.world = mul(slots_[slot.parent].world, localMatrix);
  } else {
    slot.world = localMatrix;
  }
  slot.worldDirty = false;
}

} // namespace vredit
