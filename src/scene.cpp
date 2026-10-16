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
