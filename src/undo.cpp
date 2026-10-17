#include "undo.h"

namespace vredit {

void UndoHistory::begin(SceneSnapshot const& before) {
  if (pendingOpen_) {
    // A previous edit never committed (e.g. a lost pointer release): commit it
    // against the new base rather than silently overwriting it.
    commit(before);
  }
  pending_ = before;
  pendingOpen_ = true;
}

bool UndoHistory::commit(SceneSnapshot const& current) {
  bool recorded = false;
  if (pendingOpen_ && !sameSnapshot(pending_, current)) {
    undo_.push_back(pending_);
    if (undo_.size() > kLimit) {
      undo_.erase(undo_.begin());
    }
    redo_.clear(); // a new edit invalidates the redo branch
    recorded = true;
  }
  pending_ = SceneSnapshot{};
  pendingOpen_ = false;
  return recorded;
}

bool UndoHistory::undo(Scene& scene) {
  pending_ = SceneSnapshot{};
  pendingOpen_ = false;
  if (undo_.empty()) {
    return false;
  }
  redo_.push_back(scene.snapshotTransforms());
  scene.restoreTransforms(undo_.back());
  undo_.pop_back();
  return true;
}

bool UndoHistory::redo(Scene& scene) {
  pending_ = SceneSnapshot{};
  pendingOpen_ = false;
  if (redo_.empty()) {
    return false;
  }
  undo_.push_back(scene.snapshotTransforms());
  scene.restoreTransforms(redo_.back());
  redo_.pop_back();
  return true;
}

bool UndoHistory::cancel(Scene& scene) {
  if (!pendingOpen_) {
    return false;
  }
  scene.restoreTransforms(pending_);
  pending_ = SceneSnapshot{};
  pendingOpen_ = false;
  return true;
}

void UndoHistory::clear() {
  undo_.clear();
  redo_.clear();
  pending_ = SceneSnapshot{};
  pendingOpen_ = false;
}

} // namespace vredit
