#pragma once

#include <cstddef>
#include <vector>

#include "scene.h"

namespace vredit {

// Snapshot-based undo/redo, generic over the snapshot type (needs
// sameSnapshot(a, b) and copy semantics). The editor records the state before
// an edit with begin(), then commits it when the edit completes; undo()/redo()
// swap whole states through the caller-supplied current snapshot, which the
// caller then applies to its own state. Used with SceneSnapshot for the scene
// core and with the editor's full-state snapshot (scene + visuals + groups +
// selection) in the app, so structural edits — spawning or deleting models —
// are undoable exactly like transform drags.
template <typename Snapshot>
class UndoHistory {
public:
  static constexpr std::size_t kLimit = 200U;

  void begin(Snapshot const& before) {
    if (pendingOpen_) {
      // A previous edit never committed (e.g. a lost pointer release): commit
      // it against the new base rather than silently overwriting it.
      commit(before);
    }
    pending_ = before;
    pendingOpen_ = true;
  }

  bool commit(Snapshot const& current) {
    bool recorded = false;
    if (pendingOpen_ && !sameSnapshot(pending_, current)) {
      undo_.push_back(pending_);
      if (undo_.size() > kLimit) {
        undo_.erase(undo_.begin());
      }
      redo_.clear(); // a new edit invalidates the redo branch
      recorded = true;
    }
    pending_ = Snapshot{};
    pendingOpen_ = false;
    return recorded;
  }

  bool canUndo() const { return !undo_.empty(); }
  bool canRedo() const { return !redo_.empty(); }

  // Moves `current` (the live state) onto the redo stack and pops the top
  // undo state into it. The caller applies the returned snapshot. A dangling
  // (never-committed) edit is cancelled first — rolled back, not recorded —
  // so a half-finished state can never land on the redo stack.
  bool undo(Snapshot& current) {
    (void)cancel(current);
    if (undo_.empty()) {
      return false;
    }
    redo_.push_back(current);
    current = undo_.back();
    undo_.pop_back();
    return true;
  }

  bool redo(Snapshot& current) {
    (void)cancel(current);
    if (redo_.empty()) {
      return false;
    }
    undo_.push_back(current);
    current = redo_.back();
    redo_.pop_back();
    return true;
  }

  // Drops a pending (live) edit, restoring its pre-edit state into `current`
  // without recording a step. Both stacks are left exactly as they were, so
  // cancelling a grab neither eats history nor becomes redoable. False when
  // no edit is pending.
  bool cancel(Snapshot& current) {
    if (!pendingOpen_) {
      return false;
    }
    current = pending_;
    pending_ = Snapshot{};
    pendingOpen_ = false;
    return true;
  }

  void clear() {
    undo_.clear();
    redo_.clear();
    pending_ = Snapshot{};
    pendingOpen_ = false;
  }

private:
  std::vector<Snapshot> undo_;
  std::vector<Snapshot> redo_;
  Snapshot pending_;
  bool pendingOpen_{false};
};

} // namespace vredit
