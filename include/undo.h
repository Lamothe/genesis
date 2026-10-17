#pragma once

#include <cstddef>
#include <vector>

#include "scene.h"

namespace vredit {

// Snapshot-based undo/redo for scene transforms. The editor records the state
// before an edit (a gizmo drag) with begin(), then commits it when the edit
// completes; undo()/redo() swap whole states in the scene. Selection and
// camera are deliberately not undoable — scene transforms only.
class UndoHistory {
public:
  static constexpr std::size_t kLimit = 200U;

  void begin(SceneSnapshot const& before);
  bool commit(SceneSnapshot const& current); // true when a step was recorded
  bool canUndo() const { return !undo_.empty(); }
  bool canRedo() const { return !redo_.empty(); }
  bool undo(Scene& scene); // false when nothing to undo
  bool redo(Scene& scene); // false when nothing to redo
  void clear();

private:
  std::vector<SceneSnapshot> undo_;
  std::vector<SceneSnapshot> redo_;
  SceneSnapshot pending_;
  bool pendingOpen_{false};
};

} // namespace vredit
