#pragma once

#include <cstdint>
#include <vector>

#include "picking.h"
#include "scene.h"

namespace vredit {

enum class GizmoMode : std::uint8_t { Translate, Rotate, Scale };
enum class GizmoSpace : std::uint8_t { World, Local };

// Handle expressed in the gizmo's own frame (World: world axes; Local:
// `orientation`). The renderer draws the handles and resolves which one was
// grabbed; this layer only does the drag math.
enum class GizmoHandle : std::uint8_t {
  Free,
  AxisX,
  AxisY,
  AxisZ,
  PlaneXY,
  PlaneYZ,
  PlaneZX,
};

struct GizmoSettings {
  GizmoMode mode{GizmoMode::Translate};
  GizmoSpace space{GizmoSpace::World};
  float snapStep{0.0F}; // Translate: metres; Rotate: radians; Scale: factor increments. 0 = off.
};

// Drift-free gizmo drag over a selection: every update recomputes the cumulative
// delta from the drag start and applies it to the captured start transforms, so
// repeated updates cannot accumulate rounding drift.
//
// The gizmo frame is decided by `settings.space`: World uses world axes and the
// world grid (the passed `orientation` is ignored); Local uses `orientation` for
// handle directions and for the snapping grid. Snapped free/plane deltas are
// built from the drag-plane basis, so they stay exactly in the drag plane.
class GizmoDrag {
public:
  bool begin(Scene& scene,
             std::vector<NodeId> const& selection,
             Vec3 pivot,
             Quat orientation,
             GizmoHandle handle,
             Ray const& pointerRay,
             Vec3 viewForward,
             GizmoSettings settings);
  void update(Ray const& pointerRay);
  void end();

  bool active() const { return active_; }
  // Signed metres (axis translate), metres (free/plane translate), radians
  // (rotate), or scale factor (scale; starts at 1).
  float cumulative() const { return value_; }

private:
  struct StartNode {
    NodeId id{kInvalidNode};
    Mat4 startWorld{};
    Mat4 invParent{};
  };

  bool planeIntersect(Ray const& ray, Vec3& pointOut) const;
  void applyDelta(Mat4 const& delta);

  Scene* scene_{nullptr};
  std::vector<StartNode> starts_;
  GizmoSettings settings_{};
  GizmoHandle handle_{GizmoHandle::Free};
  Quat orientation_{};
  Vec3 pivot_{};
  Vec3 planeNormal_{0.0F, 0.0F, 1.0F};
  Vec3 axis_{1.0F, 0.0F, 0.0F};
  Vec3 dragU_{1.0F, 0.0F, 0.0F};
  Vec3 dragV_{0.0F, 1.0F, 0.0F};
  Vec3 rotE1_{1.0F, 0.0F, 0.0F};
  Vec3 rotE2_{0.0F, 1.0F, 0.0F};
  Vec3 freeStart_{};
  float startParam_{0.0F};
  float value_{0.0F};
  float prevRawAngle_{0.0F};
  float angleOffset_{0.0F};
  bool active_{false};
};

} // namespace vredit
