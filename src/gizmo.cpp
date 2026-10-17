#include "gizmo.h"

#include <algorithm>
#include <cmath>

namespace vredit {

namespace {

constexpr float kPi = 3.14159265359F;

float snapValue(float v, float step) {
  if (step <= 0.0F) {
    return v;
  }
  return std::round(v / step) * step;
}

Vec3 handleAxis(GizmoHandle handle) {
  switch (handle) {
  case GizmoHandle::AxisX:
    return {1.0F, 0.0F, 0.0F};
  case GizmoHandle::AxisY:
    return {0.0F, 1.0F, 0.0F};
  case GizmoHandle::AxisZ:
    return {0.0F, 0.0F, 1.0F};
  default:
    return {};
  }
}

Vec3 handlePlaneNormal(GizmoHandle handle) {
  switch (handle) {
  case GizmoHandle::PlaneXY:
    return {0.0F, 0.0F, 1.0F};
  case GizmoHandle::PlaneYZ:
    return {1.0F, 0.0F, 0.0F};
  case GizmoHandle::PlaneZX:
    return {0.0F, 1.0F, 0.0F};
  default:
    return {};
  }
}

bool isAxisHandle(GizmoHandle handle) {
  return handle == GizmoHandle::AxisX || handle == GizmoHandle::AxisY ||
         handle == GizmoHandle::AxisZ;
}

bool isPlaneHandle(GizmoHandle handle) {
  return handle == GizmoHandle::PlaneXY || handle == GizmoHandle::PlaneYZ ||
         handle == GizmoHandle::PlaneZX;
}

} // namespace

bool GizmoDrag::planeIntersect(Ray const& ray, Vec3& pointOut) const {
  float denom = dot(ray.direction, planeNormal_);
  if (std::fabs(denom) < 1e-6F) {
    return false;
  }
  float t = dot(pivot_ - ray.origin, planeNormal_) / denom;
  if (t < 0.0F) {
    return false; // intersection behind the pointer ray origin
  }
  pointOut = ray.origin + ray.direction * t;
  return true;
}

bool GizmoDrag::begin(Scene& scene,
                      std::vector<NodeId> const& selection,
                      Vec3 pivot,
                      Quat orientation,
                      GizmoHandle handle,
                      Ray const& pointerRay,
                      Vec3 viewForward,
                      GizmoSettings settings) {
  end();
  value_ = settings.mode == GizmoMode::Scale ? 1.0F : 0.0F;
  if (length(viewForward) < 1e-6F) {
    return false;
  }
  viewForward = normalized(viewForward);

  scene_ = &scene;
  settings_ = settings;
  handle_ = handle;
  pivot_ = pivot;
  // The gizmo frame is authoritative: World space uses world axes (the passed
  // orientation is ignored); Local space uses the caller's orientation.
  orientation_ = settings.space == GizmoSpace::World ? Quat{} : normalized(orientation);

  if (isAxisHandle(handle)) {
    axis_ = rotate(orientation_, handleAxis(handle));
  } else if (isPlaneHandle(handle)) {
    axis_ = rotate(orientation_, handlePlaneNormal(handle));
  } else if (settings.mode == GizmoMode::Rotate) {
    axis_ = viewForward;
  } else {
    axis_ = {};
  }

  if (settings.mode == GizmoMode::Rotate) {
    planeNormal_ = isAxisHandle(handle) ? axis_ : viewForward;
  } else if (isAxisHandle(handle)) {
    Vec3 n = viewForward - axis_ * dot(viewForward, axis_);
    if (length(n) < 1e-4F) {
      end();
      return false;
    }
    planeNormal_ = normalized(n);
  } else {
    planeNormal_ = isPlaneHandle(handle) ? axis_ : viewForward;
  }

  // Drag-plane basis aligned to the gizmo frame. Free/plane snapping happens in
  // this basis, so snapped deltas stay exactly in the drag plane and follow the
  // world or local grid as appropriate.
  Vec3 candidate = rotate(orientation_, {1.0F, 0.0F, 0.0F});
  Vec3 u = candidate - planeNormal_ * dot(candidate, planeNormal_);
  if (length(u) < 1e-3F) {
    candidate = rotate(orientation_, {0.0F, 1.0F, 0.0F});
    u = candidate - planeNormal_ * dot(candidate, planeNormal_);
  }
  if (length(u) < 1e-3F) {
    end();
    return false;
  }
  dragU_ = normalized(u);
  dragV_ = cross(planeNormal_, dragU_);

  Vec3 startPoint;
  if (!planeIntersect(pointerRay, startPoint)) {
    end();
    return false;
  }

  prevRawAngle_ = 0.0F;
  angleOffset_ = 0.0F;

  if (settings.mode == GizmoMode::Rotate) {
    Vec3 v0 = startPoint - pivot_;
    Vec3 e1 = v0 - axis_ * dot(v0, axis_);
    if (length(e1) < 1e-6F) {
      end();
      return false;
    }
    rotE1_ = normalized(e1);
    rotE2_ = cross(axis_, rotE1_);
    startParam_ = 0.0F;
    freeStart_ = startPoint;
  } else if (settings.mode == GizmoMode::Scale) {
    if (isAxisHandle(handle)) {
      startParam_ = dot(startPoint - pivot_, axis_);
      if (std::fabs(startParam_) < 1e-4F) {
        end();
        return false;
      }
    } else {
      freeStart_ = startPoint;
      startParam_ = length(startPoint - pivot_);
      if (startParam_ < 1e-4F) {
        end();
        return false;
      }
    }
  } else if (isAxisHandle(handle)) {
    startParam_ = dot(startPoint - pivot_, axis_);
    freeStart_ = startPoint;
  } else {
    freeStart_ = startPoint;
    startParam_ = 0.0F;
  }

  starts_.clear();
  for (NodeId id : selection) {
    if (!scene.valid(id)) {
      continue;
    }
    // Rigid-group semantics: a node whose ancestor is also selected rides
    // along with that ancestor's transform instead of receiving the delta
    // itself, which would apply it twice through the hierarchy.
    bool ancestorSelected = false;
    for (NodeId p = scene.parent(id); p != kInvalidNode && scene.valid(p); p = scene.parent(p)) {
      if (std::find(selection.begin(), selection.end(), p) != selection.end()) {
        ancestorSelected = true;
        break;
      }
    }
    if (ancestorSelected) {
      continue;
    }
    StartNode start;
    start.id = id;
    NodeId parentId = scene.parent(id);
    Mat4 parentWorld;
    if (parentId != kInvalidNode && scene.valid(parentId)) {
      parentWorld = scene.world(parentId);
    }
    if (!invertAffine(parentWorld, start.invParent)) {
      // Singular parent world (a zero-scale ancestor): drop the node. Its
      // skipped descendants inherit the same singular chain and would fail
      // identically as their own entries, so nothing moves — no silent
      // half-transform.
      continue;
    }
    start.startWorld = mul(parentWorld, toMatrix(scene.local(id)));
    starts_.push_back(start);
  }

  active_ = true;
  return true;
}

void GizmoDrag::update(Ray const& pointerRay) {
  if (!active_) {
    return;
  }
  Vec3 point;
  if (!planeIntersect(pointerRay, point)) {
    return;
  }

  if (settings_.mode == GizmoMode::Rotate) {
    Vec3 v = point - pivot_;
    float raw = std::atan2(dot(v, rotE2_), dot(v, rotE1_));
    // Unwrap across the +/-pi seam so snapping stays continuous along the drag.
    float diff = raw - prevRawAngle_;
    if (diff > kPi) {
      angleOffset_ -= 2.0F * kPi;
    } else if (diff < -kPi) {
      angleOffset_ += 2.0F * kPi;
    }
    prevRawAngle_ = raw;
    value_ = snapValue(raw + angleOffset_, settings_.snapStep);
    Mat4 delta =
        mul(translationMatrix(pivot_),
            mul(rotationMatrix(quatFromAxisAngle(axis_, value_)), translationMatrix(-pivot_)));
    applyDelta(delta);
    return;
  }

  if (settings_.mode == GizmoMode::Translate) {
    Vec3 deltaVec{};
    if (isAxisHandle(handle_)) {
      float raw = dot(point - pivot_, axis_) - startParam_;
      value_ = snapValue(raw, settings_.snapStep);
      deltaVec = axis_ * value_;
    } else {
      Vec3 raw = point - freeStart_;
      float du = snapValue(dot(raw, dragU_), settings_.snapStep);
      float dv = snapValue(dot(raw, dragV_), settings_.snapStep);
      deltaVec = dragU_ * du + dragV_ * dv;
      value_ = length(deltaVec);
    }
    applyDelta(translationMatrix(deltaVec));
    return;
  }

  float factor;
  if (isAxisHandle(handle_)) {
    float current = dot(point - pivot_, axis_);
    factor = current / startParam_;
  } else {
    factor = length(point - pivot_) / startParam_;
  }
  factor = snapValue(factor, settings_.snapStep);
  if (factor < 0.01F) {
    factor = 0.01F;
  } else if (factor > 100.0F) {
    factor = 100.0F;
  }
  value_ = factor;

  Vec3 s{1.0F, 1.0F, 1.0F};
  if (isAxisHandle(handle_)) {
    if (handle_ == GizmoHandle::AxisX) {
      s.x = factor;
    } else if (handle_ == GizmoHandle::AxisY) {
      s.y = factor;
    } else {
      s.z = factor;
    }
  } else {
    s = {factor, factor, factor};
  }
  Mat4 delta =
      mul(translationMatrix(pivot_),
          mul(rotationMatrix(orientation_),
             mul(scaleMatrix(s),
                 mul(rotationMatrix(conjugate(orientation_)), translationMatrix(-pivot_)))));
  applyDelta(delta);
}

void GizmoDrag::end() {
  active_ = false;
  scene_ = nullptr;
  starts_.clear();
}

void GizmoDrag::applyDelta(Mat4 const& delta) {
  for (StartNode const& start : starts_) {
    Mat4 newWorld = mul(delta, start.startWorld);
    Mat4 newLocal = mul(start.invParent, newWorld);
    scene_->setLocal(start.id, toTransform(newLocal));
  }
}

} // namespace vredit
