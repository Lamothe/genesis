#pragma once

#include "vecmath.h"

namespace vredit {

struct HandPose {
  Vec3 position{};
  Quat orientation{};
  bool valid{false};
};

class HandGesture {
 public:
  // Acquisition (including after tracking loss) must observe a released pinch.
  bool update(bool valid, float pinch) {
    if (!valid || !std::isfinite(pinch)) {
      reset();
      return false;
    }
    if (pinch <= 0.4F) {
      armed_ = true;
      pressed_ = false;
    } else if (armed_ && !pressed_ && pinch >= 0.55F) {
      pressed_ = true;
      return true;
    }
    return false;
  }

  void reset() {
    armed_ = false;
    pressed_ = false;
  }

 private:
  bool armed_{false};
  bool pressed_{false};
};

// Panel-local coordinates, with the origin at the top centre. Invalid indices
// return the origin (which is not inside any button).
inline constexpr Vec3 handPanelButtonCentre(int index) {
  if (index < 0 || index >= 12) {
    return {};
  }
  return {(index % 2 == 0) ? -0.115F : 0.115F,
          -0.065F * (static_cast<float>(index / 2) + 0.5F), 0.0F};
}

struct HandPanel {
  Vec3 origin{};
  Quat orientation{};
  bool visible{false};

  int hit(Ray const& ray) const {
    if (!visible) {
      return -1;
    }
    auto const normal = rotate(orientation, {0.0F, 0.0F, 1.0F});
    auto const denominator = dot(ray.direction, normal);
    if (!(denominator < -1e-6F)) {
      return -1;
    }
    auto const distance = dot(origin - ray.origin, normal) / denominator;
    if (!(distance >= 0.0F) || !std::isfinite(distance)) {
      return -1;
    }
    auto const local = rotate(conjugate(orientation),
                              ray.origin + ray.direction * distance - origin);
    for (int index = 0; index < 12; ++index) {
      auto const centre = handPanelButtonCentre(index);
      if (std::fabs(local.x - centre.x) <= 0.105F &&
          std::fabs(local.y - centre.y) <= 0.028F) {
        return index;
      }
    }
    return -1;
  }
};

inline bool palmFacingUp(HandPose const& pose, bool previous) {
  return pose.valid &&
         rotate(pose.orientation, {0.0F, -1.0F, 0.0F}).y > (previous ? 0.55F : 0.75F);
}

// Poses/orientations are world-space unit quaternions, as supplied by tracking.
// Always reapply the cumulative delta to the captured initialWorld, never the
// previous result. The caller converts world to parent-local if needed.
inline Mat4 handGrabTransform(HandPose start, HandPose current, Vec3 pivot, Mat4 initialWorld) {
  if (!start.valid || !current.valid) {
    return initialWorld;
  }
  auto const delta = normalized(current.orientation * conjugate(start.orientation));
  auto const translation = current.position - start.position;
  return mul(translationMatrix(pivot + translation),
             mul(rotationMatrix(delta), mul(translationMatrix(-pivot), initialWorld)));
}

} // namespace vredit
