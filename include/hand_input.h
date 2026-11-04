#pragma once

#include "vecmath.h"

#include <initializer_list>

namespace vredit {

struct HandPose {
  Vec3 position{};
  Quat orientation{};
  bool valid{false};
};

// The centre is deliberately inert unless an exact preview surface was hit.
// Intersect the ring-facing plane so hand-ray origin does not skew side zones.
inline int carouselPinchDirection(Ray const& ray, Vec3 origin, Vec3 forward) {
  forward.y = 0;
  forward = normalized(forward);
  auto const denominator = dot(ray.direction, forward);
  if (!(denominator > .0001F)) { return 0; }
  float const distance = dot(origin + forward * 1.25F - ray.origin, forward) / denominator;
  if (!(distance > 0)) { return 0; }
  float const side = dot(ray.origin + ray.direction * distance - origin, cross(forward, Vec3{0,1,0}));
  return side < -.22F ? -1 : side > .22F ? 1 : 0;
}

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

  bool pressed() const { return pressed_; }

  void reset() {
    armed_ = false;
    pressed_ = false;
  }

 private:
  bool armed_{false};
  bool pressed_{false};
};

// Only a second released-and-repressed pinch on the selected surface starts a grab.
class HandSelectionLatch {
 public:
  bool startsGrab(bool hitSelection) const { return primed_ && hitSelection; }
  void selected(bool hasSelection) { primed_ = hasSelection; }
  void reset() { primed_ = false; }
 private:
  bool primed_{false};
};

// Exponential low-pass filtering preserves sub-millimetre motion rather than
// quantising it. Quaternion hemisphere correction prevents sign flips.
inline HandPose smoothHandPose(HandPose previous, HandPose sample, float dt) {
  if (!previous.valid || !sample.valid) { return sample; }
  float const alpha = 1.0F - std::exp(-std::fmax(0.0F, dt) / .045F);
  auto q = sample.orientation;
  auto const& p = previous.orientation;
  if (p.x*q.x + p.y*q.y + p.z*q.z + p.w*q.w < 0) { q = {-q.x,-q.y,-q.z,-q.w}; }
  return {previous.position + (sample.position - previous.position) * alpha,
          normalized(Quat{p.x + (q.x-p.x)*alpha, p.y + (q.y-p.y)*alpha,
                          p.z + (q.z-p.z)*alpha, p.w + (q.w-p.w)*alpha}), true};
}

inline float handScaleRatio(float initialDistance, float distance) {
  if (!std::isfinite(initialDistance) || !std::isfinite(distance) || initialDistance < .04F) {
    return 1.0F;
  }
  return std::fmax(.05F, std::fmin(20.0F, distance / initialDistance));
}

// Panel-local coordinates, with the origin at the top centre. Invalid indices
// return the origin (which is not inside any button).
inline constexpr Vec3 handPanelButtonCentre(int index) {
  int const slot = index == 0 ? 0 : index >= 8 && index <= 11 ? index - 7 : -1;
  if (slot < 0) { return {}; }
  return {(slot % 2 == 0) ? -0.115F : 0.115F,
          -0.065F * (static_cast<float>(slot / 2) + 0.5F), 0.0F};
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
    for (int index : {0, 8, 9, 10, 11}) {
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
  auto const& a = start.orientation;
  auto const& b = current.orientation;
  if (current.position.x == start.position.x && current.position.y == start.position.y &&
      current.position.z == start.position.z && a.x == b.x && a.y == b.y &&
      a.z == b.z && a.w == b.w) {
    return initialWorld;
  }
  auto const delta = normalized(current.orientation * conjugate(start.orientation));

  auto const translation = (current.position - start.position);
  return mul(translationMatrix(pivot + translation),
             mul(rotationMatrix(delta), mul(translationMatrix(-pivot), initialWorld)));
}

} // namespace vredit
