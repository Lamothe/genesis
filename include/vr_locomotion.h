#pragma once

#include <algorithm>
#include <cmath>

#include "vecmath.h"

namespace vredit {

// Trackpads can retain their last coordinates after contact ends. A touch
// sample is authoritative only while both actions are active; an inactive
// action is never interpreted as a physical release.
struct VrStickSample {
  Vec2 value{};
  bool active{false};
};

inline VrStickSample vrNormaliseStick(Vec2 value, bool active, bool trackpad,
                                      bool touchActive, bool touched) {
  if (!active || !std::isfinite(value.x) || !std::isfinite(value.y) ||
      (trackpad && !touchActive)) { return {}; }
  return {trackpad && !touched ? Vec2{} : value, true};
}

// Carousel browse owns the right stick and repeats at 0.25 s. Every loss of
// view or action availability requires a fresh neutral sample before cycling.
class VrCarouselBrowse {
public:
  void interrupt() { armed_ = false; cooldown_ = 0.0F; }
  int update(bool usable, Vec2 stick, float dt) {
    if (!usable || !std::isfinite(stick.x) || !std::isfinite(stick.y)) {
      interrupt(); return 0;
    }
    if (std::fabs(stick.x) <= 0.5F) { armed_ = true; cooldown_ = 0.0F; return 0; }
    if (std::isfinite(dt) && dt > 0.0F) { cooldown_ -= dt; }
    if (!armed_ || cooldown_ > 0.0F) { return 0; }
    cooldown_ = 0.25F;
    return stick.x > 0.0F ? 1 : -1;
  }
private:
  bool armed_{false};
  float cooldown_{0.0F};
};

// A rigid tracking-space -> world transform. Rotations pivot on the current
// tracked head, not the tracking origin. Raw stereo and controller poses are
// composed with this same transform; the compositor still receives raw poses.
struct VrRig {
  Vec3 position{};
  Quat orientation{};
  Mat4 world() const { return trs(position, orientation, {1.0F, 1.0F, 1.0F}); }
};

inline Vec2 vrStickDeadzone(Vec2 stick) {
  constexpr float deadzone = 0.2F;
  if (!std::isfinite(stick.x) || !std::isfinite(stick.y)) { return {}; }
  float const magnitude = std::hypot(stick.x, stick.y);
  if (!std::isfinite(magnitude) || magnitude <= deadzone) { return {}; }
  float const output = (std::min(magnitude, 1.0F) - deadzone) / (1.0F - deadzone);
  return {stick.x / magnitude * output, stick.y / magnitude * output};
}

// Each stick independently requires a neutral sample after an interruption or
// another owner (carousel/menu/grab) releases it; held input never resumes.
class VrLocomotion {
public:
  void interrupt() { leftArmed_ = false; rightArmed_ = false; }
  void update(VrRig& rig, Vec3 head, Quat headOrientation, Vec2 left, Vec2 right,
              bool leftUsable, bool rightUsable, float dt) {
    float const q2 = headOrientation.x * headOrientation.x + headOrientation.y * headOrientation.y +
                     headOrientation.z * headOrientation.z + headOrientation.w * headOrientation.w;
    if (!std::isfinite(dt) || dt <= 0.0F || !std::isfinite(head.x) ||
        !std::isfinite(head.y) || !std::isfinite(head.z) || !std::isfinite(q2) || q2 < 1e-6F) {
      interrupt(); return;
    }
    headOrientation = normalized(headOrientation);
    dt = std::min(dt, 0.05F);
    Vec2 const move = sample(left, leftUsable, leftArmed_);
    Vec2 const turn = sample(right, rightUsable, rightArmed_);
    Vec3 const pivot = rig.position + rotate(rig.orientation, head);
    Quat orientation = rig.orientation;
    // Positive X turns right; positive Y looks up. Pitch axis is horizontal
    // viewer-right, independent of physical head roll.
    orientation = normalized(quatFromAxisAngle({0.0F, 1.0F, 0.0F}, -turn.x * 1.5F * dt) * orientation);
    Vec3 forward = normalized(rotate(orientation * headOrientation, {0.0F, 0.0F, -1.0F}));
    Vec3 rightAxis = cross(forward, {0.0F, 1.0F, 0.0F});
    float const horizontal = length(rightAxis);
    if (horizontal > 1e-4F) {
      rightAxis = rightAxis / horizontal;
      float const elevation = std::asin(std::clamp(forward.y, -1.0F, 1.0F));
      constexpr float limit = 1.50F;
      float pitch = turn.y * 1.5F * dt;
      // Never snap a physically tilted head back inside the limit.
      if (pitch > 0.0F) { pitch = std::min(pitch, std::max(0.0F, limit - elevation)); }
      else { pitch = std::max(pitch, std::min(0.0F, -limit - elevation)); }
      orientation = normalized(quatFromAxisAngle(rightAxis, pitch) * orientation);
    }
    rig.orientation = orientation;
    rig.position = pivot - rotate(orientation, head);
    forward = normalized(rotate(orientation * headOrientation, {0.0F, 0.0F, -1.0F}));
    // True viewer-relative flight: forward follows head pitch; strafe stays
    // level, avoiding roll-induced sideways ascent. No diagonal speed boost.
    rightAxis = cross(forward, {0.0F, 1.0F, 0.0F});
    if (length(rightAxis) > 1e-4F) { rightAxis = normalized(rightAxis); }
    else { rightAxis = {}; }
    rig.position = rig.position + (forward * move.y + rightAxis * move.x) * (1.5F * dt);
  }
private:
  static Vec2 sample(Vec2 value, bool usable, bool& armed) {
    if (!usable || !std::isfinite(value.x) || !std::isfinite(value.y)) {
      armed = false;
      return {};
    }
    Vec2 const filtered = vrStickDeadzone(value);
    if (filtered.x == 0.0F && filtered.y == 0.0F) { armed = true; return {}; }
    return armed ? filtered : Vec2{};
  }
  bool leftArmed_{false};
  bool rightArmed_{false};
};

} // namespace vredit
