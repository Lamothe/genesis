#include "carousel.h"

#include <cmath>

namespace vredit {

namespace {
// The minimal signed wrapped difference between two angles on a circle of
// `circumference` radians — the heart of "the previous model is RIGHT THERE,
// not across the ring".
float wrapNear(float delta, float circumference) {
  if (circumference <= 0.0F) {
    return delta;
  }
  float d = std::fmod(delta, circumference);
  if (d < 0.0F) {
    d += circumference;
  }
  if (d > circumference * 0.5F) {
    d -= circumference;
  }
  return d;
}
} // namespace

void CarouselRing::setCount(int n) {
  count = n < 0 ? 0 : n;
  if (count > 0) {
    index = ((index % count) + count) % count;
  } else {
    index = 0;
  }
  // Re-anchor both accumulators on the index (only ever called when the
  // library changes; stepping keeps continuity).
  target = static_cast<float>(index) * kSlotStep;
  angle = target;
}

void CarouselRing::step(int delta) {
  if (count <= 0) {
    return;
  }
  // Advance the CONTINUOUS target by exactly one slot per step (the wrap
  // from the last model to the first goes forward one slot, never all the
  // way round the circle backwards) and wrap the visible index.
  target += static_cast<float>(delta) * kSlotStep;
  index = (((index + delta) % count) + count) % count;
}

void CarouselRing::tick(float dt) {
  if (dt <= 0.0F) {
    return;
  }
  // Exponential ease toward the target: converges without overshoot. Large
  // dt is capped so a hitch cannot teleport the ring past several slots.
  float const t = 1.0F - std::exp(-kEase * (dt > 0.1F ? 0.1F : dt));
  angle += (target - angle) * t;
}

bool CarouselRing::atRest() const {
  return std::fabs(target - angle) < 1e-3F;
}

CarouselRing::SlotPose CarouselRing::slotPose(int i, Vec3 const& eye, Vec3 const& forward) const {
  // The level frame from the viewer's heading (right = forward x worldUp).
  Vec3 fwd = forward;
  if (fwd.x == 0.0F && fwd.z == 0.0F) {
    fwd = {0.0F, 0.0F, -1.0F};
  }
  fwd = normalized(Vec3{fwd.x, 0.0F, fwd.z});
  Vec3 const right = normalized(cross(fwd, Vec3{0.0F, 1.0F, 0.0F}));
  // The MINIMAL wrapped offset of library slot i from dead ahead: a wrapped
  // neighbour (index count-1 while centred on 0) lands one slot to the side,
  // not almost a full circle away.
  float const circumference = static_cast<float>(count) * kSlotStep;
  float const offset =
      wrapNear(static_cast<float>(i) * kSlotStep - angle, circumference);
  float const s = std::sin(offset);
  float const c = std::cos(offset);
  Vec3 const position = eye + right * (s * kRadius) + fwd * (c * kRadius);
  // Facing the viewer: the world yaw whose forward points from the slot back
  // at the eye (yaw measured like the app's rig: atan2(x, -z) of forward).
  Vec3 const back = normalized(eye - position);
  float const yaw = std::atan2(back.x, -back.z);
  return SlotPose{position, yaw};
}

} // namespace vredit
