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
  // Facing the viewer: the model's art faces its own +Z (glTF convention),
  // so the yaw rotates +Z onto the direction from the slot back at the eye.
  Vec3 const back = normalized(eye - position);
  float const yaw = std::atan2(back.x, back.z);
  return SlotPose{position, yaw};
}

CarouselRing::SlotPose CarouselRing::labelPose(int i, Vec3 const& eye, Vec3 const& forward,
                                               float halfHeight, float gap) const {
  // The label belongs to its own slot: same horizontal position and the same
  // facing as the model, hung below the visible geometry. halfHeight comes
  // from the fitted preview, so a squat model's label sits close under it and
  // a tall one's does not collide with the floor ring.
  SlotPose const model = slotPose(i, eye, forward);
  float const drop = (halfHeight > 0.0F ? halfHeight : 0.0F) + (gap > 0.0F ? gap : 0.0F);
  return SlotPose{Vec3{model.position.x, model.position.y - drop, model.position.z}, model.yaw};
}

Quat CarouselRing::labelOrientation(SlotPose const& label, Vec3 const& openingEye) const {
  Vec3 const toEye = openingEye - label.position;
  float const horizontal = std::hypot(toEye.x, toEye.z);
  float const yaw = horizontal > 1e-5F ? std::atan2(toEye.x, toEye.z) : label.yaw;
  float const pitch = -std::atan2(toEye.y, horizontal);
  return normalized(quatFromAxisAngle({0.0F, 1.0F, 0.0F}, yaw) *
                    quatFromAxisAngle({1.0F, 0.0F, 0.0F}, pitch));
}

} // namespace vredit
