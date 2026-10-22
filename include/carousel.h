#pragma once

#include "vecmath.h"

namespace vredit {

// A model carousel that ROTATES IN AND OUT OF VIEW ("I expected that the
// models would rotate in/out of view with a preview of the next/previous
// model visible"): library slots sit on a ring around the viewer, the
// centred slot is dead ahead at eye height, and stepping advances the ring
// by EXACTLY one slot with a smooth eased rotation — the next/previous
// models are visible at the sides the whole time, including across the
// wrap-around (target/angle are continuous accumulators, never re-wrapped,
// so the last->first step goes one slot forward, not the long way round).
// The ring is placed in a CALLER-SUPPLIED frame — the app latches it at open
// (stationary in the world; it does not follow the viewer).
//
// Pure maths: the app supplies the viewer frame and translates SlotPose into
// nodes. `step()` advances the target; `tick()` eases `angle` toward it, so
// a switch is visible immediately even while the model behind it loads
// (loading is asynchronous — see the app's worker thread).
struct CarouselRing {
  static constexpr float kSlotStep = 0.9F; // radians between slots (~52 deg)
  static constexpr float kRadius = 1.25F;  // metres from the viewer's eye
  static constexpr float kEase = 11.0F;    // ease rate (1/s)

  int count{0};       // library size
  int index{0};       // centred library index (wrapped into [0, count))
  float target{0.0F}; // requested ring angle (radians), continuous
  float angle{0.0F};  // eased ring angle (radians), continuous

  void setCount(int n);
  void step(int delta); // advance the ring by delta slots
  void tick(float dt);  // ease angle toward target
  bool atRest() const;

  // Where library slot `i` floats and how it yaws, given the viewer's eye
  // position and LEVEL heading (forward on the horizontal plane). The offset
  // from dead ahead is the MINIMAL wrapped distance, so a wrapped slot (the
  // previous model at index count-1 while index is 0) sits right beside its
  // neighbour — never across the circle. Models face the viewer.
  struct SlotPose {
    Vec3 position;
    float yaw;
  };
  SlotPose slotPose(int i, Vec3 const& eye, Vec3 const& forward) const;
};

} // namespace vredit
