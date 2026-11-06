#include <cmath>
#include <cstdio>
#include <limits>
#include "camera.h"
#include "vr_locomotion.h"
using namespace vredit;
namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (false)
bool near(float a, float b) { return std::fabs(a-b) < 1e-4F; }
bool near(Vec3 a, Vec3 b) { return length(a-b) < 1e-4F; }
void step(VrLocomotion& l, VrRig& r, Vec3 h, Quat q, Vec2 a, Vec2 b,
          bool la = true, bool ra = true, float dt = 0.05F) {
  l.update(r,h,q,a,b,la,ra,dt);
}
}
int main() {
  CHECK(near(vrStickDeadzone({0.1F,0.1F}).x, 0));
  CHECK(near(vrStickDeadzone({2,0}).x, 1));
  CHECK(near(vrStickDeadzone({0.6F,0.8F}).x, 0.6F));
  // Vive touch gates stale coordinates, but neither inactive action nor an
  // inactive touch query fabricates a neutral sample. Quest stays unchanged.
  auto sample = vrNormaliseStick({0.9F, -0.4F}, true, true, true, false);
  CHECK(sample.active && near(sample.value.x, 0) && near(sample.value.y, 0));
  sample = vrNormaliseStick({0.9F, -0.4F}, true, true, true, true);
  CHECK(sample.active && near(sample.value.x, 0.9F));
  CHECK(!vrNormaliseStick({0.9F, 0}, false, true, true, false).active);
  CHECK(!vrNormaliseStick({0.9F, 0}, true, true, false, false).active);
  sample = vrNormaliseStick({0.9F, 0}, true, false, false, false);
  CHECK(sample.active && near(sample.value.x, 0.9F));
  VrCarouselBrowse browse;
  CHECK(browse.update(true, {}, 0.01F) == 0);
  CHECK(browse.update(true, {1,0}, 0.01F) == 1);
  CHECK(browse.update(true, {1,0}, 0.1F) == 0);
  CHECK(browse.update(false, {1,0}, 1.0F) == 0); // view loss
  CHECK(browse.update(false, {}, 1.0F) == 0); // neutral during loss is not observed
  CHECK(browse.update(true, {1,0}, 1.0F) == 0);
  CHECK(browse.update(true, {}, 0.01F) == 0);
  CHECK(browse.update(true, {-1,0}, 0.01F) == -1);
  CHECK(browse.update(true, {-1,0}, 0.25F) == -1);
  browse.interrupt(); // skipped frame
  CHECK(browse.update(true, {-1,0}, 1.0F) == 0);
  // Normalised untouch stops movement and arms only if both actions active.
  VrLocomotion padLoc; VrRig padRig;
  auto stepPad = [&](VrStickSample s) { step(padLoc, padRig, {}, {}, s.value, {}, s.active, true); };
  stepPad(vrNormaliseStick({}, true, true, true, false));
  stepPad(vrNormaliseStick({1,0}, true, true, true, true));
  Vec3 const moved = padRig.position;
  stepPad(vrNormaliseStick({1,0}, true, true, true, false));
  CHECK(near(padRig.position, moved));
  stepPad(vrNormaliseStick({1,0}, false, true, true, false));
  stepPad(vrNormaliseStick({1,0}, true, true, true, true));
  CHECK(near(padRig.position, moved));
  VrRig rig{{2,1,-3}, quatFromAxisAngle({0,1,0}, 0.4F)};
  VrLocomotion loc;
  Vec3 head{1.2F,1.7F,-0.7F};
  Quat physical = normalized(quatFromAxisAngle({0,1,0}, 0.5F) *
                             quatFromAxisAngle({1,0,0}, -0.25F) *
                             quatFromAxisAngle({0,0,1}, 0.3F));
  Vec3 eyes[2] = {head + Vec3{-0.032F,0,0}, head + Vec3{0.032F,0,0}};
  step(loc,rig,head,physical,{},{});
  Vec3 pivot = transformPoint(rig.world(),head);
  Vec3 eyeBefore[2]; Mat4 viewBefore[2];
  for (int i=0;i<2;++i) { CHECK(composeStereoEye(rig.world(),eyes[i],physical,eyeBefore[i],viewBefore[i])); }
  step(loc,rig,head,physical,{}, {0.7F,0.8F});
  CHECK(near(transformPoint(rig.world(),head),pivot));
  Vec3 eyeAfter[2]; Mat4 viewAfter[2];
  for (int i=0;i<2;++i) { CHECK(composeStereoEye(rig.world(),eyes[i],physical,eyeAfter[i],viewAfter[i])); }
  CHECK(near(length(eyeAfter[1]-eyeAfter[0]),0.064F));
  CHECK(near((eyeAfter[0]+eyeAfter[1])*0.5F,pivot));
  Vec3 expectedBaseline = rotate(rig.orientation, eyes[1]-eyes[0]);
  CHECK(near(eyeAfter[1]-eyeAfter[0], expectedBaseline));
  Vec3 physicalUp = rotate(rig.orientation * physical, {0,1,0});
  CHECK(near(transformVector(viewAfter[0],physicalUp), {0,1,0}));
  Vec3 old = rig.position;
  Vec3 forward = rotate(rig.orientation * physical, {0,0,-1});
  step(loc,rig,head,physical,{0,1},{},true,true,1.0F);
  CHECK(near(rig.position-old, forward * 0.075F)); // capped dt, pitched flight
  old = rig.position;
  step(loc,rig,head,physical,{1,0},{},true,true);
  Vec3 delta = rig.position-old;
  CHECK(near(delta, rotate(rig.orientation * physical, {1,0,0}) * 0.075F));
  CHECK(near(length(delta),0.075F)); CHECK(near(dot(delta,forward),0));
  // Virtual yaw is a world-up rotation after virtual pitch, independent of
  // physical roll. Physical turning must not rotate the scene behind the
  // viewer's back: the tracking-to-world rig stays fixed while sticks rest.
  VrRig spatial{{4.0F, -0.4F, 7.0F}, quatFromAxisAngle({0,1,0}, -0.65F)};
  VrLocomotion spatialLoc;
  Vec3 trackedHead{1.3F, 1.6F, -0.8F};
  Quat tracked = quatFromAxisAngle({1,0,0}, -0.2F);
  step(spatialLoc, spatial, trackedHead, tracked, {}, {});
  for (int i = 0; i < 8; ++i) { step(spatialLoc, spatial, trackedHead, tracked, {}, {0,1}); }
  Mat4 const beforePhysicalTurn = spatial.world();
  Vec3 const viewRightBeforePhysicalTurn = rotate(spatial.orientation * tracked, {1,0,0});
  tracked = normalized(quatFromAxisAngle({0,1,0}, 1.1F) *
                       quatFromAxisAngle({1,0,0}, -0.2F));
  step(spatialLoc, spatial, trackedHead, tracked, {}, {});
  for (int i = 0; i < 16; ++i) { CHECK(near(spatial.world().m[i], beforePhysicalTurn.m[i])); }
  Quat const actualView = normalized(spatial.orientation * tracked);
  Vec3 const actualRight = rotate(actualView, {1,0,0});
  // Physical yaw after virtual pitch tilts the apparent horizon, despite no
  // physical roll or rig movement. World-up stick yaw cannot prevent this.
  CHECK(std::fabs(viewRightBeforePhysicalTurn.y) < 1e-4F);
  CHECK(std::fabs(actualRight.y) > 0.2F);
  old = spatial.position;
  step(spatialLoc, spatial, trackedHead, tracked, {1,0}, {});
  CHECK(near(spatial.position - old, actualRight * 0.075F));
  // A physical position change, including after pitch, must translate the
  // rendered eyes while leaving the world transform unchanged.
  trackedHead = trackedHead + Vec3{.35F, -.12F, .48F};
  Mat4 const beforePhysicalMove = spatial.world();
  Vec3 const expectedMovedHead = transformPoint(beforePhysicalMove, trackedHead);
  step(spatialLoc, spatial, trackedHead, tracked, {}, {});
  for (int i = 0; i < 16; ++i) { CHECK(near(spatial.world().m[i], beforePhysicalMove.m[i])); }
  CHECK(near(transformPoint(spatial.world(), trackedHead), expectedMovedHead));
  // Physical roll is not removed; virtual yaw rotates the complete head
  // around world up. This does not promise a level apparent horizon.
  tracked = normalized(tracked * quatFromAxisAngle({0,0,1}, 0.45F));
  step(spatialLoc, spatial, trackedHead, tracked, {}, {});
  Quat const beforeYaw = normalized(spatial.orientation * tracked);
  Vec3 const fixedHead = transformPoint(spatial.world(), trackedHead);
  step(spatialLoc, spatial, trackedHead, tracked, {}, {1,0});
  Quat const yawDelta = quatFromAxisAngle({0,1,0}, -0.075F);
  Quat const afterYaw = normalized(spatial.orientation * tracked);
  CHECK(near(transformPoint(spatial.world(), trackedHead), fixedHead));
  for (Vec3 axis : {Vec3{1,0,0}, Vec3{0,1,0}, Vec3{0,0,-1}}) {
    CHECK(near(rotate(afterYaw, axis), rotate(yawDelta, rotate(beforeYaw, axis))));
  }
  CHECK(std::fabs(rotate(afterYaw, {1,0,0}).y) > 0.2F);
  // Each raw eye is inverted in the same frame as the controller aim. An
  // independently computed pose view exposes a transpose or origin mismatch.
  Mat4 const spatialWorld = spatial.world();
  Vec3 const trackedEyes[2] = {trackedHead + rotate(tracked, {-0.032F,0,0}),
                                trackedHead + rotate(tracked, {0.032F,0,0})};
  for (int i = 0; i < 2; ++i) {
    Vec3 worldEye{}; Mat4 eyeView{};
    CHECK(composeStereoEye(spatialWorld, trackedEyes[i], tracked, worldEye, eyeView));
    Mat4 const expectedEyeView = cameraViewFromPose(worldEye, afterYaw);
    Vec3 const stationaryObject{2.1F, 1.9F, -4.3F};
    CHECK(near(transformPoint(eyeView, stationaryObject),
               transformPoint(expectedEyeView, stationaryObject)));
    CHECK(near(transformPoint(eyeView, worldEye), {}));
    Vec3 const rawAimOrigin = trackedEyes[i];
    Vec3 const rawAimDir = rotate(tracked, {0,0,-1});
    Vec3 const aimOrigin = transformPoint(spatialWorld, rawAimOrigin);
    Vec3 const aimDir = transformVector(spatialWorld, rawAimDir);
    CHECK(near(transformPoint(eyeView, aimOrigin + aimDir * 3.0F), {0,0,-3}));
  }
  // Test both signs and speed in the actual eye frame, including exactly
  // vertical and inverted views, with a translated and rotated tracking rig.
  constexpr float halfPi = 1.57079632679F;
  Quat const views[] = {
    quatFromAxisAngle({1,0,0}, halfPi),
    quatFromAxisAngle({1,0,0}, -halfPi),
    normalized(quatFromAxisAngle({0,1,0}, 0.8F) *
               quatFromAxisAngle({1,0,0}, halfPi) * quatFromAxisAngle({0,0,1}, 0.9F)),
    normalized(quatFromAxisAngle({0,1,0}, -1.2F) *
               quatFromAxisAngle({1,0,0}, 2.4F) * quatFromAxisAngle({0,0,1}, -2.0F)),
    quatFromAxisAngle({0,0,1}, halfPi)
  };
  for (Quat const desiredView : views) {
    VrRig flight{{3.4F, -2.1F, 5.7F}, normalized(quatFromAxisAngle({0,1,0}, .7F) *
                                                   quatFromAxisAngle({1,0,0}, -.35F))};
    VrLocomotion flightLoc;
    Vec3 const rawHead{1.2F, 1.8F, -.9F};
    Quat const rawView = normalized(conjugate(flight.orientation) * desiredView);
    step(flightLoc, flight, rawHead, rawView, {}, {});
    for (float const sign : {-1.0F, 1.0F}) {
      Vec3 const start = transformPoint(flight.world(), rawHead);
      Quat const view = normalized(flight.orientation * rawView);
      Vec3 const viewRight = normalized(rotate(view, {1,0,0}));
      Mat4 const eyeView = cameraViewFromPose(start, view);
      step(flightLoc, flight, rawHead, rawView, {sign,0}, {});
      Vec3 const movement = transformPoint(flight.world(), rawHead) - start;
      CHECK(near(movement, viewRight * (sign * 0.075F)));
      CHECK(near(dot(movement, viewRight), sign * 0.075F));
      CHECK(near(length(movement), 0.075F)); // 1.5 m/s over 50 ms
      CHECK(near(transformVector(eyeView, movement), {sign * 0.075F,0,0}));
    }
  }
  // At either physical pole the first sample has no horizontal heading.
  // Pitch away from the pole must work, towards it must not snap, and a
  // rolled view must still produce a horizontal, unit pitch axis.
  for (float sign : {-1.0F, 1.0F}) {
    for (float roll : {0.0F, halfPi}) {
      VrRig pole; VrLocomotion poleLoc;
      Quat const view = normalized(quatFromAxisAngle({0,1,0}, 0.7F) *
          quatFromAxisAngle({1,0,0}, sign * halfPi) * quatFromAxisAngle({0,0,1}, roll));
      Vec3 const headAt{0.8F, 1.7F, -0.4F};
      step(poleLoc, pole, headAt, view, {}, {});
      Vec3 const pivotAt = transformPoint(pole.world(), headAt);
      step(poleLoc, pole, headAt, view, {}, {0,sign});
      CHECK(near(rotate(pole.orientation * view, {0,0,-1}), {0,sign,0}));
      step(poleLoc, pole, headAt, view, {}, {0,-sign});
      Vec3 const away = rotate(pole.orientation * view, {0,0,-1});
      CHECK(near(away.y, sign * std::cos(0.075F)));
      CHECK(near(transformPoint(pole.world(), headAt), pivotAt));
      step(poleLoc, pole, headAt, view, {}, {0,-sign});
      CHECK(std::fabs(rotate(pole.orientation * view, {0,0,-1}).y) < std::fabs(away.y));
    }
  }
  // Exact vertical has a deterministic view-derived fallback, including yaw.
  VrRig pole; VrLocomotion poleLoc;
  Quat const nearPole = quatFromAxisAngle({1,0,0}, halfPi - 0.05F);
  Quat const exactPole = quatFromAxisAngle({1,0,0}, halfPi);
  step(poleLoc, pole, {}, nearPole, {}, {});
  step(poleLoc, pole, {}, exactPole, {}, {1,-1});
  Vec3 const poleForward = rotate(pole.orientation * exactPole, {0,0,-1});
  CHECK(poleForward.y < 0.999F);
  CHECK(poleForward.z < 0.0F);
  // Exercise both sides of both poles with a physical yaw while inside the
  // old 0.02-radian cache zone. Pitch away must reduce absolute elevation
  // immediately and remain on the current side, including after stick yaw.
  for (float poleSign : {-1.0F, 1.0F}) {
    for (float side : {-1.0F, 1.0F}) {
      for (float stickYaw : {0.0F, 1.0F}) {
        for (float offset : {.01F, .000005F}) {
          VrRig crossing; VrLocomotion crossingLoc;
          auto physicalView = [&](float yaw, float offset) {
            return normalized(quatFromAxisAngle({0,1,0}, yaw) *
                quatFromAxisAngle({1,0,0}, poleSign * (halfPi + offset)));
          };
          step(crossingLoc, crossing, {}, physicalView(0, -side * .03F), {}, {});
          Quat const physicalNow = physicalView(1.1F, side * offset);
          Vec3 const before = rotate(physicalNow, {0,0,-1});
          step(crossingLoc, crossing, {}, physicalNow, {}, {stickYaw, -poleSign});
          Vec3 const after = rotate(crossing.orientation * physicalNow, {0,0,-1});
          Vec3 const yawedBefore = rotate(quatFromAxisAngle({0,1,0}, -stickYaw * .075F), before);
          CHECK(std::fabs(after.y) < std::fabs(before.y) - .001F);
          CHECK(dot(Vec3{after.x,0,after.z}, Vec3{yawedBefore.x,0,yawedBefore.z}) > 0);
          float const previousElevation = std::fabs(after.y);
          step(crossingLoc, crossing, {}, physicalNow, {}, {0, -poleSign});
          CHECK(std::fabs(rotate(crossing.orientation * physicalNow, {0,0,-1}).y) < previousElevation);
        }
      }
    }
  }
  // Inactive, non-finite, tracked gaps and explicit interruption all demand neutral.
  old = rig.position;
  step(loc,rig,head,physical,{1,0},{},false,true);
  step(loc,rig,head,physical,{1,0},{},true,true);
  CHECK(near(rig.position,old));
  step(loc,rig,head,physical,{},{});
  step(loc,rig,head,physical,{std::numeric_limits<float>::quiet_NaN(),0},{});
  step(loc,rig,head,physical,{1,0},{});
  CHECK(near(rig.position,old));
  step(loc,rig,head,physical,{},{}); loc.interrupt();
  step(loc,rig,head,physical,{1,0},{1,0});
  CHECK(near(rig.position,old));
  // Carousel owns right stick; closing while held cannot turn until neutral.
  Quat before = rig.orientation;
  step(loc,rig,head,physical,{}, {1,0},true,false);
  step(loc,rig,head,physical,{}, {1,0},true,true);
  CHECK(near(rotate(before,{0,0,-1}), rotate(rig.orientation,{0,0,-1})));
  step(loc,rig,head,physical,{},{});
  step(loc,rig,head,physical,{}, {1,0});
  CHECK(!near(rotate(before,{0,0,-1}), rotate(rig.orientation,{0,0,-1})));
  // Cardinal controls and pitch clamp, independent of tracking-origin offset.
  VrRig cardinal; VrLocomotion c; Vec3 h{0.7F,1.6F,0.4F};
  step(c,cardinal,h,{}, {},{});
  step(c,cardinal,h,{}, {},{1,0});
  CHECK(rotate(cardinal.orientation,{0,0,-1}).x > 0);
  Vec3 fixed = transformPoint(cardinal.world(),h);
  step(c,cardinal,h,{}, {},{0,1});
  CHECK(rotate(cardinal.orientation,{0,0,-1}).y > 0);
  for (int i=0;i<100;++i) { step(c,cardinal,h,{}, {},{0,1}); }
  CHECK(near(transformPoint(cardinal.world(),h),fixed));
  CHECK(rotate(cardinal.orientation,{0,0,-1}).y <= std::sin(1.5F)+1e-4F);
  step(c,cardinal,h,{}, {},{0,-1});
  CHECK(rotate(cardinal.orientation,{0,0,-1}).y < std::sin(1.5F)-1e-4F);
  Vec3 stable = cardinal.position;
  step(c,cardinal,h,{}, {1,0},{},true,true,std::numeric_limits<float>::infinity());
  step(c,cardinal,h,{}, {1,0},{});
  CHECK(near(cardinal.position,stable));
  step(c,cardinal,h,{}, {},{});
  step(c,cardinal,h,{}, {1,0},{});
  CHECK(!near(cardinal.position,stable));
  std::printf("VR locomotion: %s\n",failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
