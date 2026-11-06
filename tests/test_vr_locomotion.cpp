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
  CHECK(near(delta.y,0)); CHECK(near(length(delta),0.075F)); CHECK(near(dot(delta,forward),0));
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
