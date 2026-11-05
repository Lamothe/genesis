#include "hand_input.h"

#include <cstdio>
#include <initializer_list>
#include <limits>

using namespace vredit;

namespace {
int checks = 0;
int failures = 0;

void check(bool condition, char const* expression, int line) {
  ++checks;
  if (!condition) {
    ++failures;
    std::printf("FAIL test_hands.cpp:%d: %s\n", line, expression);
  }
}
#define CHECK(...) check((__VA_ARGS__), #__VA_ARGS__, __LINE__)

bool near(Vec3 a, Vec3 b) {
  return length(a - b) < 2e-5F;
}

bool near(Mat4 const& a, Mat4 const& b) {
  for (int i = 0; i < 16; ++i) {
    if (!(std::fabs(a.m[i] - b.m[i]) < 2e-5F)) {
      return false;
    }
  }
  return true;
}

void graspArbitration() {
  HandGraspGate gate;
  HandGesture grasp;
  auto step = [&](bool valid, float squeeze, bool pinch, bool captured, bool primed) {
    grasp.update(valid, squeeze);
    bool const held = handPinchHeld(valid, squeeze);
    gate.update(valid, held, pinch, captured);
    return gate.canAcquire(grasp.pressed(), primed);
  };
  CHECK(!step(true, 1, false, false, true)); // first sample cannot arm
  CHECK(!step(true, 0, false, false, true));
  CHECK(step(true, .8F, true, false, true)); // simultaneous: grasp wins
  CHECK(step(true, .8F, true, false, true)); // held may enter contact later
  CHECK(!step(false, 1, true, false, true));
  CHECK(!step(true, 1, true, false, true)); // reacquisition needs release
  CHECK(!step(true, 0, false, false, true));
  CHECK(!step(true, .8F, true, false, false)); // first pinch cannot move
  gate.reset(); grasp.reset(); // selection was made during this hold
  CHECK(!step(true, .8F, true, false, true));
  CHECK(!step(true, 0, true, false, true)); // both must release
  CHECK(!step(true, .8F, false, false, true));
  CHECK(!step(true, 0, false, false, true));
  CHECK(!step(true, .8F, false, false, false)); // unprimed selection remains safe
  CHECK(!step(true, .8F, true, true, true)); // UI capture owns overlap
  CHECK(!step(true, .8F, false, false, true));
  CHECK(!step(true, 0, false, false, true));
  CHECK(step(true, .8F, true, false, true));
}

void contactCacheLifetime() {
  HandContactCache cache;
  auto geometry = std::make_shared<PickGeometry>();
  geometry->positions = {{0,0,0},{1,0,0},{0,1,0}};
  geometry->indices = {0,1,2};
  std::weak_ptr<const PickGeometry> weak = geometry;
  auto mesh = cache.get(geometry);
  CHECK(cache.get(geometry) == mesh);
  cache.prune();
  CHECK(cache.size() == 1);
  geometry.reset();
  cache.prune(); // an external mesh still needs its geometry
  CHECK(cache.size() == 1 && !weak.expired());
  CHECK(mesh->near({.2F,.2F,0}, Mat4{}, .035F));
  mesh.reset();
  cache.prune();
  CHECK(cache.size() == 0 && weak.expired());
}

void surfaceContact() {
  auto g = std::make_shared<PickGeometry>();
  g->positions = {{0,0,0},{1,0,0},{0,1,0}, {100,100,100}};
  g->indices = {0,1,2, 0,0,0, 0,1,999}; // ignore invalid and degenerate triangles
  HandContactMesh mesh(g);
  CHECK(mesh.near({.2F,.2F,.03F}, Mat4{}, .035F));
  CHECK(!mesh.near({.8F,.8F,0}, Mat4{}, .035F)); // inside bounds, outside triangle
  CHECK(!mesh.near({.2F,.2F,.04F}, Mat4{}, .035F));
  CHECK(mesh.near({.51F,.51F,0}, Mat4{}, .035F)); // nearest edge
  CHECK(!mesh.near({100,100,100}, Mat4{}, .035F));
  auto const world = mul(translationMatrix({2,3,4}),
      mul(rotationMatrix(quatFromAxisAngle({0,1,0}, .6F)), scaleMatrix({4,.1F,2})));
  auto const surface = transformPoint(world, {.2F,.2F,0});
  auto const normal = rotate(quatFromAxisAngle({0,1,0}, .6F), {0,0,1});
  CHECK(mesh.near(surface+normal*.03F, world, .035F));
  CHECK(!mesh.near(surface+normal*.04F, world, .035F));
  CHECK(!mesh.near(surface, scaleMatrix({0,1,1}), .035F));
  CHECK(!mesh.near({NAN,0,0}, Mat4{}, .035F));
  CHECK(!mesh.near({0,0,0}, Mat4{}, NAN));
  CHECK(!mesh.near({0,0,0}, Mat4{}, -1));
  // Exercise internal BVH traversal and rejected distant regions.
  auto many = std::make_shared<PickGeometry>();
  for (unsigned i = 0; i < 1000; ++i) {
    float const x = static_cast<float>(i)*2;
    auto const base = static_cast<unsigned>(many->positions.size());
    many->positions.insert(many->positions.end(), {{x,0,0},{x+1,0,0},{x,1,0}});
    many->indices.insert(many->indices.end(), {base,base+1,base+2});
  }
  HandContactMesh large(many);
  CHECK(large.near({1998.2F,.2F,.02F}, Mat4{}, .035F));
  CHECK(!large.near({1999.2F,.2F,.02F}, Mat4{}, .035F));
  HandGesture squeeze;
  CHECK(!squeeze.update(true, 1)); // acquisition cannot start held
  CHECK(!squeeze.update(true, 0));
  CHECK(squeeze.update(true, .8F));
  CHECK(!squeeze.update(false, 1)); // readiness/tracking loss disarms
  CHECK(!squeeze.update(true, 1));
  CHECK(!squeeze.update(true, 0));
  CHECK(squeeze.update(true, .8F));
  HandPose const start{{1,2,3}, {}, true};
  auto const turn = quatFromAxisAngle({0,1,0}, 1.0F);
  HandPose const current{{1.1F,2.2F,3.3F}, turn, true};
  auto const object = translationMatrix({1.2F,2,3});
  auto const result = handGrabTransform(start, current, start.position, object);
  CHECK(near(transformPoint(result, {}), current.position + rotate(turn, {.2F,0,0})));
}

void gestures() {
  HandGesture gesture;
  CHECK(!gesture.update(false, 0.0F));
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.5F));
  CHECK(!gesture.update(true, 0.4F));
  CHECK(!gesture.update(true, 0.549F));
  CHECK(gesture.update(true, 0.55F));
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.41F));
  CHECK(!gesture.update(true, 0.55F));
  CHECK(!gesture.update(true, 0.4F));
  CHECK(gesture.update(true, 0.55F));
  CHECK(!gesture.update(false, 0.0F));
  CHECK(!gesture.update(true, 0.9F));
  CHECK(!gesture.update(true, 0.45F));
  CHECK(!gesture.update(true, 0.9F));
  CHECK(!gesture.update(true, 0.0F));
  CHECK(gesture.update(true, 0.8F));
  gesture.reset();
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.0F));
  CHECK(gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, std::numeric_limits<float>::quiet_NaN()));
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.0F));
  CHECK(gesture.update(true, 0.6F));
}

void selectionGraceAndTracking() {
  HandSelectionLatch latch;
  HandGesture gesture;
  CHECK(!gesture.update(true, 0));
  CHECK(gesture.update(true, 1));
  CHECK(!latch.startsGrab(true, true)); // first pinch cannot move anything
  latch.selected(true);
  CHECK(!latch.recentHit()); // selecting is not a post-release hover
  CHECK(!gesture.update(true, 1));
  CHECK(!gesture.update(true, 0));
  latch.rememberHit();
  latch.advance(.10F);
  CHECK(latch.recentHit());
  CHECK(gesture.update(true, 1));
  CHECK(latch.startsGrab(false, latch.recentHit()));
  latch.advance(.051F);
  CHECK(!latch.recentHit());
  CHECK(!latch.startsGrab(false, latch.recentHit()));
  CHECK(latch.startsGrab(true)); // exact hit still works after expiry
  latch.rememberHit();
  latch.clearHit(); // a different closest object, UI or stick capture
  CHECK(!latch.recentHit());
  latch.rememberHit();
  latch.selected(false); // selection changed/cleared
  CHECK(!latch.startsGrab(true, latch.recentHit()));
  latch.selected(true);
  latch.rememberHit();
  latch.advance(std::numeric_limits<float>::quiet_NaN());
  CHECK(!latch.recentHit());
  latch.rememberHit();
  latch.reset(); // tracking interruption requires new selection
  CHECK(!latch.recentHit());
  CHECK(!latch.startsGrab(true));

  // Aim loss during a held pinch must not reset the gesture. Pinch action or
  // palm tracking loss still stops edits and requires release to re-arm.
  HandGesture held;
  CHECK(!held.update(handPinchSignal(true, true, 0), 0));
  CHECK(held.update(handPinchSignal(true, true, .8F), .8F));
  bool const aimValid = false;
  CHECK(!aimValid && handPinchHeld(handPinchSignal(true, true, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, true, .8F), .8F));
  CHECK(held.pressed());
  CHECK(!handPinchHeld(handPinchSignal(true, false, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, false, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, true, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, true, 0), 0));
  CHECK(held.update(handPinchSignal(true, true, .8F), .8F));
  held.reset(); // palm tracking loss during grab
  CHECK(!held.update(true, .8F));
  CHECK(!held.update(true, 0));
  CHECK(held.update(true, .8F));
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(!handPinchHeld(handPinchSignal(true, true, nan), nan));
  CHECK(!handPinchSignal(false, true, .8F));
}

void carouselDirections() {
  // Ring-relative zones must stay correct as the viewer turns and moves.
  for (float const yaw : {0.0F, 1.1F, -2.3F}) {
    auto const orientation = quatFromAxisAngle({0, 1, 0}, yaw);
    auto const forward = rotate(orientation, {0, 0, -1});
    auto const right = rotate(orientation, {1, 0, 0});
    Vec3 const origin{2, 1, -3};
    auto const centre = origin + forward * 1.25F;
    for (Vec3 const rayOrigin : {origin, origin + right * .7F + Vec3{0, -.3F, 0}}) {
      for (float const side : {-.5F, -.219F, 0.0F, .219F, .5F}) {
        auto const target = centre + right * side;
        Ray const ray{rayOrigin, normalized(target - rayOrigin)};
        CHECK(carouselPinchDirection(ray, origin, forward) ==
              (side < -.22F ? -1 : side > .22F ? 1 : 0));
      }
    }
    // No forward intersection: away-facing, parallel, or beyond the ring.
    CHECK(carouselPinchDirection({origin, -forward}, origin, forward) == 0);
    CHECK(carouselPinchDirection({origin, right}, origin, forward) == 0);
    CHECK(carouselPinchDirection({centre + forward + right, forward},
                                origin, forward) == 0);
    CHECK(carouselPinchDirection({origin, {}}, origin, forward) == 0);
    CHECK(carouselPinchDirection({origin, forward}, origin, {}) == 0);
  }
  // Strict zone boundaries: centre misses, including the edges, are inert.
  for (float const side : {-.22F, .22F}) {
    CHECK(carouselPinchDirection({{side, 0, 0}, {0, 0, -1}}, {}, {0, 0, -1}) == 0);
  }
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(carouselPinchDirection({{}, {nan, 0, -1}}, {}, {0, 0, -1}) == 0);
  CHECK(carouselPinchDirection({{nan, 0, 0}, {0, 0, -1}}, {}, {0, 0, -1}) == 0);
  CHECK(carouselPinchDirection({{}, {0, 0, -1}}, {}, {nan, 0, -1}) == 0);
}

void panels() {
  HandPanel panel;
  auto front = [](Vec3 point) { return Ray{point + Vec3{0, 0, 1}, {0, 0, -1}}; };
  CHECK(panel.hit(front(handPanelButtonCentre(0))) == -1);
  panel.visible = true;
  for (int i : {0, 8, 9, 10, 11}) {
    auto const centre = handPanelButtonCentre(i);
    int const slot = i == 0 ? 0 : i - 7;
    CHECK(near(centre, {(slot % 2 == 0) ? -0.115F : 0.115F,
                        -0.065F * (static_cast<float>(slot / 2) + 0.5F), 0}));
    CHECK(panel.hit(front(centre)) == i);
    CHECK(panel.hit({centre - Vec3{0, 0, 1}, {0, 0, 1}}) == -1);
    CHECK(panel.hit({centre - Vec3{0, 0, 1}, {0, 0, -1}}) == -1);
    CHECK(panel.hit(front(centre + Vec3{0.1049F, 0.0279F, 0})) == i);
    CHECK(panel.hit(front(centre - Vec3{0.1049F, 0.0279F, 0})) == i);
    CHECK(panel.hit(front(centre + Vec3{0.1051F, 0, 0})) == -1);
    CHECK(panel.hit(front(centre + Vec3{0, 0.0281F, 0})) == -1);
  }
  // Exact inclusive edges on representable coordinates.
  auto const centre = handPanelButtonCentre(0);
  CHECK(panel.hit(front({centre.x + 0.105F, centre.y, 0})) == 0);
  CHECK(panel.hit(front({centre.x, centre.y + 0.028F, 0})) == 0);
  CHECK(panel.hit(front({0, -0.1F, 0})) == -1); // column gap
  CHECK(panel.hit(front({-0.115F, -0.065F, 0})) == -1); // row gap
  CHECK(panel.hit(front({0, 0, 0})) == -1);
  CHECK(panel.hit(front({0, -0.5F, 0})) == -1);
  CHECK(panel.hit({{0, 0, 1}, {1, 0, 0}}) == -1);
  CHECK(panel.hit({{0, 0, 1}, {0, 0, -0.0000001F}}) == -1);
  CHECK(panel.hit({{0, 0, 1}, {0, 0, 0}}) == -1);
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(panel.hit({{nan, 0, 1}, {0, 0, -1}}) == -1);
  CHECK(panel.hit({{0, 0, 1}, {0, 0, nan}}) == -1);
  CHECK(near(handPanelButtonCentre(-1), Vec3{}));
  CHECK(near(handPanelButtonCentre(12), Vec3{}));
  CHECK(near(handPanelButtonCentre(1000000), Vec3{}));

  panel.origin = {3, 2, -4};
  panel.orientation = quatFromAxisAngle({0, 1, 0}, 1.1F) *
                      quatFromAxisAngle({1, 0, 0}, 0.4F);
  for (int i : {0, 8, 9, 10, 11}) {
    auto const world = panel.origin + rotate(panel.orientation, handPanelButtonCentre(i));
    auto const normal = rotate(panel.orientation, {0, 0, 1});
    CHECK(panel.hit({world + normal, -normal}) == i);
    CHECK(panel.hit({world - normal, normal}) == -1);
  }
}

void palms() {
  HandPose pose;
  CHECK(!palmFacingUp(pose, false));
  pose.valid = true;
  CHECK(!palmFacingUp(pose, false)); // identity palm points down
  pose.orientation = quatFromAxisAngle({1, 0, 0}, 3.14159265F);
  CHECK(palmFacingUp(pose, false));
  pose.orientation = quatFromAxisAngle({1, 0, 0}, std::acos(-0.65F));
  CHECK(!palmFacingUp(pose, false));
  CHECK(palmFacingUp(pose, true));
  pose.orientation = quatFromAxisAngle({1, 0, 0}, std::acos(-0.5F));
  CHECK(!palmFacingUp(pose, true));
  pose.orientation = quatFromAxisAngle({0, 0, 1}, 3.14159265F);
  CHECK(palmFacingUp(pose, false));
  pose.valid = false;
  CHECK(!palmFacingUp(pose, true));
}

void projection() {
  HandGrabView const view{{0,0,0}, {}, true};
  Vec3 const hand{.1F, .1F, -.5F};
  Vec3 const moved{.15F, .08F, -.5F};
  for (float depth : {1.0F, 10.0F, 100.0F}) {
    Vec3 const surface{depth*.2F, depth*.1F, -depth};
    Vec3 const target = projectedGrabPoint(view, hand, moved, surface);
    CHECK(std::fabs(target.x / depth - surface.x / depth - .1F) < .0001F);
    CHECK(std::fabs(target.y / depth - surface.y / depth + .04F) < .0001F);
    HandPose const start{hand, {}, true};
    HandPose const now{moved, quatFromAxisAngle({0, 1, 0}, .4F), true};
    auto const delta = projectedHandGrabTransform(view, start, now, surface, Mat4{});
    CHECK(length(transformPoint(delta, surface) - target) < .0001F);
  }
  // Depth is relative to the grab-start hand and surface depth: a 10%
  // push/pull moves near and distant surfaces by the same percentage.
  for (float depth : {0.3F, 1.0F, 10.0F, 100.0F}) {
    Vec3 const surface{depth * .2F, depth * .1F, -depth};
    auto const away = projectedGrabPoint(view, hand, {hand.x, hand.y, -.55F}, surface);
    auto const closer = projectedGrabPoint(view, hand, {hand.x, hand.y, -.45F}, surface);
    CHECK(std::fabs(-away.z / depth - 1.1F) < .0001F);
    CHECK(std::fabs(-closer.z / depth - .9F) < .0001F);
    // Depth-only motion preserves the existing projection calculation.
    CHECK(std::fabs(away.x / -away.z - (surface.x / depth + hand.x / .55F - hand.x / .5F)) < .0001F);
    CHECK(std::fabs(closer.y / -closer.z - (surface.y / depth + hand.y / .45F - hand.y / .5F)) < .0001F);
    CHECK(near(projectedGrabPoint(view, hand, hand, surface), surface));
  }
  // The projection clamp must not clamp depth, including a grab started
  // inside the projection near plane. Starting in place must never jump.
  auto const closeHand = projectedGrabPoint(view, hand, {.1F,.1F,-.05F}, {0,0,-10});
  CHECK(std::fabs(closeHand.z + 1.0F) < .0001F);
  Vec3 const nearStart{.01F, .02F, -.099F};
  Vec3 const farSurface{2,1,-10};
  CHECK(near(projectedGrabPoint(view, nearStart, nearStart, farSurface), farSurface));
  auto const nearPush = projectedGrabPoint(view, nearStart, {.01F,.02F,-.198F}, farSurface);
  CHECK(std::fabs(nearPush.z + 19.9F) < .0001F);
  auto const rayStart = Vec3{.06F, 0, -.06F};
  auto const sameRay = projectedGrabPoint(view, rayStart, {.1F,0,-.1F}, {0,0,-10});
  CHECK(std::fabs(sameRay.x) < .0001F);
  auto const tinyStart = projectedGrabPoint(view, {0,0,-.0011F}, {.1F,0,-.0111F}, {0,0,-10});
  auto const behindStart = projectedGrabPoint(view, {0,0,.05F}, {.1F,0,.05F}, {0,0,-10});
  CHECK(near(tinyStart, {.1F,0,-10.01F}));
  CHECK(near(behindStart, {.1F,0,-10}));
  auto const behind = projectedGrabPoint(view, hand, {.1F,.1F,.01F}, {0,0,-10});
  CHECK(std::fabs(behind.z + .05F) < .0001F);
  // Depth and lateral motion combine without losing the proportional gain.
  auto const combined = projectedGrabPoint(view, hand, {.15F, .08F, -.55F}, {2,1,-10});
  CHECK(std::fabs(combined.z + 11.0F) < .0001F);
  CHECK(std::fabs(combined.x / 11.0F - (.2F + .15F / .55F - .1F / .5F)) < .0001F);
  CHECK(std::fabs(combined.y / 11.0F - (.1F + .08F / .55F - .1F / .5F)) < .0001F);
  // When the hand crosses the near plane, the mapping remains bounded and continuous.
  auto const nearDepthA = projectedGrabPoint(view, hand, {.2F,.1F,-.10001F}, {0,0,-10});
  auto const nearDepthB = projectedGrabPoint(view, hand, {.2F,.1F,-.09999F}, {0,0,-10});
  CHECK(length(nearDepthA - nearDepthB) < .01F);
  CHECK(-nearDepthB.z >= .05F);
  auto const orientation = quatFromAxisAngle({0, 1, 0}, .7F);
  HandGrabView const turned{{2,1,3}, orientation, true};
  auto world = [&](Vec3 v) { return turned.eye + rotate(orientation, v); };
  auto const target = projectedGrabPoint(turned, world(hand), world(moved), world({2,1,-10}));
  CHECK(length(target - world({3,.6F,-10})) < .0001F);
  auto const nearA = projectedGrabPoint(view, hand, {.2F,.1F,-.10001F}, {0,0,-10});
  auto const nearB = projectedGrabPoint(view, hand, {.2F,.1F,-.09999F}, {0,0,-10});
  CHECK(length(nearA - nearB) < .01F);
  auto const fallback = projectedGrabPoint(view, {0,0,0}, {1,0,0}, {0,0,-10});

  CHECK(near(fallback, {1,0,-10}));
}

void grabs() {

  HandPose const start{{1, 2, 3},
                       quatFromAxisAngle({1, 0, 0}, 0.4F) *
                       quatFromAxisAngle({0, 1, 0}, -0.7F), true};
  Vec3 const pivot{2, -1, 0.5F};
  Mat4 const initial = trs({4, 2, -3}, quatFromAxisAngle({0, 0, 1}, 0.3F), {2, 3, 4});
  CHECK(near(handGrabTransform(start, start, pivot, initial), initial));
  auto slight = start;
  slight.orientation = quatFromAxisAngle({0, 1, 0}, .0002F) * start.orientation;
  auto const slightResult = handGrabTransform(start, slight, pivot, initial);
  CHECK(length(transformVector(slightResult, {1, 0, 0}) - transformVector(initial, {1, 0, 0})) > .0001F);

  for (Vec3 const axis : {Vec3{0, 1, 0}, Vec3{1, 0, 0}, Vec3{0, 0, 1}}) {
    auto const delta = quatFromAxisAngle(axis, 1.2F);
    HandPose current = start;
    current.orientation = delta * start.orientation;
    current.position = start.position + Vec3{-0.3F, 0.2F, 0.7F};
    auto const result = handGrabTransform(start, current, pivot, initial);
    // Independent point/vector expectations catch quaternion order, pivot,
    // translation order, and preservation of initial orientation/scale.
    for (Vec3 const point : {Vec3{}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
      CHECK(near(transformPoint(result, point),
                 pivot + (current.position - start.position) +
                 rotate(delta, transformPoint(initial, point) - pivot)));
    }
    CHECK(near(transformVector(result, {1, 2, 3}),
               rotate(delta, transformVector(initial, {1, 2, 3}))));
  }
  HandPose current = start;
  current.position = start.position + Vec3{0.25F, -0.5F, 1.5F};
  auto const expected = mul(translationMatrix((current.position - start.position)), initial);
  for (int i = 0; i < 10000; ++i) {
    CHECK(near(handGrabTransform(start, current, pivot, initial), expected));
  }
  // Returning to the captured start must undo every intermediate sample.
  for (int i = 0; i < 1000; ++i) {
    current.position = start.position + Vec3{static_cast<float>(i) * 0.001F, 0, 0};
    current.orientation = quatFromAxisAngle({0, 1, 0}, static_cast<float>(i) * 0.01F) *
                          start.orientation;
    (void)handGrabTransform(start, current, pivot, initial);
  }
  CHECK(near(handGrabTransform(start, start, pivot, initial), initial));
  current.valid = false;
  CHECK(near(handGrabTransform(start, current, pivot, initial), initial));
  CHECK(near(handGrabTransform(current, start, pivot, initial), initial));
}
} // namespace

int main() {
  HandSelectionLatch latch;
  HandGesture deliberate;
  CHECK(!deliberate.update(true, 0));
  CHECK(deliberate.update(true, 1));
  CHECK(!latch.startsGrab(true)); // first press selects, never moves
  latch.selected(true);
  CHECK(!deliberate.update(true, 1)); // a held first press cannot grab
  CHECK(!deliberate.update(true, 0));
  CHECK(deliberate.update(true, 1));
  CHECK(latch.startsGrab(true));
  CHECK(!latch.startsGrab(false)); // a different hit selects instead
  latch.reset();
  CHECK(!latch.startsGrab(true));
  HandPose const origin{{0,0,0}, {}, true};
  HandPose const tiny{{.001F,0,0}, {}, true};
  auto const smoothed = smoothHandPose(origin, tiny, .01F);
  CHECK(smoothed.position.x > 0 && smoothed.position.x < .001F);
  CHECK(smoothHandPose(smoothed, tiny, .01F).position.x > smoothed.position.x);
  CHECK(handScaleRatio(.2F, .4F) == 2.0F);
  CHECK(handScaleRatio(.0F, .4F) == 1.0F);
  graspArbitration();
  contactCacheLifetime();
  surfaceContact();
  gestures();
  selectionGraceAndTracking();

  carouselDirections();
  panels();
  palms();
  projection();
  grabs();
  std::printf("Hand tests: %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
