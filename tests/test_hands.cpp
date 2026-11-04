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
  gestures();

  carouselDirections();
  panels();
  palms();
  grabs();
  std::printf("Hand tests: %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
