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

// All samples go through the public API; no internal state assumptions.
void swipeAcquire(HandSwipe& swipe, Vec3 position = {}, Vec3 right = {1, 0, 0},
                  int rate = 60) {
  auto const dt = 1.0F / static_cast<float>(rate);
  for (int i = 0; i <= rate * 3 / 10; ++i) {
    CHECK(swipe.update(true, true, position, right, dt) == 0);
  }
}

void swipes() {
  Vec3 const right{1, 0, 0};
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  auto const inf = std::numeric_limits<float>::infinity();

  // Acquisition needs a full stationary quarter second, not merely tracking.
  {
    HandSwipe swipe;
    CHECK(swipe.update(true, true, {}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {.2F, 0, 0}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {.2F, 0, 0}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {.2F, 0, 0}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {0, 0, 0}, right, .125F) == 1);
  }
  // Exact inclusive distance threshold and viewer-relative direction contract.
  for (float const direction : {-1.0F, 1.0F}) {
    HandSwipe swipe;
    swipeAcquire(swipe);
    CHECK(swipe.update(true, true, {direction * .159F, 0, 0}, right, .05F) == 0);
    CHECK(swipe.update(true, true, {direction * .16F, 0, 0}, right, .05F) ==
          (direction < 0 ? 1 : -1));
  }
  // Bounded tracking noise cannot fire, even over many acquisition windows.
  {
    HandSwipe swipe;
    for (int i = 0; i < 600; ++i) {
      auto const noise = (i % 2 == 0 ? 1.0F : -1.0F) * .0003F;
      CHECK(swipe.update(true, true, {noise, -noise, noise}, right, 1.0F / 60) == 0);
    }
    CHECK(swipe.update(true, true, {-.18F, 0, 0}, right, .05F) == 1);
  }
  // A fast return, continued travel and a sub-quarter-second pause do not
  // rearm. Only a new quiet interval permits a second event.
  {
    HandSwipe swipe;
    swipeAcquire(swipe);
    CHECK(swipe.update(true, true, {-.18F, 0, 0}, right, .05F) == 1);
    for (float const x : {-.09F, 0.0F, .09F, .18F, .27F}) {
      CHECK(swipe.update(true, true, {x, 0, 0}, right, .05F) == 0);
    }
    CHECK(swipe.update(true, true, {.27F, 0, 0}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {.05F, 0, 0}, right, .05F) == 0);
    CHECK(swipe.update(true, true, {.05F, 0, 0}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {.05F, 0, 0}, right, .125F) == 0);
    CHECK(swipe.update(true, true, {.23F, 0, 0}, right, .05F) == -1);
  }
  // Off-axis distance is radial (not a separate limit on each component).
  // A rejected stroke cannot recover just by returning to the swipe plane.
  for (Vec3 const offAxis : {Vec3{0, .101F, 0}, Vec3{0, 0, -.101F},
                             Vec3{0, .075F, .075F}}) {
    HandSwipe swipe;
    swipeAcquire(swipe);
    CHECK(swipe.update(true, true, Vec3{.17F, 0, 0} + offAxis, right, .05F) == 0);
    CHECK(swipe.update(true, true, {.35F, 0, 0}, right, .05F) == 0);
    swipeAcquire(swipe, {.35F, 0, 0});
    CHECK(swipe.update(true, true, {.17F, 0, 0}, right, .05F) == 1);
  }
  {
    HandSwipe swipe;
    swipeAcquire(swipe);
    CHECK(swipe.update(true, true, {.16F, .10F, 0}, right, .05F) == -1);
  }
  // A deliberate stroke near the deadline succeeds; one beyond it fails.
  for (int const frames : {6, 7}) {
    HandSwipe swipe;
    swipeAcquire(swipe);
    for (int i = 1; i < frames; ++i) {
      CHECK(swipe.update(true, true, {static_cast<float>(i) * .026F, 0, 0},
                         right, .125F) == 0);
    }
    CHECK(swipe.update(true, true, {.19F, 0, 0}, right, .125F) ==
          (frames == 6 ? -1 : 0));
  }
  // Continuous slow drift exceeds the quiet-speed cutoff but cannot reach
  // 16 cm in the 0.8 s stroke window. Later travel must not resurrect it.
  {
    HandSwipe swipe;
    swipeAcquire(swipe);
    for (int i = 1; i <= 24; ++i) {
      CHECK(swipe.update(true, true, {static_cast<float>(i) * .007F, 0, 0},
                         right, .05F) == 0);
    }
    swipeAcquire(swipe, {.168F, 0, 0});
    CHECK(swipe.update(true, true, {-.012F, 0, 0}, right, .05F) == 1);
  }
  // Invalid tracking, closed/pinching hand, bad time and non-finite positions
  // must discard both acquisition and a partially completed stroke.
  for (int invalid = 0; invalid < 12; ++invalid) {
    HandSwipe swipe;
    swipeAcquire(swipe);
    CHECK(swipe.update(true, true, {.1F, 0, 0}, right, .05F) == 0);
    bool const valid = invalid != 0;
    bool const open = invalid != 1; // caller supplies !pinching as open
    float const dt = invalid == 2 ? 0.0F : invalid == 3 ? -.01F :
                     invalid == 4 ? .151F : invalid == 5 ? nan :
                     invalid == 6 ? inf : .05F;
    Vec3 const position = invalid == 7 ? Vec3{nan, 0, 0} :
                          invalid == 8 ? Vec3{0, nan, 0} :
                          invalid == 9 ? Vec3{0, 0, nan} :
                          invalid == 10 ? Vec3{inf, 0, 0} :
                          invalid == 11 ? Vec3{0, 0, -inf} : Vec3{};
    CHECK(swipe.update(valid, open, position, right, dt) == 0);
    CHECK(swipe.update(true, true, {.3F, 0, 0}, right, .05F) == 0);
    CHECK(swipe.update(true, true, {.5F, 0, 0}, right, .05F) == 0);
    swipeAcquire(swipe, {.5F, 0, 0});
    CHECK(swipe.update(true, true, {.32F, 0, 0}, right, .05F) == 1);
  }
  // A closed hand never acquires, even while held still for long enough.
  {
    HandSwipe swipe;
    for (int i = 0; i < 60; ++i) {
      CHECK(swipe.update(true, false, {}, right, 1.0F / 60) == 0);
    }
    CHECK(swipe.update(true, false, {-.2F, 0, 0}, right, .05F) == 0);
    CHECK(swipe.update(true, true, {}, right, .05F) == 0);
    CHECK(swipe.update(true, true, {-.2F, 0, 0}, right, .05F) == 0);
  }
  // Identical physical strokes at supported frame rates and rotated viewer
  // headings yield exactly one signed step, never one per frame.
  for (int const rate : {30, 60, 90, 120}) {
    for (Vec3 const axis : {right, Vec3{0, 0, -1}, normalized(Vec3{1, 0, 1})}) {
      for (int const direction : {-1, 1}) {
        HandSwipe swipe;
        Vec3 const origin{2, 1, -3};
        swipeAcquire(swipe, origin, axis, rate);
        auto const dt = 1.0F / static_cast<float>(rate);
        auto const frames = rate * 2 / 5;
        int events = 0;
        for (int i = 1; i <= frames; ++i) {
          auto const distance = .20F * static_cast<float>(direction * i) /
                                static_cast<float>(frames);
          auto const event = swipe.update(true, true, origin + axis * distance, axis, dt);
          if (event != 0) {
            CHECK(event == -direction);
            ++events;
          }
        }
        CHECK(events == 1);
        for (int i = 1; i <= frames; ++i) {
          auto const distance = .20F * static_cast<float>(direction) *
                                (1.0F - static_cast<float>(i) / static_cast<float>(frames));
          CHECK(swipe.update(true, true, origin + axis * distance, axis, dt) == 0);
        }
      }
    }
  }
}

void panels() {
  HandPanel panel;
  auto front = [](Vec3 point) { return Ray{point + Vec3{0, 0, 1}, {0, 0, -1}}; };
  CHECK(panel.hit(front(handPanelButtonCentre(0))) == -1);
  panel.visible = true;
  for (int i = 0; i < 12; ++i) {
    auto const centre = handPanelButtonCentre(i);
    CHECK(near(centre, {(i % 2 == 0) ? -0.115F : 0.115F,
                        -0.065F * (static_cast<float>(i / 2) + 0.5F), 0}));
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
  for (int i = 0; i < 12; ++i) {
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
                 pivot + current.position - start.position +
                 rotate(delta, transformPoint(initial, point) - pivot)));
    }
    CHECK(near(transformVector(result, {1, 2, 3}),
               rotate(delta, transformVector(initial, {1, 2, 3}))));
  }
  HandPose current = start;
  current.position = start.position + Vec3{0.25F, -0.5F, 1.5F};
  auto const expected = mul(translationMatrix(current.position - start.position), initial);
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

void swipeSlowStart() {
  HandSwipe swipe;
  swipeAcquire(swipe);
  for (int i = 1; i <= 16; ++i) {
    CHECK(swipe.update(true, true, {-.01001F * i,0,0}, {1,0,0}, .125F) == 0);
  }
}

void swipeSmoothStart() {
  HandSwipe swipe;
  swipeAcquire(swipe);
  for (int i = 1; i <= 16; ++i) {
    CHECK(swipe.update(true, true, {-.01001F * i,0,0}, {1,0,0}, .05F) == (i == 16 ? 1 : 0));
  }
}

void swipeJitter() {
  for (int rate : {30, 60, 90, 120}) {
    HandSwipe swipe;
    float const dt = 1.0F / rate;
    for (int i = 0; i < rate; ++i) {
      float const noise = i % 2 ? .006F : -.006F;
      CHECK(swipe.update(true, true, {noise, -noise, noise}, {1,0,0}, dt) == 0);
    }
    CHECK(swipe.armed());
    int events = 0;
    for (int i = 1; i <= rate / 2; ++i) {
      events += swipe.update(true, true, {-.20F * i / (rate / 2),0,0}, {1,0,0}, dt);
    }
    CHECK(events == 1);
  }
}

void swipeTimingBoundary() {
  for (int hz : {10, 30, 60, 90, 120}) {
    for (bool over : {false, true}) {
      HandSwipe swipe;
      float const dt = 1.0F / static_cast<float>(hz);
      for (int i = 0; i < hz; ++i) {
        CHECK(swipe.update(true, true, {}, {1,0,0}, dt) == 0);
      }
      // First sample leaves the 2.5 cm pause envelope; subsequent samples
      // complete the stroke within (or just beyond) its time allowance.
      int const frames = hz * 4 / 5 + (over ? 1 : 0);
      int fired = 0;
      for (int i = 1; i <= frames; ++i) {
        fired += swipe.update(true, true, {-.03F - .13001F * static_cast<float>(i - 1) / (frames - 1),0,0}, {1,0,0}, dt);
      }
      CHECK(fired == (over ? 0 : 1));
    }
  }
}

int main() {
  swipeSlowStart();
  swipeSmoothStart();
  swipeJitter();
  swipeTimingBoundary();
  gestures();
  swipes();
  panels();
  palms();
  grabs();
  std::printf("Hand tests: %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
