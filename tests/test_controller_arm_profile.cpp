#include "controller_arm_profile.h"

#include <cstdio>
#include <limits>

using namespace vredit;

namespace {
int failures = 0;
#define CHECK(expr) do { if (!(expr)) { \
  std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); ++failures; \
} } while (false)

void profileTransitions() {
  ControllerArmProfileTracker profiles;
  CHECK(!profiles.known(0) && !profiles.known(1));
  profiles.sample(0, false, 42);
  profiles.sample(1, true, 20);
  CHECK(!profiles.known(0) && profiles.known(1));
  CHECK(!profiles.changed(0) && !profiles.changed(1));
  profiles.sample(0, true, 10);
  CHECK(profiles.known(0) && !profiles.changed(0));
  profiles.consume();
  profiles.sample(0, true, 10);
  profiles.sample(1, true, 20);
  CHECK(!profiles.changed(0) && !profiles.changed(1));

  profiles.sample(0, true, 11);
  CHECK(profiles.changed(0) && !profiles.changed(1));
  profiles.consume();
  CHECK(!profiles.changed(0));
  profiles.sample(1, true, 21);
  CHECK(!profiles.changed(0) && profiles.changed(1));
  profiles.sample(0, true, 12);
  CHECK(profiles.changed(0) && profiles.changed(1));
  profiles.consume();

  // Null is a successful, observable no-profile state in either direction.
  profiles.sample(0, true, 0);
  CHECK(profiles.known(0) && profiles.changed(0));
  profiles.consume();
  profiles.sample(0, true, 0);
  CHECK(!profiles.changed(0));
  profiles.sample(0, true, 12);
  CHECK(profiles.changed(0));
  profiles.consume();

  // A failed query cannot fabricate a null transition or erase the baseline.
  profiles.sample(0, false, 0);
  CHECK(!profiles.known(0) && !profiles.changed(0));
  profiles.sample(0, true, 12);
  CHECK(profiles.known(0) && !profiles.changed(0));
  profiles.sample(0, false, 999);
  profiles.sample(0, true, 13);
  CHECK(profiles.changed(0));
  // A later failure must not swallow an already observed change.
  profiles.sample(0, false, 0);
  CHECK(!profiles.known(0) && profiles.changed(0));
  profiles.consume();
  CHECK(!profiles.changed(0));

  // Multiple observations on skipped frames remain pending, even if restored.
  profiles.sample(1, true, 22);
  profiles.sample(1, true, 21);
  CHECK(profiles.changed(1));
  profiles.consume();
  CHECK(!profiles.changed(1));

  ControllerArmProfileTracker initiallyNone;
  initiallyNone.sample(0, true, 0);
  CHECK(initiallyNone.known(0) && !initiallyNone.changed(0));
  initiallyNone.sample(0, true, 10);
  CHECK(initiallyNone.changed(0));
}

void focusTransitions() {
  ControllerArmProfileTracker profiles;
  profiles.observeFocus(false);
  CHECK(!profiles.focusInterrupted());
  profiles.observeFocus(true);
  CHECK(!profiles.focusInterrupted());
  profiles.observeFocus(true);
  CHECK(!profiles.focusInterrupted());
  profiles.observeFocus(false);
  CHECK(profiles.focusInterrupted());
  profiles.observeFocus(true); // loss and regain between delivered frames
  CHECK(profiles.focusInterrupted());
  profiles.consume();
  CHECK(!profiles.focusInterrupted());
  profiles.observeFocus(false);
  CHECK(profiles.focusInterrupted());
  profiles.consume();
  profiles.observeFocus(false);
  CHECK(!profiles.focusInterrupted());
}

void visualTransitions() {
  using P = ControllerArmProfile;
  CHECK(classifyControllerArmProfile(false, 10, true, 20) == P::unknown);
  CHECK(classifyControllerArmProfile(true, 0, true, 20) == P::none);
  CHECK(classifyControllerArmProfile(true, 20, true, 20) == P::hand);
  CHECK(classifyControllerArmProfile(true, 10, true, 20) == P::controller);
  CHECK(classifyControllerArmProfile(true, 10, false, 20) == P::unknown);
  CHECK(classifyControllerArmProfile(true, 10, true, 0) == P::unknown);
  CHECK(!controllerArmGripActive(false, true));
  CHECK(!controllerArmGripActive(true, false));
  CHECK(controllerArmGripActive(true, true));

  ControllerArmContinuity continuity;
  continuity.sample(0, true);
  continuity.sample(1, true);
  CHECK(!continuity.interrupted(0) && !continuity.interrupted(1));
  continuity.sample(0, controllerArmGripActive(true, false));
  continuity.sample(0, true); // restored before delivery still resets
  CHECK(continuity.interrupted(0) && !continuity.interrupted(1));
  continuity.consume();
  CHECK(!continuity.interrupted(0));
  continuity.skipFrame();
  continuity.sample(0, true);
  continuity.sample(1, true);
  CHECK(continuity.interrupted(0) && continuity.interrupted(1));
  continuity.consume();
  CHECK(!continuity.interrupted(0) && !continuity.interrupted(1));
  continuity.sample(1, false);
  CHECK(!continuity.interrupted(0) && continuity.interrupted(1));

  // An event can signal a profile round trip entirely between samples.
  continuity.consume();
  ControllerArmProfileTracker profiles;
  profiles.sample(0, true, 10);
  profiles.sample(1, true, 20);
  continuity.profileChanged();
  profiles.sample(0, true, 10);
  profiles.sample(1, true, 20);
  continuity.sample(0, true);
  continuity.sample(1, true);
  CHECK(continuity.interrupted(0) && continuity.interrupted(1));
  CHECK(!profiles.changed(0) && !profiles.changed(1));
  CHECK(classifyControllerArmProfile(profiles.known(0), 10, true, 30) == P::controller);
  CHECK(classifyControllerArmProfile(profiles.known(1), 20, true, 30) == P::controller);
  CHECK(controllerArmVisualSample(true, true, true, 0.75F) == 0.75F);
  continuity.consume();
  CHECK(!continuity.interrupted(0) && !continuity.interrupted(1));
}

void visualSamples() {
  CHECK(controllerArmVisualSample(true, true, true, 0.75F) == 0.75F);
  CHECK(controllerArmVisualSample(true, true, true, -0.2F) == 0.0F);
  CHECK(controllerArmVisualSample(true, true, true, 1.2F) == 1.0F);
  CHECK(controllerArmVisualSample(false, true, true, 0.75F) == 0.0F);
  CHECK(controllerArmVisualSample(true, false, true, 0.75F) == 0.0F);
  CHECK(controllerArmVisualSample(true, true, false, 0.75F) == 0.0F);
  CHECK(controllerArmVisualSample(true, true, true, std::numeric_limits<float>::infinity()) == 0.0F);
  CHECK(controllerArmVisualSample(true, true, true, std::numeric_limits<float>::quiet_NaN()) == 0.0F);
}
} // namespace

int main() {
  profileTransitions();
  focusTransitions();
  visualTransitions();
  visualSamples();
  if (failures != 0) {
    return 1;
  }
  std::printf("controller arm profile tests passed\n");
  return 0;
}
