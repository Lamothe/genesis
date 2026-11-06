#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vredit {

// Retain observed changes until a frame is actually delivered. A successful
// null path is a real "no profile" observation, not a failed query. Failed
// queries leave the last successful baseline intact.
class ControllerArmProfileTracker {
public:
  void sample(int hand, bool querySucceeded, std::uint64_t profile) {
    known_[hand] = querySucceeded;
    if (!querySucceeded) {
      return;
    }
    if (haveBaseline_[hand] && profiles_[hand] != profile) {
      changed_[hand] = true;
    }
    profiles_[hand] = profile;
    haveBaseline_[hand] = true;
  }

  void observeFocus(bool focused) {
    if (haveFocus_ && focused_ && !focused) {
      focusInterrupted_ = true;
    }
    focused_ = focused;
    haveFocus_ = true;
  }

  bool known(int hand) const { return known_[hand]; }
  bool changed(int hand) const { return changed_[hand]; }
  bool focusInterrupted() const { return focusInterrupted_; }
  void consume() {
    changed_[0] = false;
    changed_[1] = false;
    focusInterrupted_ = false;
  }

private:
  std::uint64_t profiles_[2]{};
  bool haveBaseline_[2]{};
  bool known_[2]{};
  bool changed_[2]{};
  bool haveFocus_{false};
  bool focused_{false};
  bool focusInterrupted_{false};
};

enum class ControllerArmProfile { unknown, none, hand, controller };

// A failed query or unresolved hand profile cannot identify a controller.
inline ControllerArmProfile classifyControllerArmProfile(bool querySucceeded,
    std::uint64_t profile, bool handProfileKnown, std::uint64_t handProfile) {
  if (!querySucceeded) { return ControllerArmProfile::unknown; }
  if (profile == 0) { return ControllerArmProfile::none; }
  if (!handProfileKnown || handProfile == 0) { return ControllerArmProfile::unknown; }
  return profile == handProfile ? ControllerArmProfile::hand : ControllerArmProfile::controller;
}

// Interruptions remain pending until a frame is delivered to the visual solver.
class ControllerArmContinuity {
public:
  void sample(int hand, bool tracked) {
    if (!tracked) { interrupted_[hand] = true; }
  }
  void skipFrame() { interrupted_[0] = interrupted_[1] = true; }
  // OpenXR profile events identify no side, and a round trip can be missed.
  void profileChanged() { interrupted_[0] = interrupted_[1] = true; }
  bool interrupted(int hand) const { return interrupted_[hand]; }
  void consume() { interrupted_[0] = interrupted_[1] = false; }
private:
  bool interrupted_[2]{};
};

inline bool controllerArmGripActive(bool querySucceeded, bool active) {
  return querySucceeded && active;
}

// Visuals use the physical action sample, not the editing interruption latch.
inline float controllerArmVisualSample(bool syncSucceeded, bool querySucceeded,
                                       bool active, float value) {
  if (!syncSucceeded || !querySucceeded || !active || !std::isfinite(value)) {
    return 0.0F;
  }
  return std::clamp(value, 0.0F, 1.0F);
}

} // namespace vredit
