#pragma once

#include <cmath>
#include <cstdint>

#include "hand_input.h"
#include "vecmath.h"

namespace vredit {

// Who owns the one live transform grab. A grab is either a tracked hand's
// (legacy, disabled for model manipulation) or the right controller's physical
// grip pose; the two never overlap, and ending one never ends the other by
// inference.
enum class GrabOwner : std::uint8_t { none, hand, controller };

// Where the tools menu was opened from. A hand menu follows the palm; the
// controller menu is latched in the world at open so pointing at it is stable.
enum class MenuOwner : std::uint8_t { none, controller, hand };

// A physical controller pose from an XR grip space (never the aim ray and
// never a head-gaze fallback). VALID is what OpenXR promises about the pose;
// TRACKED is the measured, actively observed quality. Spatial actions that can
// move a model require both.
struct ControllerPose {
  Vec3 position{};
  Quat orientation{};
  bool valid{false};
  bool tracked{false};

  bool usable() const {
    return valid && tracked &&
           std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z) &&
           std::isfinite(orientation.x) && std::isfinite(orientation.y) &&
           std::isfinite(orientation.z) && std::isfinite(orientation.w);
  }

  HandPose handPose() const {
    return {position, orientation, usable()};
  }
};

// Trigger samples are actionable only after a physical release observed while
// active. Interruptions swallow edges until that release, including skipped
// frames. An inactive or non-finite sample is never a physical release.
struct SelectTriggerOutput {
  bool active{false};
  bool held{true};
  bool pressed{false};
  bool released{false};
};

class SelectTriggerLatch {
 public:
  void interrupt() { blocked_ = true; }

  // A view loss is an interruption even if the controller trigger is active.
  SelectTriggerOutput updateTracked(bool viewTracked, bool active, float value) {
    return update(viewTracked && active, value);
  }

  SelectTriggerOutput update(bool active, float value) {
    SelectTriggerOutput output;
    output.active = active && std::isfinite(value);
    if (!output.active) {
      interrupt();
      return output;
    }
    if (blocked_) {
      if (value <= 0.4F) {
        blocked_ = false;
        held_ = false;
        output.held = false;
      }
      return output;
    }
    bool const held = value > (held_ ? 0.4F : 0.55F);
    output.held = held;
    output.pressed = held && !held_;
    output.released = !held && held_;
    held_ = held;
    return output;
  }

 private:
  bool blocked_{true};
  bool held_{false};
};

// Left trigger either clears on release, or joins an already live
// right grab. A press held across any context transition must be released.
class LeftTriggerWorkflow {
 public:
  enum class Action : std::uint8_t { none, clearSelection, beginScale, endScale, trackingLost };
  void interrupt() { blocked_ = true; candidate_ = false; scaling_ = false; }
  Action update(bool enabled, bool active, bool held, bool pressed, bool released,
                bool rightHeld, bool grabbing, bool posesUsable, float /*dt*/) {
    if (!enabled || !active) {
      bool const lost = scaling_;
      interrupt();
      return lost ? Action::trackingLost : Action::none;
    }
    if (blocked_) {
      if (!held) { blocked_ = false; }
      return Action::none;
    }
    if (scaling_) {
      if (!grabbing || !rightHeld) { scaling_ = false; candidate_ = false; return Action::none; }
      if (!posesUsable) { interrupt(); return Action::trackingLost; }
      if (released) { scaling_ = false; return Action::endScale; }
      return Action::none;
    }
    if (rightHeld) { candidate_ = false; }
    if (pressed) {
      if (grabbing && rightHeld && posesUsable) {
        scaling_ = true;
        candidate_ = false;
        return Action::beginScale;
      }
      candidate_ = !rightHeld && !grabbing;
    }
    if (released) {
      bool const clear = candidate_ && !rightHeld && !grabbing;
      candidate_ = false;
      return clear ? Action::clearSelection : Action::none;
    }
    return Action::none;
  }
  bool scaling() const { return scaling_; }
 private:
  bool blocked_{true};
  bool candidate_{false};
  bool scaling_{false};
};

// A boolean hold is actionable only after an active physical release. Inactive
// samples and explicit interruptions block it, even when the supplied value is
// false. The first active release rearms silently; a later press is actionable.
class BooleanRearmLatch {
 public:
  void interrupt() { blocked_ = true; }

  bool updateTracked(bool viewTracked, bool active, bool held) {
    return update(viewTracked && active, held);
  }

  bool update(bool active, bool held) {
    if (!active) {
      interrupt();
      return false;
    }
    if (!held) {
      blocked_ = false;
      return false;
    }
    return !blocked_;
  }

 private:
  bool blocked_{true};
};

// Edge-triggered menu/carousel/history actions must not turn an interruption into a
// release. A pending press is also discarded if its frame is interrupted.
class BooleanPressLatch {
 public:
  void interrupt() { blocked_ = true; held_ = false; pending_ = false; }

  void updateTracked(bool viewTracked, bool active, bool held) {
    if (!viewTracked || !active) {
      interrupt();
      return;
    }
    if (!held) {
      blocked_ = false;
      held_ = false;
      return;
    }
    if (!blocked_ && !held_) { pending_ = true; }
    held_ = true;
  }

  bool pending() const { return pending_; }
  void consume() { pending_ = false; }

 private:
  bool blocked_{true};
  bool held_{false};
  bool pending_{false};
};

// Only a confirmed no-stick profile can bypass the neutral-sample check.
// A temporarily inactive or corrupt stick on any other profile remains unsafe.
inline bool carouselStickNeutral(bool active, Vec2 stick, bool confirmedNoStick = false) {
  return (!active && confirmedNoStick) || (active && std::isfinite(stick.x) && std::isfinite(stick.y) &&
                             std::fabs(stick.x) <= 0.5F);
}

// The carousel's centre trigger. Placement is no longer a pointing action:
// the trigger loads whatever the settled centre slot holds. That is only safe
// once the ring has stopped moving and THAT index has finished loading, so the
// gate arms on a short settle dwell and disarms on any index/loading change.
// Every physical press is consumed even while blocked, so holding the trigger
// through a load can never place the wrong model when the load completes.
class CarouselPlacementGate {
 public:
  static constexpr float kSettleSeconds = 0.12F;

  // Call once per rendered frame while the carousel is open. `triggerPressed`
  // is the analog trigger's rising edge; `triggerHeld` the debounced held
  // state. Returns true for exactly one frame: place the centred model now.
  bool update(bool open, int centreIndex, bool centreLoaded, bool atRest,
              bool triggerHeld, bool triggerPressed, float dt) {
    if (!open) {
      reset();
      return false;
    }
    if (!(dt >= 0.0F) || !std::isfinite(dt)) {
      dt = 0.0F;
    }
    if (!triggerHeld) {
      requireRelease_ = false;
    }
    if (!atRest || !centreLoaded || centreIndex < 0) {
      // Unsettled or loading: disarm and restart the dwell. A press in this
      // state is swallowed below.
      index_ = centreIndex;
      settle_ = 0.0F;
      armed_ = false;
    } else if (centreIndex != index_) {
      index_ = centreIndex;
      settle_ = 0.0F;
      armed_ = false;
    } else {
      settle_ += dt;
      if (settle_ >= kSettleSeconds) {
        armed_ = true;
      }
    }
    ready_ = armed_ && !requireRelease_ && atRest && centreLoaded && centreIndex >= 0 &&
             centreIndex == index_;
    bool place = false;
    if (triggerPressed) {
      place = ready_;
      requireRelease_ = true; // one press = one attempt, blocked or not
    }
    return place;
  }

  // Would a fresh trigger press place on this frame? Drives the centre
  // preview's rim highlight (readiness, not aim).
  bool ready() const { return ready_; }

  // Opens/closes and tracking interruptions: the next press must be a fresh
  // physical press (the trigger has to be released first).
  void reset() {
    index_ = -1;
    settle_ = 0.0F;
    armed_ = false;
    ready_ = false;
    requireRelease_ = true;
  }

 private:
  int index_{-1};
  float settle_{0.0F};
  bool armed_{false};
  bool ready_{false};
  bool requireRelease_{true};
};

// The right trigger's viewer workflow:
//   1. the first press SELECTS the target under the aim (or clears when the
//      press is a genuine miss) and is consumed, so it can never also grab;
//   2. release;
//   3. the next press starts a physical grab of the CURRENT selection even
//      on a tracked ray miss, unless it targets another object; the grab then
//      follows the grip pose until the trigger is released.
// A press without tracked aim is consumed without selecting, clearing or
// grabbing; it cannot replay if aim tracking returns while held.
// One physical press gives one decision. `reset()` after a tracking loss or a
// menu/gallery transition requires a real release before anything may start.
class ControllerTriggerWorkflow {
 public:
  enum class Target : std::uint8_t { none, other, selected };
  enum class Action : std::uint8_t { none, select, clearSelection, beginGrab };

  void reset() { blocked_ = true; primed_ = false; }

  Action update(bool enabled, bool triggerHeld, bool triggerPressed,
                bool aimTracked, Target target, bool hasSelection) {
    if (!hasSelection) {
      primed_ = false;
    }
    if (enabled && !triggerHeld) {
      blocked_ = false;
    }
    if (!enabled || !triggerPressed || blocked_) {
      return Action::none;
    }
    blocked_ = true; // exactly one decision per physical press
    if (!aimTracked) {
      return Action::none; // unavailable aim is not a ray miss
    }
    switch (target) {
    case Target::selected:
      if (primed_ && hasSelection) {
        return Action::beginGrab;
      }
      primed_ = true;
      return Action::select;
    case Target::other:
      primed_ = true;
      return Action::select;
    case Target::none:
      if (primed_ && hasSelection) {
        return Action::beginGrab;
      }
      [[fallthrough]];
    default:
      primed_ = false;
      return hasSelection ? Action::clearSelection : Action::none;
    }
  }

  // True until the trigger returns to release after a consumed press/reset.
  bool blocked() const { return blocked_; }
  // A previously consumed selection press can start a grab after release.
  bool primed() const { return primed_; }

 private:
  bool blocked_{false};
  bool primed_{false};
};

// The right controller's ray + trigger against the tools panel. A press
// captures exactly one button (or a miss, which is swallowed: the panel owns
// the trigger and nothing clicks through to the scene). Models/Undo/Redo/
// Backdrop fire on release when the aim is still on the same button; Delete
// is a hold that fires at full charge and cancels when released early. Sliding
// off a button cancels it, so a held trigger never retargets.
class ToolPanelPress {
 public:
  static constexpr int kModels = 0;
  static constexpr int kUndo = 8;
  static constexpr int kRedo = 9;
  static constexpr int kDelete = 10;
  static constexpr int kBackdrop = 11;
  static constexpr float kDeleteHoldSeconds = 0.6F;

  enum class Action : std::uint8_t { none, models, undo, redo, deleteNow, backdrop };

  void press(int button) {
    button_ = button;
    held_ = 0.0F;
    fired_ = false;
    cancelled_ = false;
  }

  // Call every held frame with the current hover (or -1 for a miss).
  // Leaving the captured button permanently cancels it until release.
  Action hold(bool triggerHeld, int buttonUnderAim, float dt) {
    if (!active() || !triggerHeld) {
      return Action::none;
    }
    if (button_ != buttonUnderAim) {
      cancelled_ = true;
      held_ = 0.0F;
    }
    if (cancelled_) {
      return Action::none;
    }
    if (!(dt >= 0.0F) || !std::isfinite(dt)) {
      dt = 0.0F;
    }
    if (!fired_) {
      held_ += dt;
    }
    if (button_ == kDelete && !fired_ && held_ >= kDeleteHoldSeconds) {
      fired_ = true;
      return Action::deleteNow;
    }
    return Action::none;
  }

  // Call on the trigger release with the button currently under the aim.
  // Delete is hold-only: releasing it early (or without reaching the charge)
  // cancels and never fires from this path.
  Action release(int buttonUnderAim) {
    Action action = Action::none;
    if (active() && !fired_ && !cancelled_ && button_ == buttonUnderAim && button_ != kDelete) {
      action = actionFor(button_);
    }
    clear();
    return action;
  }

  // A tracking loss, menu close or skipped frame: drop the capture without
  // firing anything. The next press starts a fresh capture.
  void cancel() { clear(); }

  bool active() const { return button_ != kInactive; }
  bool ownsTrigger() const { return active(); }
  int button() const { return button_; }

  // 0..1 red charge ramp for the Delete hold; zero for every other button.
  float charge() const {
    if (button_ != kDelete || !active() || cancelled_) {
      return 0.0F;
    }
    return std::fmin(1.0F, std::fmax(0.0F, held_ / kDeleteHoldSeconds));
  }

  static Action actionFor(int button) {
    switch (button) {
    case kModels: return Action::models;
    case kUndo: return Action::undo;
    case kRedo: return Action::redo;
    case kBackdrop: return Action::backdrop;
    case kDelete: return Action::deleteNow;
    default: return Action::none;
    }
  }

 private:
  static constexpr int kInactive = -2; // -1 is a captured miss, which still owns

  void clear() {
    button_ = kInactive;
    held_ = 0.0F;
    fired_ = false;
    cancelled_ = false;
  }

  int button_{kInactive};
  float held_{0.0F};
  bool fired_{false};
  bool cancelled_{false};
};

} // namespace vredit