#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "camera.h"
#include "controller_workflow.h"
#include "hand_input.h"
#include "vecmath.h"

namespace vredit {

// Per-frame state the app consumes, filled by XrShell::beginFrame. Eye poses
// are the raw located poses in tracking space, composed with the app's rig.
struct XrFrameInput {
  bool handOpen[2] = {false, false}; // at least three tracked extended fingers
  Vec3 handJoints[2][26]{};
  bool handJointValid[2][26]{};
  HandPose hands[2]; // palm joint poses, not runtime-filtered aim orientations
  bool handActive[2] = {false, false}; // actual hand-interaction profile
  bool pinchActive[2] = {false, false};
  float pinch[2] = {0.0F, 0.0F};
  bool graspActive{false}; // right hand grasp value and ready actions are active
  float grasp{0.0F};
  Mat4 eyeProj[2];       // per-eye projection from the runtime fov tangents
  Vec3 eyePos[2];        // located eye positions (tracking space)
  Quat eyeOrient[2];     // located eye orientations (tracking space)
  Quat headOrient{};     // raw left-eye orientation (tracking space)
  Ray pointerRay;        // right-controller aim ray (head gaze fallback)
  Ray aimRays[2];        // per-hand aim rays in XR local space (left, right)
  bool aimValid[2] = {false, false}; // aimRays[i] has a usable (VALID) pose
  // aimRays[i] is ACTIVELY tracked (VALID and TRACKED): a pose that is only
  // VALID may be inferred/last-known and must not drive spatial actions.
  bool aimTracked[2] = {false, false};
  // Physical controller grip poses, located from XR grip action spaces (left,
  // right). These are the manipulation poses: never the aim ray and never a
  // head-gaze fallback. VALID and TRACKED are recorded separately from the
  // runtime's space-location flags so the app can require measured tracking.
  ControllerPose controllerGrip[2];
  Vec3 headPosition{0.0F, 0.0F, 0.0F}; // midpoint of the eyes (tracking space)
  Vec2 leftStick{0.0F, 0.0F};  // viewer-relative flight
  Vec2 rightStick{0.0F, 0.0F}; // viewer look or carousel browsing
  bool leftStickActive{false};
  bool viewTracked{false}; // fresh position and orientation, both actively tracked
  bool rightStickActive{false}; // successful, active, finite right-stick sample
  bool rightStickAbsent{false}; // confirmed simple-controller profile, not a transient loss
  // selectActive is true only for a successful, active, finite trigger
  // sample, except that a focus/profile interruption is exposed as
  // inactive for one rendered frame so the app can cancel. On
  // inactive input the held signal is conservatively true and both edges
  // are false: cancel captures/grabs, not commit a synthetic release.
  // After an interruption, an active sample at or below the release
  // threshold silently rearms; a later physical press emits the next edge.
  bool leftSelectActive{false};
  bool leftSelectHeld{true};
  bool leftSelectPressed{false};
  bool leftSelectReleased{false};
  bool selectActive{false};
  bool selectHeld{false};
  bool selectPressed{false};
  bool selectReleased{false};
  bool menuPressed{false};
  bool menuHeld{false};    // raw held state on a safe, active sample
  bool undoPressed{false};  // latched: face button on Touch
  bool redoPressed{false};
  bool spawnPressed{false}; // A on Touch; interrupted presses are discarded
  // True only when the X action query succeeds and the action is active.
  // Inactive is not a physical release; interrupt the delete latch.
  bool deleteActive{false};
  bool deleteHeld{false};   // X on Touch held: hold-to-delete with red ramp
  // Focus/profile change, even if the current delete query is active.
  bool deleteInterrupted{false};
  float gripHeld{0.0F};
};

// Minimal OpenXR stereo shell (XR_KHR_vulkan_enable2): owns the XR instance,
// session and per-eye swapchains, plus the Vulkan instance/device handed to
// VkRenderer::initExternal. Frame flow: beginFrame fills XrFrameInput and
// acquires both eye images; the app renders via VkRenderer::drawToView into
// eyeView/eyeImage per eye; endFrame releases the images and composites.
class XrShell {
public:
  enum class InitResult {
    Ready,       // session up
    NoRuntime,   // no OpenXR runtime configured (nothing to test against)
    NoHeadset,   // runtime present but no HMD available
    Failed       // unexpected failure; details printed
  };

  // Frame diagnostics: why frames end with zero layers (WiVRn shows those as
  // passthrough) and how often.
  struct FrameStats {
    std::uint64_t begun{0};
    std::uint64_t zeroLayer{0};     // ended with no layers (passthrough flash)
    std::uint64_t renderFalse{0};   // runtime said shouldRender == false (we render anyway)
    std::uint64_t viewFail{0};      // xrLocateViews failed / wrong count
    std::uint64_t stalePose{0};     // view poses unlocated; reused the last good ones
    std::uint64_t actionFail{0};    // xrSyncActions failed
    std::uint64_t acquireFail{0};   // swapchain acquire/wait failed
    // Stick extremes seen all session (|value| high-water marks): the
    // telemetry that decides whether a dead axis is the app or the runtime.
    float leftXMax{0.0F};
    float leftYMax{0.0F};
    float rightXMax{0.0F};
    float rightYMax{0.0F};
  };

  XrShell() = default;
  ~XrShell();
  XrShell(XrShell const&) = delete;
  XrShell& operator=(XrShell const&) = delete;

  // Creates XR instance/system/session/swapchains and the Vulkan objects
  // (enable2). On Ready, hand the Vulkan handles to VkRenderer::initExternal
  // with swapchainFormat() and (eyeWidth(), eyeHeight()).
  InitResult init();
  void shutdown();

  // Vulkan handles for VkRenderer::initExternal (valid after Ready).
  VkInstance vulkanInstance() const { return instance_; }
  VkPhysicalDevice physicalDevice() const { return physical_; }
  VkDevice device() const { return device_; }
  std::uint32_t queueFamily() const { return queueFamily_; }
  VkFormat swapchainFormat() const { return swapchainFormat_; }
  std::uint32_t eyeWidth() const { return eyeWidth_; }
  std::uint32_t eyeHeight() const { return eyeHeight_; }

  // Frame flow. beginFrame paces on the runtime (xrWaitFrame) and returns:
  //   Render  — frame begun and ready to draw into the eye targets;
  //   Skipped — frame begun and already ended (nothing to draw this time);
  //   Pump    — no frame begun yet (session starting); retry after a short sleep;
  //   Exit    — session over or a fatal failure; stop the loop.
  // `shouldRender` is treated as a hint only: the spec allows layers in every
  // xrEndFrame, and zero-layer frames display as a cleared (passthrough) frame
  // on WiVRn, so a skip is never worth a visible blank.
  enum class FrameStatus { Render, Skipped, Pump, Exit };
  FrameStatus beginFrame(XrFrameInput& inputOut);
  // "Show the model without the background": composite ALPHA_BLEND over the
  // real world instead of an opaque backdrop. Unsupported runtimes keep the
  // opaque frame and say so once.
  void setPassthrough(bool enabled) { passthrough_ = enabled; }
  // False when the runtime cannot composite ALPHA_BLEND frames (or the
  // swapchain has no alpha): the toggle then keeps an opaque void instead.
  bool passthroughSupported() const { return alphaBlendSupported_; }
  VkImageView eyeView(int eye) const { return eyeViews_[eye]; }
  VkImage eyeImage(int eye) const { return eyeImages_[eye]; }
  bool endFrame();

  FrameStats const& frameStats() const { return stats_; }

private:
  bool createActions();
  bool createSwapchains();
  void pollEvents(bool& running);
  bool syncActions(XrFrameInput& input, std::uint64_t displayTime);

  struct ImplData;
  ImplData* xr_{nullptr}; // XR handles (kept out of this header)

  VkInstance instance_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  std::uint32_t queueFamily_{0};
  VkFormat swapchainFormat_{VK_FORMAT_R8G8B8A8_UNORM};
  std::uint32_t eyeWidth_{0};
  std::uint32_t eyeHeight_{0};
  VkImage eyeImages_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkImageView eyeViews_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  bool imagesAcquired_[2] = {false, false};
  bool passthrough_{false};        // setPassthrough: alpha-blend compositing
  bool alphaBlendSupported_{false}; // runtime accepts ALPHA_BLEND (checked at init)
  bool passthroughWarned_{false};
  FrameStats stats_;
};

} // namespace vredit
