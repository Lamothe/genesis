#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "camera.h"
#include "vecmath.h"

namespace vredit {

// Per-frame state the app consumes, filled by XrShell::beginFrame. The eye
// poses are the corrected (roll-stripped) ones in TRACKING space; the app
// composes them with its rig into the final views, which is where the
// combined basis gets its own horizon lock (a pitched rig composed with a
// yawed head produces phantom roll otherwise).
struct XrFrameInput {
  Mat4 eyeProj[2];       // per-eye projection from the runtime fov tangents
  Vec3 eyePos[2];        // corrected eye positions (tracking space)
  Quat eyeOrient[2];     // corrected eye orientations (tracking space)
  Quat headOrient{};     // roll-stripped head orientation (tracking space)
  Ray pointerRay;        // right-controller aim ray (head gaze fallback)
  Ray aimRays[2];        // per-hand aim rays in XR local space (left, right)
  bool aimValid[2] = {false, false}; // aimRays[i] is tracked this frame
  Vec3 headPosition{0.0F, 0.0F, 0.0F}; // midpoint of the eyes (tracking space)
  Vec2 moveStick{0.0F, 0.0F}; // left stick: x = strafe, y = forward
  Vec2 turnStick{0.0F, 0.0F}; // right stick: x = yaw, y = pitch
  bool selectHeld{false};
  bool selectPressed{false};
  bool selectReleased{false};
  bool menuPressed{false};
  bool undoPressed{false};  // latched: Face button on Touch, trackpad on Vive
  bool redoPressed{false};
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
    float moveXMax{0.0F};
    float moveYMax{0.0F};
    float turnXMax{0.0F};
    float turnYMax{0.0F};
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
