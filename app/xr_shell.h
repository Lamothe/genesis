#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "camera.h"
#include "vecmath.h"

namespace vredit {

// Per-frame state the app consumes, filled by XrShell::beginFrame.
struct XrFrameInput {
  Mat4 eyeViewProj[2];   // proj * view per eye (renderer conventions)
  Ray pointerRay;        // right-controller aim ray (head gaze fallback)
  Vec3 headPosition{0.0F, 0.0F, 0.0F}; // midpoint of the eyes
  Vec3 headForward{0.0F, 0.0F, -1.0F};
  Vec2 moveStick{0.0F, 0.0F}; // left stick: x = strafe, y = forward
  Vec2 turnStick{0.0F, 0.0F}; // right stick: x = yaw, y = pitch
  bool selectHeld{false};
  bool selectPressed{false};
  bool selectReleased{false};
  bool menuPressed{false};
  float gripHeld{0.0F};
  bool poseValid{false}; // pointer ray is tracked this frame
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

  // Frame flow. beginFrame returns false when the session is not rendering
  // (skip the frame but keep pumping), and sets `running` false on exit.
  bool beginFrame(XrFrameInput& inputOut, bool& running);
  VkImageView eyeView(int eye) const { return eyeViews_[eye]; }
  VkImage eyeImage(int eye) const { return eyeImages_[eye]; }
  bool endFrame();

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
};

} // namespace vredit
