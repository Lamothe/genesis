#include "xr_shell.h"

#include <openxr/openxr.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vredit {

namespace {

// Type-safe struct initialisers (the C headers' designated patterns trip
// -Wmissing-field-initializers; these value-initialise every field).
template <typename T>
T xrStruct(XrStructureType type) {
  T value{};
  value.type = type;
  return value;
}

template <typename T>
T vkStruct(VkStructureType type) {
  T value{};
  value.sType = type;
  return value;
}

bool xrOk(XrResult result, char const* what) {
  if (XR_FAILED(result)) {
    std::printf("[xr] %s failed (%d)\n", what, static_cast<int>(result));
    return false;
  }
  return true;
}

Ray rayFromPose(XrPosef const& pose) {
  // OpenXR orientation is world <- local; forward is -Z, right is +X.
  Quat q{pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
  Vec3 forward = rotate(q, {0.0F, 0.0F, -1.0F});
  Vec3 origin{pose.position.x, pose.position.y, pose.position.z};
  return {origin, normalized(forward)};
}

} // namespace

struct XrShell::ImplData {
  XrInstance instance{XR_NULL_HANDLE};
  XrSystemId system{XR_NULL_SYSTEM_ID};
  XrSession session{XR_NULL_HANDLE};
  XrSpace space{XR_NULL_HANDLE};
  XrActionSet actionSet{XR_NULL_HANDLE};
  XrAction triggerAction{XR_NULL_HANDLE};
  XrAction leftStickAction{XR_NULL_HANDLE};
  XrAction rightStickAction{XR_NULL_HANDLE};
  XrAction menuAction{XR_NULL_HANDLE};
  XrAction undoAction{XR_NULL_HANDLE}; // B on Touch
  XrAction redoAction{XR_NULL_HANDLE}; // Y on Touch
  XrAction spawnAction{XR_NULL_HANDLE}; // A on Touch: model chooser (stage/place)
  XrAction deleteAction{XR_NULL_HANDLE}; // X on Touch: hold to delete
  XrAction gripAction{XR_NULL_HANDLE};
  XrAction aimAction{XR_NULL_HANDLE};
  XrSpace aimSpaces[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE}; // left, right
  XrSwapchain swapchains[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
  std::vector<XrSwapchainImageVulkan2KHR> images[2];
  std::vector<VkImageView> imageViews[2];
  std::uint32_t acquiredIndex[2] = {0, 0};
  XrView views[2];
  XrView correctedViews[2]; // horizon-locked poses the images are rendered from
  bool haveGoodViews{false}; // views[] holds the last successfully located pair
  XrViewConfigurationView viewConfig[2];
  XrSessionState sessionState{XR_SESSION_STATE_UNKNOWN};
  bool sessionRunning{false};
  bool sessionVisible{false};
  bool frameBegun{false};
  bool frameReady{false}; // views located and both eye images acquired
  XrTime predictedDisplayTime{0};
  XrFrameState frameState{};
  bool selectPrev{false};
  bool menuPrev{false};
  bool undoPrev{false};
  bool redoPrev{false};
  bool spawnPrev{false};
  bool undoPending{false}; // latched until a frame renders and consumes it
  bool redoPending{false};
  bool spawnPending{false};

  // XR_KHR_vulkan_enable2 entry points.
  PFN_xrCreateVulkanInstanceKHR pfnCreateVulkanInstance{nullptr};
  PFN_xrGetVulkanGraphicsDevice2KHR pfnGetVulkanGraphicsDevice2{nullptr};
  PFN_xrCreateVulkanDeviceKHR pfnCreateVulkanDevice{nullptr};
  PFN_xrGetVulkanGraphicsRequirements2KHR pfnGetVulkanGraphicsRequirements2{nullptr};
};

XrShell::~XrShell() {
  shutdown();
}

XrShell::InitResult XrShell::init() {
  xr_ = new ImplData();

  XrInstanceCreateInfo instanceInfo = xrStruct<XrInstanceCreateInfo>(XR_TYPE_INSTANCE_CREATE_INFO);
  std::strncpy(instanceInfo.applicationInfo.applicationName, "vr-editor",
               XR_MAX_APPLICATION_NAME_SIZE - 1);
  instanceInfo.applicationInfo.applicationVersion = 1;
  std::strncpy(instanceInfo.applicationInfo.engineName, "vredit",
               XR_MAX_ENGINE_NAME_SIZE - 1);
  instanceInfo.applicationInfo.engineVersion = 1;
  instanceInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
  char const* extensions[] = {"XR_KHR_vulkan_enable2"};
  instanceInfo.enabledExtensionCount = 1;
  instanceInfo.enabledExtensionNames = extensions;

  XrResult result = xrCreateInstance(&instanceInfo, &xr_->instance);
  if (result == XR_ERROR_RUNTIME_UNAVAILABLE) {
    std::printf("[xr] no OpenXR runtime configured\n");
    return InitResult::NoRuntime;
  }
  if (!xrOk(result, "xrCreateInstance")) {
    return InitResult::Failed;
  }

  XrSystemGetInfo systemInfo = xrStruct<XrSystemGetInfo>(XR_TYPE_SYSTEM_GET_INFO);
  systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  result = xrGetSystem(xr_->instance, &systemInfo, &xr_->system);
  if (result == XR_ERROR_FORM_FACTOR_UNAVAILABLE) {
    std::printf("[xr] runtime present but no headset available\n");
    return InitResult::NoHeadset;
  }
  if (!xrOk(result, "xrGetSystem")) {
    return InitResult::Failed;
  }
  // Does the runtime composite ALPHA_BLEND frames (passthrough under models)?
  std::uint32_t blendCount = 0;
  xrEnumerateEnvironmentBlendModes(xr_->instance, xr_->system,
                                   XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &blendCount,
                                   nullptr);
  if (blendCount > 0U) {
    std::vector<XrEnvironmentBlendMode> blendModes(blendCount);
    xrEnumerateEnvironmentBlendModes(xr_->instance, xr_->system,
                                     XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, blendCount,
                                     &blendCount, blendModes.data());
    for (XrEnvironmentBlendMode mode : blendModes) {
      if (mode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) {
        alphaBlendSupported_ = true;
      }
    }
    std::printf("[xr] blend modes: %u (%s)\n", blendCount,
                alphaBlendSupported_ ? "alpha-blend supported" : "opaque only");
  }

  // Enable2 entry points.
  auto load = [this](char const* name, PFN_xrVoidFunction* out) {
    XrResult r = xrGetInstanceProcAddr(xr_->instance, name, out);
    if (XR_FAILED(r) || *out == nullptr) {
      std::printf("[xr] missing entry point %s (%d)\n", name, static_cast<int>(r));
      return false;
    }
    return true;
  };
  if (!load("xrCreateVulkanInstanceKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&xr_->pfnCreateVulkanInstance)) ||
      !load("xrGetVulkanGraphicsDevice2KHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&xr_->pfnGetVulkanGraphicsDevice2)) ||
      !load("xrCreateVulkanDeviceKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&xr_->pfnCreateVulkanDevice)) ||
      !load("xrGetVulkanGraphicsRequirements2KHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&xr_->pfnGetVulkanGraphicsRequirements2))) {
    return InitResult::Failed;
  }

  XrGraphicsRequirementsVulkan2KHR requirements = xrStruct<XrGraphicsRequirementsVulkan2KHR>(XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR);
  if (!xrOk(xr_->pfnGetVulkanGraphicsRequirements2(xr_->instance, xr_->system, &requirements),
            "xrGetVulkanGraphicsRequirements2KHR")) {
    return InitResult::Failed;
  }
  // These fields are spec-encoded with XR_MAKE_VERSION(major, minor, 0)
  // (major<<48 | minor<<32), but some runtimes return VK_MAKE_VERSION values.
  // An XR-encoded value carries bits above 32; a VK-encoded one always fits in
  // 32 bits. Normalise both instead of trusting one convention — reading XR
  // values through VK_API_VERSION_* macros yields a bogus "0.0".
  auto vulkanVersion = [](XrVersion version) -> std::uint32_t {
    if ((version >> 32) != 0) {
      return VK_MAKE_API_VERSION(0, static_cast<std::uint32_t>((version >> 48) & 0xffffULL),
                                 static_cast<std::uint32_t>((version >> 32) & 0xffffULL), 0);
    }
    return static_cast<std::uint32_t>(version);
  };
  std::uint32_t const minSupported = vulkanVersion(requirements.minApiVersionSupported);
  std::uint32_t const maxSupported = vulkanVersion(requirements.maxApiVersionSupported);
  // maxApiVersionSupported is advisory ("highest version tested; newer may
  // still work"), and the renderer needs 1.3 core features, so request1.3 and
  // let xrCreateVulkanInstanceKHR arbitrate. A runtime floor above 1.3 is
  // honoured rather than refused: newer instances still expose 1.3 features.
  std::uint32_t apiVersionRequested = VK_API_VERSION_1_3;
  if (minSupported > apiVersionRequested) {
    apiVersionRequested = minSupported;
  }
  if (maxSupported != 0 && maxSupported < VK_API_VERSION_1_3) {
    std::printf("[xr] runtime only tested Vulkan up to %u.%u; trying 1.3 anyway\n",
                VK_API_VERSION_MAJOR(maxSupported), VK_API_VERSION_MINOR(maxSupported));
  }

  // The enable2 create calls make the runtime merge its own Vulkan extensions
  // into these create infos; we only list what the renderer itself needs
  // (nothing: no surface in XR mode, no VK_KHR_swapchain for runtime images).
  VkApplicationInfo vkAppInfo = vkStruct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
  vkAppInfo.pApplicationName = "vr-editor";
  vkAppInfo.pEngineName = "vredit";
  vkAppInfo.apiVersion = apiVersionRequested;
  VkInstanceCreateInfo vkInstanceInfo = vkStruct<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  vkInstanceInfo.pApplicationInfo = &vkAppInfo;

  XrVulkanInstanceCreateInfoKHR xrVkInstance = xrStruct<XrVulkanInstanceCreateInfoKHR>(XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR);
  xrVkInstance.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
  xrVkInstance.systemId = xr_->system;
  xrVkInstance.vulkanCreateInfo = &vkInstanceInfo;
  xrVkInstance.vulkanAllocator = nullptr;
  VkResult vkResult = VK_SUCCESS;
  if (!xrOk(xr_->pfnCreateVulkanInstance(xr_->instance, &xrVkInstance, &instance_, &vkResult),
            "xrCreateVulkanInstanceKHR") ||
      vkResult != VK_SUCCESS) {
    std::printf("[xr] xrCreateVulkanInstanceKHR returned VkResult %d\n",
                static_cast<int>(vkResult));
    return InitResult::Failed;
  }

  XrVulkanGraphicsDeviceGetInfoKHR deviceInfo =
      xrStruct<XrVulkanGraphicsDeviceGetInfoKHR>(XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR);
  deviceInfo.systemId = xr_->system;
  deviceInfo.vulkanInstance = instance_;
  if (!xrOk(xr_->pfnGetVulkanGraphicsDevice2(xr_->instance, &deviceInfo, &physical_),
            "xrGetVulkanGraphicsDevice2KHR")) {
    return InitResult::Failed;
  }

  // Graphics queue family on the runtime's physical device.
  std::uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &familyCount, families.data());
  bool familyFound = false;
  for (std::uint32_t i = 0; i < familyCount; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U) {
      queueFamily_ = i;
      familyFound = true;
      break;
    }
  }
  if (!familyFound) {
    std::printf("[xr] no graphics queue family on the XR device\n");
    return InitResult::Failed;
  }

  float priority = 1.0F;
  VkDeviceQueueCreateInfo queueInfo = vkStruct<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
  queueInfo.queueFamilyIndex = queueFamily_;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;
  VkPhysicalDeviceVulkan13Features features13 =
      vkStruct<VkPhysicalDeviceVulkan13Features>(
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;
  VkDeviceCreateInfo vkDeviceInfo = vkStruct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
  vkDeviceInfo.pNext = &features13;
  vkDeviceInfo.queueCreateInfoCount = 1;
  vkDeviceInfo.pQueueCreateInfos = &queueInfo;

  XrVulkanDeviceCreateInfoKHR xrVkDevice = xrStruct<XrVulkanDeviceCreateInfoKHR>(XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR);
  xrVkDevice.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
  xrVkDevice.systemId = xr_->system;
  xrVkDevice.vulkanPhysicalDevice = physical_;
  xrVkDevice.vulkanCreateInfo = &vkDeviceInfo;
  xrVkDevice.vulkanAllocator = nullptr;
  if (!xrOk(xr_->pfnCreateVulkanDevice(xr_->instance, &xrVkDevice, &device_, &vkResult),
            "xrCreateVulkanDeviceKHR") ||
      vkResult != VK_SUCCESS) {
    std::printf("[xr] xrCreateVulkanDeviceKHR returned VkResult %d\n",
                static_cast<int>(vkResult));
    return InitResult::Failed;
  }

  // Session bound to the Vulkan device.
  XrGraphicsBindingVulkan2KHR binding = xrStruct<XrGraphicsBindingVulkan2KHR>(XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR);
  binding.instance = instance_;
  binding.physicalDevice = physical_;
  binding.device = device_;
  binding.queueFamilyIndex = queueFamily_;
  binding.queueIndex = 0;
  XrSessionCreateInfo sessionInfo = xrStruct<XrSessionCreateInfo>(XR_TYPE_SESSION_CREATE_INFO);
  sessionInfo.next = &binding;
  sessionInfo.systemId = xr_->system;
  if (!xrOk(xrCreateSession(xr_->instance, &sessionInfo, &xr_->session), "xrCreateSession")) {
    return InitResult::Failed;
  }

  XrReferenceSpaceCreateInfo spaceInfo = xrStruct<XrReferenceSpaceCreateInfo>(XR_TYPE_REFERENCE_SPACE_CREATE_INFO);
  spaceInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  spaceInfo.poseInReferenceSpace.orientation.w = 1.0F;
  if (!xrOk(xrCreateReferenceSpace(xr_->session, &spaceInfo, &xr_->space),
            "xrCreateReferenceSpace")) {
    return InitResult::Failed;
  }

  std::uint32_t viewCount = 0;
  xr_->viewConfig[0] = xrStruct<XrViewConfigurationView>(XR_TYPE_VIEW_CONFIGURATION_VIEW);
  xr_->viewConfig[1] = xrStruct<XrViewConfigurationView>(XR_TYPE_VIEW_CONFIGURATION_VIEW);
  if (!xrOk(xrEnumerateViewConfigurationViews(xr_->instance, xr_->system,
                                              XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2,
                                              &viewCount, xr_->viewConfig),
            "xrEnumerateViewConfigurationViews") ||
      viewCount != 2U) {
    std::printf("[xr] expected 2 stereo views, got %u\n", viewCount);
    return InitResult::Failed;
  }
  eyeWidth_ = xr_->viewConfig[0].recommendedImageRectWidth > xr_->viewConfig[1].recommendedImageRectWidth
                  ? xr_->viewConfig[0].recommendedImageRectWidth
                  : xr_->viewConfig[1].recommendedImageRectWidth;
  eyeHeight_ = xr_->viewConfig[0].recommendedImageRectHeight > xr_->viewConfig[1].recommendedImageRectHeight
                   ? xr_->viewConfig[0].recommendedImageRectHeight
                   : xr_->viewConfig[1].recommendedImageRectHeight;

  if (!createActions() || !createSwapchains()) {
    return InitResult::Failed;
  }

  XrSessionActionSetsAttachInfo attachInfo =
      xrStruct<XrSessionActionSetsAttachInfo>(XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO);
  attachInfo.countActionSets = 1;
  attachInfo.actionSets = &xr_->actionSet;
  if (!xrOk(xrAttachSessionActionSets(xr_->session, &attachInfo),
            "xrAttachSessionActionSets")) {
    return InitResult::Failed;
  }

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_, &props);
  std::printf("[xr] session ready: %ux%u per eye on %s\n", eyeWidth_, eyeHeight_,
              props.deviceName);
  return InitResult::Ready;
}

bool XrShell::createActions() {
  XrActionSetCreateInfo setInfo = xrStruct<XrActionSetCreateInfo>(XR_TYPE_ACTION_SET_CREATE_INFO);
  std::strncpy(setInfo.actionSetName, "editor", XR_MAX_ACTION_SET_NAME_SIZE - 1);
  std::strncpy(setInfo.localizedActionSetName, "Editor", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE - 1);
  if (!xrOk(xrCreateActionSet(xr_->instance, &setInfo, &xr_->actionSet), "xrCreateActionSet")) {
    return false;
  }

  XrPath leftPath = XR_NULL_PATH;
  XrPath rightPath = XR_NULL_PATH;
  if (!xrOk(xrStringToPath(xr_->instance, "/user/hand/left", &leftPath), "path /user/hand/left") ||
      !xrOk(xrStringToPath(xr_->instance, "/user/hand/right", &rightPath), "path /user/hand/right")) {
    return false;
  }
  XrPath hands[2] = {leftPath, rightPath};

  auto makeFloat = [&](char const* name, XrAction* out) {
    XrActionCreateInfo info = xrStruct<XrActionCreateInfo>(XR_TYPE_ACTION_CREATE_INFO);
    std::strncpy(info.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
    std::strncpy(info.localizedActionName, name, XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
    info.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
    info.countSubactionPaths = 2;
    info.subactionPaths = hands;
    return xrOk(xrCreateAction(xr_->actionSet, &info, out), name);
  };
  auto makeVec2 = [&](char const* name, XrAction* out) {
    XrActionCreateInfo info = xrStruct<XrActionCreateInfo>(XR_TYPE_ACTION_CREATE_INFO);
    std::strncpy(info.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
    std::strncpy(info.localizedActionName, name, XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
    info.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
    info.countSubactionPaths = 2;
    info.subactionPaths = hands;
    return xrOk(xrCreateAction(xr_->actionSet, &info, out), name);
  };
  auto makeBool = [&](char const* name, XrAction* out) {
    XrActionCreateInfo info = xrStruct<XrActionCreateInfo>(XR_TYPE_ACTION_CREATE_INFO);
    std::strncpy(info.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
    std::strncpy(info.localizedActionName, name, XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
    info.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
    info.countSubactionPaths = 2;
    info.subactionPaths = hands;
    return xrOk(xrCreateAction(xr_->actionSet, &info, out), name);
  };
  auto makePose = [&](char const* name, XrAction* out) {
    XrActionCreateInfo info = xrStruct<XrActionCreateInfo>(XR_TYPE_ACTION_CREATE_INFO);
    std::strncpy(info.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
    std::strncpy(info.localizedActionName, name, XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
    info.actionType = XR_ACTION_TYPE_POSE_INPUT;
    info.countSubactionPaths = 2;
    info.subactionPaths = hands;
    return xrOk(xrCreateAction(xr_->actionSet, &info, out), name);
  };

  if (!makeFloat("trigger", &xr_->triggerAction) || !makeVec2("leftstick", &xr_->leftStickAction) ||
      !makeVec2("rightstick", &xr_->rightStickAction) || !makeBool("menu", &xr_->menuAction) ||
      !makeBool("undo", &xr_->undoAction) || !makeBool("redo", &xr_->redoAction) ||
      !makeBool("spawn", &xr_->spawnAction) || !makeBool("delete", &xr_->deleteAction) ||
      !makeFloat("grip", &xr_->gripAction) || !makePose("aim", &xr_->aimAction)) {
    return false;
  }

  // Suggested bindings per interaction profile. Float actions accept the
  // boolean trigger sources on wand-style controllers (false/true -> 0/1).
  struct Binding {
    char const* profile;
    std::vector<std::pair<char const*, XrAction>> paths;
  };
  std::vector<Binding> bindings = {
      {"/interaction_profiles/khr/simple_controller",
       {{"/user/hand/right/input/select/click", xr_->triggerAction},
        {"/user/hand/left/input/select/click", xr_->triggerAction},
        {"/user/hand/right/input/menu/click", xr_->menuAction},
        {"/user/hand/left/input/menu/click", xr_->menuAction},
        {"/user/hand/right/input/aim/pose", xr_->aimAction},
        {"/user/hand/left/input/aim/pose", xr_->aimAction}}}, // no thumbstick on this profile
      {"/interaction_profiles/oculus/touch_controller",
       {{"/user/hand/right/input/trigger/value", xr_->triggerAction},
        {"/user/hand/left/input/trigger/value", xr_->triggerAction},
        {"/user/hand/left/input/thumbstick", xr_->leftStickAction},
        {"/user/hand/right/input/thumbstick", xr_->rightStickAction},
        {"/user/hand/left/input/menu/click", xr_->menuAction},
        {"/user/hand/right/input/squeeze/value", xr_->gripAction},
        {"/user/hand/left/input/squeeze/value", xr_->gripAction},
        {"/user/hand/right/input/aim/pose", xr_->aimAction},
        {"/user/hand/left/input/aim/pose", xr_->aimAction},
        // Undo/redo/spawn on the face buttons, NOT the stick clicks: the
        // sticks are the editing controls now and clicking one mid-gesture is
        // a natural accident. A = spawn, X (held) = delete.
        {"/user/hand/right/input/b/click", xr_->undoAction},
        {"/user/hand/left/input/y/click", xr_->redoAction},
        {"/user/hand/right/input/a/click", xr_->spawnAction},
        {"/user/hand/left/input/x/click", xr_->deleteAction}}},
      {"/interaction_profiles/htc/vive_controller",
       {{"/user/hand/right/input/trigger/click", xr_->triggerAction},
        {"/user/hand/left/input/trigger/click", xr_->triggerAction},
        {"/user/hand/left/input/trackpad", xr_->leftStickAction},
        {"/user/hand/right/input/trackpad", xr_->rightStickAction},
        {"/user/hand/right/input/menu/click", xr_->menuAction},
        {"/user/hand/left/input/menu/click", xr_->menuAction},
        {"/user/hand/right/input/squeeze/click", xr_->gripAction},
        {"/user/hand/left/input/squeeze/click", xr_->gripAction},
        {"/user/hand/right/input/aim/pose", xr_->aimAction},
        {"/user/hand/left/input/aim/pose", xr_->aimAction},
        {"/user/hand/right/input/trackpad/click", xr_->undoAction},
        {"/user/hand/left/input/trackpad/click", xr_->redoAction}}},
  };

  for (Binding const& binding : bindings) {
    XrPath profilePath = XR_NULL_PATH;
    if (!xrOk(xrStringToPath(xr_->instance, binding.profile, &profilePath),
              "xrStringToPath(profile)")) {
      return false;
    }
    std::vector<XrActionSuggestedBinding> suggested;
    for (auto const& [pathText, action] : binding.paths) {
      XrPath path = XR_NULL_PATH;
      if (!xrOk(xrStringToPath(xr_->instance, pathText, &path), "xrStringToPath(binding)")) {
        return false;
      }
      XrActionSuggestedBinding entry{};
      entry.binding = path;
      entry.action = action;
      suggested.push_back(entry);
    }
    XrInteractionProfileSuggestedBinding suggest =
        xrStruct<XrInteractionProfileSuggestedBinding>(
            XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING);
    suggest.interactionProfile = profilePath;
    suggest.countSuggestedBindings = static_cast<uint32_t>(suggested.size());
    suggest.suggestedBindings = suggested.data();
    // Unknown profiles may be rejected (XR_ERROR_PATH_UNSUPPORTED) and are
    // fine; anything else is worth knowing about.
    XrResult suggestResult = xrSuggestInteractionProfileBindings(xr_->instance, &suggest);
    if (suggestResult != XR_SUCCESS && suggestResult != XR_ERROR_PATH_UNSUPPORTED) {
      std::printf("[xr] %s bindings suggestion failed (%d)\n", binding.profile,
                  static_cast<int>(suggestResult));
    }
  }

  // Aim spaces for both controller pointers (left, right).
  XrActionSpaceCreateInfo spaceInfo = xrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
  spaceInfo.action = xr_->aimAction;
  spaceInfo.poseInActionSpace.orientation.w = 1.0F;
  XrPath const handPaths[2] = {leftPath, rightPath};
  for (int hand = 0; hand < 2; ++hand) {
    spaceInfo.subactionPath = handPaths[hand];
    if (!xrOk(xrCreateActionSpace(xr_->session, &spaceInfo, &xr_->aimSpaces[hand]),
              "xrCreateActionSpace")) {
      return false;
    }
  }
  return true;
}

bool XrShell::createSwapchains() {
  std::uint32_t formatCount = 0;
  xrEnumerateSwapchainFormats(xr_->session, 0, &formatCount, nullptr);
  std::vector<int64_t> formats(formatCount);
  xrEnumerateSwapchainFormats(xr_->session, formatCount, &formatCount, formats.data());

  // Prefer UNORM8 (straight to sRGB-free display like the desktop path).
  swapchainFormat_ = VK_FORMAT_R8G8B8A8_UNORM;
  bool found = false;
  for (int64_t format : formats) {
    if (format == static_cast<int64_t>(VK_FORMAT_R8G8B8A8_UNORM)) {
      found = true;
      break;
    }
  }
  if (!found) {
    for (int64_t format : formats) {
      if (format == static_cast<int64_t>(VK_FORMAT_B8G8R8A8_UNORM)) {
        swapchainFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
        found = true;
        break;
      }
    }
  }
  if (!found && !formats.empty()) {
    // Passthrough composites through the alpha channel, so prefer a format
    // that has one; if none does, say so (the alpha-0 clear would be a lie).
    auto hasAlpha = [](VkFormat f) {
      return f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB ||
             f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB ||
             f == VK_FORMAT_R8G8B8A8_USCALED || f == VK_FORMAT_B8G8R8A8_USCALED;
    };
    for (int64_t format : formats) {
      if (hasAlpha(static_cast<VkFormat>(format))) {
        swapchainFormat_ = static_cast<VkFormat>(format);
        found = true;
        break;
      }
    }
    if (!found) {
      swapchainFormat_ = static_cast<VkFormat>(formats[0]);
      std::printf("[xr] runtime-preferred format %d has no alpha channel; "
                  "passthrough will show black\n",
                  static_cast<int>(swapchainFormat_));
    }
  } else if (!found) {
    std::printf("[xr] no swapchain formats reported\n");
    return false;
  }

  for (int eye = 0; eye < 2; ++eye) {
    XrSwapchainCreateInfo swapInfo = xrStruct<XrSwapchainCreateInfo>(XR_TYPE_SWAPCHAIN_CREATE_INFO);
    swapInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    swapInfo.format = static_cast<int64_t>(swapchainFormat_);
    swapInfo.sampleCount = 1;
    swapInfo.width = eyeWidth_;
    swapInfo.height = eyeHeight_;
    swapInfo.faceCount = 1;
    swapInfo.arraySize = 1;
    swapInfo.mipCount = 1;
    if (!xrOk(xrCreateSwapchain(xr_->session, &swapInfo, &xr_->swapchains[eye]),
              "xrCreateSwapchain")) {
      return false;
    }

    std::uint32_t imageCount = 0;
    xrEnumerateSwapchainImages(xr_->swapchains[eye], 0, &imageCount, nullptr);
    xr_->images[eye].resize(imageCount);
    for (XrSwapchainImageVulkan2KHR& image : xr_->images[eye]) {
      image = xrStruct<XrSwapchainImageVulkan2KHR>(XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR);
    }
    if (!xrOk(xrEnumerateSwapchainImages(
                  xr_->swapchains[eye], imageCount, &imageCount,
                  reinterpret_cast<XrSwapchainImageBaseHeader*>(xr_->images[eye].data())),
              "xrEnumerateSwapchainImages")) {
      return false;
    }
    if (imageCount == 0U) {
      std::printf("[xr] swapchain reports no images\n");
      return false;
    }

    xr_->imageViews[eye].resize(imageCount, VK_NULL_HANDLE);
    for (std::uint32_t i = 0; i < imageCount; ++i) {
      VkImageViewCreateInfo viewInfo = vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
      viewInfo.image = xr_->images[eye][i].image;
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = swapchainFormat_;
      viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.subresourceRange.levelCount = 1;
      viewInfo.subresourceRange.layerCount = 1;
      if (vkCreateImageView(device_, &viewInfo, nullptr, &xr_->imageViews[eye][i]) !=
          VK_SUCCESS) {
        std::printf("[xr] swapchain image view creation failed\n");
        return false;
      }
    }
  }
  return true;
}

void XrShell::pollEvents(bool& running) {
  XrEventDataBuffer event = xrStruct<XrEventDataBuffer>(XR_TYPE_EVENT_DATA_BUFFER);
  while (xrPollEvent(xr_->instance, &event) == XR_SUCCESS) {
    if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
      auto* changed = reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
      xr_->sessionState = changed->state;
      switch (changed->state) {
      case XR_SESSION_STATE_READY: {
        XrSessionBeginInfo beginInfo = xrStruct<XrSessionBeginInfo>(XR_TYPE_SESSION_BEGIN_INFO);
        beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        if (xrOk(xrBeginSession(xr_->session, &beginInfo), "xrBeginSession")) {
          xr_->sessionRunning = true;
        } else {
          running = false; // fatal: the session can never start
        }
        break;
      }
      case XR_SESSION_STATE_STOPPING:
        if (!xrOk(xrEndSession(xr_->session), "xrEndSession")) {
          running = false; // fatal: the session can never end cleanly
        }
        xr_->sessionRunning = false;
        break;      case XR_SESSION_STATE_EXITING:
      case XR_SESSION_STATE_LOSS_PENDING:
        running = false;
        break;
      default:
        break;
      }
    } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
      running = false;
    }
    event = xrStruct<XrEventDataBuffer>(XR_TYPE_EVENT_DATA_BUFFER);
  }
}

bool XrShell::syncActions(XrFrameInput& input, std::uint64_t displayTime) {
  XrActiveActionSet active{xr_->actionSet, XR_NULL_PATH};
  XrActionsSyncInfo syncInfo = xrStruct<XrActionsSyncInfo>(XR_TYPE_ACTIONS_SYNC_INFO);
  syncInfo.countActiveActionSets = 1;
  syncInfo.activeActionSets = &active;
  if (!xrOk(xrSyncActions(xr_->session, &syncInfo), "xrSyncActions")) {
    return false;
  }

  XrPath leftPath = XR_NULL_PATH;
  XrPath rightPath = XR_NULL_PATH;
  xrStringToPath(xr_->instance, "/user/hand/left", &leftPath);
  xrStringToPath(xr_->instance, "/user/hand/right", &rightPath);

  XrActionStateGetInfo getInfo = xrStruct<XrActionStateGetInfo>(XR_TYPE_ACTION_STATE_GET_INFO);

  XrActionStateFloat triggerState = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
  getInfo.action = xr_->triggerAction;
  getInfo.subactionPath = rightPath;
  xrGetActionStateFloat(xr_->session, &getInfo, &triggerState);
  // Hysteresis on the analog trigger: hovering at the press threshold would
  // otherwise drop and re-press every other frame, and each re-press restarts
  // the grab at the current position — the model creeps and undo fills with
  // micro-steps instead of tracking the hand 1:1.
  float const trigger = triggerState.isActive ? triggerState.currentState : 0.0F;
  bool selectNow = trigger > (xr_->selectPrev ? 0.4F : 0.55F);
  input.selectHeld = selectNow;
  input.selectPressed = selectNow && !xr_->selectPrev;
  input.selectReleased = !selectNow && xr_->selectPrev;
  xr_->selectPrev = selectNow;

  XrActionStateVector2f leftStickState = xrStruct<XrActionStateVector2f>(XR_TYPE_ACTION_STATE_VECTOR2F);
  getInfo.action = xr_->leftStickAction;
  getInfo.subactionPath = leftPath;
  xrGetActionStateVector2f(xr_->session, &getInfo, &leftStickState);
  if (leftStickState.isActive) {
    input.leftStick = {leftStickState.currentState.x, leftStickState.currentState.y};
    stats_.leftXMax = std::max(stats_.leftXMax, std::abs(input.leftStick.x));
    stats_.leftYMax = std::max(stats_.leftYMax, std::abs(input.leftStick.y));
  }

  XrActionStateVector2f rightStickState = xrStruct<XrActionStateVector2f>(XR_TYPE_ACTION_STATE_VECTOR2F);
  getInfo.action = xr_->rightStickAction;
  getInfo.subactionPath = rightPath;
  xrGetActionStateVector2f(xr_->session, &getInfo, &rightStickState);
  if (rightStickState.isActive) {
    input.rightStick = {rightStickState.currentState.x, rightStickState.currentState.y};
    stats_.rightXMax = std::max(stats_.rightXMax, std::abs(input.rightStick.x));
    stats_.rightYMax = std::max(stats_.rightYMax, std::abs(input.rightStick.y));
  }

  XrActionStateBoolean menuLeft = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->menuAction;
  getInfo.subactionPath = leftPath;
  xrGetActionStateBoolean(xr_->session, &getInfo, &menuLeft);
  XrActionStateBoolean menuRight = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.subactionPath = rightPath;
  xrGetActionStateBoolean(xr_->session, &getInfo, &menuRight);
  bool menuNow = (menuLeft.isActive && menuLeft.currentState) ||
                 (menuRight.isActive && menuRight.currentState);
  input.menuPressed = menuNow && !xr_->menuPrev;
  xr_->menuPrev = menuNow;

  // Undo/redo: face buttons on Touch (B = undo, Y = redo), trackpad click on
  // Vive. The press is latched until a frame actually renders, so a
  // skipped/acquire-failed frame cannot swallow it, and an inactive action
  // keeps its previous state rather than manufacturing a new edge.
  XrActionStateBoolean undoState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->undoAction;
  getInfo.subactionPath = rightPath;
  xrGetActionStateBoolean(xr_->session, &getInfo, &undoState);
  bool undoNow = undoState.isActive ? undoState.currentState : xr_->undoPrev;
  xr_->undoPending = xr_->undoPending || (undoNow && !xr_->undoPrev);
  xr_->undoPrev = undoNow;
  XrActionStateBoolean redoState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->redoAction;
  getInfo.subactionPath = leftPath;
  xrGetActionStateBoolean(xr_->session, &getInfo, &redoState);
  bool redoNow = redoState.isActive ? redoState.currentState : xr_->redoPrev;
  xr_->redoPending = xr_->redoPending || (redoNow && !xr_->redoPrev);
  xr_->redoPrev = redoNow;
  input.undoPressed = xr_->undoPending;
  input.redoPressed = xr_->redoPending;

  // Spawn (A on Touch): latched like undo/redo so a skipped frame cannot
  // swallow it. Delete (X on Touch) is a HELD state — the app turns the hold
  // into a charge ramp and fires at full charge.
  XrActionStateBoolean spawnState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->spawnAction;
  getInfo.subactionPath = rightPath;
  xrGetActionStateBoolean(xr_->session, &getInfo, &spawnState);
  bool spawnNow = spawnState.isActive ? spawnState.currentState : xr_->spawnPrev;
  xr_->spawnPending = xr_->spawnPending || (spawnNow && !xr_->spawnPrev);
  xr_->spawnPrev = spawnNow;
  input.spawnPressed = xr_->spawnPending;
  XrActionStateBoolean deleteState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->deleteAction;
  getInfo.subactionPath = leftPath;
  xrGetActionStateBoolean(xr_->session, &getInfo, &deleteState);
  input.deleteHeld = deleteState.isActive && deleteState.currentState;

  XrActionStateFloat gripState = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
  getInfo.action = xr_->gripAction;
  getInfo.subactionPath = rightPath;
  xrGetActionStateFloat(xr_->session, &getInfo, &gripState);
  input.gripHeld = gripState.isActive ? gripState.currentState : 0.0F;

  // Per-hand aim rays for the drawn pointers; the right-hand aim doubles as
  // the manipulation pointer (head gaze fallback when it is untracked).
  uint32_t tracked = XR_SPACE_LOCATION_POSITION_VALID_BIT |
                     XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
  for (int hand = 0; hand < 2; ++hand) {
    XrSpaceLocation location = xrStruct<XrSpaceLocation>(XR_TYPE_SPACE_LOCATION);
    XrResult locate = xrLocateSpace(xr_->aimSpaces[hand], xr_->space,
                                    static_cast<XrTime>(displayTime), &location);
    if (XR_SUCCEEDED(locate) && (location.locationFlags & tracked) == tracked) {
      input.aimRays[hand] = rayFromPose(location.pose);
      input.aimValid[hand] = true;
    }
  }
  if (input.aimValid[1]) {
    input.pointerRay = input.aimRays[1];
  } else {
    // Head gaze fallback: the same horizon-locked eye pose the view is
    // rendered from, so the drag anchor matches what the user sees.
    input.pointerRay = rayFromPose(xr_->correctedViews[0].pose);
  }
  return true;
}

XrShell::FrameStatus XrShell::beginFrame(XrFrameInput& inputOut) {
  if (xr_ == nullptr) {
    return FrameStatus::Exit;
  }
  inputOut = XrFrameInput{};
  bool running = true;
  pollEvents(running);
  if (!running) {
    return FrameStatus::Exit;
  }
  if (!xr_->sessionRunning) {
    return FrameStatus::Pump; // session still starting; keep pumping
  }

  XrFrameWaitInfo waitInfo = xrStruct<XrFrameWaitInfo>(XR_TYPE_FRAME_WAIT_INFO);
  xr_->frameState = xrStruct<XrFrameState>(XR_TYPE_FRAME_STATE);
  if (!xrOk(xrWaitFrame(xr_->session, &waitInfo, &xr_->frameState), "xrWaitFrame")) {
    return FrameStatus::Exit;
  }
  XrFrameBeginInfo beginInfo = xrStruct<XrFrameBeginInfo>(XR_TYPE_FRAME_BEGIN_INFO);
  if (!xrOk(xrBeginFrame(xr_->session, &beginInfo), "xrBeginFrame")) {
    return FrameStatus::Exit;
  }
  xr_->frameBegun = true;
  xr_->frameReady = false;
  xr_->predictedDisplayTime = xr_->frameState.predictedDisplayTime;
  ++stats_.begun;
  // Pairing rule: every xrBeginFrame gets exactly one xrEndFrame. When a
  // frame cannot be rendered, end it here (zero layers) before bailing.
  auto failFrame = [this]() {
    endFrame(); // endFrame counts the zero-layer outcome
    return FrameStatus::Skipped;
  };
  // shouldRender is a power hint (spec: "should avoid heavy GPU work"), not a
  // constraint on xrEndFrame. Zero layers mean "clear the display" — WiVRn
  // shows that as a passthrough flash — so a hint must never blank the frame.
  // Rendering is ~3 ms here; the hint is only counted.
  if (!xr_->frameState.shouldRender) {
    ++stats_.renderFalse;
  }

  // Views. Locate into a scratch pair first: unlocated poses (transient
  // tracking gaps) keep the last good ones rather than failing the frame.
  XrViewLocateInfo locateInfo = xrStruct<XrViewLocateInfo>(XR_TYPE_VIEW_LOCATE_INFO);
  locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  locateInfo.displayTime = xr_->frameState.predictedDisplayTime;
  locateInfo.space = xr_->space;
  XrViewState viewState = xrStruct<XrViewState>(XR_TYPE_VIEW_STATE);
  XrView located[2] = {xrStruct<XrView>(XR_TYPE_VIEW), xrStruct<XrView>(XR_TYPE_VIEW)};
  std::uint32_t viewCount = 0;
  bool locateOk = xrOk(xrLocateViews(xr_->session, &locateInfo, &viewState, 2, &viewCount, located),
                       "xrLocateViews");
  std::uint32_t const poseFlags = XR_VIEW_STATE_POSITION_VALID_BIT |
                                  XR_VIEW_STATE_ORIENTATION_VALID_BIT;
  if (locateOk && viewCount == 2U && (viewState.viewStateFlags & poseFlags) == poseFlags) {
    xr_->views[0] = located[0];
    xr_->views[1] = located[1];
    xr_->haveGoodViews = true;
  } else if (xr_->haveGoodViews) {
    ++stats_.stalePose; // transient gap: reuse the last good poses
  } else {
    ++stats_.viewFail;
    return failFrame(); // no poses ever; nothing to show yet
  }

  // Horizon lock: strip the head's physical roll from both eye poses — one
  // correction for the pair, applied to the orientations and to the eye
  // positions around the eye midpoint so the stereo pair stays physically
  // consistent. The look direction is preserved exactly (the correction
  // rotates about it), so the horizon stays level while turning or
  // stick-pitching instead of swinging with the user's head roll.
  Vec3 const leftEye{xr_->views[0].pose.position.x, xr_->views[0].pose.position.y,
                     xr_->views[0].pose.position.z};
  Vec3 const rightEye{xr_->views[1].pose.position.x, xr_->views[1].pose.position.y,
                      xr_->views[1].pose.position.z};
  Vec3 const headMid = (leftEye + rightEye) * 0.5F;
  Quat const rawHead{xr_->views[0].pose.orientation.x, xr_->views[0].pose.orientation.y,
                     xr_->views[0].pose.orientation.z, xr_->views[0].pose.orientation.w};
  Quat const correction = gravityAlign(rawHead) * conjugate(normalized(rawHead));

  for (int eye = 0; eye < 2; ++eye) {
    XrFovf const& fov = xr_->views[eye].fov;
    Mat4 proj = cameraProjectionFromTangents(std::tan(fov.angleLeft), std::tan(fov.angleRight),
                                             std::tan(fov.angleUp), std::tan(fov.angleDown),
                                             0.05F, 200.0F); // match the desktop camera
    XrPosef const& pose = xr_->views[eye].pose;
    Vec3 const rawPos{pose.position.x, pose.position.y, pose.position.z};
    Quat const rawOrient{pose.orientation.x, pose.orientation.y, pose.orientation.z,
                         pose.orientation.w};
    Vec3 const correctedPos = headMid + rotate(correction, rawPos - headMid);
    Quat const correctedOrient = correction * rawOrient;
    // The corrected pair is what the images are rendered from AND what the
    // compositor must be told in endFrame — the runtime reprojects against
    // the submitted pose, and a raw/corrected mismatch makes the image swim
    // exactly while a rolled head moves.
    xr_->correctedViews[eye].fov = fov;
    xr_->correctedViews[eye].pose.position = {correctedPos.x, correctedPos.y, correctedPos.z};
    xr_->correctedViews[eye].pose.orientation = {correctedOrient.x, correctedOrient.y,
                                                 correctedOrient.z, correctedOrient.w};
    // Hand the app the composable pieces, not a baked view: the rig (its own
    // yaw/pitch) must be composed and horizon-locked as a whole, since a
    // pitched rig multiplied by a yawed head view produces phantom roll.
    inputOut.eyeProj[eye] = proj;
    inputOut.eyePos[eye] = correctedPos;
    inputOut.eyeOrient[eye] = correctedOrient;
  }

  // Head pose: eye midpoint for gizmo sizing, aligned orientation for the
  // flight and ray directions (forward is unchanged by the horizon lock).
  inputOut.headPosition = headMid;
  inputOut.headOrient = gravityAlign(rawHead);

  if (!syncActions(inputOut, static_cast<std::uint64_t>(xr_->frameState.predictedDisplayTime))) {
    ++stats_.actionFail;
    return failFrame();
  }

  // Acquire both eye images for the app to render into. The image is marked
  // owned the moment acquire succeeds, so any later failure still releases it
  // via endFrame.
  for (int eye = 0; eye < 2; ++eye) {
    XrSwapchainImageAcquireInfo acquireInfo = xrStruct<XrSwapchainImageAcquireInfo>(XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO);
    std::uint32_t index = 0;
    if (!xrOk(xrAcquireSwapchainImage(xr_->swapchains[eye], &acquireInfo, &index),
              "xrAcquireSwapchainImage")) {
      ++stats_.acquireFail;
      return failFrame();
    }
    xr_->acquiredIndex[eye] = index;
    eyeImages_[eye] = xr_->images[eye][index].image;
    eyeViews_[eye] = xr_->imageViews[eye][index];
    imagesAcquired_[eye] = true;
    XrSwapchainImageWaitInfo imageWait = xrStruct<XrSwapchainImageWaitInfo>(XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO);
    imageWait.timeout = XR_INFINITE_DURATION;
    if (!xrOk(xrWaitSwapchainImage(xr_->swapchains[eye], &imageWait), "xrWaitSwapchainImage")) {
      ++stats_.acquireFail;
      return failFrame();
    }
  }
  xr_->frameReady = true;
  // The app consumes this frame's undo/redo presses now; a later skip must
  // not swallow a fresh press or replay this one.
  xr_->undoPending = false;
  xr_->redoPending = false;
  xr_->spawnPending = false;
  return FrameStatus::Render;
}

bool XrShell::endFrame() {
  if (xr_ == nullptr || !xr_->frameBegun) {
    return false;
  }

  XrCompositionLayerProjectionView projectionViews[2];
  for (int eye = 0; eye < 2; ++eye) {
    projectionViews[eye] =
        xrStruct<XrCompositionLayerProjectionView>(XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW);
    projectionViews[eye].pose = xr_->correctedViews[eye].pose;
    projectionViews[eye].fov = xr_->views[eye].fov;
    projectionViews[eye].subImage.swapchain = xr_->swapchains[eye];
    projectionViews[eye].subImage.imageRect.offset = {0, 0};
    projectionViews[eye].subImage.imageRect.extent = {static_cast<int32_t>(eyeWidth_),
                                                      static_cast<int32_t>(eyeHeight_)};
    projectionViews[eye].subImage.imageArrayIndex = 0;
    if (imagesAcquired_[eye]) {
      XrSwapchainImageReleaseInfo releaseInfo = xrStruct<XrSwapchainImageReleaseInfo>(XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO);
      xrOk(xrReleaseSwapchainImage(xr_->swapchains[eye], &releaseInfo),
           "xrReleaseSwapchainImage");
      imagesAcquired_[eye] = false;
    }
  }

  XrCompositionLayerProjection layer = xrStruct<XrCompositionLayerProjection>(XR_TYPE_COMPOSITION_LAYER_PROJECTION);
  layer.space = xr_->space;
  layer.viewCount = 2;
  layer.views = projectionViews;
  // Passthrough: alpha-blend the layer over the real world. The clear colour
  // alpha is 0 in this mode, so empty pixels show the passthrough cameras.
  bool const alphaBlend = passthrough_ && alphaBlendSupported_;
  if (passthrough_ && !alphaBlendSupported_ && !passthroughWarned_) {
    passthroughWarned_ = true;
    std::printf("[xr] runtime does not support ALPHA_BLEND; passthrough stays off\n");
  }
  if (alphaBlend) {
    layer.layerFlags |= XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  }
  XrFrameEndInfo endInfo = xrStruct<XrFrameEndInfo>(XR_TYPE_FRAME_END_INFO);
  endInfo.displayTime = xr_->frameState.predictedDisplayTime;
  endInfo.environmentBlendMode =
      alphaBlend ? XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND : XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  // Layers must outlive the xrEndFrame call below — the runtime reads through
  // endInfo.layers during the call, so the array cannot be scoped to the branch.
  XrCompositionLayerBaseHeader const* layers[] = {
      reinterpret_cast<XrCompositionLayerBaseHeader const*>(&layer)};
  // frameReady is the only gate: we render despite shouldRender (a hint), and
  // zero layers mean "clear the display" (passthrough on WiVRn).
  if (xr_->frameReady) {
    endInfo.layerCount = 1;
    endInfo.layers = layers;
  } else {
    // Skipped or failed frames end with zero layers (never with unlocated
    // view poses).
    ++stats_.zeroLayer;
    endInfo.layerCount = 0;
    endInfo.layers = nullptr;
  }
  bool ok = xrOk(xrEndFrame(xr_->session, &endInfo), "xrEndFrame");
  xr_->frameBegun = false;
  xr_->frameReady = false;
  return ok;
}

void XrShell::shutdown() {
  if (xr_ == nullptr) {
    return;
  }
  for (int eye = 0; eye < 2; ++eye) {
    for (VkImageView& view : xr_->imageViews[eye]) {
      if (view != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view, nullptr);
      }
    }
    xr_->imageViews[eye].clear();
    if (xr_->swapchains[eye] != XR_NULL_HANDLE) {
      xrDestroySwapchain(xr_->swapchains[eye]);
      xr_->swapchains[eye] = XR_NULL_HANDLE;
    }
    imagesAcquired_[eye] = false;
  }
  for (XrSpace& aimSpace : xr_->aimSpaces) {
    if (aimSpace != XR_NULL_HANDLE) {
      xrDestroySpace(aimSpace);
      aimSpace = XR_NULL_HANDLE;
    }
  }
  if (xr_->space != XR_NULL_HANDLE) {
    xrDestroySpace(xr_->space);
  }
  if (xr_->session != XR_NULL_HANDLE) {
    xrDestroySession(xr_->session);
  }
  if (xr_->actionSet != XR_NULL_HANDLE) {
    xrDestroyActionSet(xr_->actionSet);
  }
  if (xr_->instance != XR_NULL_HANDLE) {
    xrDestroyInstance(xr_->instance);
    xr_->instance = XR_NULL_HANDLE;
  }
  // The enable2 calls created the Vulkan objects for us: the app owns them.
  // VkRenderer::shutdown() has already released its device resources.
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  physical_ = VK_NULL_HANDLE;
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
  delete xr_;
  xr_ = nullptr;
}

} // namespace vredit
