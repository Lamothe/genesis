#include "xr_shell.h"
#include "xr_view.h"
#include "controller_workflow.h"
#include "controller_arm_profile.h"
#include "vr_locomotion.h"

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

} // namespace

struct XrShell::ImplData {
  XrInstance instance{XR_NULL_HANDLE};
  XrSystemId system{XR_NULL_SYSTEM_ID};
  XrSession session{XR_NULL_HANDLE};
  XrSpace space{XR_NULL_HANDLE};
  XrActionSet actionSet{XR_NULL_HANDLE};
  XrAction triggerAction{XR_NULL_HANDLE};
  XrAction pinchAction{XR_NULL_HANDLE};
  XrAction pinchReadyAction{XR_NULL_HANDLE};
  XrAction graspAction{XR_NULL_HANDLE};
  XrAction graspReadyAction{XR_NULL_HANDLE};
  XrAction leftStickAction{XR_NULL_HANDLE};
  XrAction rightStickAction{XR_NULL_HANDLE};
  XrAction trackpadTouchAction{XR_NULL_HANDLE};
  XrAction menuAction{XR_NULL_HANDLE};
  XrAction undoAction{XR_NULL_HANDLE}; // B on Touch
  XrAction redoAction{XR_NULL_HANDLE}; // Y on Touch
  XrAction spawnAction{XR_NULL_HANDLE}; // A on Touch: model chooser (stage/place)
  XrAction deleteAction{XR_NULL_HANDLE}; // X on Touch: hold to delete
  XrAction gripAction{XR_NULL_HANDLE};
  XrAction aimAction{XR_NULL_HANDLE};
  XrAction gripPoseAction{XR_NULL_HANDLE}; // physical controller pose
  XrSpace aimSpaces[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};   // left, right
  XrSpace gripSpaces[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};  // left, right
  XrSwapchain swapchains[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
  std::vector<XrSwapchainImageVulkan2KHR> images[2];
  std::vector<VkImageView> imageViews[2];
  std::uint32_t acquiredIndex[2] = {0, 0};
  XrView views[2]{};
  bool haveGoodViews{false}; // views[] holds the last successfully located pair
  XrViewConfigurationView viewConfig[2];
  XrSessionState sessionState{XR_SESSION_STATE_UNKNOWN};
  bool sessionRunning{false};
  bool sessionVisible{false};
  bool frameBegun{false};
  bool frameBroken{false}; // cleanup or xrEndFrame failed; do not start another frame
  bool frameReady{false}; // views located and both eye images acquired
  XrTime predictedDisplayTime{0};
  XrFrameState frameState{};
  SelectTriggerLatch selectTrigger;
  SelectTriggerLatch leftSelectTrigger;
  bool selectInterruptedPending{false};
  ControllerArmProfileTracker controllerProfiles;
  ControllerArmContinuity controllerContinuity;
  BooleanPressLatch menuPress;
  BooleanPressLatch spawnPress;
  BooleanPressLatch undoPress;
  BooleanPressLatch redoPress;
  bool handTracking{false};
  XrHandTrackerEXT trackers[2]{XR_NULL_HANDLE, XR_NULL_HANDLE};
  PFN_xrCreateHandTrackerEXT createHandTracker{nullptr};
  PFN_xrDestroyHandTrackerEXT destroyHandTracker{nullptr};
  PFN_xrLocateHandJointsEXT locateHandJoints{nullptr};
  bool handInteraction{false}; // XR_EXT_hand_interaction enabled at instance creation
  XrPath handPaths[2]{XR_NULL_PATH, XR_NULL_PATH};    // "/user/hand/left|right" (diagnostics)
  XrPath handProfiles[2]{XR_NULL_PATH, XR_NULL_PATH}; // last active profile seen per hand
  int profileCheckIn{0};                              // frames until the next profile poll
  int menuSigPrev{-1}; // last menu (active, down) pair printed; -1 prints once at startup

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
  // Optional input extension: hand tracking. The hand-interaction profile
  // only exists for an instance that enabled XR_EXT_hand_interaction, so ask
  // the runtime rather than assuming WiVRn is in front of us.
  std::vector<char const*> extensions{"XR_KHR_vulkan_enable2"};
  std::uint32_t availableCount = 0;
  std::vector<XrExtensionProperties> available;
  if (XR_SUCCEEDED(xrEnumerateInstanceExtensionProperties(nullptr, 0, &availableCount, nullptr)) &&
      availableCount > 0) {
    available.assign(availableCount, xrStruct<XrExtensionProperties>(XR_TYPE_EXTENSION_PROPERTIES));
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, availableCount, &availableCount,
                                                         available.data()))) {
      available.clear(); // unknown extension set: controllers only, never fatal
    } else {
      available.resize(availableCount);
    }
  }
  for (XrExtensionProperties const& offered : available) {
    if (std::strcmp(offered.extensionName, XR_EXT_HAND_INTERACTION_EXTENSION_NAME) == 0) {
      extensions.push_back(XR_EXT_HAND_INTERACTION_EXTENSION_NAME);
      xr_->handInteraction = true;
      break;
    }
  }
  for (auto const& offered : available) {
    if (std::strcmp(offered.extensionName, XR_EXT_HAND_TRACKING_EXTENSION_NAME) == 0) {
      extensions.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
      xr_->handTracking = true;
    }
  }
  instanceInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
  instanceInfo.enabledExtensionNames = extensions.data();

  XrResult result = xrCreateInstance(&instanceInfo, &xr_->instance);
  if (result == XR_ERROR_RUNTIME_UNAVAILABLE) {
    std::printf("[xr] no OpenXR runtime configured\n");
    return InitResult::NoRuntime;
  }
  if (!xrOk(result, "xrCreateInstance")) {
    return InitResult::Failed;
  }
  if (xr_->handInteraction) {
    std::printf("[xr] hand interaction: enabled (right pinch selects, hand aim is the pointer)\n");
  } else {
    std::printf("[xr] hand interaction: not offered by this runtime; controllers only\n");
  }

  if (xr_->handTracking) {
    xrGetInstanceProcAddr(xr_->instance, "xrCreateHandTrackerEXT",
                         reinterpret_cast<PFN_xrVoidFunction*>(&xr_->createHandTracker));
    xrGetInstanceProcAddr(xr_->instance, "xrDestroyHandTrackerEXT",
                         reinterpret_cast<PFN_xrVoidFunction*>(&xr_->destroyHandTracker));
    xrGetInstanceProcAddr(xr_->instance, "xrLocateHandJointsEXT",
                         reinterpret_cast<PFN_xrVoidFunction*>(&xr_->locateHandJoints));
    xr_->handTracking = xr_->createHandTracker && xr_->destroyHandTracker && xr_->locateHandJoints;
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
  if (xr_->handTracking) {
    auto hands = xrStruct<XrSystemHandTrackingPropertiesEXT>(XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT);
    auto properties = xrStruct<XrSystemProperties>(XR_TYPE_SYSTEM_PROPERTIES);
    properties.next = &hands;
    xr_->handTracking = XR_SUCCEEDED(xrGetSystemProperties(xr_->instance, xr_->system, &properties)) &&
        hands.supportsHandTracking;
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
  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(physical_, &supported);
  if (!supported.shaderSampledImageArrayDynamicIndexing) {
    std::printf("[xr] sampled image dynamic indexing unavailable\n");
    return InitResult::Failed;
  }
  VkPhysicalDeviceFeatures enabled{};
  enabled.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
  VkDeviceCreateInfo vkDeviceInfo = vkStruct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
  vkDeviceInfo.pEnabledFeatures = &enabled;

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

  if (xr_->handTracking) {
    for (int hand = 0; hand < 2; ++hand) {
      auto info = xrStruct<XrHandTrackerCreateInfoEXT>(XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT);
      info.hand = hand == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
      info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
      xrOk(xr_->createHandTracker(xr_->session, &info, &xr_->trackers[hand]), "create hand tracker");
    }
  }
  std::printf("[hands] palm joints: left %s, right %s (aim/pinch still available without joints)\n",
      xr_->trackers[0] != XR_NULL_HANDLE ? "enabled" : "unavailable",
      xr_->trackers[1] != XR_NULL_HANDLE ? "enabled" : "unavailable");
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
  xr_->handPaths[0] = leftPath;
  xr_->handPaths[1] = rightPath;

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

  if (!makeFloat("trigger", &xr_->triggerAction) || !makeFloat("pinch", &xr_->pinchAction) || !makeBool("pinch_ready", &xr_->pinchReadyAction) || !makeFloat("hand_grasp", &xr_->graspAction) || !makeBool("grasp_ready", &xr_->graspReadyAction) || !makeVec2("leftstick", &xr_->leftStickAction) ||
      !makeVec2("rightstick", &xr_->rightStickAction) ||
      !makeBool("trackpad_touch", &xr_->trackpadTouchAction) || !makeBool("menu", &xr_->menuAction) ||
      !makeBool("undo", &xr_->undoAction) || !makeBool("redo", &xr_->redoAction) ||
      !makeBool("spawn", &xr_->spawnAction) || !makeBool("delete", &xr_->deleteAction) ||
      !makeFloat("grip", &xr_->gripAction) || !makePose("aim", &xr_->aimAction) ||
      !makePose("grip_pose", &xr_->gripPoseAction)) {
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
        {"/user/hand/left/input/menu/click", xr_->menuAction},
        {"/user/hand/right/input/aim/pose", xr_->aimAction},
        {"/user/hand/left/input/aim/pose", xr_->aimAction},
        {"/user/hand/right/input/grip/pose", xr_->gripPoseAction},
        {"/user/hand/left/input/grip/pose", xr_->gripPoseAction}}}, // no thumbstick on this profile
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
        // The physical grip pose drives view-relative model manipulation;
        // the aim ray only selects or points at UI.
        {"/user/hand/right/input/grip/pose", xr_->gripPoseAction},
        {"/user/hand/left/input/grip/pose", xr_->gripPoseAction},
        // Undo/redo/spawn on the face buttons. Thumbstick clicks are
        // deliberately UNBOUND (the user's binding): the carousel is the
        // left menu button's TAP. A = spawn, X (held) = delete.
        {"/user/hand/right/input/b/click", xr_->undoAction},
        {"/user/hand/left/input/y/click", xr_->redoAction},
        {"/user/hand/right/input/a/click", xr_->spawnAction},
        {"/user/hand/left/input/x/click", xr_->deleteAction}}},
      {"/interaction_profiles/htc/vive_controller",
       {{"/user/hand/right/input/trigger/click", xr_->triggerAction},
        {"/user/hand/left/input/trigger/click", xr_->triggerAction},
        {"/user/hand/left/input/trackpad", xr_->leftStickAction},
        {"/user/hand/right/input/trackpad", xr_->rightStickAction},
        {"/user/hand/left/input/trackpad/touch", xr_->trackpadTouchAction},
        {"/user/hand/right/input/trackpad/touch", xr_->trackpadTouchAction},
        {"/user/hand/left/input/menu/click", xr_->menuAction},
        {"/user/hand/right/input/squeeze/click", xr_->gripAction},
        {"/user/hand/left/input/squeeze/click", xr_->gripAction},
        {"/user/hand/right/input/aim/pose", xr_->aimAction},
        {"/user/hand/left/input/aim/pose", xr_->aimAction},
        {"/user/hand/right/input/grip/pose", xr_->gripPoseAction},
        {"/user/hand/left/input/grip/pose", xr_->gripPoseAction}}}, // trackpad clicks are not history shortcuts
  };
  if (xr_->handInteraction) {
    // Preserve aim activation for the existing trigger path, but use the
    // explicit pinch action for hand interactions and two-hand manipulation.
    bindings.push_back(
        {"/interaction_profiles/ext/hand_interaction_ext",
         {{"/user/hand/right/input/aim/pose", xr_->aimAction},
          {"/user/hand/left/input/aim/pose", xr_->aimAction},
          {"/user/hand/left/input/aim_activate_ext/value", xr_->triggerAction},
          {"/user/hand/right/input/aim_activate_ext/value", xr_->triggerAction},
          {"/user/hand/left/input/pinch_ext/value", xr_->pinchAction},
          {"/user/hand/right/input/pinch_ext/value", xr_->pinchAction},
          {"/user/hand/left/input/pinch_ext/ready_ext", xr_->pinchReadyAction},
          {"/user/hand/right/input/pinch_ext/ready_ext", xr_->pinchReadyAction},
          {"/user/hand/right/input/grasp_ext/value", xr_->graspAction},
          {"/user/hand/right/input/grasp_ext/ready_ext", xr_->graspReadyAction}}});
  }

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

  // Aim spaces for the pointers and grip spaces for the physical poses
  // (left, right). The grip space is what model manipulation follows; the
  // aim space only selects and points at UI.
  XrActionSpaceCreateInfo spaceInfo = xrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
  spaceInfo.poseInActionSpace.orientation.w = 1.0F;
  XrPath const handPaths[2] = {leftPath, rightPath};
  for (int hand = 0; hand < 2; ++hand) {
    spaceInfo.action = xr_->aimAction;
    spaceInfo.subactionPath = handPaths[hand];
    if (!xrOk(xrCreateActionSpace(xr_->session, &spaceInfo, &xr_->aimSpaces[hand]),
              "xrCreateActionSpace(aim)")) {
      return false;
    }
    spaceInfo.action = xr_->gripPoseAction;
    if (!xrOk(xrCreateActionSpace(xr_->session, &spaceInfo, &xr_->gripSpaces[hand]),
              "xrCreateActionSpace(grip)")) {
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

void XrShell::pollEvents(bool& running, bool& failed) {
  XrEventDataBuffer event = xrStruct<XrEventDataBuffer>(XR_TYPE_EVENT_DATA_BUFFER);
  while (xrPollEvent(xr_->instance, &event) == XR_SUCCESS) {
    if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
      auto* changed = reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
      if (xr_->sessionState == XR_SESSION_STATE_FOCUSED &&
          changed->state != XR_SESSION_STATE_FOCUSED) {
        xr_->selectTrigger.interrupt();
        xr_->leftSelectTrigger.interrupt();
        xr_->menuPress.interrupt();
        xr_->spawnPress.interrupt();
        xr_->undoPress.interrupt();
        xr_->redoPress.interrupt();
        xr_->selectInterruptedPending = true;
      }
      xr_->controllerProfiles.observeFocus(changed->state == XR_SESSION_STATE_FOCUSED);
      xr_->sessionState = changed->state;
      switch (changed->state) {
      case XR_SESSION_STATE_READY: {
        XrSessionBeginInfo beginInfo = xrStruct<XrSessionBeginInfo>(XR_TYPE_SESSION_BEGIN_INFO);
        beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        if (xrOk(xrBeginSession(xr_->session, &beginInfo), "xrBeginSession")) {
          xr_->sessionRunning = true;
        } else {
          running = false; // fatal: the session can never start
          failed = true;
        }
        break;
      }
      case XR_SESSION_STATE_STOPPING:
        if (!xrOk(xrEndSession(xr_->session), "xrEndSession")) {
          running = false; // fatal: the session can never end cleanly
          failed = true;
        }
        xr_->sessionRunning = false;
        break;      case XR_SESSION_STATE_EXITING:
      case XR_SESSION_STATE_LOSS_PENDING:
        running = false;
        break;
      default:
        break;
      }
    } else if (event.type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) {
      // A profile can change without an inactive sample in between.
      xr_->controllerContinuity.profileChanged();
      xr_->selectTrigger.interrupt();
      xr_->leftSelectTrigger.interrupt();
      xr_->menuPress.interrupt();
      xr_->spawnPress.interrupt();
      xr_->undoPress.interrupt();
      xr_->redoPress.interrupt();
      xr_->selectInterruptedPending = true;
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
  XrResult const syncResult = xrSyncActions(xr_->session, &syncInfo);
  if (!xrOk(syncResult, "xrSyncActions")) {
    return false;
  }

  XrPath leftPath = XR_NULL_PATH;
  XrPath rightPath = XR_NULL_PATH;
  xrStringToPath(xr_->instance, "/user/hand/left", &leftPath);
  xrStringToPath(xr_->instance, "/user/hand/right", &rightPath);

  XrPath handProfile = XR_NULL_PATH;
  bool const handProfileKnown = XR_SUCCEEDED(xrStringToPath(
      xr_->instance, "/interaction_profiles/ext/hand_interaction_ext", &handProfile));
  for (int hand = 0; hand < 2; ++hand) {
    auto profile = xrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
    bool const ok = xr_->handPaths[hand] != XR_NULL_PATH &&
        XR_SUCCEEDED(xrGetCurrentInteractionProfile(xr_->session, xr_->handPaths[hand], &profile));
    xr_->controllerProfiles.sample(hand, ok, profile.interactionProfile);
    input.controllerProfileKnown[hand] = xr_->controllerProfiles.known(hand);
    input.controllerProfileChanged[hand] = xr_->controllerProfiles.changed(hand);
    auto const classification = classifyControllerArmProfile(
        ok, profile.interactionProfile, handProfileKnown, handProfile);
    input.controllerProfileIsController[hand] = classification == ControllerArmProfile::controller;
    input.handActive[hand] = xr_->handInteraction && classification == ControllerArmProfile::hand;
  }
  input.controllerVisualsInterrupted = xr_->controllerProfiles.focusInterrupted();

  XrActionStateGetInfo getInfo = xrStruct<XrActionStateGetInfo>(XR_TYPE_ACTION_STATE_GET_INFO);

  XrActionStateFloat triggerState = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
  getInfo.action = xr_->triggerAction;
  getInfo.subactionPath = rightPath;
  XrResult const triggerResult = xrGetActionStateFloat(xr_->session, &getInfo, &triggerState);
  // Hysteresis prevents chatter; inactive or failed queries are not releases.
  auto const select = xr_->selectTrigger.updateTracked(input.viewTracked, syncResult == XR_SUCCESS &&
                                                XR_SUCCEEDED(triggerResult) && triggerState.isActive &&
                                                !xr_->selectInterruptedPending, triggerState.currentState);
  input.controllerTrigger[1] = controllerArmVisualSample(syncResult == XR_SUCCESS,
      XR_SUCCEEDED(triggerResult), triggerState.isActive, triggerState.currentState);
  input.selectActive = select.active;
  input.selectHeld = select.held;
  input.selectPressed = select.pressed;
  input.selectReleased = select.released;

  triggerState = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
  getInfo.subactionPath = leftPath;
  XrResult const leftTriggerResult = xrGetActionStateFloat(xr_->session, &getInfo, &triggerState);
  auto const leftSelect = xr_->leftSelectTrigger.updateTracked(input.viewTracked, syncResult == XR_SUCCESS &&
      XR_SUCCEEDED(leftTriggerResult) && triggerState.isActive &&
      !xr_->selectInterruptedPending, triggerState.currentState);
  input.controllerTrigger[0] = controllerArmVisualSample(syncResult == XR_SUCCESS,
      XR_SUCCEEDED(leftTriggerResult), triggerState.isActive, triggerState.currentState);
  input.leftSelectActive = leftSelect.active;
  input.leftSelectHeld = leftSelect.held;
  input.leftSelectPressed = leftSelect.pressed;
  input.leftSelectReleased = leftSelect.released;

  XrPath viveProfile = XR_NULL_PATH;
  bool const vivePathValid = XR_SUCCEEDED(xrStringToPath(
      xr_->instance, "/interaction_profiles/htc/vive_controller", &viveProfile));
  auto sampleStick = [&](XrAction action, XrPath hand) {
    auto state = xrStruct<XrActionStateVector2f>(XR_TYPE_ACTION_STATE_VECTOR2F);
    getInfo.action = action;
    getInfo.subactionPath = hand;
    XrResult const result = xrGetActionStateVector2f(xr_->session, &getInfo, &state);
    auto profile = xrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
    bool const profileValid = vivePathValid &&
        XR_SUCCEEDED(xrGetCurrentInteractionProfile(xr_->session, hand, &profile));
    bool const trackpad = profileValid && profile.interactionProfile == viveProfile;
    auto touch = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
    bool touchActive = false;
    if (trackpad) {
      getInfo.action = xr_->trackpadTouchAction;
      touchActive = XR_SUCCEEDED(xrGetActionStateBoolean(xr_->session, &getInfo, &touch)) &&
                    touch.isActive;
    }
    return vrNormaliseStick({state.currentState.x, state.currentState.y},
        profileValid && syncResult == XR_SUCCESS && XR_SUCCEEDED(result) && state.isActive &&
            !xr_->selectInterruptedPending, trackpad, touchActive, touch.currentState);
  };
  auto const leftStick = sampleStick(xr_->leftStickAction, leftPath);
  auto const rightStick = sampleStick(xr_->rightStickAction, rightPath);
  input.leftStickActive = leftStick.active;
  input.leftStick = leftStick.value;
  input.rightStickActive = rightStick.active;
  input.rightStick = rightStick.value;
  if (input.leftStickActive) {
    stats_.leftXMax = std::max(stats_.leftXMax, std::abs(input.leftStick.x));
    stats_.leftYMax = std::max(stats_.leftYMax, std::abs(input.leftStick.y));
  }
  if (input.rightStickActive) {
    stats_.rightXMax = std::max(stats_.rightXMax, std::abs(input.rightStick.x));
    stats_.rightYMax = std::max(stats_.rightYMax, std::abs(input.rightStick.y));
  }
  // Inactivity alone does not prove a stick is absent: allow the fallback
  // only for a successfully identified simple-controller profile. Query on
  // every inactive sample so profile switches cannot leave stale permission.
  if (!input.rightStickActive && syncResult == XR_SUCCESS &&
      input.viewTracked && !xr_->selectInterruptedPending) {
    XrPath simpleProfile = XR_NULL_PATH;
    auto profile = xrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
    input.rightStickAbsent =
        XR_SUCCEEDED(xrStringToPath(xr_->instance, "/interaction_profiles/khr/simple_controller", &simpleProfile)) &&
        XR_SUCCEEDED(xrGetCurrentInteractionProfile(xr_->session, rightPath, &profile)) &&
        profile.interactionProfile == simpleProfile;
  }

  // Both toggle actions need a tracked view and an active physical release
  // after any interruption. Never carry a press across a lost/skipped frame.
  bool const toggleSafe = input.viewTracked && syncResult == XR_SUCCESS &&
      xr_->sessionState == XR_SESSION_STATE_FOCUSED && !xr_->selectInterruptedPending;
  XrActionStateBoolean menuLeft = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->menuAction;
  getInfo.subactionPath = leftPath;
  XrResult const menuResult = xrGetActionStateBoolean(xr_->session, &getInfo, &menuLeft);
  xr_->menuPress.updateTracked(toggleSafe, XR_SUCCEEDED(menuResult) && menuLeft.isActive,
                               menuLeft.currentState);
  input.menuPressed = xr_->menuPress.pending();
  input.menuHeld = toggleSafe && XR_SUCCEEDED(menuResult) && menuLeft.isActive && menuLeft.currentState;
  // "The menu button does nothing" has two very different causes: the button
  // never reaches us (the runtime has the hand on a profile without a menu
  // binding, or reserved the button), or it reaches us and the app drops it.
  // Print every change of (active, down) so the console says which — the
  // startup line shows whether the binding is live before you even press it.
  int const menuSig = (menuLeft.isActive ? 2 : 0) + (menuLeft.currentState ? 1 : 0);
  if (menuSig != xr_->menuSigPrev) {
    std::printf("[xr] left menu: %s, button %s\n",
                menuLeft.isActive ? "active" : "inactive (no menu binding on this profile)",
                menuLeft.currentState ? "down" : "up");
    xr_->menuSigPrev = menuSig;
  }

  // B/Y history uses the same interruption-aware edge semantics as Menu/A.
  // Failed or inactive queries are not releases; an active release on a safe
  // frame must rearm each button before a later press can trigger history.
  XrActionStateBoolean undoState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->undoAction;
  getInfo.subactionPath = rightPath;
  XrResult const undoResult = xrGetActionStateBoolean(xr_->session, &getInfo, &undoState);
  xr_->undoPress.updateTracked(toggleSafe, XR_SUCCEEDED(undoResult) && undoState.isActive,
                               undoState.currentState);
  input.undoPressed = xr_->undoPress.pending();
  XrActionStateBoolean redoState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->redoAction;
  getInfo.subactionPath = leftPath;
  XrResult const redoResult = xrGetActionStateBoolean(xr_->session, &getInfo, &redoState);
  xr_->redoPress.updateTracked(toggleSafe, XR_SUCCEEDED(redoResult) && redoState.isActive,
                               redoState.currentState);
  input.redoPressed = xr_->redoPress.pending();

  // A toggles the carousel under the same interruption rules as Menu.
  XrActionStateBoolean spawnState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->spawnAction;
  getInfo.subactionPath = rightPath;
  XrResult const spawnResult = xrGetActionStateBoolean(xr_->session, &getInfo, &spawnState);
  xr_->spawnPress.updateTracked(toggleSafe, XR_SUCCEEDED(spawnResult) && spawnState.isActive,
                                spawnState.currentState);
  input.spawnPressed = xr_->spawnPress.pending();
  XrActionStateBoolean deleteState = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.action = xr_->deleteAction;
  getInfo.subactionPath = leftPath;
  XrResult const deleteResult = xrGetActionStateBoolean(xr_->session, &getInfo, &deleteState);
  input.deleteActive = syncResult == XR_SUCCESS && XR_SUCCEEDED(deleteResult) && deleteState.isActive;
  input.deleteHeld = input.deleteActive && deleteState.currentState;
  input.deleteInterrupted = xr_->selectInterruptedPending;
  input.controllerVisualsFocused = xr_->sessionState == XR_SESSION_STATE_FOCUSED;

  XrActionStateFloat gripState = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
  getInfo.action = xr_->gripAction;
  getInfo.subactionPath = rightPath;
  xrGetActionStateFloat(xr_->session, &getInfo, &gripState);
  input.gripHeld = gripState.isActive ? gripState.currentState : 0.0F;
  // Do not change the existing modifier query above. These are visual-only.
  for (int hand = 0; hand < 2; ++hand) {
    auto visualGrip = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
    getInfo.subactionPath = xr_->handPaths[hand];
    XrResult const result = xrGetActionStateFloat(xr_->session, &getInfo, &visualGrip);
    input.controllerSqueeze[hand] = controllerArmVisualSample(syncResult == XR_SUCCESS,
        XR_SUCCEEDED(result), visualGrip.isActive, visualGrip.currentState);
  }

  // Keep hand-only gestures separate from controllers (including inferred joints).
  for (int hand = 0; hand < 2; ++hand) {
    auto pinch = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
    getInfo.action = xr_->pinchAction;
    getInfo.subactionPath = xr_->handPaths[hand];
    xrGetActionStateFloat(xr_->session, &getInfo, &pinch);
    auto ready = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
    getInfo.action = xr_->pinchReadyAction;
    xrGetActionStateBoolean(xr_->session, &getInfo, &ready);
    input.pinchActive[hand] = pinch.isActive && ready.isActive && ready.currentState;

    input.pinch[hand] = pinch.isActive ? pinch.currentState : 0.0F;
    if (!input.handActive[hand] || xr_->trackers[hand] == XR_NULL_HANDLE) {
      continue;
    }
    XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT]{};
    auto locations = xrStruct<XrHandJointLocationsEXT>(XR_TYPE_HAND_JOINT_LOCATIONS_EXT);
    locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
    locations.jointLocations = joints;
    auto locate = xrStruct<XrHandJointsLocateInfoEXT>(XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT);
    locate.baseSpace = xr_->space;
    locate.time = static_cast<XrTime>(displayTime);
    if (XR_SUCCEEDED(xr_->locateHandJoints(xr_->trackers[hand], &locate, &locations)) &&
        locations.isActive) {
      static_assert(XR_HAND_JOINT_COUNT_EXT == 26);
      for (int joint = 0; joint < XR_HAND_JOINT_COUNT_EXT; ++joint) {
        constexpr auto flags = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT;
        if ((joints[joint].locationFlags & flags) == flags) {
          auto const& p = joints[joint].pose.position;
          input.handJoints[hand][joint] = {p.x, p.y, p.z};
          input.handJointValid[hand][joint] = true;
        }
      }
      int extended = 0;

      for (int finger = 0; finger < 4; ++finger) {
        int const proximal = XR_HAND_JOINT_INDEX_PROXIMAL_EXT + finger * 5;
        auto const& a = joints[proximal];
        auto const& b = joints[proximal + 1];
        auto const& c = joints[proximal + 3];
        constexpr auto trackedPosition = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT;
        if ((a.locationFlags & trackedPosition) == trackedPosition &&
            (b.locationFlags & trackedPosition) == trackedPosition &&
            (c.locationFlags & trackedPosition) == trackedPosition) {
          auto point = [](XrHandJointLocationEXT const& joint) {
            return Vec3{joint.pose.position.x, joint.pose.position.y, joint.pose.position.z};
          };
          if (dot(normalized(point(b) - point(a)), normalized(point(c) - point(b))) > .75F) {
            ++extended;
          }
        }
      }
      input.handOpen[hand] = extended >= 3;
      auto const& palm = joints[XR_HAND_JOINT_PALM_EXT];
      constexpr XrSpaceLocationFlags required = XR_SPACE_LOCATION_POSITION_VALID_BIT |
          XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT |
          XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
      if ((palm.locationFlags & required) == required) {
        auto const& p = palm.pose;
        input.hands[hand] = {{p.position.x, p.position.y, p.position.z},
            {p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w}, true};
      }
    }
  }
  // Grasp is an independent hand-only action; controller squeeze remains grip.
  auto grasp = xrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
  auto graspReady = xrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
  getInfo.subactionPath = rightPath;
  getInfo.action = xr_->graspAction;
  xrGetActionStateFloat(xr_->session, &getInfo, &grasp);
  getInfo.action = xr_->graspReadyAction;
  xrGetActionStateBoolean(xr_->session, &getInfo, &graspReady);
  input.graspActive = input.handActive[1] && grasp.isActive && graspReady.isActive && graspReady.currentState;
  input.grasp = grasp.isActive ? grasp.currentState : 0.0F;
  // Per-hand aim rays for the drawn pointers; the right-hand aim doubles as
  // the manipulation pointer (head gaze fallback when it is untracked).
  // VALID means the pose is usable; TRACKED means it is actively observed.
  // OpenXR keeps VALID set while clearing TRACKED for an inferred or
  // last-known pose, so the carousel's press and highlight require both.
  uint32_t const aimValidFlags = XR_SPACE_LOCATION_POSITION_VALID_BIT |
                                 XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
  uint32_t const aimTrackedFlags = aimValidFlags |
                                   XR_SPACE_LOCATION_POSITION_TRACKED_BIT |
                                   XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
  for (int hand = 0; hand < 2; ++hand) {
    XrSpaceLocation location = xrStruct<XrSpaceLocation>(XR_TYPE_SPACE_LOCATION);
    XrResult locate = xrLocateSpace(xr_->aimSpaces[hand], xr_->space,
                                    static_cast<XrTime>(displayTime), &location);
    if (XR_SUCCEEDED(locate) && (location.locationFlags & aimValidFlags) == aimValidFlags) {
      input.aimRays[hand] = rayFromXrPose(location.pose);
      input.aimValid[hand] = true;
      input.aimTracked[hand] = (location.locationFlags & aimTrackedFlags) == aimTrackedFlags;
    }
    // The pose-action state gates only the visual; controller editing remains unchanged.
    auto gripPoseState = xrStruct<XrActionStatePose>(XR_TYPE_ACTION_STATE_POSE);
    getInfo.action = xr_->gripPoseAction;
    getInfo.subactionPath = xr_->handPaths[hand];
    XrResult const poseResult = xrGetActionStatePose(xr_->session, &getInfo, &gripPoseState);
    input.controllerGripPoseActive[hand] = controllerArmGripActive(
        XR_SUCCEEDED(poseResult), gripPoseState.isActive);
    // The physical grip pose is measured independently: a VALID-but-untracked
    // pose is recorded as exactly that, never silently promoted to usable.
    XrSpaceLocation grip = xrStruct<XrSpaceLocation>(XR_TYPE_SPACE_LOCATION);
    locate = xrLocateSpace(xr_->gripSpaces[hand], xr_->space,
                           static_cast<XrTime>(displayTime), &grip);
    if (XR_SUCCEEDED(locate) && (grip.locationFlags & aimValidFlags) == aimValidFlags) {
      auto const& p = grip.pose;
      input.controllerGrip[hand].position = {p.position.x, p.position.y, p.position.z};
      input.controllerGrip[hand].orientation = {p.orientation.x, p.orientation.y,
                                               p.orientation.z, p.orientation.w};
      input.controllerGrip[hand].valid = true;
      input.controllerGrip[hand].tracked =
          (grip.locationFlags & aimTrackedFlags) == aimTrackedFlags;
    }
  }
  for (int hand = 0; hand < 2; ++hand) {
    xr_->controllerContinuity.sample(hand, input.viewTracked && input.controllerVisualsFocused &&
        input.controllerProfileIsController[hand] && input.controllerGripPoseActive[hand] &&
        input.controllerGrip[hand].usable());
    input.controllerTrackingInterrupted[hand] = xr_->controllerContinuity.interrupted(hand);
  }
  // Head gaze fallback uses the same raw eye pose as the rendered view.
  input.pointerRay = xrPointerRay(input.aimValid[1], input.aimRays[1], xr_->views[0]);

  // Which input device is actually driving each hand? A profile change is the
  // difference between "hands are being tracked" and "the runtime is ignoring
  // them" — and a dead menu button is usually the runtime having moved the
  // hand onto a profile that has no menu binding, which looks identical to
  // "never pressed" from here. Say it once per change instead of guessing.
  if (--xr_->profileCheckIn <= 0) {
    xr_->profileCheckIn = 60; // ~1 s at headset frame rates
    static char const* const kHandNames[2] = {"left", "right"};
    for (int hand = 0; hand < 2; ++hand) {
      XrInteractionProfileState profile =
          xrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
      if (XR_FAILED(xrGetCurrentInteractionProfile(xr_->session, xr_->handPaths[hand], &profile)) ||
          profile.interactionProfile == xr_->handProfiles[hand]) {
        continue;
      }
      xr_->handProfiles[hand] = profile.interactionProfile;
      char name[XR_MAX_PATH_LENGTH]{};
      std::uint32_t written = 0;
      if (profile.interactionProfile == XR_NULL_PATH ||
          XR_FAILED(xrPathToString(xr_->instance, profile.interactionProfile, sizeof(name), &written,
                                   name))) {
        std::printf("[xr] %s hand input: none\n", kHandNames[hand]);
      } else {
        std::printf("[xr] %s hand input: %s\n", kHandNames[hand], name);
      }
    }
  }
  return true;
}

XrShell::FrameStatus XrShell::beginFrame(XrFrameInput& inputOut) {
  if (xr_ == nullptr || xr_->frameBegun || xr_->frameBroken) {
    return FrameStatus::Failed;
  }
  inputOut = XrFrameInput{};
  bool running = true;
  bool failed = false;
  pollEvents(running, failed);
  if (!running) {
    return failed ? FrameStatus::Failed : FrameStatus::Exit;
  }
  if (!xr_->sessionRunning) {
    xr_->controllerContinuity.skipFrame();
    xr_->selectTrigger.interrupt();
    xr_->leftSelectTrigger.interrupt();
    xr_->menuPress.interrupt();
    xr_->spawnPress.interrupt();
    xr_->undoPress.interrupt();
    xr_->redoPress.interrupt();
    return FrameStatus::Pump; // session still starting; keep pumping
  }

  XrFrameWaitInfo waitInfo = xrStruct<XrFrameWaitInfo>(XR_TYPE_FRAME_WAIT_INFO);
  xr_->frameState = xrStruct<XrFrameState>(XR_TYPE_FRAME_STATE);
  if (!xrOk(xrWaitFrame(xr_->session, &waitInfo, &xr_->frameState), "xrWaitFrame")) {
    return FrameStatus::Failed;
  }
  XrFrameBeginInfo beginInfo = xrStruct<XrFrameBeginInfo>(XR_TYPE_FRAME_BEGIN_INFO);
  if (!xrOk(xrBeginFrame(xr_->session, &beginInfo), "xrBeginFrame")) {
    return FrameStatus::Failed;
  }
  xr_->frameBegun = true;
  xr_->frameReady = false;
  xr_->predictedDisplayTime = xr_->frameState.predictedDisplayTime;
  ++stats_.begun;
  // Pairing rule: every xrBeginFrame gets exactly one xrEndFrame. When a
  // frame cannot be rendered, end it here (zero layers) before bailing.
  auto failFrame = [this]() {
    xr_->controllerContinuity.skipFrame();
    xr_->selectTrigger.interrupt();
    xr_->leftSelectTrigger.interrupt();
    xr_->menuPress.interrupt();
    xr_->spawnPress.interrupt();
    xr_->undoPress.interrupt();
    xr_->redoPress.interrupt();
    // A failed release or end leaves the runtime in an uncertain state.
    return abortFrame() ? FrameStatus::Skipped : FrameStatus::Failed;
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

  // Render and submit the same raw located poses, preserving the natural
  // eye baseline and roll in the same tracking space as hands and controllers.
  inputFromXrViews(xr_->views, inputOut);
  std::uint32_t const trackedFlags = XR_VIEW_STATE_POSITION_TRACKED_BIT |
                                     XR_VIEW_STATE_ORIENTATION_TRACKED_BIT;
  inputOut.viewTracked = locateOk && viewCount == 2U &&
      (viewState.viewStateFlags & (poseFlags | trackedFlags)) == (poseFlags | trackedFlags);

  if (!syncActions(inputOut, static_cast<std::uint64_t>(xr_->frameState.predictedDisplayTime))) {
    ++stats_.actionFail;
    return failFrame();
  }

  // Mark ownership immediately; abortFrame waits before releasing even if
  // the initial wait fails.
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
    XrResult waitResult;
    do {
      waitResult = xrWaitSwapchainImage(xr_->swapchains[eye], &imageWait);
    } while (waitResult == XR_TIMEOUT_EXPIRED);
    if (!xrOk(waitResult, "xrWaitSwapchainImage")) {
      ++stats_.acquireFail;
      return failFrame();
    }
    imagesWaited_[eye] = true;
  }
  xr_->frameReady = true;
  xr_->selectInterruptedPending = false;
  xr_->controllerProfiles.consume();
  xr_->controllerContinuity.consume();
  // The app consumes this frame's button presses now; never replay them.
  xr_->undoPress.consume();
  xr_->redoPress.consume();
  xr_->spawnPress.consume();
  xr_->menuPress.consume();
  return FrameStatus::Render;
}

bool XrShell::endFrame() {
  return finishFrame(true);
}

bool XrShell::abortFrame() {
  return finishFrame(false);
}

bool XrShell::finishFrame(bool submitProjection) {
  if (xr_ == nullptr || !xr_->frameBegun) {
    return false;
  }

  bool released = true;
  XrCompositionLayerProjectionView projectionViews[2];
  for (int eye = 0; eye < 2; ++eye) {
    projectionViews[eye] =
        xrStruct<XrCompositionLayerProjectionView>(XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW);
    projectionFromXrView(xr_->views[eye], projectionViews[eye]);
    projectionViews[eye].subImage.swapchain = xr_->swapchains[eye];
    projectionViews[eye].subImage.imageRect.offset = {0, 0};
    projectionViews[eye].subImage.imageRect.extent = {static_cast<int32_t>(eyeWidth_),
                                                      static_cast<int32_t>(eyeHeight_)};
    projectionViews[eye].subImage.imageArrayIndex = 0;
    if (imagesAcquired_[eye]) {
      if (!imagesWaited_[eye]) {
        XrSwapchainImageWaitInfo waitInfo = xrStruct<XrSwapchainImageWaitInfo>(XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO);
        waitInfo.timeout = XR_INFINITE_DURATION;
        XrResult waitResult;
        do {
          waitResult = xrWaitSwapchainImage(xr_->swapchains[eye], &waitInfo);
        } while (waitResult == XR_TIMEOUT_EXPIRED);
        if (!xrOk(waitResult, "xrWaitSwapchainImage during cleanup")) {
          released = false;
          continue; // releasing an image that was not waited is invalid
        }
        imagesWaited_[eye] = true;
      }
      XrSwapchainImageReleaseInfo releaseInfo = xrStruct<XrSwapchainImageReleaseInfo>(XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO);
      if (!xrOk(xrReleaseSwapchainImage(xr_->swapchains[eye], &releaseInfo),
                "xrReleaseSwapchainImage")) {
        released = false;
        continue;
      }
      imagesAcquired_[eye] = false;
      imagesWaited_[eye] = false;
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
  // A projection is safe only after both draws and releases succeeded.
  if (submitProjection && xr_->frameReady && released) {
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
  xr_->frameBroken = !ok || !released;
  return ok && released;
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
    imagesWaited_[eye] = false;
  }
  for (XrSpace& aimSpace : xr_->aimSpaces) {
    if (aimSpace != XR_NULL_HANDLE) {
      xrDestroySpace(aimSpace);
      aimSpace = XR_NULL_HANDLE;
    }
  }
  for (XrSpace& gripSpace : xr_->gripSpaces) {
    if (gripSpace != XR_NULL_HANDLE) {
      xrDestroySpace(gripSpace);
      gripSpace = XR_NULL_HANDLE;
    }
  }
  for (auto tracker : xr_->trackers) {
    if (tracker != XR_NULL_HANDLE && xr_->destroyHandTracker) {
      xr_->destroyHandTracker(tracker);
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
