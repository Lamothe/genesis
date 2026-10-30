#include "vk_renderer.h"

#include "triangle_sort.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <xcb/xcb.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xcb.h>

namespace vredit {

namespace {

constexpr std::size_t kMaxLineVerts = 65536U;

bool checkVk(VkResult result, char const* what) {
  if (result != VK_SUCCESS) {
    std::printf("[vk] error %d: %s\n", static_cast<int>(result), what);
    return false;
  }
  return true;
}

#define VK_CHECK(expr)                                                                             \
  do {                                                                                             \
    if (!checkVk((expr), #expr)) {                                                                 \
      return false;                                                                                \
    }                                                                                              \
  } while (false)

// Zero-initialised create-info with sType set; keeps -Wextra clean without
// partial aggregate initialisation.
template <typename T>
T vkStruct(VkStructureType sType) {
  T value{};
  value.sType = sType;
  return value;
}

bool readBinaryFile(char const* path, std::vector<char>& out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::printf("[vk] cannot open shader %s\n", path);
    return false;
  }
  std::streamsize size = file.tellg();
  file.seekg(0);
  out.resize(static_cast<std::size_t>(size));
  return static_cast<bool>(file.read(out.data(), size));
}

// Mirrors the GLSL push_constant block exactly.
struct PushData {
  float model[16];
  float tint[4];
  float params[4];
  float normalParams[4]; // x: has normal map, y: texture index, z: scale, w: highlight
};
static_assert(sizeof(PushData) == 112, "push constants must match the shader layout");

} // namespace

// The blend sort re-runs only when the view moves past this (about 0.2
// degrees of rotation or 4 mm of translation in view terms): closer views
// give the same visible triangle order, and head-tracking jitter is smaller.
// The bound is on the matrix, so the effective depth error scales with the
// mesh's position-sum magnitudes: for huge meshes far from the origin the
// safe band is proportionally narrower.
constexpr float kSortReuseEpsilon = 0.004F;

inline bool matricesEqual(Mat4 const& a, Mat4 const& b) {
  return std::memcmp(a.m, b.m, sizeof(a.m)) == 0;
}

inline bool matricesClose(Mat4 const& a, Mat4 const& b, float epsilon) {
  for (std::size_t i = 0; i < 16U; ++i) {
    if (std::fabs(a.m[i] - b.m[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

struct VkRenderer::Impl {
  struct MeshRecord {
    VkBuffer vertexBuffer{VK_NULL_HANDLE};
    VkDeviceMemory vertexMemory{VK_NULL_HANDLE};
    VkBuffer indexBuffer{VK_NULL_HANDLE};
    VkDeviceMemory indexMemory{VK_NULL_HANDLE};
    std::uint32_t indexCount{0};
    // CPU-side source data for the per-view blend sort (see recordCommands):
    // per-triangle position sums (the depth key), the source indices (the
    // rewrite input) and the mesh-space bounds centre (the item-ordering key).
    std::vector<Vec3> triPosSums;
    std::vector<std::uint32_t> indices;
    Vec3 boundsCentre{0.0F, 0.0F, 0.0F};
  };

  // Backdrop void colour (setClearColour).
  float clearR{0.08F};
  float clearA{1.0F};
  float clearG{0.09F};
  float clearB{0.11F};

  bool windowed{false};
  bool external{false}; // caller-owned instance/device (XR path)
  std::uint32_t width{0};
  std::uint32_t height{0};
  char deviceName[256] = {0};

  VkInstance instance{VK_NULL_HANDLE};
  VkSurfaceKHR surface{VK_NULL_HANDLE};
  VkPhysicalDevice physical{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  std::uint32_t queueFamily{0};
  VkCommandPool commandPool{VK_NULL_HANDLE};
  VkCommandBuffer command{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};

  VkDescriptorSetLayout setLayout{VK_NULL_HANDLE};
  VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
  VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
  VkShaderModule vertModule{VK_NULL_HANDLE};
  VkShaderModule fragModule{VK_NULL_HANDLE};
  VkPipeline meshPipeline{VK_NULL_HANDLE};
  VkPipeline blendPipeline{VK_NULL_HANDLE};

  static constexpr std::size_t kMaxTextures = 256; // matches the shader array
  // Upload budget per texture side: past this the resample costs more VRAM
  // than detail ("run out of memory after a few models" — one 12288^2 map
  // upscaled to 16384^2 was 1.4 GB alone). 8K is ~2.8x the per-eye target.
  static constexpr std::uint32_t kUploadBudget = 8192U;
  std::size_t textureBytes{0}; // uploaded texture VRAM (mips included)

  struct TextureRecord {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};
  };
  std::vector<TextureRecord> textures; // 0 = white fallback
  VkPipeline lineDepthPipeline{VK_NULL_HANDLE};
  VkPipeline lineOverlayPipeline{VK_NULL_HANDLE};

  VkBuffer lineBuffer{VK_NULL_HANDLE};
  VkDeviceMemory lineMemory{VK_NULL_HANDLE};
  void* lineMapped{nullptr};
  VkBuffer uniformBuffer{VK_NULL_HANDLE};
  VkDeviceMemory uniformMemory{VK_NULL_HANDLE};
  void* uniformMapped{nullptr};
  VkBuffer readbackBuffer{VK_NULL_HANDLE};
  VkDeviceMemory readbackMemory{VK_NULL_HANDLE};
  void* readbackMapped{nullptr};

  VkImage depthImage{VK_NULL_HANDLE};
  VkDeviceMemory depthMemory{VK_NULL_HANDLE};
  VkImageView depthView{VK_NULL_HANDLE};

  VkImage colourImage{VK_NULL_HANDLE};
  VkDeviceMemory colourMemory{VK_NULL_HANDLE};
  VkImageView colourView{VK_NULL_HANDLE};

  VkFormat swapchainFormat{VK_FORMAT_B8G8R8A8_UNORM};
  VkSurfaceFormatKHR surfaceFormat{};
  VkFormat pipelineColourFormat{VK_FORMAT_B8G8R8A8_UNORM}; // set by createCommonResources
  VkSwapchainKHR swapchain{VK_NULL_HANDLE};
  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainViews;
  std::vector<MeshRecord> meshes; // 0 = built-in cube
  std::uint32_t swapchainIndex{0};
  VkSemaphore imageAvailable{VK_NULL_HANDLE};
  VkSemaphore renderFinished{VK_NULL_HANDLE};

  std::uint32_t findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags flags) const {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      if ((typeBits & (1U << i)) != 0U &&
          (props.memoryTypes[i].propertyFlags & flags) == flags) {
        return i;
      }
    }
    return UINT32_MAX;
  }

  // Memory queries honour VkMemoryDedicatedRequirements: large resources may
  // require or prefer a dedicated allocation, and ignoring that lands the
  // driver's suballocator in placement states where huge transfers fault the
  // channel (observed: a 12288^2 texture upload lost the device whenever
  // other buffers had been allocated first).
  struct Requirements {
    VkMemoryRequirements memory{};
    bool dedicated{false};
  };

  Requirements bufferRequirements(VkBuffer buffer) {
    Requirements out{};
    VkBufferMemoryRequirementsInfo2 info = vkStruct<VkBufferMemoryRequirementsInfo2>(
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2);
    info.buffer = buffer;
    VkMemoryDedicatedRequirements dedicated = vkStruct<VkMemoryDedicatedRequirements>(
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS);
    VkMemoryRequirements2 chain =
        vkStruct<VkMemoryRequirements2>(VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2);
    chain.pNext = &dedicated;
    vkGetBufferMemoryRequirements2(device, &info, &chain);
    out.memory = chain.memoryRequirements;
    out.dedicated = dedicated.prefersDedicatedAllocation == VK_TRUE ||
                    dedicated.requiresDedicatedAllocation == VK_TRUE;
    return out;
  }

  Requirements imageRequirements(VkImage image) {
    Requirements out{};
    VkImageMemoryRequirementsInfo2 info = vkStruct<VkImageMemoryRequirementsInfo2>(
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2);
    info.image = image;
    VkMemoryDedicatedRequirements dedicated = vkStruct<VkMemoryDedicatedRequirements>(
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS);
    VkMemoryRequirements2 chain =
        vkStruct<VkMemoryRequirements2>(VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2);
    chain.pNext = &dedicated;
    vkGetImageMemoryRequirements2(device, &info, &chain);
    out.memory = chain.memoryRequirements;
    out.dedicated = dedicated.prefersDedicatedAllocation == VK_TRUE ||
                    dedicated.requiresDedicatedAllocation == VK_TRUE;
    return out;
  }

  bool createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memFlags,
                    VkBuffer& buffer,
                    VkDeviceMemory& memory,
                    void** mapped) {
    VkBufferCreateInfo bufferInfo = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));
    Requirements reqs = bufferRequirements(buffer);
    VkMemoryRequirements requirements = reqs.memory;
    std::uint32_t memoryType = findMemoryType(requirements.memoryTypeBits, memFlags);
    if (memoryType == UINT32_MAX) {
      std::printf("[vk] no suitable memory type\n");
      vkDestroyBuffer(device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      return false;
    }
    VkMemoryDedicatedAllocateInfo dedicatedInfo =
        vkStruct<VkMemoryDedicatedAllocateInfo>(
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
    dedicatedInfo.buffer = buffer;
    VkMemoryAllocateInfo allocInfo =
        vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = memoryType;
    if (reqs.dedicated || requirements.size >= (64ULL << 20)) {
      allocInfo.pNext = &dedicatedInfo; // big resources never share a slab
    }
    VkResult allocResult = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (allocResult != VK_SUCCESS) {
      vkDestroyBuffer(device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      return checkVk(allocResult, "vkAllocateMemory");
    }
    VkResult bindResult = vkBindBufferMemory(device, buffer, memory, 0);
    if (bindResult != VK_SUCCESS) {
      vkFreeMemory(device, memory, nullptr);
      memory = VK_NULL_HANDLE;
      vkDestroyBuffer(device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      return checkVk(bindResult, "vkBindBufferMemory");
    }
    if (mapped != nullptr) {
      VkResult mapResult = vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, mapped);
      if (mapResult != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return checkVk(mapResult, "vkMapMemory");
      }
    }
    return true;
  }

  bool createImage2D(VkFormat format,
                     VkImageUsageFlags usage,
                     VkImageAspectFlags aspect,
                     VkImage& image,
                     VkDeviceMemory& memory,
                     VkImageView& view) {
    VkImageCreateInfo imageInfo = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image));
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image, &requirements);
    std::uint32_t memoryType =
        findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryType == UINT32_MAX) {
      return false;
    }
    VkMemoryAllocateInfo allocInfo =
        vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = memoryType;
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindImageMemory(device, image, memory, 0));
    VkImageViewCreateInfo viewInfo =
        vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspect;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &view));
    return true;
  }

  void destroyImage2D(VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, view, nullptr);
      view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE) {
      vkDestroyImage(device, image, nullptr);
      image = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, nullptr);
      memory = VK_NULL_HANDLE;
    }
  }

  // Picks the surface format up front so the pipelines bake the format the
  // swapchain will actually use (the two must match in dynamic rendering).
  bool chooseSurfaceFormat() {
    std::uint32_t formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, formats.data()));
    if (formats.empty()) {
      std::printf("[vk] surface reports no formats\n");
      return false;
    }
    surfaceFormat = formats[0];
    for (VkSurfaceFormatKHR const& format : formats) {
      if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
          format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        surfaceFormat = format;
        break;
      }
    }
    swapchainFormat = surfaceFormat.format;
    return true;
  }

  // Recreates the swapchain and the depth image as one consistent set. Order
  // matters: the swapchain reports the extent it actually got (configure
  // events lag the real window size during an interactive resize), and the
  // depth image must follow it — rendering with a depth image sized for a
  // different extent is invalid and has crashed the driver on resize.
  bool recreateTargets() {
    if (!createSwapchain()) {
      return false;
    }
    if (width == 0U || height == 0U) {
      // Minimised: the swapchain guard kept the old targets alive, so keep the
      // old depth too — creating a 0x0 image here would be invalid.
      return true;
    }
    destroyImage2D(depthImage, depthMemory, depthView);
    return createImage2D(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                         VK_IMAGE_ASPECT_DEPTH_BIT, depthImage, depthMemory, depthView);
  }

  bool createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps));

    VkSurfaceFormatKHR chosen = surfaceFormat;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
      extent.width = width;
      extent.height = height;
    }
    if (extent.width == 0U || extent.height == 0U) {
      // Minimised / zero-extent surface: nothing can be created. Keep the old
      // targets; draw() skips rendering while width/height stay zero.
      width = 0;
      height = 0;
      return true;
    }

    std::uint32_t imageCount = caps.minImageCount + 1U;
    if (caps.maxImageCount != 0U && imageCount > caps.maxImageCount) {
      imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo =
        vkStruct<VkSwapchainCreateInfoKHR>(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = chosen.format;
    createInfo.imageColorSpace = chosen.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = swapchain;
    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapchain));
    destroySwapchain();
    swapchain = newSwapchain;

    std::uint32_t actualCount = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &actualCount, nullptr));
    swapchainImages.resize(actualCount);
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &actualCount, swapchainImages.data()));
    swapchainViews.resize(actualCount);
    for (std::uint32_t i = 0; i < actualCount; ++i) {
      VkImageViewCreateInfo viewInfo =
          vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
      viewInfo.image = swapchainImages[i];
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = chosen.format;
      viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.subresourceRange.levelCount = 1;
      viewInfo.subresourceRange.layerCount = 1;
      if (vkCreateImageView(device, &viewInfo, nullptr, &swapchainViews[i]) != VK_SUCCESS) {
        // A half-filled view array must never reach draw(): retire the whole
        // swapchain so the next frame fails cleanly instead of crashing.
        std::printf("[vk] swapchain view creation failed\n");
        destroySwapchain();
        return false;
      }
    }
    // Commit the extent only after full success so renderArea/viewport always
    // match the live attachments.
    width = extent.width;
    height = extent.height;
    return true;
  }

  void destroySwapchain() {
    for (VkImageView view : swapchainViews) {
      vkDestroyImageView(device, view, nullptr);
    }
    swapchainViews.clear();
    swapchainImages.clear();
    if (swapchain != VK_NULL_HANDLE) {
      vkDestroySwapchainKHR(device, swapchain, nullptr);
      swapchain = VK_NULL_HANDLE;
    }
  }

  bool createPipeline(bool depthTest, bool depthWrite, VkPrimitiveTopology topology,
                      bool alphaBlend, VkPipeline& pipeline) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = vkStruct<VkPipelineShaderStageCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1] = vkStruct<VkPipelineShaderStageCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(RendererVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attributes[4] = {};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RendererVertex, pos)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RendererVertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RendererVertex, colour)};
    attributes[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(RendererVertex, uv)};
    VkPipelineVertexInputStateCreateInfo vertexInput =
        vkStruct<VkPipelineVertexInputStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly =
        vkStruct<VkPipelineInputAssemblyStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
    inputAssembly.topology = topology;

    VkPipelineViewportStateCreateInfo viewportState =
        vkStruct<VkPipelineViewportStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster =
        vkStruct<VkPipelineRasterizationStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0F;

    VkPipelineMultisampleStateCreateInfo multisample =
        vkStruct<VkPipelineMultisampleStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil =
        vkStruct<VkPipelineDepthStencilStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO);
    depthStencil.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (alphaBlend) {
      blendAttachment.blendEnable = VK_TRUE;
      blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
      blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo blend =
        vkStruct<VkPipelineColorBlendStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic =
        vkStruct<VkPipelineDynamicStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO);
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkFormat colourFormat = pipelineColourFormat;
    VkPipelineRenderingCreateInfo rendering =
        vkStruct<VkPipelineRenderingCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &colourFormat;
    rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo createInfo =
        vkStruct<VkGraphicsPipelineCreateInfo>(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
    createInfo.pNext = &rendering;
    createInfo.stageCount = 2;
    createInfo.pStages = stages;
    createInfo.pVertexInputState = &vertexInput;
    createInfo.pInputAssemblyState = &inputAssembly;
    createInfo.pViewportState = &viewportState;
    createInfo.pRasterizationState = &raster;
    createInfo.pMultisampleState = &multisample;
    createInfo.pDepthStencilState = &depthStencil;
    createInfo.pColorBlendState = &blend;
    createInfo.pDynamicState = &dynamic;
    createInfo.layout = pipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline));
    return true;
  }

  // Creates command/fence, shaders, descriptors, depth image and pipelines on
  // the already-created device. Shared by init() and initExternal().

  enum class Target { Window, Readback, External };

  bool recordCommands(std::size_t depthLineCount,
                      std::size_t overlayLineCount,
                      VkImageView colourViewUsed,
                      VkImage colourImageUsed,
                      Target target,
                      Mat4 const& sortView,
                      Vec3 const& sortCamPos) {
    VkCommandBufferBeginInfo beginInfo =
        vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(command, &beginInfo));

    VkImageMemoryBarrier2 colourBarrier =
        vkStruct<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
    colourBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    colourBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colourBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colourBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colourBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colourBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colourBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colourBarrier.image = colourImageUsed;
    colourBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colourBarrier.subresourceRange.levelCount = 1;
    colourBarrier.subresourceRange.layerCount = 1;

    VkImageMemoryBarrier2 depthBarrier = colourBarrier;
    depthBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthBarrier.image = depthImage;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkDependencyInfo dependency =
        vkStruct<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
    VkImageMemoryBarrier2 barriers[2] = {colourBarrier, depthBarrier};
    dependency.imageMemoryBarrierCount = 2;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command, &dependency);

    VkRenderingAttachmentInfo colourAttachment =
        vkStruct<VkRenderingAttachmentInfo>(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
    colourAttachment.imageView = colourViewUsed;
    colourAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colourAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colourAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colourAttachment.clearValue.color = {{clearR, clearG, clearB, clearA}};

    VkRenderingAttachmentInfo depthAttachment =
        vkStruct<VkRenderingAttachmentInfo>(VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
    depthAttachment.imageView = depthView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0F, 0};

    VkRenderingInfo rendering = vkStruct<VkRenderingInfo>(VK_STRUCTURE_TYPE_RENDERING_INFO);
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = {width, height};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &colourAttachment;
    rendering.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(command, &rendering);

    VkViewport viewport{0.0F, 0.0F, static_cast<float>(width), static_cast<float>(height), 0.0F,
                        1.0F};
    VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptorSet, 0, nullptr);

    // Draw order: opaque/mask items keep the app's order; blend items draw
    // farthest-centre-first with a stable tie-break — the Centre Stage rule
    // ("hair shells painted over strands unsorted"). Per view, so both eyes
    // sort for their own camera. Within a blend item the same rule applies
    // per triangle (see sortTrianglesBackToFront).
    std::vector<std::size_t> order(items.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::vector<float> centreDist2(items.size(), 0.0F);
    for (std::size_t i = 0; i < items.size(); ++i) {
      DrawItem const& item = items[i];
      if (item.alpha != AlphaMode::Blend) {
        continue;
      }
      std::size_t const centreHandle = item.mesh < meshes.size() ? item.mesh : 0U;
      Vec3 const b = meshes[centreHandle].boundsCentre;
      Vec3 const c{item.model.m[0] * b.x + item.model.m[4] * b.y + item.model.m[8] * b.z +
                       item.model.m[12],
                   item.model.m[1] * b.x + item.model.m[5] * b.y + item.model.m[9] * b.z +
                       item.model.m[13],
                   item.model.m[2] * b.x + item.model.m[6] * b.y + item.model.m[10] * b.z +
                       item.model.m[14]};
      Vec3 const d = c - sortCamPos;
      centreDist2[i] = d.x * d.x + d.y * d.y + d.z * d.z;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      bool const aBlend = items[a].alpha == AlphaMode::Blend;
      bool const bBlend = items[b].alpha == AlphaMode::Blend;
      if (aBlend != bBlend) {
        return !aBlend; // non-blend first
      }
      if (!aBlend) {
        return false; // non-blend items keep the app's order
      }
      return centreDist2[a] > centreDist2[b]; // blend: farthest centre first
    });

    // Per-triangle back-to-front rewrite of each blend item's indices, uploaded
    // through one transient staging buffer into per-blend-ordinal DEVICE_LOCAL
    // slots before any blend draw (one transfer->index barrier covers the
    // batch).
    std::size_t blendCount = 0;
    VkDeviceSize stagingBytes = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (items[i].alpha != AlphaMode::Blend) {
        continue;
      }
      ++blendCount;
      std::size_t const countHandle = items[i].mesh < meshes.size() ? items[i].mesh : 0U;
      stagingBytes += sizeof(std::uint32_t) * meshes[countHandle].indices.size();
    }
    if (blendScratch.size() < blendCount) {
      blendScratch.resize(blendCount);
    }
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void* stagingMapped = nullptr;
    bool sortActive = blendCount > 0 && stagingBytes > 0;
    if (sortActive) {
      sortActive = createBuffer(stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                stagingBuffer, stagingMemory, &stagingMapped);
    }
    VkDeviceSize stagingOffset = 0;
    std::size_t blendSlot = 0;
    if (sortActive) {
      auto* staging = static_cast<std::uint8_t*>(stagingMapped);
      for (std::size_t oi = 0; oi < order.size() && sortActive; ++oi) {
        DrawItem const& item = items[order[oi]];
        if (item.alpha != AlphaMode::Blend) {
          continue;
        }
        std::size_t const handle = item.mesh < meshes.size() ? item.mesh : 0U;
        MeshRecord const& record = meshes[handle];
        BlendScratch& slot = blendScratch[blendSlot];
        VkDeviceSize const bytes = sizeof(std::uint32_t) * record.indices.size();
        std::size_t const triCount = record.indices.size() / 3U;
        if (record.triPosSums.empty() || record.indices.empty() ||
            record.indices.size() % 3U != 0U || record.triPosSums.size() != triCount) {
          sortActive = false; // unsortable: draws fall back to index order
          break;
        }
        // Triangle depth sums through the draw's exact chain (view * model):
        // the sum of a triangle's three view-space depths equals row 2 of the
        // composed matrix dotted with the precomputed triangle position sum,
        // plus three times the row's translation (column-major).
        Mat4 const vm = mul(sortView, item.model);
        bool const reuse = slot.hasSort && slot.indexBuffer != VK_NULL_HANDLE &&
                           slot.capacity >= bytes && slot.lastMesh == static_cast<MeshHandle>(handle) &&
                           matricesEqual(slot.lastModel, item.model) &&
                           matricesClose(slot.lastView, vm, kSortReuseEpsilon);
        if (reuse) {
          // Head jitter: the previous order still applies and the scratch
          // already holds it on the GPU — nothing to recompute or re-upload.
          ++blendSlot;
          continue;
        }
        if (slot.capacity < bytes) {
          freeLater(slot.indexBuffer, slot.indexMemory);
          slot.indexBuffer = VK_NULL_HANDLE;
          slot.indexMemory = VK_NULL_HANDLE;
          slot.capacity = 0;
          slot.hasSort = false; // the buffer its state described is gone
          if (!createBuffer(bytes,
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, slot.indexBuffer,
                            slot.indexMemory, nullptr)) {
            sortActive = false;
            break;
          }
          slot.capacity = bytes;
        }
        blendDepths.resize(triCount);
        float const rx = vm.m[2];
        float const ry = vm.m[6];
        float const rz = vm.m[10];
        float const rw = 3.0F * vm.m[14];
        for (std::size_t t = 0; t < triCount; ++t) {
          Vec3 const& sum = record.triPosSums[t];
          blendDepths[t] = rx * sum.x + ry * sum.y + rz * sum.z + rw;
        }
        blendSorted.resize(record.indices.size());
        blendKeys.resize(2U * triCount);
        sortTrianglesBackToFrontSums(blendDepths.data(), record.indices.data(),
                                     static_cast<std::uint32_t>(triCount), blendSorted.data(),
                                     blendKeys.data());
        std::memcpy(staging + stagingOffset, blendSorted.data(),
                    static_cast<std::size_t>(bytes));
        VkBufferCopy copy{stagingOffset, 0, bytes};
        vkCmdCopyBuffer(command, stagingBuffer, slot.indexBuffer, 1, &copy);
        stagingOffset += bytes;
        slot.hasSort = true;
        slot.lastMesh = static_cast<MeshHandle>(handle);
        slot.lastModel = item.model;
        slot.lastView = vm;
        ++blendSlot;
      }
    }
    if (stagingOffset > 0) {
      // Recorded copies must become visible to the index reads of the draws
      // below (and of any later deadband reuse) — regardless of whether the
      // draws this frame end up reading the sorted buffers or falling back.
      VkMemoryBarrier2 toIndex = vkStruct<VkMemoryBarrier2>(VK_STRUCTURE_TYPE_MEMORY_BARRIER_2);
      toIndex.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      toIndex.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      toIndex.dstStageMask = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
      toIndex.dstAccessMask = VK_ACCESS_2_INDEX_READ_BIT;
      VkDependencyInfo sortDependency =
          vkStruct<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
      sortDependency.memoryBarrierCount = 1;
      sortDependency.pMemoryBarriers = &toIndex;
      vkCmdPipelineBarrier2(command, &sortDependency);
    }
    if (stagingBuffer != VK_NULL_HANDLE) {
      freeLater(stagingBuffer, stagingMemory); // read this frame; freed after the fence wait
    }

    // Meshes: one indexed draw per item in the computed order (opaque+mask
    // first, blends last; blend draws read their per-view sorted indices).
    std::size_t drawBlendSlot = 0;
    for (std::size_t oi = 0; oi < order.size(); ++oi) {
      DrawItem const& item = items[order[oi]];
      std::size_t handle = item.mesh < meshes.size() ? item.mesh : 0U;
      MeshRecord const& record = meshes[handle];
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        item.alpha == AlphaMode::Blend ? blendPipeline : meshPipeline);
      VkDeviceSize zero = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &record.vertexBuffer, &zero);
      VkBuffer indexUsed = record.indexBuffer;
      if (sortActive && item.alpha == AlphaMode::Blend) {
        indexUsed = blendScratch[drawBlendSlot++].indexBuffer;
      }
      vkCmdBindIndexBuffer(command, indexUsed, 0, VK_INDEX_TYPE_UINT32);
      PushData push{};
      std::memcpy(push.model, item.model.m, sizeof(float) * 16U);
      push.tint[0] = item.colour.x;
      push.tint[1] = item.colour.y;
      push.tint[2] = item.colour.z;
      push.tint[3] = item.alphaFactor;
      push.params[0] = item.lit ? 1.0F : 0.0F;
      push.params[1] = item.texture < textures.size() ? static_cast<float>(item.texture) : 0.0F;
      push.params[2] = item.alphaCutoff;
      push.params[3] = item.alpha == AlphaMode::Mask    ? 1.0F
                       : item.alpha == AlphaMode::Blend ? 2.0F
                                                        : 0.0F;
      push.normalParams[0] =
          (item.normalMapped && item.normalTexture < textures.size()) ? 1.0F : 0.0F;
      push.normalParams[1] =
          item.normalTexture < textures.size() ? static_cast<float>(item.normalTexture) : 0.0F;
      push.normalParams[2] = item.normalScale;
      push.normalParams[3] = item.highlight;
      vkCmdPushConstants(command, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushData), &push);
      vkCmdDrawIndexed(command, record.indexCount, 1, 0, 0, 0);
    }

    VkDeviceSize stride = sizeof(RendererVertex);
    // Line passes always push their own identity transform: overlay lines must
    // never inherit the last mesh item's model/tint/highlight.
    PushData linePush{};
    linePush.model[0] = 1.0F;
    linePush.model[5] = 1.0F;
    linePush.model[10] = 1.0F;
    linePush.model[15] = 1.0F;
    linePush.tint[0] = 1.0F;
    linePush.tint[1] = 1.0F;
    linePush.tint[2] = 1.0F;
    linePush.tint[3] = 1.0F;
    if (depthLineCount > 0U) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, lineDepthPipeline);
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &lineBuffer, &offset);
      vkCmdPushConstants(command, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushData), &linePush);
      vkCmdDraw(command, static_cast<std::uint32_t>(depthLineCount), 1, 0, 0);
    }
    if (overlayLineCount > 0U) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, lineOverlayPipeline);
      VkDeviceSize offset = stride * depthLineCount;
      vkCmdBindVertexBuffers(command, 0, 1, &lineBuffer, &offset);
      vkCmdPushConstants(command, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushData), &linePush);
      vkCmdDraw(command, static_cast<std::uint32_t>(overlayLineCount), 1, 0, 0);
    }

    vkCmdEndRendering(command);

    VkImageMemoryBarrier2 afterBarrier = colourBarrier;
    afterBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    afterBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    afterBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (target == Target::Window) {
      afterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
      afterBarrier.dstAccessMask = VK_ACCESS_2_NONE;
      afterBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    } else if (target == Target::Readback) {
      afterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      afterBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      afterBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    } else {
      // XR swapchain images are released to the runtime in
      // COLOR_ATTACHMENT_OPTIMAL; writes just need to be visible.
      afterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
      afterBarrier.dstAccessMask = VK_ACCESS_2_NONE;
      afterBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &afterBarrier;
    vkCmdPipelineBarrier2(command, &dependency);

    if (target == Target::Readback) {
      VkBufferImageCopy copy{};
      copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      copy.imageSubresource.layerCount = 1;
      copy.imageExtent = {width, height, 1};
      vkCmdCopyImageToBuffer(command, colourImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readbackBuffer, 1, &copy);
    }

    if (!checkVk(vkEndCommandBuffer(command), "vkEndCommandBuffer")) {
      // The recorded copies may never execute: the slot state must not claim
      // sorted contents a later deadband reuse could bind (see the submit).
      for (BlendScratch& slot : blendScratch) {
        slot.hasSort = false;
      }
      return false;
    }

    VkSubmitInfo submit = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (target == Target::Window) {
      submit.waitSemaphoreCount = 1;
      submit.pWaitSemaphores = &imageAvailable;
      submit.pWaitDstStageMask = &waitStage;
      submit.signalSemaphoreCount = 1;
      submit.pSignalSemaphores = &renderFinished;
    }
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    // Reset immediately before submit: any earlier failure leaves the fence
    // signalled, so a failed or skipped frame can never deadlock the next one.
    if (!checkVk(vkResetFences(device, 1, &fence), "vkResetFences")) {
      for (BlendScratch& slot : blendScratch) {
        slot.hasSort = false;
      }
      return false;
    }
    VkResult submitResult = vkQueueSubmit(queue, 1, &submit, fence);
    if (submitResult != VK_SUCCESS) {
      // A failed submit leaves the fence unsignalled forever: recreate it
      // signalled so the next frame's wait cannot deadlock.
      vkDestroyFence(device, fence, nullptr);
      VkFenceCreateInfo freshInfo = vkStruct<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
      freshInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      fence = VK_NULL_HANDLE;
      vkCreateFence(device, &freshInfo, nullptr, &fence);
      // The copies may not have executed: never leave slots claiming sorted
      // contents — a later deadband reuse would bind stale indices.
      for (BlendScratch& slot : blendScratch) {
        slot.hasSort = false;
      }
      return checkVk(submitResult, "vkQueueSubmit");
    }
    return true;
  }

  std::vector<DrawItem> items; // set by draw() before recordCommands()

  // Per-view blend sort scratch (see recordCommands). Sorted indices upload
  // through a transient staging buffer into DEVICE_LOCAL slots — persistent
  // HOST_VISIBLE mesh memory triggers the NPOT texture driver bug (see
  // addMesh). One slot per blend item of the current draw, reused only after
  // the entry fence wait, so nothing in flight is ever rewritten.
  struct BlendScratch {
    VkBuffer indexBuffer{VK_NULL_HANDLE};
    VkDeviceMemory indexMemory{VK_NULL_HANDLE};
    VkDeviceSize capacity{0};
    // What the buffer currently holds: the sorted indices of `lastMesh` under
    // `lastModel` * `lastView`. Reused unchanged when the new view is within
    // kSortReuseEpsilon (head jitter), so a still frame costs nothing.
    MeshHandle lastMesh{UINT32_MAX};
    Mat4 lastModel{};
    Mat4 lastView{};
    bool hasSort{false};
  };
  std::vector<BlendScratch> blendScratch;
  std::vector<std::pair<VkBuffer, VkDeviceMemory>> pendingFree; // flushed after the fence wait
  std::vector<float> blendDepths;         // CPU scratch: per-triangle depth sums
  std::vector<std::uint64_t> blendKeys;   // CPU scratch: sort keys
  std::vector<std::uint32_t> blendSorted; // CPU scratch: sorted indices

  void freeLater(VkBuffer buffer, VkDeviceMemory memory) {
    if (buffer != VK_NULL_HANDLE || memory != VK_NULL_HANDLE) {
      pendingFree.emplace_back(buffer, memory);
    }
  }

  void flushFrees() {
    for (auto const& [buffer, memory] : pendingFree) {
      if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
      }
      if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
      }
    }
    pendingFree.clear();
  }
};

VkRenderer::VkRenderer() : impl_(std::make_unique<Impl>()) {}

VkRenderer::~VkRenderer() {
  shutdown();
}

bool VkRenderer::init(Config const& config) {
  Impl& g = *impl_;
  g.windowed = config.windowed;
  g.width = config.width;
  g.height = config.height;

  std::vector<char const*> instanceExtensions;
  if (g.windowed) {
    instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    instanceExtensions.push_back("VK_KHR_xcb_surface");
  }
  std::vector<char const*> layers;
  if (std::getenv("VREDITOR_VALIDATE") != nullptr) {
    std::uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layerProps(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layerProps.data());
    for (VkLayerProperties const& prop : layerProps) {
      if (std::strcmp(prop.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
      }
    }
  }

  VkApplicationInfo appInfo =
      vkStruct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
  appInfo.pApplicationName = "vr-editor";
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo instanceInfo =
      vkStruct<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  instanceInfo.pApplicationInfo = &appInfo;
  instanceInfo.enabledExtensionCount = static_cast<std::uint32_t>(instanceExtensions.size());
  instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
  instanceInfo.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
  instanceInfo.ppEnabledLayerNames = layers.data();
  VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &g.instance));

  if (g.windowed) {
    VkXcbSurfaceCreateInfoKHR surfaceInfo =
        vkStruct<VkXcbSurfaceCreateInfoKHR>(VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR);
    surfaceInfo.connection = static_cast<xcb_connection_t*>(config.xcbConnection);
    surfaceInfo.window = config.xcbWindow;
    VK_CHECK(vkCreateXcbSurfaceKHR(g.instance, &surfaceInfo, nullptr, &g.surface));
  }

  std::uint32_t deviceCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(g.instance, &deviceCount, nullptr));
  if (deviceCount == 0U) {
    std::printf("[vk] no Vulkan devices\n");
    return false;
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(g.instance, &deviceCount, devices.data()));
  VkPhysicalDevice chosen = VK_NULL_HANDLE;
  std::uint32_t chosenFamily = 0;
  bool chosenIsDiscrete = false;
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) {
      continue;
    }
    std::uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());
    for (std::uint32_t f = 0; f < familyCount; ++f) {
      if ((families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U) {
        continue;
      }
      if (g.windowed) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, f, g.surface, &present);
        if (present == VK_FALSE) {
          continue;
        }
      }
      bool discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
      if (chosen == VK_NULL_HANDLE || (discrete && !chosenIsDiscrete)) {
        chosen = device;
        chosenFamily = f;
        chosenIsDiscrete = discrete;
      }
      break;
    }
  }
  if (chosen == VK_NULL_HANDLE) {
    std::printf("[vk] no suitable Vulkan 1.3 device\n");
    return false;
  }
  g.physical = chosen;
  g.queueFamily = chosenFamily;
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g.physical, &props);
    std::snprintf(g.deviceName, sizeof(g.deviceName), "%s", props.deviceName);
    // The shader's texture array is fixed at kMaxTextures; a device below
    // that cannot create the descriptor layout (the spec minimum is far
    // lower). Fail loudly at init instead of failing mysteriously later.
    if (props.limits.maxPerStageDescriptorSampledImages <
        static_cast<std::uint32_t>(Impl::kMaxTextures)) {
      std::printf("[vk] %s supports %u sampled images per stage, need %zu\n",
                  props.deviceName, props.limits.maxPerStageDescriptorSampledImages,
                  Impl::kMaxTextures);
      return false;
    }
  }

  float queuePriority = 1.0F;
  VkDeviceQueueCreateInfo queueInfo =
      vkStruct<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
  queueInfo.queueFamilyIndex = g.queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceVulkan13Features features13 =
      vkStruct<VkPhysicalDeviceVulkan13Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;

  std::vector<char const*> deviceExtensions;
  if (g.windowed) {
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  VkDeviceCreateInfo deviceInfo =
      vkStruct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
  deviceInfo.pNext = &features13;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
  deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
  VK_CHECK(vkCreateDevice(g.physical, &deviceInfo, nullptr, &g.device));
  vkGetDeviceQueue(g.device, g.queueFamily, 0, &g.queue);

  if (g.windowed && !g.chooseSurfaceFormat()) {
    return false;
  }
  if (!createCommonResources(config.shaderDir,
                             g.windowed ? g.swapchainFormat : VK_FORMAT_R8G8B8A8_UNORM)) {
    return false;
  }
  if (g.windowed) {
    VkSemaphoreCreateInfo semaphoreInfo =
        vkStruct<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    VK_CHECK(vkCreateSemaphore(g.device, &semaphoreInfo, nullptr, &g.imageAvailable));
    VK_CHECK(vkCreateSemaphore(g.device, &semaphoreInfo, nullptr, &g.renderFinished));
    if (!g.recreateTargets()) {
      return false;
    }
  } else {
    if (!g.createImage2D(VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, g.colourImage, g.colourMemory,
                         g.colourView)) {
      return false;
    }
    if (!g.createBuffer(static_cast<VkDeviceSize>(g.width) * g.height * 4U,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        g.readbackBuffer, g.readbackMemory, &g.readbackMapped)) {
      return false;
    }
  }
  std::printf("[vk] device: %s (%s)\n", g.deviceName, g.windowed ? "windowed" : "offscreen");
  return true;
}

bool VkRenderer::createCommonResources(char const* shaderDir, VkFormat colourFormat) {
  Impl& g = *impl_;
  g.pipelineColourFormat = colourFormat;
  VkCommandPoolCreateInfo poolInfo =
      vkStruct<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = g.queueFamily;
  VK_CHECK(vkCreateCommandPool(g.device, &poolInfo, nullptr, &g.commandPool));
  VkCommandBufferAllocateInfo commandInfo =
      vkStruct<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
  commandInfo.commandPool = g.commandPool;
  commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandInfo.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(g.device, &commandInfo, &g.command));
  VkFenceCreateInfo fenceInfo = vkStruct<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VK_CHECK(vkCreateFence(g.device, &fenceInfo, nullptr, &g.fence));

  std::vector<char> vertCode;
  std::vector<char> fragCode;
  std::string vertPath = std::string(shaderDir) + "/mesh.vert.spv";
  std::string fragPath = std::string(shaderDir) + "/mesh.frag.spv";
  if (!readBinaryFile(vertPath.c_str(), vertCode) || !readBinaryFile(fragPath.c_str(), fragCode)) {
    return false;
  }
  VkShaderModuleCreateInfo moduleInfo =
      vkStruct<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
  moduleInfo.codeSize = vertCode.size();
  moduleInfo.pCode = reinterpret_cast<std::uint32_t const*>(vertCode.data());
  VK_CHECK(vkCreateShaderModule(g.device, &moduleInfo, nullptr, &g.vertModule));
  moduleInfo.codeSize = fragCode.size();
  moduleInfo.pCode = reinterpret_cast<std::uint32_t const*>(fragCode.data());
  VK_CHECK(vkCreateShaderModule(g.device, &moduleInfo, nullptr, &g.fragModule));

  VkDescriptorSetLayoutBinding bindings[2] = {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[0].descriptorCount = 1;
  // FRAGMENT too: the fragment stage reads frame.cameraPos for the
  // selection rim — binding it vertex-only made that read a spec violation.
  bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[1].descriptorCount = static_cast<std::uint32_t>(Impl::kMaxTextures);
  bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo setLayoutInfo =
      vkStruct<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
  setLayoutInfo.bindingCount = 2;
  setLayoutInfo.pBindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(g.device, &setLayoutInfo, nullptr, &g.setLayout));

  VkPushConstantRange pushRange{};
  pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(PushData);
  VkPipelineLayoutCreateInfo layoutInfo =
      vkStruct<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &g.setLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushRange;
  VK_CHECK(vkCreatePipelineLayout(g.device, &layoutInfo, nullptr, &g.pipelineLayout));

  VkDescriptorPoolSize poolSizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<std::uint32_t>(Impl::kMaxTextures)}};
  VkDescriptorPoolCreateInfo poolCreateInfo =
      vkStruct<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
  poolCreateInfo.maxSets = 1;
  poolCreateInfo.poolSizeCount = 2;
  poolCreateInfo.pPoolSizes = poolSizes;
  VK_CHECK(vkCreateDescriptorPool(g.device, &poolCreateInfo, nullptr, &g.descriptorPool));
  VkDescriptorSetAllocateInfo setAllocInfo =
      vkStruct<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
  setAllocInfo.descriptorPool = g.descriptorPool;
  setAllocInfo.descriptorSetCount = 1;
  setAllocInfo.pSetLayouts = &g.setLayout;
  VK_CHECK(vkAllocateDescriptorSets(g.device, &setAllocInfo, &g.descriptorSet));

  std::vector<MeshVertex> cubeVerts;
  appendCube(cubeVerts);
  MeshData cubeMesh;
  cubeMesh.vertices = cubeVerts;
  cubeMesh.indices.resize(cubeVerts.size());
  for (std::size_t i = 0; i < cubeMesh.indices.size(); ++i) {
    cubeMesh.indices[i] = static_cast<std::uint32_t>(i);
  }
  if (addMesh(cubeMesh) != 0U || g.meshes.size() != 1U) {
    std::printf("[vk] built-in cube must register as mesh 0\n");
    return false;
  }
  if (!g.createBuffer(sizeof(RendererVertex) * kMaxLineVerts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      g.lineBuffer, g.lineMemory, &g.lineMapped)) {
    return false;
  }
  if (!g.createBuffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      g.uniformBuffer, g.uniformMemory, &g.uniformMapped)) {
    return false;
  }

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = g.uniformBuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = 80; // the Frame block: viewProj (64) + cameraPos (16)

  // Texture 0 is the built-in white fallback; every slot starts on it.
  Image white;
  white.width = 1;
  white.height = 1;
  white.rgba = {255U, 255U, 255U, 255U};
  if (addTexture(white, true, true) != 0U || g.textures.size() != 1U) {
    std::printf("[vk] white fallback texture must register as texture 0\n");
    return false;
  }
  std::vector<VkDescriptorImageInfo> imageInfos(Impl::kMaxTextures);
  for (VkDescriptorImageInfo& info : imageInfos) {
    info.sampler = g.textures[0].sampler;
    info.imageView = g.textures[0].view;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }

  VkWriteDescriptorSet writes[2] = {};
  writes[0] = vkStruct<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
  writes[0].dstSet = g.descriptorSet;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[0].pBufferInfo = &bufferInfo;
  writes[1] = vkStruct<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
  writes[1].dstSet = g.descriptorSet;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = static_cast<std::uint32_t>(imageInfos.size());
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].pImageInfo = imageInfos.data();
  vkUpdateDescriptorSets(g.device, 2, writes, 0, nullptr);

  if (!g.createImage2D(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT, g.depthImage, g.depthMemory, g.depthView)) {
    return false;
  }
  if (!g.createPipeline(true, true, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, g.meshPipeline) ||
      !g.createPipeline(true, false, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, g.blendPipeline) ||
      !g.createPipeline(true, false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, g.lineDepthPipeline) ||
      !g.createPipeline(false, false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false,
                        g.lineOverlayPipeline)) {
    return false;
  }
  return true;
}

void VkRenderer::shutdown() {
  if (!impl_) {
    return;
  }
  Impl& g = *impl_;
  if (g.device == VK_NULL_HANDLE) {
    return;
  }
  vkDeviceWaitIdle(g.device);
  g.flushFrees();
  for (Impl::BlendScratch& slot : g.blendScratch) {
    if (slot.indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, slot.indexBuffer, nullptr);
    }
    if (slot.indexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, slot.indexMemory, nullptr);
    }
  }
  g.blendScratch.clear();
  for (VkPipeline pipeline :
       {g.meshPipeline, g.blendPipeline, g.lineDepthPipeline, g.lineOverlayPipeline}) {
    if (pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(g.device, pipeline, nullptr);
    }
  }
  for (Impl::TextureRecord& record : g.textures) {
    if (record.sampler != VK_NULL_HANDLE) {
      vkDestroySampler(g.device, record.sampler, nullptr);
    }
    if (record.view != VK_NULL_HANDLE) {
      vkDestroyImageView(g.device, record.view, nullptr);
    }
    if (record.image != VK_NULL_HANDLE) {
      vkDestroyImage(g.device, record.image, nullptr);
    }
    if (record.memory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, record.memory, nullptr);
    }
  }
  g.textures.clear();
  if (g.pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(g.device, g.pipelineLayout, nullptr);
  }
  if (g.descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(g.device, g.descriptorPool, nullptr);
  }
  if (g.setLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(g.device, g.setLayout, nullptr);
  }
  if (g.vertModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(g.device, g.vertModule, nullptr);
  }
  if (g.fragModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(g.device, g.fragModule, nullptr);
  }
  g.destroySwapchain();
  g.destroyImage2D(g.depthImage, g.depthMemory, g.depthView);
  g.destroyImage2D(g.colourImage, g.colourMemory, g.colourView);
  for (Impl::MeshRecord& record : g.meshes) {
    if (record.vertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, record.vertexBuffer, nullptr);
    }
    if (record.vertexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, record.vertexMemory, nullptr);
    }
    if (record.indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, record.indexBuffer, nullptr);
    }
    if (record.indexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, record.indexMemory, nullptr);
    }
  }
  g.meshes.clear();
  for (auto [buffer, memory] : {std::pair<VkBuffer, VkDeviceMemory>{g.lineBuffer, g.lineMemory},
                                std::pair<VkBuffer, VkDeviceMemory>{g.uniformBuffer, g.uniformMemory},
                                std::pair<VkBuffer, VkDeviceMemory>{g.readbackBuffer,
                                                                   g.readbackMemory}}) {
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, memory, nullptr);
    }
  }
  if (g.imageAvailable != VK_NULL_HANDLE) {
    vkDestroySemaphore(g.device, g.imageAvailable, nullptr);
  }
  if (g.renderFinished != VK_NULL_HANDLE) {
    vkDestroySemaphore(g.device, g.renderFinished, nullptr);
  }
  if (g.fence != VK_NULL_HANDLE) {
    vkDestroyFence(g.device, g.fence, nullptr);
  }
  if (g.commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(g.device, g.commandPool, nullptr);
  }
  if (!g.external) {
    vkDestroyDevice(g.device, nullptr);
  }
  g.device = VK_NULL_HANDLE; // external: the caller owns the device teardown
  if (g.surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
    g.surface = VK_NULL_HANDLE;
  }
  if (g.instance != VK_NULL_HANDLE && !g.external) {
    vkDestroyInstance(g.instance, nullptr);
    g.instance = VK_NULL_HANDLE;
  }
}

MeshHandle VkRenderer::addMesh(MeshData const& data) {
  Impl& g = *impl_;
  if (g.device == VK_NULL_HANDLE || data.vertices.empty() || data.indices.empty() ||
      data.indices.size() > UINT32_MAX || data.vertices.size() > UINT32_MAX) {
    return kInvalidMesh;
  }
  vkDeviceWaitIdle(g.device); // buffers may be rewritten between frames

  Impl::MeshRecord record{};
  // CPU-side data for the per-view blend sort, built first so a mesh with
  // out-of-range indices is rejected before any GPU allocation: per-triangle
  // position sums (the depth key), the source indices (the rewrite input) and
  // the mesh-space bounds centre (the item-ordering key).
  Vec3 bmin = data.vertices.front().pos;
  Vec3 bmax = bmin;
  for (MeshVertex const& vertex : data.vertices) {
    bmin.x = std::min(bmin.x, vertex.pos.x);
    bmin.y = std::min(bmin.y, vertex.pos.y);
    bmin.z = std::min(bmin.z, vertex.pos.z);
    bmax.x = std::max(bmax.x, vertex.pos.x);
    bmax.y = std::max(bmax.y, vertex.pos.y);
    bmax.z = std::max(bmax.z, vertex.pos.z);
  }
  record.boundsCentre = (bmin + bmax) * 0.5F;
  record.indices = data.indices;
  // Validate every index (including any trailing partial triangle) before the
  // CPU mirror reads through them.
  for (std::uint32_t const index : data.indices) {
    if (index >= data.vertices.size()) {
      std::printf("[vk] mesh index out of range\n");
      return kInvalidMesh;
    }
  }
  record.triPosSums.resize(data.indices.size() / 3U);
  for (std::size_t t = 0; t < record.triPosSums.size(); ++t) {
    Vec3 sum{0.0F, 0.0F, 0.0F};
    for (std::size_t k = 0; k < 3U; ++k) {
      sum = sum + data.vertices[data.indices[3U * t + k]].pos;
    }
    record.triPosSums[t] = sum;
  }
  VkDeviceSize vertexBytes = sizeof(MeshVertex) * data.vertices.size();
  VkDeviceSize indexBytes = sizeof(std::uint32_t) * data.indices.size();

  // Vertex and index data lives in DEVICE_LOCAL memory (the standard layout,
  // and faster for draws). A side effect matters here too: persistent
  // HOST_VISIBLE mesh buffers trigger a driver bug where a later NPOT texture
  // upload (e.g. 12288^2) faults the graphics channel.
  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  void* stagingMapped = nullptr;
  bool ok = g.createBuffer(vertexBytes + indexBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           stagingBuffer, stagingMemory, &stagingMapped);
  if (ok) {
    ok = g.createBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, record.vertexBuffer,
                        record.vertexMemory, nullptr);
  }
  if (ok) {
    ok = g.createBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, record.indexBuffer,
                        record.indexMemory, nullptr);
  }
  if (!ok) {
    // createBuffer cleans its own outputs; drop anything created here.
    if (stagingBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, stagingBuffer, nullptr);
    }
    if (stagingMemory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, stagingMemory, nullptr);
    }
    if (record.vertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, record.vertexBuffer, nullptr);
    }
    if (record.vertexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, record.vertexMemory, nullptr);
    }
    if (record.indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g.device, record.indexBuffer, nullptr);
    }
    if (record.indexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, record.indexMemory, nullptr);
    }
    return kInvalidMesh;
  }

  auto* stagingBytes = static_cast<std::uint8_t*>(stagingMapped);
  std::memcpy(stagingBytes, data.vertices.data(), static_cast<std::size_t>(vertexBytes));
  std::memcpy(stagingBytes + vertexBytes, data.indices.data(), static_cast<std::size_t>(indexBytes));

  VkCommandBufferBeginInfo beginInfo =
      vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkBufferCopy vertexCopy{0, 0, vertexBytes};
  VkBufferCopy indexCopy{vertexBytes, 0, indexBytes};
  if (vkBeginCommandBuffer(g.command, &beginInfo) != VK_SUCCESS) {
    std::printf("[vk] mesh upload begin failed\n");
    vkDestroyBuffer(g.device, stagingBuffer, nullptr);
    vkFreeMemory(g.device, stagingMemory, nullptr);
    vkDestroyBuffer(g.device, record.vertexBuffer, nullptr);
    vkFreeMemory(g.device, record.vertexMemory, nullptr);
    vkDestroyBuffer(g.device, record.indexBuffer, nullptr);
    vkFreeMemory(g.device, record.indexMemory, nullptr);
    return kInvalidMesh;
  }
  vkCmdCopyBuffer(g.command, stagingBuffer, record.vertexBuffer, 1, &vertexCopy);
  vkCmdCopyBuffer(g.command, stagingBuffer, record.indexBuffer, 1, &indexCopy);
  // Transfer writes must be made visible to later vertex/index reads — a wait
  // alone only orders execution, it does not flush the writes.
  VkMemoryBarrier2 toVertex =
      vkStruct<VkMemoryBarrier2>(VK_STRUCTURE_TYPE_MEMORY_BARRIER_2);
  toVertex.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  toVertex.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  toVertex.dstStageMask =
      VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
  toVertex.dstAccessMask =
      VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
  VkDependencyInfo dependency = vkStruct<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &toVertex;
  vkCmdPipelineBarrier2(g.command, &dependency);
  if (vkEndCommandBuffer(g.command) != VK_SUCCESS) {
    std::printf("[vk] mesh upload end failed\n");
    vkDestroyBuffer(g.device, stagingBuffer, nullptr);
    vkFreeMemory(g.device, stagingMemory, nullptr);
    vkDestroyBuffer(g.device, record.vertexBuffer, nullptr);
    vkFreeMemory(g.device, record.vertexMemory, nullptr);
    vkDestroyBuffer(g.device, record.indexBuffer, nullptr);
    vkFreeMemory(g.device, record.indexMemory, nullptr);
    return kInvalidMesh;
  }
  VkSubmitInfo submitInfo = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &g.command;
  VkResult submitResult = vkQueueSubmit(g.queue, 1, &submitInfo, VK_NULL_HANDLE);
  VkResult waitResult = vkQueueWaitIdle(g.queue);
  vkDestroyBuffer(g.device, stagingBuffer, nullptr);
  vkFreeMemory(g.device, stagingMemory, nullptr);
  if (submitResult != VK_SUCCESS || waitResult != VK_SUCCESS) {
    std::printf("[vk] mesh upload submit failed (submit=%d wait=%d)\n",
                static_cast<int>(submitResult), static_cast<int>(waitResult));
    vkDestroyBuffer(g.device, record.vertexBuffer, nullptr);
    vkFreeMemory(g.device, record.vertexMemory, nullptr);
    vkDestroyBuffer(g.device, record.indexBuffer, nullptr);
    vkFreeMemory(g.device, record.indexMemory, nullptr);
    return kInvalidMesh;
  }

  record.indexCount = static_cast<std::uint32_t>(data.indices.size());
  g.meshes.push_back(std::move(record));
  return static_cast<MeshHandle>(g.meshes.size() - 1U);
}

// True when `image` needs the upload-budget resample: oversize or NPOT
// (NPOT image extents fault the NVIDIA DMA path; see fitToBudget).
static bool needsUploadFit(Image const& image, std::uint32_t maxDim) {
  auto const isPot = [](std::uint32_t v) { return (v & (v - 1U)) == 0U; };
  return image.width > maxDim || image.height > maxDim || !isPot(image.width) ||
         !isPot(image.height);
}

TextureHandle VkRenderer::addTexture(Image const& image, bool repeatU, bool repeatV) {
  Impl& g = *impl_;
  if (g.device == VK_NULL_HANDLE || image.width == 0U || image.height == 0U ||
      image.rgba.size() != static_cast<std::size_t>(image.width) * image.height * 4U ||
      g.textures.size() >= Impl::kMaxTextures) {
    return kInvalidTexture;
  }
  vkDeviceWaitIdle(g.device);

  // Full mip chain: large textures would otherwise lose their detail to
  // minification (sampling a few texels of an 8K skin map looks smooth).
  // Level 0 is `image` itself (not copied); levels 1..N are derived.
  // fitToBudget: any texture over the upload budget is scaled DOWN (not up),
  // and NPOT extents become POT — an NVIDIA driver bug faults the DMA path
  // for NPOT images in some allocation states. POT images within budget pass
  // through untouched (zero copy).
  Image fitted;
  Image const* source = &image;
  if (needsUploadFit(image, Impl::kUploadBudget)) {
    fitted = fitToBudget(image, Impl::kUploadBudget);
    source = &fitted;
    std::printf("[vk] texture %ux%u -> %ux%u (upload budget, NPOT/oversize)\n", image.width,
                image.height, source->width, source->height);
  }
  std::vector<Image> levels = buildMipLevels(*source);
  std::uint32_t const mipLevels = static_cast<std::uint32_t>(levels.size() + 1U);
  // VRAM accounting ("we run out of memory after a few models"): every
  // upload reports its footprint and the running total.
  {
    std::size_t bytes = static_cast<std::size_t>(source->width) * source->height * 4U;
    for (Image const& level : levels) {
      bytes += static_cast<std::size_t>(level.width) * level.height * 4U;
    }
    g.textureBytes += bytes;
    std::printf("[vk] texture VRAM: +%zu MB, total %zu MB\n", bytes / (1024U * 1024U),
                g.textureBytes / (1024U * 1024U));
  }

  Impl::TextureRecord record{};
  VkImageCreateInfo imageInfo = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.extent = {source->width, source->height, 1};
  imageInfo.mipLevels = mipLevels;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(g.device, &imageInfo, nullptr, &record.image) != VK_SUCCESS) {
    std::printf("[vk] texture image creation failed\n");
    return kInvalidTexture;
  }

  Impl::Requirements reqs = g.imageRequirements(record.image);
  VkMemoryRequirements requirements = reqs.memory;
  std::uint32_t memoryType =
      g.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkMemoryDedicatedAllocateInfo dedicatedInfo =
      vkStruct<VkMemoryDedicatedAllocateInfo>(
          VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
  dedicatedInfo.image = record.image;
  VkMemoryAllocateInfo allocInfo =
      vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
  allocInfo.allocationSize = requirements.size;
  allocInfo.memoryTypeIndex = memoryType;
  if (reqs.dedicated || requirements.size >= (64ULL << 20)) {
    allocInfo.pNext = &dedicatedInfo; // big resources never share a slab
  }
  if (memoryType == UINT32_MAX ||
      vkAllocateMemory(g.device, &allocInfo, nullptr, &record.memory) != VK_SUCCESS ||
      vkBindImageMemory(g.device, record.image, record.memory, 0) != VK_SUCCESS) {
    std::printf("[vk] texture memory allocation failed\n");
    if (record.memory != VK_NULL_HANDLE) {
      vkFreeMemory(g.device, record.memory, nullptr);
    }
    vkDestroyImage(g.device, record.image, nullptr);
    return kInvalidTexture;
  }

  VkImageViewCreateInfo viewInfo =
      vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
  viewInfo.image = record.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.levelCount = mipLevels;
  viewInfo.subresourceRange.layerCount = 1;
  if (vkCreateImageView(g.device, &viewInfo, nullptr, &record.view) != VK_SUCCESS) {
    std::printf("[vk] texture view creation failed\n");
    vkFreeMemory(g.device, record.memory, nullptr);
    vkDestroyImage(g.device, record.image, nullptr);
    return kInvalidTexture;
  }

  VkSamplerCreateInfo samplerInfo =
      vkStruct<VkSamplerCreateInfo>(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; // trilinear
  samplerInfo.addressModeU = repeatU ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                     : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = repeatV ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                     : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.maxLod = static_cast<float>(mipLevels);
  if (vkCreateSampler(g.device, &samplerInfo, nullptr, &record.sampler) != VK_SUCCESS) {
    std::printf("[vk] texture sampler creation failed\n");
    vkDestroyImageView(g.device, record.view, nullptr);
    vkFreeMemory(g.device, record.memory, nullptr);
    vkDestroyImage(g.device, record.image, nullptr);
    return kInvalidTexture;
  }

  // Tiled staging upload through one small window. Whole-image copies of very
  // large textures have been observed to kill the graphics channel on this
  // driver whenever other buffers were allocated first (VK_ERROR_DEVICE_LOST
  // during a 12288^2 upload after meshes loaded); 2048x2048 tiles keep every
  // copy dimension, row pitch and the staging footprint far from engine
  // limits regardless of what was allocated before.
  constexpr std::uint32_t kTile = 2048U;
  constexpr VkDeviceSize kWindowBytes = 64ULL * 1024ULL * 1024ULL; // four tiles

  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  void* stagingMapped = nullptr;
  if (!g.createBuffer(kWindowBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuffer, stagingMemory, &stagingMapped)) {
    vkDestroySampler(g.device, record.sampler, nullptr);
    vkDestroyImageView(g.device, record.view, nullptr);
    vkFreeMemory(g.device, record.memory, nullptr);
    vkDestroyImage(g.device, record.image, nullptr);
    return kInvalidTexture;
  }
  auto* window = static_cast<std::uint8_t*>(stagingMapped);

  // Failure cleanup: staging plus the record's Vulkan objects. Without this,
  // a failed upload would both leak and masquerade as texture 0.
  auto cleanup = [&]() {
    vkDestroyBuffer(g.device, stagingBuffer, nullptr);
    vkFreeMemory(g.device, stagingMemory, nullptr);
    vkDestroySampler(g.device, record.sampler, nullptr);
    vkDestroyImageView(g.device, record.view, nullptr);
    vkDestroyImage(g.device, record.image, nullptr);
    vkFreeMemory(g.device, record.memory, nullptr);
    return kInvalidTexture;
  };

  VkImageMemoryBarrier2 toTransfer =
      vkStruct<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
  toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
  toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toTransfer.image = record.image;
  toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  toTransfer.subresourceRange.levelCount = mipLevels;
  toTransfer.subresourceRange.layerCount = 1;
  VkImageMemoryBarrier2 toSample = toTransfer;
  toSample.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  toSample.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  toSample.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  toSample.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
  toSample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toSample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  std::vector<VkBufferImageCopy> regions; // tile copies staged in the window
  VkDeviceSize windowUsed = 0;
  bool transferStarted = false;

  auto submitBatch = [&](bool last) -> bool {
    VkCommandBufferBeginInfo beginInfo =
        vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g.command, &beginInfo) != VK_SUCCESS) {
      std::printf("[vk] texture upload begin failed\n");
      return false;
    }
    VkDependencyInfo dependency = vkStruct<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
    dependency.imageMemoryBarrierCount = 1;
    if (!transferStarted) {
      dependency.pImageMemoryBarriers = &toTransfer;
      vkCmdPipelineBarrier2(g.command, &dependency);
      transferStarted = true;
    }
    for (VkBufferImageCopy const& region : regions) {
      vkCmdCopyBufferToImage(g.command, stagingBuffer, record.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    if (last) {
      dependency.pImageMemoryBarriers = &toSample;
      vkCmdPipelineBarrier2(g.command, &dependency);
    }
    if (vkEndCommandBuffer(g.command) != VK_SUCCESS) {
      std::printf("[vk] texture upload end failed\n");
      return false;
    }
    VkSubmitInfo submitInfo = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &g.command;
    VkResult submitResult = vkQueueSubmit(g.queue, 1, &submitInfo, VK_NULL_HANDLE);
    VkResult waitResult = vkQueueWaitIdle(g.queue);
    if (submitResult != VK_SUCCESS || waitResult != VK_SUCCESS) {
      std::printf("[vk] texture upload submit failed (submit=%d wait=%d)\n",
                  static_cast<int>(submitResult), static_cast<int>(waitResult));
      return false;
    }
    regions.clear();
    windowUsed = 0;
    return true;
  };

  for (std::uint32_t level = 0; level < mipLevels; ++level) {
    std::uint8_t const* levelSource =
        level == 0U ? source->rgba.data() : levels[level - 1U].rgba.data();
    std::uint32_t levelWidth = level == 0U ? source->width : levels[level - 1U].width;
    std::uint32_t levelHeight = level == 0U ? source->height : levels[level - 1U].height;
    for (std::uint32_t tileY = 0; tileY < levelHeight; tileY += kTile) {
      for (std::uint32_t tileX = 0; tileX < levelWidth; tileX += kTile) {
        std::uint32_t tileW = std::min(kTile, levelWidth - tileX);
        std::uint32_t tileH = std::min(kTile, levelHeight - tileY);
        VkDeviceSize tileBytes = static_cast<VkDeviceSize>(tileW) * tileH * 4U;
        if (windowUsed + tileBytes > kWindowBytes) {
          if (!submitBatch(false)) {
            return cleanup();
          }
        }
        std::uint8_t* tileDst = window + windowUsed;
        for (std::uint32_t row = 0; row < tileH; ++row) {
          std::memcpy(tileDst + static_cast<std::size_t>(row) * tileW * 4U,
                      levelSource +
                          (static_cast<std::size_t>(tileY + row) * levelWidth + tileX) * 4U,
                      static_cast<std::size_t>(tileW) * 4U);
        }
        VkBufferImageCopy copy{};
        copy.bufferOffset = windowUsed;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = level;
        copy.imageSubresource.layerCount = 1;
        copy.imageOffset = {static_cast<std::int32_t>(tileX), static_cast<std::int32_t>(tileY), 0};
        copy.imageExtent = {tileW, tileH, 1};
        regions.push_back(copy);
        windowUsed += tileBytes;
      }
    }
  }
  if (!submitBatch(true)) {
    return cleanup();
  }
  vkDestroyBuffer(g.device, stagingBuffer, nullptr);
  vkFreeMemory(g.device, stagingMemory, nullptr);

  g.textures.push_back(record);
  TextureHandle handle = static_cast<TextureHandle>(g.textures.size() - 1U);

  VkDescriptorImageInfo descriptorInfo{record.sampler, record.view,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet write = vkStruct<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
  write.dstSet = g.descriptorSet;
  write.dstBinding = 1;
  write.dstArrayElement = handle;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &descriptorInfo;
  vkUpdateDescriptorSets(g.device, 1, &write, 0, nullptr);
  return handle;
}

bool VkRenderer::resize(std::uint32_t width, std::uint32_t height) {
  Impl& g = *impl_;
  if (!g.windowed || width == 0U || height == 0U) {
    return true;
  }
  if (width == g.width && height == g.height) {
    return true;
  }
  vkDeviceWaitIdle(g.device);
  // Requested size is a fallback for surfaces that report no fixed extent;
  // createSwapchain() overwrites it with the extent the swapchain really got.
  g.width = width;
  g.height = height;
  return g.recreateTargets();
}
void VkRenderer::setClearColour(float r, float g, float b, float a) {
  impl_->clearR = r;
  impl_->clearG = g;
  impl_->clearB = b;
  impl_->clearA = a;
}


bool VkRenderer::draw(Camera const& camera,
                      std::vector<DrawItem> const& items,
                      std::vector<RendererVertex> const& depthLines,
                      std::vector<RendererVertex> const& overlayLines) {
  Impl& g = *impl_;
  if (g.meshes.empty()) {
    std::printf("[vk] no meshes registered\n");
    return false;
  }
  if (depthLines.size() + overlayLines.size() > kMaxLineVerts) {
    std::printf("[vk] line buffer overflow\n");
    return false;
  }

  VK_CHECK(vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX));
  g.flushFrees(); // completed frame: release retired scratch/staging buffers
  if (g.width == 0U || g.height == 0U) {
    return true; // minimised / zero-extent swapchain: nothing to render
  }

  float aspect = static_cast<float>(g.width) / static_cast<float>(g.height);
  Mat4 viewProj = cameraViewProjection(camera, aspect);
  std::memcpy(g.uniformMapped, viewProj.m, sizeof(float) * 16U);
  float const camPos[4] = {camera.position.x, camera.position.y, camera.position.z, 1.0F};
  std::memcpy(static_cast<char*>(g.uniformMapped) + sizeof(float) * 16U, camPos, sizeof(camPos));

  std::size_t depthCount = depthLines.size();
  if (depthCount > 0U) {
    std::memcpy(g.lineMapped, depthLines.data(), sizeof(RendererVertex) * depthCount);
  }
  if (!overlayLines.empty()) {
    std::memcpy(static_cast<char*>(g.lineMapped) + sizeof(RendererVertex) * depthCount,
                overlayLines.data(), sizeof(RendererVertex) * overlayLines.size());
  }

  VkImageView colourViewUsed = g.colourView;
  if (g.windowed) {
    VkResult acquire = vkAcquireNextImageKHR(g.device, g.swapchain, UINT64_MAX, g.imageAvailable,
                                             VK_NULL_HANDLE, &g.swapchainIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
      vkDeviceWaitIdle(g.device);
      return g.recreateTargets();
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
      return checkVk(acquire, "vkAcquireNextImageKHR");
    }
    colourViewUsed = g.swapchainViews[g.swapchainIndex];
  }

  g.items = items;
  Impl::Target target = g.windowed ? Impl::Target::Window : Impl::Target::Readback;
  VkImage colourImageUsed = g.windowed ? g.swapchainImages[g.swapchainIndex] : g.colourImage;
  if (!g.recordCommands(depthCount, overlayLines.size(), colourViewUsed, colourImageUsed, target,
                        cameraView(camera), camera.position)) {
    return false;
  }

  if (g.windowed) {
    VkPresentInfoKHR presentInfo =
        vkStruct<VkPresentInfoKHR>(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &g.renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &g.swapchain;
    presentInfo.pImageIndices = &g.swapchainIndex;
    VkResult presentResult = vkQueuePresentKHR(g.queue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
      vkDeviceWaitIdle(g.device);
      return g.recreateTargets();
    }
    if (presentResult != VK_SUCCESS) {
      return checkVk(presentResult, "vkQueuePresentKHR");
    }
  }
  return true;
}

bool VkRenderer::initExternal(Config const& config, VkInstance instance, VkPhysicalDevice physical,
                              VkDevice device, std::uint32_t queueFamily, VkFormat colourFormat) {
  Impl& g = *impl_;
  g.windowed = false;
  g.external = true;
  g.width = config.width;
  g.height = config.height;
  g.instance = instance;
  g.physical = physical;
  g.device = device;
  g.queueFamily = queueFamily;
  vkGetDeviceQueue(g.device, g.queueFamily, 0, &g.queue);
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(g.physical, &props);
  std::snprintf(g.deviceName, sizeof(g.deviceName), "%s", props.deviceName);
  if (!createCommonResources(config.shaderDir, colourFormat)) {
    return false;
  }
  std::printf("[vk] device: %s (external)\n", g.deviceName);
  return true;
}

bool VkRenderer::drawToView(VkImageView target, VkImage targetImage, Mat4 const& view,
                            Mat4 const& viewProj, Vec3 cameraPos,
                            std::vector<DrawItem> const& items,
                            std::vector<RendererVertex> const& depthLines,
                            std::vector<RendererVertex> const& overlayLines) {
  Impl& g = *impl_;
  if (g.meshes.empty()) {
    std::printf("[vk] no meshes registered\n");
    return false;
  }
  if (depthLines.size() + overlayLines.size() > kMaxLineVerts) {
    std::printf("[vk] line buffer overflow\n");
    return false;
  }

  VK_CHECK(vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX));
  g.flushFrees(); // completed frame: release retired scratch/staging buffers
  std::memcpy(g.uniformMapped, viewProj.m, sizeof(float) * 16U);
  float const camPos[4] = {cameraPos.x, cameraPos.y, cameraPos.z, 1.0F};
  std::memcpy(static_cast<char*>(g.uniformMapped) + sizeof(float) * 16U, camPos,
              sizeof(camPos));

  std::size_t depthCount = depthLines.size();
  if (depthCount > 0U) {
    std::memcpy(g.lineMapped, depthLines.data(), sizeof(RendererVertex) * depthCount);
  }
  if (!overlayLines.empty()) {
    std::memcpy(static_cast<char*>(g.lineMapped) + sizeof(RendererVertex) * depthCount,
                overlayLines.data(), sizeof(RendererVertex) * overlayLines.size());
  }

  g.items = items;
  if (!g.recordCommands(depthCount, overlayLines.size(), target, targetImage,
                        Impl::Target::External, view, cameraPos)) {
    return false;
  }
  // The caller releases the swapchain image only after rendering completes.
  VK_CHECK(vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX));
  return true;
}

bool VkRenderer::readback(std::vector<std::uint8_t>& pixelsOut) {
  Impl& g = *impl_;
  if (g.windowed || g.readbackMapped == nullptr) {
    return false;
  }
  VK_CHECK(vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX));
  std::size_t byteCount = static_cast<std::size_t>(g.width) * g.height * 4U;
  pixelsOut.resize(byteCount);
  std::memcpy(pixelsOut.data(), g.readbackMapped, byteCount);
  return true;
}

std::uint32_t VkRenderer::width() const {
  return impl_->width;
}

std::uint32_t VkRenderer::height() const {
  return impl_->height;
}

char const* VkRenderer::deviceName() const {
  return impl_->deviceName;
}

} // namespace vredit
