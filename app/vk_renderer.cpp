#include "vk_renderer.h"

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
};
static_assert(sizeof(PushData) == 96, "push constants must match the shader layout");

} // namespace

struct VkRenderer::Impl {
  struct MeshRecord {
    VkBuffer vertexBuffer{VK_NULL_HANDLE};
    VkDeviceMemory vertexMemory{VK_NULL_HANDLE};
    VkBuffer indexBuffer{VK_NULL_HANDLE};
    VkDeviceMemory indexMemory{VK_NULL_HANDLE};
    std::uint32_t indexCount{0};
  };

  bool windowed{false};
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
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    std::uint32_t memoryType = findMemoryType(requirements.memoryTypeBits, memFlags);
    if (memoryType == UINT32_MAX) {
      std::printf("[vk] no suitable memory type\n");
      vkDestroyBuffer(device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      return false;
    }
    VkMemoryAllocateInfo allocInfo =
        vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = memoryType;
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

  bool createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps));

    std::uint32_t formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, formats.data()));
    VkSurfaceFormatKHR chosen = formats[0];
    for (VkSurfaceFormatKHR const& format : formats) {
      if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
          format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        chosen = format;
        break;
      }
    }
    swapchainFormat = chosen.format;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
      extent.width = width;
      extent.height = height;
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
      VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &swapchainViews[i]));
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
                      VkPipeline& pipeline) {
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
    VkVertexInputAttributeDescription attributes[3] = {};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RendererVertex, pos)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RendererVertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RendererVertex, colour)};
    VkPipelineVertexInputStateCreateInfo vertexInput =
        vkStruct<VkPipelineVertexInputStateCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 3;
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

    VkFormat colourFormat = windowed ? swapchainFormat : VK_FORMAT_R8G8B8A8_UNORM;
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

  bool recordCommands(std::size_t depthLineCount,
                      std::size_t overlayLineCount,
                      VkImageView colourViewUsed) {
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
    colourBarrier.image = windowed ? swapchainImages[swapchainIndex] : colourImage;
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
    colourAttachment.clearValue.color = {{0.08F, 0.09F, 0.11F, 1.0F}};

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

    // Meshes: one indexed draw per item.
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
    for (DrawItem const& item : items) {
      std::size_t handle = item.mesh < meshes.size() ? item.mesh : 0U;
      MeshRecord const& record = meshes[handle];
      VkDeviceSize zero = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &record.vertexBuffer, &zero);
      vkCmdBindIndexBuffer(command, record.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
      PushData push{};
      std::memcpy(push.model, item.model.m, sizeof(float) * 16U);
      push.tint[0] = item.colour.x;
      push.tint[1] = item.colour.y;
      push.tint[2] = item.colour.z;
      push.tint[3] = 1.0F;
      push.params[0] = item.lit ? 1.0F : 0.0F;
      vkCmdPushConstants(command, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushData), &push);
      vkCmdDrawIndexed(command, record.indexCount, 1, 0, 0, 0);
    }

    VkDeviceSize stride = sizeof(RendererVertex);
    if (depthLineCount > 0U) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, lineDepthPipeline);
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &lineBuffer, &offset);
      PushData push{};
      push.model[0] = 1.0F;
      push.model[5] = 1.0F;
      push.model[10] = 1.0F;
      push.model[15] = 1.0F;
      push.tint[0] = 1.0F;
      push.tint[1] = 1.0F;
      push.tint[2] = 1.0F;
      push.tint[3] = 1.0F;
      vkCmdPushConstants(command, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushData), &push);
      vkCmdDraw(command, static_cast<std::uint32_t>(depthLineCount), 1, 0, 0);
    }
    if (overlayLineCount > 0U) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, lineOverlayPipeline);
      VkDeviceSize offset = stride * depthLineCount;
      vkCmdBindVertexBuffers(command, 0, 1, &lineBuffer, &offset);
      vkCmdDraw(command, static_cast<std::uint32_t>(overlayLineCount), 1, 0, 0);
    }

    vkCmdEndRendering(command);

    VkImageMemoryBarrier2 afterBarrier = colourBarrier;
    afterBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    afterBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    afterBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (windowed) {
      afterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
      afterBarrier.dstAccessMask = VK_ACCESS_2_NONE;
      afterBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    } else {
      afterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      afterBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      afterBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &afterBarrier;
    vkCmdPipelineBarrier2(command, &dependency);

    if (!windowed) {
      VkBufferImageCopy copy{};
      copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      copy.imageSubresource.layerCount = 1;
      copy.imageExtent = {width, height, 1};
      vkCmdCopyImageToBuffer(command, colourImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readbackBuffer, 1, &copy);
    }

    VK_CHECK(vkEndCommandBuffer(command));

    VkSubmitInfo submit = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (windowed) {
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
    VK_CHECK(vkResetFences(device, 1, &fence));
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    return true;
  }

  std::vector<DrawItem> items; // set by draw() before recordCommands()
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
  if (g.windowed) {
    VkSemaphoreCreateInfo semaphoreInfo =
        vkStruct<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    VK_CHECK(vkCreateSemaphore(g.device, &semaphoreInfo, nullptr, &g.imageAvailable));
    VK_CHECK(vkCreateSemaphore(g.device, &semaphoreInfo, nullptr, &g.renderFinished));
  }

  std::vector<char> vertCode;
  std::vector<char> fragCode;
  std::string vertPath = std::string(config.shaderDir) + "/mesh.vert.spv";
  std::string fragPath = std::string(config.shaderDir) + "/mesh.frag.spv";
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

  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  VkDescriptorSetLayoutCreateInfo setLayoutInfo =
      vkStruct<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
  setLayoutInfo.bindingCount = 1;
  setLayoutInfo.pBindings = &binding;
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

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
  VkDescriptorPoolCreateInfo poolCreateInfo =
      vkStruct<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
  poolCreateInfo.maxSets = 1;
  poolCreateInfo.poolSizeCount = 1;
  poolCreateInfo.pPoolSizes = &poolSize;
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
  bufferInfo.range = 64;
  VkWriteDescriptorSet write =
      vkStruct<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
  write.dstSet = g.descriptorSet;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.pBufferInfo = &bufferInfo;
  vkUpdateDescriptorSets(g.device, 1, &write, 0, nullptr);

  if (!g.createImage2D(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT, g.depthImage, g.depthMemory, g.depthView)) {
    return false;
  }
  if (g.windowed) {
    if (!g.createSwapchain()) {
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
  if (!g.createPipeline(true, true, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, g.meshPipeline) ||
      !g.createPipeline(true, false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, g.lineDepthPipeline) ||
      !g.createPipeline(false, false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, g.lineOverlayPipeline)) {
    return false;
  }

  std::printf("[vk] device: %s (%s)\n", g.deviceName, g.windowed ? "windowed" : "offscreen");
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
  for (VkPipeline pipeline : {g.meshPipeline, g.lineDepthPipeline, g.lineOverlayPipeline}) {
    if (pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(g.device, pipeline, nullptr);
    }
  }
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
  vkDestroyDevice(g.device, nullptr);
  g.device = VK_NULL_HANDLE;
  if (g.surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
    g.surface = VK_NULL_HANDLE;
  }
  if (g.instance != VK_NULL_HANDLE) {
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
  void* vertexMapped = nullptr;
  void* indexMapped = nullptr;
  bool ok = g.createBuffer(sizeof(MeshVertex) * data.vertices.size(),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           record.vertexBuffer, record.vertexMemory, &vertexMapped);
  if (ok) {
    ok = g.createBuffer(sizeof(std::uint32_t) * data.indices.size(),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        record.indexBuffer, record.indexMemory, &indexMapped);
  }
  if (!ok) {
    // createBuffer cleans its own outputs; drop anything created here.
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

  std::memcpy(vertexMapped, data.vertices.data(), sizeof(MeshVertex) * data.vertices.size());
  std::memcpy(indexMapped, data.indices.data(), sizeof(std::uint32_t) * data.indices.size());
  record.indexCount = static_cast<std::uint32_t>(data.indices.size());
  g.meshes.push_back(record);
  return static_cast<MeshHandle>(g.meshes.size() - 1U);
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
  g.width = width;
  g.height = height;
  g.destroyImage2D(g.depthImage, g.depthMemory, g.depthView);
  if (!g.createImage2D(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT, g.depthImage, g.depthMemory, g.depthView)) {
    return false;
  }
  return g.createSwapchain();
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
  if (g.width == 0U || g.height == 0U) {
    return true; // minimised / zero-extent swapchain: nothing to render
  }

  float aspect = static_cast<float>(g.width) / static_cast<float>(g.height);
  Mat4 viewProj = cameraViewProjection(camera, aspect);
  std::memcpy(g.uniformMapped, viewProj.m, sizeof(float) * 16U);

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
      return g.createSwapchain();
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
      return checkVk(acquire, "vkAcquireNextImageKHR");
    }
    colourViewUsed = g.swapchainViews[g.swapchainIndex];
  }

  g.items = items;
  if (!g.recordCommands(depthCount, overlayLines.size(), colourViewUsed)) {
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
      return g.createSwapchain();
    }
    if (presentResult != VK_SUCCESS) {
      return checkVk(presentResult, "vkQueuePresentKHR");
    }
  }
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
