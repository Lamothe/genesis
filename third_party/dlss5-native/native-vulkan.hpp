// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Minimal compute-capable Vulkan context for the native GPU stage: instance,
// a preferred-discrete physical device with a compute queue, logical device,
// command pool, device-local buffers with staging transfers, timestamp
// queries and one-shot dispatches. The CPU path in the family headers stays
// the deterministic reference; every GPU family is parity-tested against it.
//
// Shader sources are GLSL compiled by the test/tool runner with glslc (no
// compiled binaries are stored in the repository). There is no vendor
// extension requirement: plain Vulkan 1.1 compute with push constants.
#ifndef DLSS5_NR_NATIVE_VULKAN_HPP
#define DLSS5_NR_NATIVE_VULKAN_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "native-pre-post.hpp"
#include "native-vit.hpp"

#if defined(__linux__) && !defined(DLSS5_NR_NO_VULKAN)
#define DLSS5_NR_HAS_VULKAN 1
#include <vulkan/vulkan.h>
#else
#define DLSS5_NR_HAS_VULKAN 0
#endif

namespace dlss5_nr {

#if DLSS5_NR_HAS_VULKAN

// ---- the device -----------------------------------------------------------------------------------

// TEST-ONLY forced-failure seam (fault INJECTION for tests). Armed
// counters make the Nth call of a run-path operation fail synthetically
// WITHOUT touching the driver; -1 (the default) means never. This is not
// a runtime or capture probe: it never queries the driver and changes
// nothing when unarmed. Production behaviour is byte-identical to the
// previous build when the counters are left at -1.
struct VulkanFaultPlan {
    // Fires on the Nth vkMapMemory (0 = the next call).
    int fail_map_countdown = -1;
    // Fires on the Nth vkQueueSubmit (0 = the next call).
    int fail_submit_countdown = -1;
};
inline VulkanFaultPlan g_vulkan_faults{};

// Consumes one call from the countdown: returns true when the call must
// fail. The counter arms for exactly one failure (then resets to -1).
inline bool vulkan_fault_consume(int& countdown) {
    if (countdown < 0) {
        return false;
    }
    if (countdown == 0) {
        countdown = -1;
        return true;
    }
    --countdown;
    return false;
}

struct VulkanDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    float timestamp_period = 0.0f;   // nanoseconds per timestamp tick
    bool has_timestamps = false;
    char name[256] = {};
};

inline void vulkan_close(VulkanDevice& d) {
    if (d.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(d.device);
        if (d.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(d.device, d.pool, nullptr);
        }
        vkDestroyDevice(d.device, nullptr);
    }
    if (d.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(d.instance, nullptr);
    }
    d = VulkanDevice{};
}

// Opens a compute-capable device. `opened` is set false when no usable
// Vulkan implementation exists (the callers record an environment gap and
// skip; that is not a kernel failure). `prefer_discrete` selects the
// discrete GPU when several devices qualify.
inline Status vulkan_open(VulkanDevice& out, bool& opened,
                          bool prefer_discrete = true) {
    opened = false;
    out = VulkanDevice{};
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dlss5-native";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &out.instance) != VK_SUCCESS) {
        out = VulkanDevice{};
        return Status::kOk;  // nothing created; `opened` stays false
    }
    std::uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(out.instance, &dev_count, nullptr);
    if (dev_count == 0) {
        vulkan_close(out);
        return Status::kOk;
    }
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(out.instance, &dev_count, devices.data());
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    std::uint32_t chosen_family = 0;
    float chosen_period = 0.0f;
    bool chosen_ts = false;
    int chosen_score = -1;
    for (std::uint32_t i = 0; i < dev_count; ++i) {
        std::uint32_t fam_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &fam_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> fams(fam_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &fam_count,
                                                 fams.data());
        for (std::uint32_t f = 0; f < fam_count; ++f) {
            if ((fams[f].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
                continue;
            }
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(devices[i], &props);
            // Prefer a timestamp-capable compute family on the same device.
            if (chosen == devices[i] && chosen_ts
                && fams[f].timestampValidBits == 0) {
                continue;
            }
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score = prefer_discrete ? 2 : 1;
            } else if (props.deviceType
                       == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score = prefer_discrete ? 1 : 2;
            }
            if (score > chosen_score) {
                chosen_score = score;
                chosen = devices[i];
                chosen_family = f;
                chosen_period = props.limits.timestampPeriod;
                chosen_ts = fams[f].timestampValidBits > 0;
                std::snprintf(out.name, sizeof(out.name), "%s",
                              props.deviceName);
            }
            break;  // one queue family per device suffices
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        vulkan_close(out);
        return Status::kOk;
    }
    out.physical = chosen;
    out.queue_family = chosen_family;
    out.timestamp_period = chosen_period;
    out.has_timestamps = chosen_ts;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = chosen_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(chosen, &dci, nullptr, &out.device) != VK_SUCCESS) {
        vulkan_close(out);
        return Status::kOk;
    }
    vkGetDeviceQueue(out.device, chosen_family, 0, &out.queue);
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = chosen_family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(out.device, &pci, nullptr, &out.pool)
        != VK_SUCCESS) {
        vulkan_close(out);
        return Status::kOk;
    }
    opened = true;
    return Status::kOk;
}

// ---- buffers --------------------------------------------------------------------------------------

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::size_t size = 0;
};

inline void vulkan_buffer_destroy(const VulkanDevice& d, VulkanBuffer& b) {
    if (b.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(d.device, b.buffer, nullptr);
    }
    if (b.memory != VK_NULL_HANDLE) {
        vkFreeMemory(d.device, b.memory, nullptr);
    }
    b = VulkanBuffer{};
}

inline Status vulkan_buffer_create(const VulkanDevice& d, VulkanBuffer& out,
                                   std::size_t bytes, VkBufferUsageFlags usage,
                                   bool host_visible) {
    out = VulkanBuffer{};
    if (bytes == 0) {
        return Status::kInvalidBuffer;
    }
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d.device, &bci, nullptr, &out.buffer) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d.device, out.buffer, &req);
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(d.physical, &props);
    const VkMemoryPropertyFlags want = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    std::uint32_t index = UINT32_MAX;
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) != 0
            && (props.memoryTypes[i].propertyFlags & want) == want) {
            index = i;
            break;
        }
    }
    if (index == UINT32_MAX) {
        vulkan_buffer_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = index;
    if (vkAllocateMemory(d.device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
        vulkan_buffer_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    if (vkBindBufferMemory(d.device, out.buffer, out.memory, 0)
        != VK_SUCCESS) {
        vulkan_buffer_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    out.size = bytes;
    return Status::kOk;
}

inline Status vulkan_upload(const VulkanDevice& d, VulkanBuffer& staging,
                            VulkanBuffer& device_local, const void* data,
                            std::size_t bytes) {
    if (data == nullptr || bytes > staging.size || bytes > device_local.size) {
        return Status::kInvalidBuffer;
    }
    void* mapped = nullptr;
    if (vulkan_fault_consume(g_vulkan_faults.fail_map_countdown)
        || vkMapMemory(d.device, staging.memory, 0, bytes, 0, &mapped)
               != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(d.device, staging.memory);
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(cmd, staging.buffer, device_local.buffer, 1, &copy);
    ok = ok && vkEndCommandBuffer(cmd) == VK_SUCCESS;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    {
        bool submit_ok = false;
        if (!vulkan_fault_consume(g_vulkan_faults.fail_submit_countdown)) {
            submit_ok = vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE)
                        == VK_SUCCESS;
        }
        ok = ok && submit_ok;
    }
    ok = ok && vkQueueWaitIdle(d.queue) == VK_SUCCESS;
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    return ok ? Status::kOk : Status::kInvalidBuffer;
}

inline Status vulkan_download(const VulkanDevice& d, VulkanBuffer& staging,
                              VulkanBuffer& device_local, void* data,
                              std::size_t bytes) {
    if (data == nullptr || bytes > staging.size || bytes > device_local.size) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(cmd, device_local.buffer, staging.buffer, 1, &copy);
    ok = ok && vkEndCommandBuffer(cmd) == VK_SUCCESS;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    {
        bool submit_ok = false;
        if (!vulkan_fault_consume(g_vulkan_faults.fail_submit_countdown)) {
            submit_ok = vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE)
                        == VK_SUCCESS;
        }
        ok = ok && submit_ok;
    }
    ok = ok && vkQueueWaitIdle(d.queue) == VK_SUCCESS;
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    if (!ok) {
        return Status::kInvalidBuffer;
    }
    void* mapped = nullptr;
    if (vulkan_fault_consume(g_vulkan_faults.fail_map_countdown)
        || vkMapMemory(d.device, staging.memory, 0, bytes, 0, &mapped)
               != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    std::memcpy(data, mapped, bytes);
    vkUnmapMemory(d.device, staging.memory);
    return Status::kOk;
}

// ---- shader modules --------------------------------------------------------------------------------

// Loads SPIR-V produced by the runner's glslc step (never stored in-tree).
inline Status vulkan_shader(const VulkanDevice& d, const std::uint32_t* words,
                            std::size_t word_count, VkShaderModule& out) {
    if (words == nullptr || word_count == 0) {
        return Status::kInvalidBuffer;
    }
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = word_count * sizeof(std::uint32_t);
    ci.pCode = words;
    if (vkCreateShaderModule(d.device, &ci, nullptr, &out) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// ---- the projection matmul (the hot family) --------------------------------------------------------

// out[t][n] = sum_k in[t][k] * w[n][k] on the GPU: the CPU's matmul_e4m3
// shape with e4m3 weights uploaded as exact float32 decodes. The shader's
// `precise` accumulation keeps the k-ordered scalar sum uncontracted and
// unreassociated for CPU bit-parity.
struct VulkanMatmul {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    VulkanBuffer in_dev, w_dev, out_dev, staging;
    // Bumped on every create (the grow-only rebuild frees in_dev/out_dev);
    // the weight cache's stale-set check compares generations, which VkBuffer
    // handle values cannot defeat (handles are reusable after free).
    std::uint32_t generation = 0;
    int max_tokens = 0;
    int max_c_in = 0;
    int max_rows = 0;
};

struct VulkanMatmulTiming {
    double upload_ms = 0.0;
    double kernel_ms = 0.0;
    double download_ms = 0.0;
};

inline void vulkan_matmul_destroy(const VulkanDevice& d, VulkanMatmul& m) {
    // The generation is the matmul's IDENTITY (the cache's stale-set check
    // compares it), not a resource: destroying frees the handles but keeps
    // the counter, so destroy + create (the grow-only rebuild) advances it
    // and a rebuilt matmul cannot silently reuse an old descriptor.
    const std::uint32_t preserved_generation = m.generation;
    if (d.device == VK_NULL_HANDLE) {
        m = VulkanMatmul{};
        m.generation = preserved_generation;
        return;
    }
    vkDeviceWaitIdle(d.device);
    vulkan_buffer_destroy(d, m.in_dev);
    vulkan_buffer_destroy(d, m.w_dev);
    vulkan_buffer_destroy(d, m.out_dev);
    vulkan_buffer_destroy(d, m.staging);
    if (m.queries != VK_NULL_HANDLE) {
        vkDestroyQueryPool(d.device, m.queries, nullptr);
    }
    if (m.dpool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(d.device, m.dpool, nullptr);
    }
    if (m.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(d.device, m.pipeline, nullptr);
    }
    if (m.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(d.device, m.layout, nullptr);
    }
    if (m.dsl != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(d.device, m.dsl, nullptr);
    }
    m = VulkanMatmul{};
    m.generation = preserved_generation;
}

inline Status vulkan_matmul_create(const VulkanDevice& d, VulkanMatmul& out,
                                   const std::uint32_t* spirv,
                                   std::size_t word_count, int max_tokens,
                                   int max_c_in, int max_rows) {
    if (spirv == nullptr || word_count == 0 || max_tokens < 1 || max_c_in < 1
        || max_rows < 1) {
        return Status::kInvalidValues;
    }
    // Preserve the generation across the destroy + struct reset so a
    // grow-only rebuild (destroy + create) is distinguishable from the
    // first create: the cache's stale-set check compares this counter, and
    // VkBuffer handles are reusable after free (a handle comparison alone
    // cannot tell an old descriptor from a reused handle). Read BEFORE the
    // destroy (destroy zeroes the struct).
    const std::uint32_t next_generation = out.generation + 1;
    vulkan_matmul_destroy(d, out);  // safe on a zeroed or live handle set
    out = VulkanMatmul{};
    out.generation = next_generation;
    out.max_tokens = max_tokens;
    out.max_c_in = max_c_in;
    out.max_rows = max_rows;
    VkShaderModule module = VK_NULL_HANDLE;
    {
        const Status st = vulkan_shader(d, spirv, word_count, module);
        if (st != Status::kOk) {
            return st;
        }
    }
    VkDescriptorSetLayoutBinding binds[3]{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 3 * sizeof(std::uint32_t);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &out.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    // Create layout and pipeline; on any failure clean up and report.
    bool ok = vkCreateDescriptorSetLayout(d.device, &dslci, nullptr,
                                          &out.dsl) == VK_SUCCESS;
    plci.pSetLayouts = &out.dsl;
    ok = ok && vkCreatePipelineLayout(d.device, &plci, nullptr, &out.layout)
                     == VK_SUCCESS;
    cpci.layout = out.layout;
    ok = ok
        && vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                    &out.pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(d.device, module, nullptr);
    VkDescriptorPoolSize psize{};
    psize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psize;
    ok = ok && vkCreateDescriptorPool(d.device, &dpci, nullptr, &out.dpool)
                     == VK_SUCCESS;
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = out.dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &out.dsl;
    ok = ok
        && vkAllocateDescriptorSets(d.device, &dsai, &out.set) == VK_SUCCESS;
    const std::size_t in_bytes = static_cast<std::size_t>(max_tokens)
        * static_cast<std::size_t>(max_c_in) * sizeof(float);
    const std::size_t w_bytes = static_cast<std::size_t>(max_rows)
        * static_cast<std::size_t>(max_c_in) * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(max_tokens)
        * static_cast<std::size_t>(max_rows) * sizeof(float);
    const std::size_t staging_bytes =
        in_bytes > w_bytes ? (in_bytes > out_bytes ? in_bytes : out_bytes)
                           : (w_bytes > out_bytes ? w_bytes : out_bytes);
    ok = ok
        && vulkan_buffer_create(d, out.in_dev, in_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                false)
               == Status::kOk;
    ok = ok
        && vulkan_buffer_create(d, out.w_dev, w_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                false)
               == Status::kOk;
    ok = ok
        && vulkan_buffer_create(d, out.out_dev, out_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                false)
               == Status::kOk;
    ok = ok
        && vulkan_buffer_create(d, out.staging, staging_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                true)
               == Status::kOk;
    VkDescriptorBufferInfo infos[3]{};
    infos[0] = {out.in_dev.buffer, 0, in_bytes};
    infos[1] = {out.w_dev.buffer, 0, w_bytes};
    infos[2] = {out.out_dev.buffer, 0, out_bytes};
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = out.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    if (ok) {
        vkUpdateDescriptorSets(d.device, 3, writes, 0, nullptr);
    }
    if (d.has_timestamps) {
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        ok = ok
            && vkCreateQueryPool(d.device, &qpci, nullptr, &out.queries)
                   == VK_SUCCESS;
    }
    if (!ok) {
        vulkan_matmul_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// Runs out[t][n] = sum_k in[t][k] * w[n][k] with the CPU's exact element
// order; `in`/`weights` are plain row-major float arrays (the weights are
// the CPU's exact e4m3 decodes).
// The resident-weight variant: `w_dev` is a persistent weight buffer (the
// G5 weight cache) already holding the [rows, c_in] float32 matrix. Only
// the input stages; the dispatch and download are the same as
// vulkan_matmul_run, so results are bit-identical between the two.
inline Status vulkan_matmul_run_resident(const VulkanDevice& d, VulkanMatmul& m,
                                         const float* in, int tokens, int c_in,
                                         const VulkanBuffer& w_dev, int rows,
                                         float* out,
                                         VulkanMatmulTiming& timing);

// A small persistent weight cache keyed by the caller's weight pointer (the
// EvalWeights record source): the FIRST sight of a key uploads its decoded
// float matrix once; later runs reuse the buffer (no per-run staging).
// Fixed capacity, linear search — the plan's matrix count is well under it.
struct VulkanWeightCache {
    static constexpr int kCapacity = 512;
    struct Entry {
        const void* key = nullptr;
        int rows = 0;
        int cols = 0;
        VulkanBuffer buf;
        VkDescriptorSet set = VK_NULL_HANDLE;
        // The matmul in/out buffers the descriptor was written against; the
        // matmul's grow-only rebuild rewrites the stale set at the next run.
        VkBuffer written_in = VK_NULL_HANDLE;
        VkBuffer written_out = VK_NULL_HANDLE;
        std::uint32_t written_gen = 0;
        // The content hash of the codes/floats this buffer holds: a pointer
        // key alone is unsound when a provider recycles per-call storage, so
        // every hit verifies the content and re-uploads on mismatch.
        std::uint64_t hash = 0;
    };
    Entry entries[kCapacity];
    int count = 0;
    VkDescriptorPool pool = VK_NULL_HANDLE;

    Status init(const VulkanDevice& d) {
        if (pool != VK_NULL_HANDLE) {
            return Status::kOk;
        }
        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = kCapacity;
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = kCapacity;
        pci.poolSizeCount = 1;
        pci.pPoolSizes = &size;
        return vkCreateDescriptorPool(d.device, &pci, nullptr, &pool)
                   == VK_SUCCESS
            ? Status::kOk
            : Status::kInvalidBuffer;
    }

    void destroy(const VulkanDevice& d) {
        for (int i = 0; i < count; ++i) {
            vulkan_buffer_destroy(d, entries[i].buf);
        }
        count = 0;
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(d.device, pool, nullptr);
            pool = VK_NULL_HANDLE;
        }
    }
};

inline void vulkan_weight_cache_destroy(const VulkanDevice& d,
                                        VulkanWeightCache& cache) {
    cache.destroy(d);
}

// The cache key is the float matrix's data pointer with its shape; a key
// whose shape changed re-uploads (the old buffer is reused in place).
// Writes/rewrites the entry's descriptor against the matmul's current
// in/out buffers (the grow-only rebuild makes old sets stale).
// FNV-1a over the record bytes; word-wise would be faster but the byte form
// is exact and the cost (a few ms per large record) is still far below the
// staging path it replaces.
inline std::uint64_t vulkan_weight_hash(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Re-uploads an existing entry's buffer in place (same shape, new content).
inline Status vulkan_weight_cache_update(const VulkanDevice& d,
                                         VulkanWeightCache::Entry& e,
                                         const float* weights,
                                         std::uint64_t hash) {
    const std::size_t bytes = static_cast<std::size_t>(e.rows)
        * static_cast<std::size_t>(e.cols) * sizeof(float);
    VulkanBuffer upload_staging;
    const Status sc = vulkan_buffer_create(d, upload_staging, bytes,
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           true);
    if (sc != Status::kOk) {
        return sc;
    }
    const Status up = vulkan_upload(d, upload_staging, e.buf, weights, bytes);
    vulkan_buffer_destroy(d, upload_staging);
    if (up == Status::kOk) {
        e.hash = hash;
    }
    return up;
}

// The stale-set rewrite: the matmul's grow-only rebuild frees in_dev/out_dev
// (and VkBuffer handles are reusable after free), so every hit rewrites the
// entry's descriptor when the matmul's generation or buffers changed. The
// generation check is the authoritative one.
inline void vulkan_weight_cache_refresh(VulkanWeightCache::Entry& e,
                                        const VulkanDevice& d,
                                        const VulkanMatmul& m) {
    if (e.written_gen == m.generation
        && e.written_in == m.in_dev.buffer
        && e.written_out == m.out_dev.buffer) {
        return;
    }
    VkDescriptorBufferInfo infos[3]{};
    infos[0] = {m.in_dev.buffer, 0, VK_WHOLE_SIZE};
    infos[1] = {e.buf.buffer, 0, VK_WHOLE_SIZE};
    infos[2] = {m.out_dev.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t w = 0; w < 3; ++w) {
        writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[w].dstSet = e.set;
        writes[w].dstBinding = w;
        writes[w].descriptorCount = 1;
        writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[w].pBufferInfo = &infos[w];
    }
    vkUpdateDescriptorSets(d.device, 3, writes, 0, nullptr);
    e.written_in = m.in_dev.buffer;
    e.written_out = m.out_dev.buffer;
    e.written_gen = m.generation;
}

inline Status vulkan_weight_cache_ensure(VulkanWeightCache& cache,
                                         const VulkanDevice& d,
                                         const VulkanMatmul& m,
                                         const void* key,
                                         const float* weights, int rows,
                                         int cols, std::uint64_t content_hash,
                                         VulkanWeightCache::Entry** out) {
    for (int i = 0; i < cache.count; ++i) {
        if (cache.entries[i].key == key && cache.entries[i].rows == rows
            && cache.entries[i].cols == cols) {
            *out = &cache.entries[i];
            vulkan_weight_cache_refresh(cache.entries[i], d, m);
            if (cache.entries[i].hash != content_hash) {
                // The provider recycled the pointer for different content:
                // re-upload in place (the decode still happened once).
                return vulkan_weight_cache_update(d, cache.entries[i], weights,
                                                  content_hash);
            }
            return Status::kOk;
        }
    }
    if (key == nullptr || weights == nullptr || rows < 1 || cols < 1) {
        return Status::kInvalidValues;
    }
    if (cache.count >= VulkanWeightCache::kCapacity) {
        // The cache is an optimisation: exhaustion must not fail the frame.
        return Status::kInvalidBuffer;  // callers fall back to staging
    }
    const Status init_st = cache.init(d);
    if (init_st != Status::kOk) {
        return init_st;
    }
    VulkanWeightCache::Entry& e = cache.entries[cache.count];
    const std::size_t bytes = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(cols) * sizeof(float);
    const Status st = vulkan_buffer_create(d, e.buf, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    if (st != Status::kOk) {
        return st;
    }
    // One staging buffer for the one-time upload; destroyed after (the
    // resident path never stages weights again).
    VulkanBuffer upload_staging;
    const Status sc = vulkan_buffer_create(d, upload_staging, bytes,
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           true);
    if (sc != Status::kOk) {
        vulkan_buffer_destroy(d, e.buf);
        return sc;
    }
    const Status up = vulkan_upload(d, upload_staging, e.buf, weights, bytes);
    vulkan_buffer_destroy(d, upload_staging);
    if (up != Status::kOk) {
        vulkan_buffer_destroy(d, e.buf);
        return up;
    }
    // The descriptor set against the matmul's current buffers.
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = cache.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &m.dsl;
    if (vkAllocateDescriptorSets(d.device, &dsai, &e.set) != VK_SUCCESS) {
        vulkan_buffer_destroy(d, e.buf);
        return Status::kInvalidBuffer;
    }
    VkDescriptorBufferInfo infos[3]{};
    infos[0] = {m.in_dev.buffer, 0, VK_WHOLE_SIZE};
    infos[1] = {e.buf.buffer, 0, VK_WHOLE_SIZE};
    infos[2] = {m.out_dev.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t w = 0; w < 3; ++w) {
        writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[w].dstSet = e.set;
        writes[w].dstBinding = w;
        writes[w].descriptorCount = 1;
        writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[w].pBufferInfo = &infos[w];
    }
    vkUpdateDescriptorSets(d.device, 3, writes, 0, nullptr);
    e.key = key;
    e.rows = rows;
    e.cols = cols;
    e.written_in = m.in_dev.buffer;
    e.written_out = m.out_dev.buffer;
    e.written_gen = m.generation;
    e.hash = content_hash;
    *out = &e;
    ++cache.count;
    return Status::kOk;
}

// The resident-weight variant: the entry's descriptor (holding the
// persistent weight buffer) is bound instead of m.set; only the input
// stages. The dispatch and download are identical to vulkan_matmul_run, so
// the results are bit-identical between the two paths.
inline Status vulkan_matmul_run_resident(const VulkanDevice& d, VulkanMatmul& m,
                                         const float* in, int tokens, int c_in,
                                         const VulkanWeightCache::Entry& e,
                                         int rows, float* out,
                                         VulkanMatmulTiming& timing) {
    if (in == nullptr || out == nullptr || e.set == VK_NULL_HANDLE) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || c_in < 1 || rows < 1 || tokens > m.max_tokens
        || c_in > m.max_c_in || rows > m.max_rows
        || e.rows != rows || e.cols != c_in) {
        return Status::kInvalidDimensions;
    }
    // The same dispatch-extent checks as vulkan_matmul_run.
    if (tokens > 65535 || rows > 65535 * 128
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(rows)
               > UINT32_MAX
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(c_in)
               > UINT32_MAX
        || static_cast<std::uint64_t>(rows)
                   * static_cast<std::uint64_t>(c_in)
               > UINT32_MAX) {
        return Status::kInvalidDimensions;
    }
    timing = VulkanMatmulTiming{};
    const std::size_t in_bytes = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(c_in) * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(rows) * sizeof(float);
    {
        const auto t0 = std::chrono::steady_clock::now();
        const Status st = vulkan_upload(d, m.staging, m.in_dev, in, in_bytes);
        const auto t1 = std::chrono::steady_clock::now();
        timing.upload_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (st != Status::kOk) {
            return st;
        }
    }
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &cbi) != VK_SUCCESS) {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return Status::kInvalidBuffer;
    }
    if (m.queries != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m.queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m.queries,
                            0);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m.layout, 0,
                            1, &e.set, 0, nullptr);
    const std::uint32_t push[3] = {static_cast<std::uint32_t>(tokens),
                                   static_cast<std::uint32_t>(c_in),
                                   static_cast<std::uint32_t>(rows)};
    vkCmdPushConstants(cmd, m.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), push);
    vkCmdDispatch(cmd, (push[2] + 127) / 128, push[0], 1);
    if (m.queries != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            m.queries, 1);
    }
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return Status::kInvalidBuffer;
    }
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    bool submitted = false;
    {
        const auto t0 = std::chrono::steady_clock::now();
        submitted = !vulkan_fault_consume(g_vulkan_faults.fail_submit_countdown)
                    && vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE)
                           == VK_SUCCESS;
        submitted = (vkQueueWaitIdle(d.queue) == VK_SUCCESS) && submitted;
        const auto t1 = std::chrono::steady_clock::now();
        timing.kernel_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    if (!submitted) {
        return Status::kInvalidBuffer;
    }
    if (m.queries != VK_NULL_HANDLE && d.has_timestamps) {
        std::uint64_t stamps[2] = {0, 0};
        if (vkGetQueryPoolResults(d.device, m.queries, 0, 2, sizeof(stamps),
                                  stamps, sizeof(stamps[0]),
                                  VK_QUERY_RESULT_64_BIT)
            == VK_SUCCESS) {
            const double ticks = static_cast<double>(stamps[1] - stamps[0]);
            timing.kernel_ms = ticks * static_cast<double>(d.timestamp_period)
                / 1.0e6;
        }
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        const Status st = vulkan_download(d, m.staging, m.out_dev, out,
                                          out_bytes);
        const auto t1 = std::chrono::steady_clock::now();
        timing.download_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (st != Status::kOk) {
            return st;
        }
    }
    return Status::kOk;
}

inline Status vulkan_matmul_run(const VulkanDevice& d, VulkanMatmul& m,
                                const float* in, int tokens, int c_in,
                                const float* weights, int rows, float* out,
                                VulkanMatmulTiming& timing) {
    if (in == nullptr || weights == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || c_in < 1 || rows < 1 || tokens > m.max_tokens
        || c_in > m.max_c_in || rows > m.max_rows) {
        return Status::kInvalidDimensions;
    }
    // The 2D dispatch count must stay within the guaranteed workgroup
    // count limits and the shader's uint32 indexing.
    if (static_cast<std::uint64_t>(tokens) > 65535u
        || static_cast<std::uint64_t>(rows) > (65535u * 128u)
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(rows)
               > UINT32_MAX) {
        return Status::kInvalidDimensions;
    }
    timing = VulkanMatmulTiming{};
    const std::size_t in_bytes = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(c_in) * sizeof(float);
    const std::size_t w_bytes = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(c_in) * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(rows) * sizeof(float);
    double ms = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        Status st = vulkan_upload(d, m.staging, m.in_dev, in, in_bytes);
        if (st == Status::kOk) {
            st = vulkan_upload(d, m.staging, m.w_dev, weights, w_bytes);
        }
        const auto t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing.upload_ms = ms;
        if (st != Status::kOk) {
            return st;
        }
    }
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return Status::kInvalidBuffer;
    }
    if (m.queries != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m.queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m.queries,
                            0);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m.layout, 0,
                            1, &m.set, 0, nullptr);
    const std::uint32_t push[3] = {static_cast<std::uint32_t>(tokens),
                                   static_cast<std::uint32_t>(c_in),
                                   static_cast<std::uint32_t>(rows)};
    vkCmdPushConstants(cmd, m.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), push);
    // 2D dispatch: x = output rows (128-wide), y = tokens.
    vkCmdDispatch(cmd, (push[2] + 127) / 128, push[0], 1);
    if (m.queries != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            m.queries, 1);
    }
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return Status::kInvalidBuffer;
    }
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    bool submitted = false;
    {
        const auto t0 = std::chrono::steady_clock::now();
        submitted = !vulkan_fault_consume(g_vulkan_faults.fail_submit_countdown)
                    && vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE)
                           == VK_SUCCESS;
        submitted = (vkQueueWaitIdle(d.queue) == VK_SUCCESS) && submitted;
        const auto t1 = std::chrono::steady_clock::now();
        timing.kernel_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    if (!submitted) {
        return Status::kInvalidBuffer;
    }
    if (m.queries != VK_NULL_HANDLE && d.has_timestamps) {
        std::uint64_t stamps[2] = {0, 0};
        if (vkGetQueryPoolResults(d.device, m.queries, 0, 2, sizeof(stamps),
                                  stamps, sizeof(stamps[0]),
                                  VK_QUERY_RESULT_64_BIT)
            == VK_SUCCESS) {
            const double ticks =
                static_cast<double>(stamps[1] - stamps[0]);
            timing.kernel_ms = ticks * static_cast<double>(d.timestamp_period)
                / 1.0e6;
        }
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        const Status st = vulkan_download(d, m.staging, m.out_dev, out,
                                          out_bytes);
        const auto t1 = std::chrono::steady_clock::now();
        timing.download_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (st != Status::kOk) {
            return st;
        }
    }
    return Status::kOk;
}

// ---- the projection executor (the G2 eval-dispatch backend) ---------------------------------------

// Binds a VulkanMatmul as a ProjectionExecutor for the eval dispatch's vit
// projection rows. The e4m3 codes are decoded host-side (exact in float32)
// per call; G4 weight residency replaces this with persistent buffers.
struct VulkanProjectionSite {
    VulkanDevice* device = nullptr;
    // G5 weight residency: with a cache set, the first sight of a weight
    // key uploads it once and every later call binds the persistent buffer
    // (no per-run staging). Null keeps the per-call staging path.
    VulkanWeightCache* cache = nullptr;
    VulkanMatmul matmul;         // grow-only: resized to cover row shapes
    std::uint32_t* spirv = nullptr;
    std::size_t spirv_words = 0;
    std::vector<float> decoded;
    std::uint64_t calls = 0;       // executor entries (routing evidence)
    double decode_ms = 0.0;
    double kernel_ms = 0.0;
    double upload_ms = 0.0;
    double download_ms = 0.0;
};

inline void vulkan_projection_close(VulkanProjectionSite& site) {
    if (site.device != nullptr) {
        vulkan_matmul_destroy(*site.device, site.matmul);
    }
    site = VulkanProjectionSite{};
}

// The float-weight variant (the swin family's decoded weights): raw float32
// rows [rows, cols] with no e4m3 codes and no decode/quantisation error. The
// GPU shader is float-native. Bit-parity with the CPU kernel additionally
// assumes grid-exact products (the swin seam validates both operands onto
// the e4m3 grid) and the documented fp32 normal-range contract.
inline Status vulkan_projection_matmul_f32(void* ctx, const float* in,
                                           int tokens, int c_in,
                                           const float* weights, int rows,
                                           float* out) {
    auto* site = static_cast<VulkanProjectionSite*>(ctx);
    if (site == nullptr || site->device == nullptr || site->spirv == nullptr) {
        return Status::kNullPointer;
    }
    if (in == nullptr || weights == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || c_in < 1 || rows < 1) {
        return Status::kInvalidDimensions;
    }
    ++site->calls;
    if (tokens > 65535 || rows > 65535 * 128
        || static_cast<std::uint64_t>(tokens) * rows > UINT32_MAX
        || static_cast<std::uint64_t>(tokens) * c_in > UINT32_MAX
        || static_cast<std::uint64_t>(rows) * c_in > UINT32_MAX) {
        return Status::kInvalidDimensions;
    }
    if (site->matmul.pipeline == VK_NULL_HANDLE
        || tokens > site->matmul.max_tokens || c_in > site->matmul.max_c_in
        || rows > site->matmul.max_rows) {
        int max_tokens = site->matmul.max_tokens > tokens
            ? site->matmul.max_tokens : tokens;
        int max_c_in = site->matmul.max_c_in > c_in ? site->matmul.max_c_in
                                                    : c_in;
        int max_rows = site->matmul.max_rows > rows ? site->matmul.max_rows
                                                    : rows;
        if (max_tokens < 4096) { max_tokens = 4096; }
        if (max_c_in < 1024) { max_c_in = 1024; }
        if (max_rows < 1024) { max_rows = 1024; }
        vulkan_matmul_destroy(*site->device, site->matmul);
        const Status st = vulkan_matmul_create(*site->device, site->matmul,
                                               site->spirv, site->spirv_words,
                                               max_tokens, max_c_in, max_rows);
        if (st != Status::kOk) {
            return st;
        }
    }
    VulkanMatmulTiming timing;
    Status st = Status::kOk;
    if (site->cache != nullptr) {
        VulkanWeightCache::Entry* entry = nullptr;
        const std::uint64_t w_hash = vulkan_weight_hash(
            weights, static_cast<std::size_t>(rows) * c_in
                         * sizeof(float));
        st = vulkan_weight_cache_ensure(*site->cache, *site->device,
                                       site->matmul, weights, weights, rows,
                                       c_in, w_hash, &entry);
        if (st == Status::kOk) {
            st = vulkan_matmul_run_resident(*site->device, site->matmul, in,
                                            tokens, c_in, *entry, rows, out,
                                            timing);
        } else {
            // The cache is an optimisation (capacity or allocation): the
            // staging path stays available and correct.
            st = vulkan_matmul_run(*site->device, site->matmul, in, tokens,
                                   c_in, weights, rows, out, timing);
        }
    } else {
        st = vulkan_matmul_run(*site->device, site->matmul, in, tokens, c_in,
                               weights, rows, out, timing);
    }
    if (st == Status::kOk) {
        site->kernel_ms += timing.kernel_ms;
        site->upload_ms += timing.upload_ms;
        site->download_ms += timing.download_ms;
    }
    return st;
}

inline Status vulkan_projection_matmul(void* ctx, const float* in, int tokens,
                                       int c_in, const E4m3Matrix& w,
                                       float* out) {
    auto* site = static_cast<VulkanProjectionSite*>(ctx);
    if (site == nullptr || site->device == nullptr || site->spirv == nullptr) {
        return Status::kNullPointer;
    }
    if (in == nullptr || w.bits == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || c_in < 1 || w.rows < 1 || w.cols != c_in) {
        return Status::kInvalidDimensions;
    }
    ++site->calls;
    if (tokens > 65535 || w.rows > 65535 * 128
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(w.rows)
               > UINT32_MAX
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(c_in)
               > UINT32_MAX
        || static_cast<std::uint64_t>(w.rows)
                   * static_cast<std::uint64_t>(c_in)
               > UINT32_MAX) {
        return Status::kInvalidDimensions;
    }
    // Grow-only coverage: the plan's row shapes stabilise after a few
    // resizes; the buffers never shrink.
    if (site->matmul.pipeline == VK_NULL_HANDLE
        || tokens > site->matmul.max_tokens || c_in > site->matmul.max_c_in
        || w.rows > site->matmul.max_rows) {
        // Grow-only: never shrink, so alternating row shapes (expand
        // 1024->4096 / contract 4096->1024) settle after at most one
        // rebuild per dimension and stop rebuilding. The old maxima are
        // read before the destroy zeroes them.
        int max_tokens = site->matmul.max_tokens > tokens
            ? site->matmul.max_tokens : tokens;
        int max_c_in = site->matmul.max_c_in > c_in ? site->matmul.max_c_in
                                                    : c_in;
        int max_rows = site->matmul.max_rows > w.rows ? site->matmul.max_rows
                                                      : w.rows;
        if (max_tokens < 4096) { max_tokens = 4096; }
        if (max_c_in < 1024) { max_c_in = 1024; }
        if (max_rows < 1024) { max_rows = 1024; }
        vulkan_matmul_destroy(*site->device, site->matmul);
        const Status st = vulkan_matmul_create(*site->device, site->matmul,
                                               site->spirv, site->spirv_words,
                                               max_tokens, max_c_in, max_rows);
        if (st != Status::kOk) {
            return st;
        }
    }
    VulkanMatmulTiming timing;
    Status st = Status::kOk;
    // The cache key is the CODES pointer (the EvalWeights record source); on
    // a hit the decode is skipped entirely (the resident buffer holds the
    // decoded float matrix).
    VulkanWeightCache::Entry* hit = nullptr;
    if (site->cache != nullptr) {
        for (int i = 0; i < site->cache->count; ++i) {
            VulkanWeightCache::Entry& en = site->cache->entries[i];
            if (en.key == w.bits && en.rows == w.rows && en.cols == w.cols) {
                hit = &en;
                break;
            }
        }
    }
    const std::size_t code_count = static_cast<std::size_t>(w.rows)
        * static_cast<std::size_t>(c_in);
    if (hit != nullptr) {
        vulkan_weight_cache_refresh(*hit, *site->device, site->matmul);
        // Verify the content: a recycled pointer key with different codes
        // must not be served from the resident buffer.
        const std::uint64_t code_hash = vulkan_weight_hash(w.bits, code_count);
        if (hit->hash == code_hash) {
            st = vulkan_matmul_run_resident(*site->device, site->matmul, in,
                                            tokens, c_in, *hit, w.rows, out,
                                            timing);
        } else {
            // Content changed behind the pointer: decode and re-upload.
            const auto d0 = std::chrono::steady_clock::now();
            site->decoded.resize(code_count);
            for (std::size_t i = 0; i < code_count; ++i) {
                site->decoded[i] = e4m3_from_bits(w.bits[i]);
            }
            const auto d1 = std::chrono::steady_clock::now();
            site->decode_ms +=
                std::chrono::duration<double, std::milli>(d1 - d0).count();
            const Status up = vulkan_weight_cache_update(*site->device, *hit,
                                                         site->decoded.data(),
                                                         code_hash);
            if (up == Status::kOk) {
                st = vulkan_matmul_run_resident(*site->device, site->matmul,
                                                in, tokens, c_in, *hit,
                                                w.rows, out, timing);
            } else {
                st = up;
            }
        }
    } else {
        const std::size_t count = static_cast<std::size_t>(w.rows)
            * static_cast<std::size_t>(c_in);
        const auto d0 = std::chrono::steady_clock::now();
        site->decoded.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            site->decoded[i] = e4m3_from_bits(w.bits[i]);
        }
        const auto d1 = std::chrono::steady_clock::now();
        site->decode_ms +=
            std::chrono::duration<double, std::milli>(d1 - d0).count();
        if (site->cache != nullptr) {
            VulkanWeightCache::Entry* entry = nullptr;
            const std::uint64_t code_hash = vulkan_weight_hash(
                w.bits, code_count);
            st = vulkan_weight_cache_ensure(*site->cache, *site->device,
                                           site->matmul, w.bits,
                                           site->decoded.data(), w.rows,
                                           w.cols, code_hash, &entry);
            if (st == Status::kOk) {
                st = vulkan_matmul_run_resident(*site->device, site->matmul,
                                                in, tokens, c_in, *entry,
                                                w.rows, out, timing);
            } else {
                // Cache exhaustion or allocation failure: the staging path.
                st = vulkan_matmul_run(*site->device, site->matmul, in, tokens,
                                       c_in, site->decoded.data(), w.rows, out,
                                       timing);
            }
        } else {
            st = vulkan_matmul_run(*site->device, site->matmul, in, tokens,
                                   c_in, site->decoded.data(), w.rows, out,
                                   timing);
        }
    }
    site->kernel_ms += timing.kernel_ms;
    site->upload_ms += timing.upload_ms;
    site->download_ms += timing.download_ms;
    return st;
}

inline ProjectionExecutor vulkan_projection_executor(VulkanProjectionSite*
                                                         site) {
    ProjectionExecutor ex;
    ex.ctx = site;
    ex.matmul = &vulkan_projection_matmul;
    ex.matmul_f32 = &vulkan_projection_matmul_f32;
    return ex;
}

#endif  // DLSS5_NR_HAS_VULKAN

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_VULKAN_HPP
