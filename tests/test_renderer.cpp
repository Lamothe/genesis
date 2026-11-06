// Standalone offscreen renderer regression checks; requires a Vulkan 1.3 device.
// After run-editor.sh compiles the shaders:
// g++ -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude -Iapp tests/test_renderer.cpp src/camera.cpp src/handles.cpp src/triangle_sort.cpp src/mesh.cpp src/image.cpp app/geometry.cpp app/vk_renderer.cpp -lvulkan -lxcb -lz -o .build/test-renderer
// .build/test-renderer
#include "vk_renderer.h"

#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <new>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

// Fail one C++ allocation on the calling thread, without affecting driver
// worker threads. The injection is disabled outside the registration call.
thread_local int allocationsBeforeFailure = -1;

void* operator new(std::size_t size) {
  if (allocationsBeforeFailure == 0) {
    allocationsBeforeFailure = -1;
    throw std::bad_alloc();
  }
  if (allocationsBeforeFailure > 0) {
    --allocationsBeforeFailure;
  }
  if (void* p = std::malloc(size == 0 ? 1 : size)) {
    return p;
  }
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
__attribute__((noinline)) void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// Interpose only waits on this test thread. A simulated device loss first
// performs the real wait, so it cannot cause a synthetic GPU use-after-free.
thread_local VkResult fenceWaitFault = VK_SUCCESS;
thread_local int fenceWaitFaultSkip = 0;
thread_local VkResult queueWaitFault = VK_SUCCESS;
thread_local VkResult deviceWaitFault = VK_SUCCESS;
thread_local int deviceWaitCalls = 0;
thread_local int deviceWaitFaultSkip = 0;

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(
    VkDevice device, uint32_t count, const VkFence* fences, VkBool32 all, uint64_t timeout) {
  auto real = reinterpret_cast<PFN_vkWaitForFences>(dlsym(RTLD_NEXT, "vkWaitForFences"));
  if (fenceWaitFault != VK_SUCCESS && fenceWaitFaultSkip-- <= 0) {
    VkResult fault = fenceWaitFault;
    fenceWaitFault = VK_SUCCESS;
    if (fault == VK_ERROR_DEVICE_LOST && real(device, count, fences, all, timeout) != VK_SUCCESS) { std::abort(); }
    return fault;
  }
  return real(device, count, fences, all, timeout);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue queue) {
  auto real = reinterpret_cast<PFN_vkQueueWaitIdle>(dlsym(RTLD_NEXT, "vkQueueWaitIdle"));
  if (queueWaitFault != VK_SUCCESS) {
    VkResult fault = queueWaitFault;
    queueWaitFault = VK_SUCCESS;
    if (fault == VK_ERROR_DEVICE_LOST && real(queue) != VK_SUCCESS) { std::abort(); }
    return fault;
  }
  return real(queue);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice device) {
  auto real = reinterpret_cast<PFN_vkDeviceWaitIdle>(dlsym(RTLD_NEXT, "vkDeviceWaitIdle"));
  ++deviceWaitCalls;
  if (deviceWaitFault != VK_SUCCESS && deviceWaitFaultSkip-- <= 0) {
    VkResult fault = deviceWaitFault;
    deviceWaitFault = VK_SUCCESS;
    if (fault == VK_ERROR_DEVICE_LOST && real(device) != VK_SUCCESS) { std::abort(); }
    return fault;
  }
  return real(device);
}

using namespace vredit;

namespace {

MeshVertex vertex(Vec3 pos, Vec3 normal, Vec3 colour) {
  return {pos, normal, colour, {0.0F, 0.0F}};
}

bool render(VkRenderer& renderer, Camera const& camera, DrawItem const& item,
            std::vector<std::uint8_t>& pixels) {
  return renderer.draw(camera, {item}, {}, {}) && renderer.readback(pixels);
}

bool testWaitFailures() {
  VkRenderer::Config config;
  config.width = 96;
  config.height = 96;
  VkRenderer renderer;
  if (!renderer.init(config)) { return false; }
  MeshData mesh;
  mesh.vertices = {vertex({-0.8F, -0.8F, -3}, {0, 0, 1}, {1, 1, 1}),
                   vertex({0.8F, -0.8F, -3}, {0, 0, 1}, {1, 1, 1}),
                   vertex({0, 0.8F, -3}, {0, 0, 1}, {1, 1, 1})};
  mesh.indices = {0, 1, 2};
  Image image;
  image.width = 1;
  image.height = 1;
  image.rgba = {0, 255, 0, 255};

  // Registration's first device wait succeeds; the second is the fallback
  // after a successful submit and an injected queue-wait failure.
  deviceWaitCalls = 0;
  queueWaitFault = VK_ERROR_OUT_OF_HOST_MEMORY;
  if (renderer.addMesh(mesh) != kInvalidMesh || deviceWaitCalls != 2 ||
      renderer.hasFatalError() || renderer.addMesh(mesh) == kInvalidMesh) { return false; }
  deviceWaitCalls = 0;
  queueWaitFault = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  if (renderer.addTexture(image, true, true) != kInvalidTexture || deviceWaitCalls != 2 ||
      renderer.hasFatalError() || renderer.addTexture(image, true, true) == kInvalidTexture) { return false; }

  auto checkpoint = renderer.checkpointResources();
  if (!checkpoint || renderer.addMesh(mesh) == kInvalidMesh) { return false; }
  deviceWaitFault = VK_ERROR_OUT_OF_HOST_MEMORY;
  if (renderer.rollbackResources(checkpoint) || renderer.hasFatalError() ||
      !renderer.rollbackResources(checkpoint)) { return false; }

  checkpoint = renderer.checkpointResources();
  if (!checkpoint) { return false; }
  deviceWaitFault = VK_ERROR_DEVICE_LOST;
  if (renderer.rollbackResources(checkpoint) || !renderer.hasFatalError() ||
      renderer.rollbackResources(checkpoint) || renderer.commitResources(checkpoint) ||
      renderer.addMesh(mesh) != kInvalidMesh || renderer.addTexture(image, true, true) != kInvalidTexture) {
    return false;
  }
  renderer.shutdown();

  VkRenderer uploadLost;
  if (!uploadLost.init(config)) { return false; }
  queueWaitFault = VK_ERROR_DEVICE_LOST;
  if (uploadLost.addMesh(mesh) != kInvalidMesh || !uploadLost.hasFatalError() ||
      uploadLost.addMesh(mesh) != kInvalidMesh) { return false; }
  uploadLost.shutdown();
  // An upload cannot release its GPU objects if both retirement waits fail.
  // The child must abort rather than return and unwind those objects.
  pid_t child = fork();
  if (child < 0) { return false; }
  if (child == 0) {
    rlimit noCore{0, 0};
    setrlimit(RLIMIT_CORE, &noCore);
    VkRenderer failedWait;
    if (!failedWait.init(config)) { _exit(2); }
    queueWaitFault = VK_ERROR_OUT_OF_HOST_MEMORY;
    deviceWaitFault = VK_ERROR_OUT_OF_HOST_MEMORY;
    deviceWaitFaultSkip = 1; // registration's initial wait must succeed
    failedWait.addMesh(mesh);
    _exit(3);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFSIGNALED(status) ||
      WTERMSIG(status) != SIGABRT) { return false; }
  return true;
}

bool testRegistrationAllocationFailures() {
  VkRenderer::Config config;
  config.width = 96;
  config.height = 96;
  MeshData mesh;
  mesh.vertices = {vertex({-0.8F, -0.8F, -3}, {0, 0, 1}, {1, 1, 1}),
                   vertex({0.8F, -0.8F, -3}, {0, 0, 1}, {1, 1, 1}),
                   vertex({0, 0.8F, -3}, {0, 0, 1}, {1, 1, 1})};
  mesh.indices = {0, 1, 2};
  Image image;
  image.width = 1;
  image.height = 1;
  image.rgba = {0, 255, 0, 255};
  Camera camera;
  camera.position = {0, 0, 0};
  camera.pitch = 0;
  for (int scenario = 0; scenario < 4; ++scenario) {
    VkRenderer renderer;
    if (!renderer.init(config)) { return false; }
    if (scenario == 0) {
      bool checkpointThrew = false;
      allocationsBeforeFailure = 0;
      try {
        renderer.checkpointResources();
      } catch (std::bad_alloc const&) {
        checkpointThrew = true;
      }
      allocationsBeforeFailure = -1;
      if (!checkpointThrew) { return false; }
    }
    auto checkpoint = renderer.checkpointResources();
    if (!checkpoint) { return false; }
    bool threw = false;
    // Mesh: indices, triangle sums, then vector growth after GPU upload.
    // Texture: the first copy-region allocation or the texture-vector
    // growth after upload; hand: mesh-vector growth after buffer allocation.
    allocationsBeforeFailure = scenario == 0 ? 2 : (scenario == 3 ? 1 : 0);
    try {
      if (scenario == 0) { renderer.addMesh(mesh, true); }
      if (scenario == 1 || scenario == 3) { renderer.addTexture(image, true, true); }
      if (scenario == 2) { renderer.updateHandMesh(0, mesh); }
    } catch (std::bad_alloc const&) {
      threw = true;
    }
    allocationsBeforeFailure = -1;
    if (!threw) { return false; }
    auto const handle = scenario == 2 ? renderer.updateHandMesh(0, mesh)
                                      : renderer.addMesh(mesh, true);
    auto const texture = renderer.addTexture(image, true, true);
    DrawItem item;
    item.mesh = handle;
    item.texture = texture;
    item.lit = false;
    std::vector<std::uint8_t> pixels;
    auto const centre = (48U * config.width + 48U) * 4U;
    if (handle != 1U || texture != 1U || !render(renderer, camera, item, pixels) ||
        pixels[centre + 1] < 240 || !renderer.rollbackResources(checkpoint)) {
      return false;
    }
  }
  return true;
}

bool testResourceRollback() {
  VkRenderer renderer;
  VkRenderer other;
  VkRenderer::ResourceCheckpoint empty;
  if (renderer.checkpointResources() || renderer.commitResources(empty) ||
      renderer.rollbackResources(empty)) {
    return false;
  }
  VkRenderer::Config config;
  config.width = 96;
  config.height = 96;
  if (!renderer.init(config) || !other.init(config)) { return false; }
  Camera camera;
  camera.position = {0, 0, 0};
  camera.pitch = 0;
  MeshData mesh;
  mesh.vertices = {vertex({-0.8F, -0.8F, -3}, {0, 0, 1}, {1, 1, 1}),
                   vertex({0.8F, -0.8F, -3}, {0, 0, 1}, {1, 1, 1}),
                   vertex({0, 0.8F, -3}, {0, 0, 1}, {1, 1, 1})};
  mesh.indices = {0, 1, 2};
  Image red;
  red.width = 1;
  red.height = 1;
  red.rgba = {255, 0, 0, 255};
  Image green = red;
  green.rgba = {0, 255, 0, 255};
  Image blue = red;
  blue.rgba = {0, 0, 255, 255};
  auto const oldMesh = renderer.addMesh(mesh, true);
  auto const oldTexture = renderer.addTexture(red, true, true);
  auto const oldHand = renderer.updateHandMesh(0, mesh);
  DrawItem item;
  item.mesh = oldMesh;
  item.texture = oldTexture;
  item.lit = false;
  std::vector<std::uint8_t> before;
  std::vector<std::uint8_t> pixels;
  auto const centre = (48U * config.width + 48U) * 4U;
  if (oldMesh == kInvalidMesh || oldTexture == kInvalidTexture || oldHand == kInvalidMesh ||
      !render(renderer, camera, item, before) || before[centre] < 240 || before[centre + 1] > 10) {
    return false;
  }

  auto checkpoint = renderer.checkpointResources();
  auto copy = checkpoint;
  auto rejected = renderer.checkpointResources();
  auto foreign = other.checkpointResources();
  if (!checkpoint || rejected || !foreign || renderer.commitResources(rejected) ||
      renderer.rollbackResources(foreign) || other.rollbackResources(checkpoint)) {
    return false;
  }
  MeshData changedHand = mesh;
  for (auto& v : changedHand.vertices) { v.colour = {0, 1, 0}; }
  if (renderer.updateHandMesh(0, changedHand) != oldHand) { return false; }
  auto const transient = renderer.addMesh(mesh, true);
  auto const hand = renderer.updateHandMesh(1, mesh);
  auto const transientStatic = renderer.addMesh(mesh);
  auto const greenTexture = renderer.addTexture(green, true, true);
  auto const blueTexture = renderer.addTexture(blue, false, false);
  if (transient == kInvalidMesh || hand == kInvalidMesh || transientStatic == kInvalidMesh ||
      greenTexture == kInvalidTexture || blueTexture == kInvalidTexture ||
      renderer.addMesh(MeshData{}) != kInvalidMesh ||
      renderer.addTexture(Image{}, true, true) != kInvalidTexture ||
      !renderer.updateMeshVertices(transient, mesh.vertices)) {
    return false;
  }
  item.mesh = transient;
  item.texture = blueTexture;
  if (!render(renderer, camera, item, pixels) || pixels[centre + 2] < 240 ||
      !renderer.updateMeshVertices(oldMesh, mesh.vertices) ||
      !renderer.updateMeshVertices(transient, mesh.vertices) ||
      !renderer.rollbackResources(copy) || renderer.rollbackResources(checkpoint) ||
      renderer.commitResources(checkpoint) || renderer.updateMeshVertices(transient, mesh.vertices)) {
    return false;
  }
  item.mesh = oldMesh;
  item.texture = oldTexture;
  if (!render(renderer, camera, item, pixels) || pixels != before) { return false; }
  item.mesh = oldHand;
  item.texture = 0;
  if (!render(renderer, camera, item, pixels) || pixels[centre + 1] < 240 || pixels[centre] > 10 ||
      renderer.updateHandMesh(0, mesh) != oldHand) {
    return false;
  }
  // A removed texture handle resolves to white until its slot is reused.
  item.mesh = oldMesh;
  item.texture = greenTexture;
  if (!render(renderer, camera, item, pixels) || pixels[centre] < 240 ||
      pixels[centre + 1] < 240 || pixels[centre + 2] < 240) {
    return false;
  }

  // The removed handles are reusable, with the descriptor now sampling the
  // newly registered image rather than either a stale image or white fallback.
  auto const replacement = renderer.addMesh(mesh, true);
  auto const replacementTexture = renderer.addTexture(green, true, true);
  auto const replacementHand = renderer.updateHandMesh(1, mesh);
  if (replacement != transient || replacementTexture != greenTexture || replacementHand != hand) {
    return false;
  }
  item.mesh = replacement;
  item.texture = replacementTexture;
  if (!render(renderer, camera, item, pixels) || pixels[centre + 1] < 240 || pixels[centre] > 10) {
    return false;
  }
  item.mesh = oldMesh;
  item.texture = oldTexture;
  if (!render(renderer, camera, item, pixels) || pixels != before) { return false; }

  auto committed = renderer.checkpointResources();
  auto const keptMesh = renderer.addMesh(mesh);
  auto const keptTexture = renderer.addTexture(blue, true, true);
  if (!committed || keptMesh == kInvalidMesh || keptTexture == kInvalidTexture ||
      !renderer.commitResources(committed) || renderer.commitResources(committed) ||
      renderer.rollbackResources(committed)) {
    return false;
  }
  auto next = renderer.checkpointResources();
  if (!next || renderer.rollbackResources(committed) || !renderer.rollbackResources(next)) {
    return false;
  }
  item.mesh = keptMesh;
  item.texture = keptTexture;
  if (!render(renderer, camera, item, pixels) || pixels[centre + 2] < 240 ||
      pixels[centre] > 10 || !other.rollbackResources(foreign)) {
    return false;
  }
  // A rolled-back blended mesh may leave sorted GPU indices in a scratch
  // slot. Reusing its handle for a mesh with opposite depth order must sort
  // again, not replay the old mesh's indices.
  MeshData layers;
  layers.vertices = {vertex({-0.8F, -0.8F, -2}, {0, 0, 1}, {1, 0, 0}),
                     vertex({0.8F, -0.8F, -2}, {0, 0, 1}, {1, 0, 0}),
                     vertex({0, 0.8F, -2}, {0, 0, 1}, {1, 0, 0}),
                     vertex({-1.2F, -1.2F, -3}, {0, 0, 1}, {0, 0, 1}),
                     vertex({1.2F, -1.2F, -3}, {0, 0, 1}, {0, 0, 1}),
                     vertex({0, 1.2F, -3}, {0, 0, 1}, {0, 0, 1})};
  layers.indices = {0, 1, 2, 3, 4, 5};
  auto blendCheckpoint = renderer.checkpointResources();
  DrawItem blended;
  blended.mesh = renderer.addMesh(layers);
  blended.alpha = AlphaMode::Blend;
  blended.alphaFactor = 0.5F;
  blended.lit = false;
  if (!blendCheckpoint || blended.mesh == kInvalidMesh ||
      !render(renderer, camera, blended, pixels) ||
      pixels[centre] < pixels[centre + 2] + 40 ||
      !renderer.rollbackResources(blendCheckpoint)) {
    return false;
  }
  auto const removedBlend = blended.mesh;
  layers.indices = {3, 4, 5, 0, 1, 2};
  blended.mesh = renderer.addMesh(layers);
  if (blended.mesh != removedBlend || !render(renderer, camera, blended, pixels) ||
      pixels[centre] < pixels[centre + 2] + 40) {
    return false;
  }

  // Repeated full-capacity transactions prove all texture slots can be
  // reclaimed, including the last shader descriptor and failed registrations.
  for (int pass = 0; pass < 2; ++pass) {
    auto full = renderer.checkpointResources();
    if (!full) { return false; }
    TextureHandle last = 0;
    TextureHandle highest = 0;
    do {
      last = renderer.addTexture(green, false, false);
      if (last != kInvalidTexture) { highest = last; }
    } while (last != kInvalidTexture);
    if (highest != 255U || !renderer.rollbackResources(full)) { return false; }
    item.mesh = oldMesh;
    item.texture = oldTexture;
    if (!render(renderer, camera, item, pixels) || pixels != before) { return false; }
  }
  auto stale = renderer.checkpointResources();
  renderer.shutdown();
  if (renderer.commitResources(stale) || renderer.rollbackResources(stale)) { return false; }
  return true;
}

} // namespace

bool testExternalWaitFailures();
bool testVertexUpdateAllocationFailure();

int main() {
  if (!testExternalWaitFailures() || !testVertexUpdateAllocationFailure()) {
    std::printf("renderer test: XR retirement or update transaction failed\n");
    return 1;
  }
  if (!testWaitFailures()) {
    std::printf("renderer test: Vulkan wait failure handling failed\n");
    return 1;
  }
  std::printf("renderer tests: Vulkan wait failure handling passed\n");
  if (!testRegistrationAllocationFailures()) {
    std::printf("renderer test: registration allocation failure recovery failed\n");
    return 1;
  }
  std::printf("renderer tests: registration allocation failure recovery passed\n");
  if (!testResourceRollback()) {
    std::printf("renderer test: resource checkpoint/rollback failed\n");
    return 1;
  }
  std::printf("renderer tests: resource checkpoint/rollback passed\n");
  VkRenderer renderer;
  VkRenderer::Config config;
  config.width = 96;
  config.height = 96;
  if (!renderer.init(config)) {
    std::printf("renderer test: init failed\n");
    return 1;
  }
  Camera camera;
  camera.position = {0.0F, 0.0F, 0.0F};
  camera.pitch = 0.0F;

  // Compare an object-space scaled draw against identical pre-transformed
  // geometry with explicitly inverse-transposed normals. This exercises the
  // actual vertex shader rather than a CPU copy of its normal maths.
  MeshData source;
  Vec3 const normal{1.0F, 1.0F, 1.0F};
  source.vertices = {vertex({-0.7F, -0.7F, 0.0F}, normal, {1, 1, 1}),
                     vertex({0.7F, -0.7F, 0.0F}, normal, {1, 1, 1}),
                     vertex({0.0F, 0.7F, 0.0F}, normal, {1, 1, 1})};
  source.indices = {0, 1, 2};
  MeshData baked = source;
  for (auto& v : baked.vertices) {
    v.pos.x *= 2.0F;
    v.normal = {0.5F, 1.0F, 1.0F};
  }
  auto const sourceHandle = renderer.addMesh(source);
  auto const bakedHandle = renderer.addMesh(baked);
  if (sourceHandle == kInvalidMesh || bakedHandle == kInvalidMesh) {
    return 1;
  }
  DrawItem item;
  item.mesh = sourceHandle;
  item.model = mul(translationMatrix({0, 0, -3}), scaleMatrix({2, 1, 1}));
  std::vector<std::uint8_t> scaledPixels;
  std::vector<std::uint8_t> bakedPixels;
  if (!render(renderer, camera, item, scaledPixels)) {
    return 1;
  }
  item.mesh = bakedHandle;
  item.model = translationMatrix({0, 0, -3});
  if (!render(renderer, camera, item, bakedPixels)) {
    return 1;
  }
  auto const centre = (48U * config.width + 48U) * 4U;
  if (scaledPixels[centre] < 80 ||
      std::abs(static_cast<int>(scaledPixels[centre]) - static_cast<int>(bakedPixels[centre])) > 1) {
    std::printf("renderer test: nonuniform normal mismatch %u vs %u\n",
                scaledPixels[centre], bakedPixels[centre]);
    return 1;
  }

  // Render the actual shader with opposed light directions and ambient fill.
  MeshData lightMesh = source;
  for (auto& v : lightMesh.vertices) { v.normal = {0, 0, 1}; v.colour = {.5F, .5F, .5F}; }
  DrawItem lightItem;
  lightItem.mesh = renderer.addMesh(lightMesh);
  lightItem.model = translationMatrix({0, 0, -3});
  SceneLighting lighting;
  lighting.direction = {0, 0, 1}; lighting.ambient = 0.1F;
  renderer.setLighting(lighting);
  std::vector<std::uint8_t> lit, dark, ambient, unlitA, unlitB;
  if (lightItem.mesh == kInvalidMesh || !render(renderer, camera, lightItem, lit)) { return 1; }
  lightItem.model = mul(translationMatrix({0, 0, -3}), scaleMatrix({-1, 1, 1}));
  std::vector<std::uint8_t> mirroredLighting;
  if (!render(renderer, camera, lightItem, mirroredLighting) ||
      std::abs(static_cast<int>(mirroredLighting[centre]) - static_cast<int>(lit[centre])) > 1) {
    std::printf("renderer test: mirrored face lighting mismatch %u vs %u\n", lit[centre], mirroredLighting[centre]);
    return 1;
  }
  lightItem.model = translationMatrix({0, 0, -3});
  lighting.direction = {0, 0, -1}; renderer.setLighting(lighting);
  if (!render(renderer, camera, lightItem, dark)) { return 1; }
  lighting.ambient = 0.6F; renderer.setLighting(lighting);
  if (!render(renderer, camera, lightItem, ambient)) { return 1; }
  if (lit[centre] <= dark[centre] + 70 || ambient[centre] <= dark[centre] + 40 ||
      lit[centre] <= ambient[centre] + 30) {
    std::printf("renderer test: lighting not applied: %u %u %u\n", lit[centre], dark[centre], ambient[centre]);
    return 1;
  }
  std::printf("renderer lighting RGB centre: directional=%u opposed=%u ambient=%u\n", lit[centre], dark[centre], ambient[centre]);
  lightItem.lit = false;
  if (!render(renderer, camera, lightItem, unlitA)) { return 1; }
  renderer.setLighting({});
  if (!render(renderer, camera, lightItem, unlitB) || unlitA != unlitB) {
    std::printf("renderer test: unlit material changed with lighting\n"); return 1;
  }

  // Vertex alpha participates in BLEND and MASK, but OPAQUE ignores it.
  MeshData alphaMesh = source;
  for (auto& v : alphaMesh.vertices) { v.alpha = 0.25F; }
  item = DrawItem{};
  item.mesh = renderer.addMesh(alphaMesh);
  item.model = translationMatrix({0, 0, -3});
  item.lit = false;
  if (item.mesh == kInvalidMesh) { return 1; }
  std::vector<std::uint8_t> opaqueAlpha, maskAlpha, blendAlpha, blankAlpha;
  if (!render(renderer, camera, item, opaqueAlpha)) { return 1; }
  item.alpha = AlphaMode::Mask; item.alphaCutoff = 0.5F;
  if (!render(renderer, camera, item, maskAlpha)) { return 1; }
  item.alpha = AlphaMode::Blend;
  if (!render(renderer, camera, item, blendAlpha)) { return 1; }
  for (auto& v : alphaMesh.vertices) { v.alpha = 0.0F; }
  item.mesh = renderer.addMesh(alphaMesh);
  if (item.mesh == kInvalidMesh || !render(renderer, camera, item, blankAlpha)) { return 1; }
  if (opaqueAlpha[centre] < 240 || maskAlpha[centre] != blankAlpha[centre] ||
      blendAlpha[centre] <= blankAlpha[centre] + 30 ||
      blendAlpha[centre] >= opaqueAlpha[centre] - 100) {
    std::printf("renderer test: vertex alpha modes opaque=%u mask=%u blend=%u blank=%u\n",
                opaqueAlpha[centre], maskAlpha[centre], blendAlpha[centre], blankAlpha[centre]);
    return 1;
  }

  // Selection adds a neutral, grazing-only rim, not a warm fill. Use a
  // deliberately edge-on normal and compare actual shader output at centre.
  MeshData rimMesh = source;
  for (auto& v : rimMesh.vertices) { v.normal = {1, 0, 0}; v.colour = {.15F, .15F, .15F}; }
  item = DrawItem{};
  item.mesh = renderer.addMesh(rimMesh);
  item.model = translationMatrix({0, 0, -3});
  item.lit = false;
  std::vector<std::uint8_t> rimBase;
  std::vector<std::uint8_t> rimSelected;
  if (item.mesh == kInvalidMesh || !render(renderer, camera, item, rimBase)) { return 1; }
  item.highlight = 5.0F;
  if (!render(renderer, camera, item, rimSelected)) { return 1; }
  int const dr = static_cast<int>(rimSelected[centre]) - rimBase[centre];
  int const dg = static_cast<int>(rimSelected[centre+1]) - rimBase[centre+1];
  int const db = static_cast<int>(rimSelected[centre+2]) - rimBase[centre+2];
  if (dr < 20 || std::abs(dr-dg) > 2 || std::abs(dr-db) > 2) {
    std::printf("renderer test: selection rim not neutral (%d %d %d)\n", dr, dg, db);
    return 1;
  }
  for (auto& v : rimMesh.vertices) { v.normal = {0, 0, 1}; }
  item.mesh = renderer.addMesh(rimMesh);
  item.highlight = 0.0F;
  if (item.mesh == kInvalidMesh || !render(renderer, camera, item, rimBase)) { return 1; }
  item.highlight = 5.0F;
  if (!render(renderer, camera, item, rimSelected) ||
      std::abs(static_cast<int>(rimSelected[centre]) - rimBase[centre]) > 1) {
    std::printf("renderer test: selection adds face fill\n");
    return 1;
  }

  // Mirroring changes the determinant sign as well as the normal direction.
  item = DrawItem{};
  MeshData mirrored = source;
  for (auto& v : mirrored.vertices) {
    v.pos.x *= -2.0F;
    v.normal = {-0.5F, 1.0F, 1.0F};
  }
  // A baked mirror must reverse its winding to preserve the original front.
  std::swap(mirrored.indices[1], mirrored.indices[2]);
  auto const mirrorHandle = renderer.addMesh(mirrored);
  item.mesh = sourceHandle;
  item.model = mul(translationMatrix({0, 0, -3}), scaleMatrix({-2, 1, 1}));
  if (mirrorHandle == kInvalidMesh || !render(renderer, camera, item, scaledPixels)) {
    return 1;
  }
  item.mesh = mirrorHandle;
  item.model = translationMatrix({0, 0, -3});
  if (!render(renderer, camera, item, bakedPixels) || scaledPixels[centre] < 80 ||
      std::abs(static_cast<int>(scaledPixels[centre]) - static_cast<int>(bakedPixels[centre])) > 1) {
    std::printf("renderer test: mirrored normal mismatch\n");
    return 1;
  }

  // A flattened model has no inverse. It should use the finite forward
  // transform fallback, including when the model is singular.
  MeshData flattened = source;
  for (auto& v : flattened.vertices) {
    v.normal = {1.0F, 1.0F, 0.0F};
  }
  auto const flatHandle = renderer.addMesh(flattened);
  item.mesh = sourceHandle;
  item.model = mul(translationMatrix({0, 0, -3}), scaleMatrix({1, 1, 0}));
  if (flatHandle == kInvalidMesh || !render(renderer, camera, item, scaledPixels)) {
    return 1;
  }
  item.mesh = flatHandle;
  item.model = translationMatrix({0, 0, -3});
  if (!render(renderer, camera, item, bakedPixels) || scaledPixels[centre] < 80 ||
      std::abs(static_cast<int>(scaledPixels[centre]) - static_cast<int>(bakedPixels[centre])) > 1) {
    std::printf("renderer test: singular normal fallback failed\n");
    return 1;
  }

  // Near triangle deliberately precedes far triangle in the source indices.
  // Correct blend index upload and sort puts the red near triangle on top.
  MeshData blend;
  blend.vertices = {vertex({-0.8F, -0.8F, -2.0F}, normal, {1, 0, 0}),
                    vertex({0.8F, -0.8F, -2.0F}, normal, {1, 0, 0}),
                    vertex({0.0F, 0.8F, -2.0F}, normal, {1, 0, 0}),
                    vertex({-1.2F, -1.2F, -3.0F}, normal, {0, 0, 1}),
                    vertex({1.2F, -1.2F, -3.0F}, normal, {0, 0, 1}),
                    vertex({0.0F, 1.2F, -3.0F}, normal, {0, 0, 1})};
  blend.indices = {0, 1, 2, 3, 4, 5};
  item = DrawItem{};
  item.mesh = renderer.addMesh(blend);
  item.model = Mat4{};
  item.alpha = AlphaMode::Blend;
  item.alphaFactor = 0.5F;
  item.lit = false;
  if (item.mesh == kInvalidMesh) {
    return 1;
  }
  std::vector<std::uint8_t> pixels;
  for (int frame = 0; frame < 3; ++frame) {
    if (frame == 2) {
      camera.position.z = 0.1F; // leave the reuse deadband and upload again
    }
    if (!render(renderer, camera, item, pixels) || pixels[centre] < pixels[centre + 2] + 40) {
      std::printf("renderer test: blend ordering failed on frame %d\n", frame);
      return 1;
    }
  }
  // Centre Stage hair treatment: at alpha 0.5, k=3 yields 0.875 per
  // fragment. Compare the actual GPU output with the authored-alpha path.
  MeshData opacity = source;
  for (auto& v : opacity.vertices) { v.colour = {1, 1, 1}; }
  DrawItem opacityItem;
  opacityItem.mesh = renderer.addMesh(opacity);
  opacityItem.model = translationMatrix({0, 0, -3});
  opacityItem.alpha = AlphaMode::Blend;
  opacityItem.alphaFactor = 0.5F;
  opacityItem.lit = false;
  std::vector<std::uint8_t> unboosted;
  if (opacityItem.mesh == kInvalidMesh || !render(renderer, camera, opacityItem, unboosted)) {
    return 1;
  }
  opacityItem.hairOpacityBoost = true;
  if (!render(renderer, camera, opacityItem, pixels) ||
      pixels[centre] < unboosted[centre] + 70 || pixels[centre] < 215 || pixels[centre] > 230) {
    std::printf("renderer test: hair opacity boost failed (%u -> %u)\n", unboosted[centre], pixels[centre]);
    return 1;
  }
  opacityItem.alpha = AlphaMode::Opaque;
  if (!render(renderer, camera, opacityItem, pixels) || pixels[centre] < 250) {
    std::printf("renderer test: hair boost changed opaque mode\n");
    return 1;
  }
  opacityItem.alpha = AlphaMode::Mask;
  opacityItem.alphaCutoff = 0.6F;
  opacityItem.hairOpacityBoost = false;
  if (!render(renderer, camera, opacityItem, unboosted)) { return 1; }
  opacityItem.hairOpacityBoost = true;
  if (!render(renderer, camera, opacityItem, pixels) || pixels[centre] != unboosted[centre] ||
      pixels[centre] > 30) {
    std::printf("renderer test: hair boost changed mask cutoff\n");
    return 1;
  }
  // Two stable hand slots update in place and share scene depth.
  camera.position.z = 0.0F;
  MeshData hand;
  hand.vertices = {vertex({-0.8F, -0.8F, -3.0F}, normal, {1, 0, 0}),
                   vertex({0.8F, -0.8F, -3.0F}, normal, {1, 0, 0}),
                   vertex({0.0F, 0.8F, -3.0F}, normal, {1, 0, 0})};
  hand.indices = {0, 1, 2};
  auto const handHandle = renderer.updateHandMesh(0, hand);
  auto const otherHandle = renderer.updateHandMesh(1, hand);
  if (handHandle == kInvalidMesh || otherHandle == kInvalidMesh || handHandle == otherHandle ||
      renderer.updateHandMesh(2, hand) != kInvalidMesh) {
    std::printf("renderer test: hand slots failed\n");
    return 1;
  }
  MeshData occluder = hand;
  for (auto& v : occluder.vertices) {
    v.pos.x *= 4.0F / 3.0F;
    v.pos.y *= 4.0F / 3.0F;
    v.pos.z = -4.0F;
    v.colour = {0, 0, 1};
  }
  DrawItem sceneItem;
  sceneItem.mesh = renderer.addMesh(occluder);
  sceneItem.lit = false;
  DrawItem handItem;
  handItem.mesh = handHandle;
  handItem.lit = false;
  handItem.alpha = AlphaMode::Blend; // forced opaque
  handItem.alphaFactor = 0.1F;
  auto drawHand = [&]() {
    // Hand listed first deliberately; opaque scene must be drawn first.
    return renderer.draw(camera, {handItem, sceneItem}, {}, {}) && renderer.readback(pixels);
  };
  if (sceneItem.mesh == kInvalidMesh || !drawHand() || pixels[centre] < 240 ||
      pixels[centre + 2] > 10) {
    std::printf("renderer test: hand in front of opaque scene failed\n");
    return 1;
  }
  // Inverting the depth must hide the hand, regardless of submission order.
  DrawItem nearScene = sceneItem;
  nearScene.model = scaleMatrix({0.5F, 0.5F, 0.5F});
  for (int order = 0; order < 2; ++order) {
    auto draws = order == 0 ? std::vector<DrawItem>{handItem, nearScene}
                            : std::vector<DrawItem>{nearScene, handItem};
    if (!renderer.draw(camera, draws, {}, {}) || !renderer.readback(pixels) ||
        pixels[centre] > 10 || pixels[centre + 2] < 240) {
      std::printf("renderer test: opaque in front of hand failed, order %d\n", order);
      return 1;
    }
  }

  // A sloping hand crosses the scene plane: both visible regions must coexist.
  MeshData crossing = hand;
  crossing.vertices[0].pos = {-0.8F * 2.0F / 3.0F, -0.8F * 2.0F / 3.0F, -2.0F};
  crossing.vertices[1].pos = {0.8F * 4.0F / 3.0F, -0.8F * 4.0F / 3.0F, -4.0F};
  DrawItem middleScene = sceneItem;
  middleScene.model = scaleMatrix({0.75F, 0.75F, 0.75F});
  auto sample = [&](std::uint32_t x, std::uint32_t y, unsigned channel) {
    return pixels[(y * config.width + x) * 4U + channel];
  };
  if (renderer.updateHandMesh(0, crossing) != handHandle ||
      !renderer.draw(camera, {handItem, middleScene}, {}, {}) || !renderer.readback(pixels) ||
      sample(40, 52, 0) < 240 || sample(40, 52, 2) > 10 ||
      sample(56, 52, 2) < 240 || sample(56, 52, 0) > 10 ||
      renderer.updateHandMesh(0, hand) != handHandle) {
    std::printf("renderer test: hand/scene intersection failed\n");
    return 1;
  }

  // A cutout in front writes depth only for retained mask fragments.
  Image cutout;
  cutout.width = 2;
  cutout.height = 1;
  cutout.rgba = {255, 255, 255, 0, 255, 255, 255, 255};
  auto const cutoutTexture = renderer.addTexture(cutout, false, false);
  MeshData mask;
  mask.vertices = {vertex({-0.8F, -0.8F, -2.0F}, normal, {0, 0, 1}),
                   vertex({0.8F, -0.8F, -2.0F}, normal, {0, 0, 1}),
                   vertex({0.8F, 0.8F, -2.0F}, normal, {0, 0, 1}),
                   vertex({-0.8F, 0.8F, -2.0F}, normal, {0, 0, 1})};
  mask.vertices[0].uv = {0, 0};
  mask.vertices[1].uv = {1, 0};
  mask.vertices[2].uv = {1, 1};
  mask.vertices[3].uv = {0, 1};
  mask.indices = {0, 1, 2, 0, 2, 3};
  DrawItem maskItem;
  maskItem.mesh = renderer.addMesh(mask);
  maskItem.texture = cutoutTexture;
  maskItem.alpha = AlphaMode::Mask;
  maskItem.alphaCutoff = 0.5F;
  maskItem.lit = false;
  if (cutoutTexture == kInvalidTexture || maskItem.mesh == kInvalidMesh ||
      !renderer.draw(camera, {handItem, maskItem}, {}, {}) || !renderer.readback(pixels) ||
      sample(40, 52, 0) < 240 || sample(40, 52, 2) > 10 ||
      sample(56, 52, 2) < 240 || sample(56, 52, 0) > 10) {
    std::printf("renderer test: mask cutout depth failed\n");
    return 1;
  }

  // Front blend composites over the hand; a blend behind it fails depth.
  // The second blend exercises scratch slot indexing and reuse across frames.
  DrawItem frontBlend = nearScene;
  frontBlend.alpha = AlphaMode::Blend;
  frontBlend.alphaFactor = 0.5F;
  DrawItem backBlend = sceneItem;
  backBlend.alpha = AlphaMode::Blend;
  backBlend.alphaFactor = 0.5F;
  for (int frame = 0; frame < 3; ++frame) {
    // Alternate eye-like views to exercise per-view ordering and scratch reuse.
    camera.position.x = frame == 0 ? -0.03F : (frame == 1 ? 0.03F : -0.03F);
    if (!renderer.draw(camera, {frontBlend, handItem, backBlend}, {}, {}) ||
        !renderer.readback(pixels) || pixels[centre] < 115 || pixels[centre] > 140 ||
        pixels[centre + 1] > 10 || pixels[centre + 2] < 115 || pixels[centre + 2] > 140 ||
        !renderer.draw(camera, {backBlend, handItem}, {}, {}) || !renderer.readback(pixels) ||
        pixels[centre] < 240 || pixels[centre + 2] > 10) {
      std::printf("renderer test: blended scene/hand depth failed, frame %d\n", frame);
      return 1;
    }
  }
  // Distinct blend meshes on both scratch slots: the first has deliberately
  // reversed triangle indices; the second has only one triangle. The hand is
  // behind both and must not consume a sorted-index slot.
  DrawItem farHandItem = handItem;
  farHandItem.model = translationMatrix({0, 0, -2});
  DrawItem closestBlend = frontBlend;
  closestBlend.model = scaleMatrix({0.375F, 0.375F, 0.375F});
  for (int frame = 0; frame < 3; ++frame) {
    camera.position.x = frame == 1 ? 0.03F : -0.03F;
    if (!renderer.draw(camera, {closestBlend, farHandItem, item}, {}, {}) ||
        !renderer.readback(pixels) || pixels[centre] < 85 || pixels[centre] > 105 ||
        pixels[centre + 1] > 10 || pixels[centre + 2] < 150 || pixels[centre + 2] > 170) {
      std::printf("renderer test: blend scratch alignment failed, frame %d\n", frame);
      return 1;
    }
  }
  camera.position.x = 0.0F;

  // Even transparent textures and hostile material settings cannot make a
  // hand translucent or discard it. Check alpha over a transparent clear and
  // colour over scene geometry, for both BLEND and MASK input modes.
  Image transparentTexture;
  transparentTexture.width = 1;
  transparentTexture.height = 1;
  transparentTexture.rgba = {255, 255, 255, 0};
  auto const transparentHandle = renderer.addTexture(transparentTexture, true, true);
  if (transparentHandle == kInvalidTexture || transparentHandle == 0U) {
    std::printf("renderer test: transparent texture registration failed\n");
    return 1;
  }
  DrawItem alphaHand = handItem;
  alphaHand.texture = transparentHandle;
  alphaHand.alphaFactor = 0.1F;
  alphaHand.alphaCutoff = 0.9F;
  renderer.setClearColour(0.0F, 0.0F, 0.0F, 0.0F);
  for (AlphaMode mode : {AlphaMode::Blend, AlphaMode::Mask}) {
    alphaHand.alpha = mode;
    if (!render(renderer, camera, alphaHand, pixels) || pixels[centre] < 240 ||
        pixels[centre + 1] > 10 || pixels[centre + 2] > 10 || pixels[centre + 3] != 255 ||
        !renderer.draw(camera, {alphaHand, sceneItem}, {}, {}) || !renderer.readback(pixels) ||
        pixels[centre] < 240 || pixels[centre + 1] > 10 || pixels[centre + 2] > 10 ||
        pixels[centre + 3] != 255) {
      std::printf("renderer test: hand alpha override failed for %s\n",
                  mode == AlphaMode::Blend ? "blend" : "mask");
      return 1;
    }
  }
  renderer.setClearColour(0.08F, 0.09F, 0.11F, 1.0F);
  // Overlapping triangles in one hand must resolve by depth, not index order.
  MeshData overlapping = hand;
  for (auto const& v : hand.vertices) {
    auto farVertex = v;
    farVertex.pos.x *= 4.0F / 3.0F;
    farVertex.pos.y *= 4.0F / 3.0F;
    farVertex.pos.z = -4.0F;
    farVertex.colour = {0, 1, 0};
    overlapping.vertices.push_back(farVertex);
  }
  for (int order = 0; order < 2; ++order) {
    overlapping.indices = order == 0 ? std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5}
                                     : std::vector<std::uint32_t>{3, 4, 5, 0, 1, 2};
    if (renderer.updateHandMesh(0, overlapping) != handHandle ||
        !renderer.draw(camera, {handItem, sceneItem}, {}, {}) ||
        !renderer.readback(pixels) || pixels[centre] < 240 ||
        pixels[centre + 1] > 10 || pixels[centre + 2] > 10) {
      std::printf("renderer test: hand triangle depth failed for order %d\n", order);
      return 1;
    }
  }

  // Resolve overlap between slots as well, in either draw order and with the
  // nearer surface assigned to either slot.
  MeshData farHand = hand;
  for (auto& v : farHand.vertices) {
    v.pos.x *= 4.0F / 3.0F;
    v.pos.y *= 4.0F / 3.0F;
    v.pos.z = -4.0F;
    v.colour = {0, 1, 0};
  }
  DrawItem secondHand = handItem;
  secondHand.mesh = otherHandle;
  for (int nearSlot = 0; nearSlot < 2; ++nearSlot) {
    if (renderer.updateHandMesh(0, nearSlot == 0 ? hand : farHand) != handHandle ||
        renderer.updateHandMesh(1, nearSlot == 1 ? hand : farHand) != otherHandle) {
      std::printf("renderer test: overlapping slot update failed\n");
      return 1;
    }
    for (int order = 0; order < 2; ++order) {
      std::vector<DrawItem> draws = order == 0
          ? std::vector<DrawItem>{handItem, secondHand, sceneItem}
          : std::vector<DrawItem>{secondHand, handItem, sceneItem};
      if (!renderer.draw(camera, draws, {}, {}) || !renderer.readback(pixels) ||
          pixels[centre] < 240 || pixels[centre + 1] > 10 || pixels[centre + 2] > 10) {
        std::printf("renderer test: hand slot depth failed for slot %d order %d\n",
                    nearSlot, order);
        return 1;
      }
    }
  }
  if (renderer.updateHandMesh(0, hand) != handHandle ||
      renderer.updateHandMesh(1, hand) != otherHandle) {
    return 1;
  }

  // Depth lines respect both scene and hands; overlay lines ignore depth.
  std::vector<RendererVertex> line = {
      vertex({-0.4F, 0.0F, -1.5F}, normal, {0, 1, 0}),
      vertex({0.4F, 0.0F, -1.5F}, normal, {0, 1, 0})};
  auto greenInCentre = [&]() {
    for (std::uint32_t y = 44; y < 52; ++y) {
      for (std::uint32_t x = 44; x < 52; ++x) {
        auto const offset = (y * config.width + x) * 4U;
        if (pixels[offset + 1] > 240 && pixels[offset] < 10 && pixels[offset + 2] < 10) {
          return true;
        }
      }
    }
    return false;
  };
  if (!renderer.draw(camera, {sceneItem}, line, {}) || !renderer.readback(pixels) ||
      !greenInCentre() ||
      !renderer.draw(camera, {sceneItem, handItem}, line, {}) || !renderer.readback(pixels) ||
      !greenInCentre()) {
    std::printf("renderer test: scene depth line ordering failed\n");
    return 1;
  }
  for (auto& v : line) {
    v.pos.z = -3.5F;
  }
  if (!renderer.draw(camera, {sceneItem, handItem}, line, {}) || !renderer.readback(pixels) ||
      greenInCentre() || !renderer.draw(camera, {sceneItem, handItem}, {}, line) || !renderer.readback(pixels) ||
      !greenInCentre()) {
    std::printf("renderer test: overlay line ordering failed\n");
    return 1;
  }

  // Exercise an update directly after submission, varying both counts and
  // contents. Repeated frames must retain the same handle and current data.
  hand.vertices.push_back(vertex({0.8F, 0.8F, -3.0F}, normal, {0, 1, 0}));
  for (auto& v : hand.vertices) {
    v.colour = {0, 1, 0};
  }
  hand.indices = {0, 1, 2, 1, 3, 2};
  for (int frame = 0; frame < 8; ++frame) {
    if (renderer.updateHandMesh(0, hand) != handHandle ||
        !renderer.draw(camera, {handItem, sceneItem}, {}, {})) {
      std::printf("renderer test: dynamic hand update failed\n");
      return 1;
    }
  }
  if (!renderer.readback(pixels) || pixels[centre + 1] < 240 || pixels[centre + 2] > 10) {
    std::printf("renderer test: dynamic hand readback failed\n");
    return 1;
  }
  if (!renderer.draw(camera, {secondHand, sceneItem}, {}, {}) || !renderer.readback(pixels) ||
      pixels[centre] < 240 || pixels[centre + 1] > 10) {
    std::printf("renderer test: other hand slot was changed\n");
    return 1;
  }
  hand.vertices.resize(3);
  hand.indices = {0, 1, 2};
  if (renderer.updateHandMesh(0, hand) != handHandle || !drawHand() || pixels[centre + 1] < 240) {
    std::printf("renderer test: shrinking hand mesh failed\n");
    return 1;
  }
  MeshData farther = occluder;
  for (auto& v : farther.vertices) {
    v.pos.z = -5.0F;
    v.pos.x *= 1.25F;
    v.pos.y *= 1.25F;
    v.colour = {1, 0, 0};
  }
  DrawItem fartherItem;
  fartherItem.mesh = renderer.addMesh(farther);
  fartherItem.lit = false;
  if (fartherItem.mesh == kInvalidMesh ||
      !renderer.draw(camera, {sceneItem, fartherItem}, {}, {}) || !renderer.readback(pixels) ||
      pixels[centre + 2] < 240 || pixels[centre] > 10) {
    std::printf("renderer test: ordinary scene depth failed\n");
    return 1;
  }
  MeshData invalid = hand;
  invalid.indices[0] = static_cast<std::uint32_t>(invalid.vertices.size());
  if (renderer.updateHandMesh(0, invalid) != kInvalidMesh || !drawHand() ||
      pixels[centre + 1] < 240) {
    std::printf("renderer test: invalid update modified hand\n");
    return 1;
  }
  invalid.vertices.resize(VkRenderer::kHandMeshMaxVertices + 1U);
  if (renderer.updateHandMesh(0, invalid) != kInvalidMesh ||
      renderer.updateHandMesh(0, MeshData{}) != handHandle || !drawHand() ||
      pixels[centre + 2] < 240 || pixels[centre + 1] > 10) {
    std::printf("renderer test: hand bounds or clear failed\n");
    return 1;
  }
  if (renderer.updateHandMesh(0, hand) != handHandle || !drawHand() || pixels[centre + 1] < 240) {
    std::printf("renderer test: hand reactivation failed\n");
    return 1;
  }
  // Dynamic (skinned) mesh path: the same handle is updated in place, with no
  // topology change, and the next draw sees the new vertices.
  MeshData dynamic;
  dynamic.vertices = {vertex({-0.6F, -0.6F, -3.0F}, normal, {1, 1, 1}),
                      vertex({0.6F, -0.6F, -3.0F}, normal, {1, 1, 1}),
                      vertex({0.0F, 0.6F, -3.0F}, normal, {1, 1, 1})};
  dynamic.indices = {0, 1, 2};
  MeshHandle const dynamicHandle = renderer.addMesh(dynamic, true);
  DrawItem dynamicItem;
  dynamicItem.mesh = dynamicHandle;
  dynamicItem.lit = false;
  if (dynamicHandle == kInvalidMesh ||
      renderer.updateMeshVertices(sourceHandle, dynamic.vertices) ||
      !render(renderer, camera, dynamicItem, pixels) || pixels[centre] < 200) {
    std::printf("renderer test: dynamic mesh registration/static rejection failed\n");
    return 1;
  }
  std::vector<MeshVertex> moved = dynamic.vertices;
  for (MeshVertex& v : moved) {
    v.pos.x += 1.8F;
  }
  std::vector<MeshVertex> wrongSize = moved;
  wrongSize.pop_back();
  std::vector<MeshVertex> nonFinite = moved;
  nonFinite[0].pos.x = std::numeric_limits<float>::quiet_NaN();
  if (!renderer.updateMeshVertices(dynamicHandle, moved) ||
      renderer.updateMeshVertices(dynamicHandle, wrongSize) ||
      renderer.updateMeshVertices(dynamicHandle, nonFinite) ||
      !render(renderer, camera, dynamicItem, pixels) || pixels[centre] > 60) {
    std::printf("renderer test: dynamic mesh update/validation failed\n");
    return 1;
  }
  if (!renderer.updateMeshVertices(dynamicHandle, dynamic.vertices) ||
      !render(renderer, camera, dynamicItem, pixels) || pixels[centre] < 200) {
    std::printf("renderer test: dynamic mesh restore failed\n");
    return 1;
  }
  // Pending data is a plain assignment: when the app updates twice before a
  // frame, the last payload is what the next draw shows. An opaque skinned
  // registration (trackTriangleSums=false, the production path for characters
  // without blending) must update and draw too.
  MeshHandle const opaqueHandle = renderer.addMesh(dynamic, true, false);
  DrawItem opaqueItem = dynamicItem;
  opaqueItem.mesh = opaqueHandle;
  if (opaqueHandle == kInvalidMesh ||
      !renderer.updateMeshVertices(opaqueHandle, moved) ||
      !renderer.updateMeshVertices(opaqueHandle, dynamic.vertices) ||
      !render(renderer, camera, opaqueItem, pixels) || pixels[centre] < 200) {
    std::printf("renderer test: opaque dynamic mesh last-update-wins failed\n");
    return 1;
  }

  std::printf("renderer tests: dynamic vertex update and validation passed\n");
  std::printf("renderer tests: scene-correct hand depth, slots, lines, bounds and clear passed\n");
  std::printf("renderer tests: nonuniform/mirrored/singular normals and blend sort/reuse passed\n");
  return 0;
}

namespace {
// A real external Vulkan target exercises the same path as an XR swapchain.
struct ExternalTarget {
  VkInstance instance{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  VkImage image{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkImageView view{VK_NULL_HANDLE};
  VkPhysicalDevice physical{VK_NULL_HANDLE};
  std::uint32_t family{0};
  VkRenderer renderer;

  ~ExternalTarget() {
    renderer.shutdown();
    if (device != VK_NULL_HANDLE) {
      vkDestroyImageView(device, view, nullptr);
      vkDestroyImage(device, image, nullptr);
      vkFreeMemory(device, memory, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) { vkDestroyInstance(instance, nullptr); }
  }

  bool init() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) { return false; }
    std::uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) { return false; }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) { return false; }
    for (auto candidate : devices) {
      VkPhysicalDeviceProperties props{};
      VkPhysicalDeviceFeatures features{};
      vkGetPhysicalDeviceProperties(candidate, &props);
      vkGetPhysicalDeviceFeatures(candidate, &features);
      if (props.apiVersion < VK_API_VERSION_1_3 || !features.shaderSampledImageArrayDynamicIndexing ||
          props.limits.maxPerStageDescriptorSampledImages < 256 ||
          props.limits.maxPerStageDescriptorSamplers < 256) { continue; }
      std::uint32_t n = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
      std::vector<VkQueueFamilyProperties> families(n);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, families.data());
      for (std::uint32_t i = 0; i < n; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
          physical = candidate;
          family = i;
          break;
        }
      }
      if (physical != VK_NULL_HANDLE) { break; }
    }
    if (physical == VK_NULL_HANDLE) { return false; }
    float priority = 1.0F;
    VkDeviceQueueCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures features{};
    features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    VkDeviceCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.pNext = &f13;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.pEnabledFeatures = &features;
    if (vkCreateDevice(physical, &di, nullptr, &device) != VK_SUCCESS) { return false; }
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {96, 96, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (vkCreateImage(device, &ii, nullptr, &image) != VK_SUCCESS) { return false; }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, image, &req);
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    std::uint32_t type = 0;
    while (type < props.memoryTypeCount && (req.memoryTypeBits & (1U << type)) == 0) { ++type; }
    if (type == props.memoryTypeCount) { return false; }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS ||
        vkBindImageMemory(device, image, memory, 0) != VK_SUCCESS) { return false; }
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ii.format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &vi, nullptr, &view) != VK_SUCCESS) { return false; }
    VkRenderer::Config config;
    config.width = 96;
    config.height = 96;
    return renderer.initExternal(config, instance, physical, device, family, ii.format);
  }
  bool draw() { return renderer.drawToView(view, image, Mat4{}, Mat4{}, {}, {}, {}, {}); }
};
}

bool testExternalWaitFailures() {
  for (int scenario = 0; scenario < 3; ++scenario) {
    ExternalTarget target;
    if (!target.init()) { return false; }
    deviceWaitCalls = 0;
    fenceWaitFaultSkip = 1; // initial fence succeeds; fail only after submission
    fenceWaitFault = scenario == 1 ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
    if (scenario == 2) { deviceWaitFault = VK_ERROR_DEVICE_LOST; }
    if (target.draw() || fenceWaitFault != VK_SUCCESS ||
        deviceWaitCalls != (scenario == 1 ? 0 : 1) ||
        target.renderer.hasFatalError() != (scenario != 0)) { return false; }
    if (scenario == 0 && !target.draw()) { return false; }
    if (scenario != 0 && target.draw()) { return false; }
  }
  pid_t child = fork();
  if (child < 0) { return false; }
  if (child == 0) {
    rlimit noCore{0, 0};
    setrlimit(RLIMIT_CORE, &noCore);
    ExternalTarget target;
    if (!target.init()) { _exit(2); }
    fenceWaitFaultSkip = 1;
    fenceWaitFault = VK_ERROR_OUT_OF_HOST_MEMORY;
    deviceWaitFault = VK_ERROR_OUT_OF_HOST_MEMORY;
    deviceWaitFaultSkip = 0;
    target.draw();
    _exit(3); // must not return an image whose retirement is unknown
  }
  int status = 0;
  return waitpid(child, &status, 0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

bool testVertexUpdateAllocationFailure() {
  VkRenderer::Config config;
  config.width = 96;
  config.height = 96;
  VkRenderer renderer;
  if (!renderer.init(config)) { return false; }
  MeshData mesh;
  auto triangle = [&](float z, Vec3 colour) {
    mesh.vertices.push_back(vertex({-0.8F, -0.8F, z}, {0, 0, 1}, colour));
    mesh.vertices.push_back(vertex({0.8F, -0.8F, z}, {0, 0, 1}, colour));
    mesh.vertices.push_back(vertex({0, 0.8F, z}, {0, 0, 1}, colour));
  };
  triangle(-4, {1, 0, 0});
  triangle(-3, {0, 1, 0});
  mesh.indices = {0, 1, 2, 3, 4, 5};
  MeshHandle handle = renderer.addMesh(mesh, true);
  if (handle == kInvalidMesh) { return false; }
  MeshData other = mesh;
  other.vertices.resize(3);
  other.indices = {0, 1, 2};
  for (auto& v : other.vertices) { v.pos.z = -4.0F; v.colour = {0, 0, 1}; }
  MeshHandle otherHandle = renderer.addMesh(other);
  if (otherHandle == kInvalidMesh) { return false; }
  DrawItem item;
  item.mesh = handle;
  item.alpha = AlphaMode::Blend;
  item.alphaFactor = 0.5F;
  item.lit = false;
  DrawItem otherItem = item;
  otherItem.mesh = otherHandle;
  Camera camera;
  camera.position = {0, 0, 0};
  camera.pitch = 0;
  std::vector<std::uint8_t> expected, actual;
  auto draw = [&](std::vector<std::uint8_t>& pixels) {
    return renderer.draw(camera, {item, otherItem}, {}, {}) && renderer.readback(pixels);
  };
  // The first update has no allocated pending vector: the old implementation
  // used to change the blend keys before its allocation could throw.
  if (!draw(expected)) { return false; }
  auto rejected = mesh.vertices;
  for (std::size_t i = 0; i < rejected.size(); ++i) {
    rejected[i].pos.z = i < 3 ? -2.0F : -6.0F;
  }
  Vec3 centre{0, 0, -9};
  bool threw = false;
  allocationsBeforeFailure = 0;
  try { renderer.updateMeshVertices(handle, rejected, &centre); }
  catch (std::bad_alloc const&) { threw = true; }
  allocationsBeforeFailure = -1;
  if (!threw || !draw(actual) || actual != expected) { return false; }
  // Keep the item key unchanged and force a fresh triangle sort by moving
  // the camera. This independently catches partially changed triangle keys.
  camera.position.x = 0.2F;
  std::vector<std::uint8_t> movedExpected;
  if (!draw(movedExpected)) { return false; }
  camera.position.x = 0;
  if (!draw(actual)) { return false; }
  Vec3 originalCentre{0, 0, -3.5F};
  threw = false;
  allocationsBeforeFailure = 0;
  try { renderer.updateMeshVertices(handle, rejected, &originalCentre); }
  catch (std::bad_alloc const&) { threw = true; }
  allocationsBeforeFailure = -1;
  camera.position.x = 0.2F;
  if (!threw || !draw(actual) || actual != movedExpected) { return false; }
  camera.position.x = 0;
  // Prove the test is sensitive to the changed pose and ordering.
  if (!renderer.updateMeshVertices(handle, rejected, &centre) || !draw(actual) ||
      actual == expected) { return false; }
  // Once reserved, repeated updates do not allocate, even when a pose is
  // already pending; the last successful update wins.
  if (!renderer.updateMeshVertices(handle, mesh.vertices)) { return false; }
  allocationsBeforeFailure = 0;
  bool updated = false;
  try { updated = renderer.updateMeshVertices(handle, rejected, &centre); }
  catch (std::bad_alloc const&) {}
  allocationsBeforeFailure = -1;
  return updated && draw(actual) && actual != expected;
}
