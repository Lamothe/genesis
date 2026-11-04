// Standalone offscreen renderer regression checks; requires a Vulkan 1.3 device.
// After run-editor.sh compiles the shaders:
// g++ -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude -Iapp tests/test_renderer.cpp src/camera.cpp src/handles.cpp src/triangle_sort.cpp src/mesh.cpp src/image.cpp app/geometry.cpp app/vk_renderer.cpp -lvulkan -lxcb -lz -o .build/test-renderer
// .build/test-renderer
#include "vk_renderer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vredit;

namespace {

MeshVertex vertex(Vec3 pos, Vec3 normal, Vec3 colour) {
  return {pos, normal, colour, {0.0F, 0.0F}};
}

bool render(VkRenderer& renderer, Camera const& camera, DrawItem const& item,
            std::vector<std::uint8_t>& pixels) {
  return renderer.draw(camera, {item}, {}, {}) && renderer.readback(pixels);
}

} // namespace

int main() {
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

  // Mirroring changes the determinant sign as well as the normal direction.
  MeshData mirrored = source;
  for (auto& v : mirrored.vertices) {
    v.pos.x *= -2.0F;
    v.normal = {-0.5F, 1.0F, 1.0F};
  }
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
  std::printf("renderer tests: nonuniform/mirrored/singular normals and blend sort/reuse passed\n");
  return 0;
}
