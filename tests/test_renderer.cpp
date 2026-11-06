// Standalone offscreen renderer regression checks; requires a Vulkan 1.3 device.
// After run-editor.sh compiles the shaders:
// g++ -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude -Iapp tests/test_renderer.cpp src/camera.cpp src/handles.cpp src/triangle_sort.cpp src/mesh.cpp src/image.cpp app/geometry.cpp app/vk_renderer.cpp -lvulkan -lxcb -lz -o .build/test-renderer
// .build/test-renderer
#include "vk_renderer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
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
