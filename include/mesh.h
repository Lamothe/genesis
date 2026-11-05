#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "animation.h"
#include "image.h"
#include "scene.h"

namespace vredit {

using MeshHandle = std::uint32_t;

// Returned by renderer mesh registration on failure; never a valid draw handle
// (0 is the built-in cube).
constexpr MeshHandle kInvalidMesh = UINT32_MAX;

// Renderer-registered texture handle; 0 is the built-in white fallback.
using TextureHandle = std::uint32_t;
constexpr TextureHandle kInvalidTexture = UINT32_MAX;

// How a mesh's alpha channel is rendered.
enum class AlphaMode : std::uint8_t { Opaque, Mask, Blend };

// Single vertex format for the viewport pipeline and all model loaders.
struct MeshVertex {
  Vec3 pos;
  Vec3 normal;
  Vec3 colour;
  Vec2 uv;
};

struct MeshData {
  std::vector<MeshVertex> vertices;
  std::vector<std::uint32_t> indices;
};

// One vertex's four glTF joint influences. joints index ModelSkin::joints;
// weights are non-negative and sanitised to sum to 1 at load (all zero means
// the vertex stays at its bind position).
struct SkinVertexWeight {
  std::array<std::uint16_t, 4> joints{};
  std::array<float, 4> weights{};
};

// A glTF skin. joints are Model::nodes indices (already mapped from the glTF
// file's node indices). inverseBindMatrices has one entry per joint and is in
// the same column-major convention as Mat4.
struct ModelSkin {
  std::vector<std::int32_t> joints;
  std::vector<Mat4> inverseBindMatrices;
};

// A loaded model: meshes plus a node hierarchy with TRS transforms and mesh
// references (mesh index -1 = empty node).
struct ModelNode {
  std::string name;
  Transform local;
  std::int32_t parent{-1};
  std::int32_t mesh{-1};
  // glTF skin index (-1 = unskinned). skinNode is the node whose global
  // transform the skin's mesh-local matrices are relative to; it equals the
  // node itself except for the synthetic children of a multi-primitive mesh,
  // whose skin belongs to their wrapper node.
  std::int32_t skin{-1};
  std::int32_t skinNode{-1};
};

struct ModelMesh {
  MeshData data;
  Vec3 colour{1.0F, 1.0F, 1.0F}; // material base colour multiplier
  float alphaFactor{1.0F};       // material alpha (dissolve / baseColorFactor.a)
  std::int32_t texture{-1};      // index into Model::textures; -1 = untextured
  std::int32_t normalTexture{-1}; // tangent-space normal map; -1 = flat
  float normalScale{1.0F};        // glTF normalTexture.scale
  AlphaMode alpha{AlphaMode::Opaque};
  float alphaCutoff{0.5F};
  // JOINTS_0 / WEIGHTS_0 for this primitive; empty = rigid. Parsed only when
  // the file defines skins, so the static library models (which carry unused
  // joint attributes) pay no per-vertex memory cost.
  std::vector<SkinVertexWeight> skinWeights;
};

// A loaded texture image with sampler-relevant wrap modes.
struct ModelTexture {
  Image image;
  bool repeatU{true};
  bool repeatV{true};
};

struct Model {
  std::vector<ModelMesh> meshes;
  std::vector<ModelNode> nodes;
  std::vector<std::int32_t> roots;
  std::vector<ModelTexture> textures;
  std::vector<AnimationClip> clips;
  // A malformed skin is stored empty and treated as unskinned; the model's
  // geometry and rigid animation remain usable.
  std::vector<ModelSkin> skins;
};

// Axis-aligned bounds of mesh data (local space).
Aabb meshBounds(MeshData const& mesh);

} // namespace vredit
