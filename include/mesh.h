#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

// A loaded model: meshes plus a node hierarchy with TRS transforms and mesh
// references (mesh index -1 = empty node).
struct ModelNode {
  std::string name;
  Transform local;
  std::int32_t parent{-1};
  std::int32_t mesh{-1};
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
};

// Axis-aligned bounds of mesh data (local space).
Aabb meshBounds(MeshData const& mesh);

} // namespace vredit
