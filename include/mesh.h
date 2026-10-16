#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "scene.h"

namespace vredit {

using MeshHandle = std::uint32_t;

// Returned by renderer mesh registration on failure; never a valid draw handle
// (0 is the built-in cube).
constexpr MeshHandle kInvalidMesh = UINT32_MAX;

// Single vertex format for the viewport pipeline and all model loaders.
struct MeshVertex {
  Vec3 pos;
  Vec3 normal;
  Vec3 colour;
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
};

struct Model {
  std::vector<ModelMesh> meshes;
  std::vector<ModelNode> nodes;
  std::vector<std::int32_t> roots;
};

// Axis-aligned bounds of mesh data (local space).
Aabb meshBounds(MeshData const& mesh);

} // namespace vredit
