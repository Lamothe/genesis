#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "mesh.h"

namespace vredit {

// An .mtl material: diffuse colour, optional texture paths (resolved to images
// by loadObjFile) and dissolve (1 = opaque, < 1 = blended).
struct MtlMaterial {
  std::string name;
  Vec3 colour{1.0F, 1.0F, 1.0F};
  float dissolve{1.0F};
  std::string mapKd;
  std::string mapBump;                   // map_Bump / bump / norm (a normal map)
  float normalScale{1.0F};               // -bm <scale> on the bump statement
  std::int32_t texture{-1};              // index into the textures passed to loadObj
  std::int32_t normalTexture{-1};        // ditto for the normal map; -1 = flat
};

using MtlMaterials = std::vector<MtlMaterial>;

// Wavefront OBJ: v/vt/vn/f (polygons fan-triangulated, negative indices
// supported), o/g groups become model nodes, usemtl selects colours/textures
// from `materials` (a mid-group material change splits the node). Faces without
// normals get flat per-face normals. Vertices are emitted per face.
bool loadObj(std::string_view text,
             MtlMaterials const& materials,
             std::vector<ModelTexture> const& textures,
             Model& out,
             std::string& error);

// Parses .mtl content (best effort): `newmtl` names with `Kd`, `map_Kd` and
// `d`/`Tr` entries.
bool parseMtl(std::string_view text, MtlMaterials& out);

// Loads an OBJ file; a sibling .mtl named by `mtllib` is loaded when present
// and its `map_Kd` images are decoded (missing images load untextured).
bool loadObjFile(char const* path, Model& out, std::string& error);

} // namespace vredit
