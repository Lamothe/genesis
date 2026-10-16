#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mesh.h"

namespace vredit {

using MtlColours = std::vector<std::pair<std::string, Vec3>>;

// Wavefront OBJ: v/vn/f (polygons fan-triangulated, negative indices supported),
// o/g groups become model nodes, usemtl selects colours from `materials`. Faces
// without normals get flat per-face normals. Vertices are emitted per face.
bool loadObj(std::string_view text, MtlColours const& materials, Model& out, std::string& error);

// Parses .mtl content (best effort): `newmtl` names with `Kd` diffuse colours.
bool parseMtl(std::string_view text, MtlColours& out);

// Loads an OBJ file; a sibling .mtl named by `mtllib` is loaded when present.
bool loadObjFile(char const* path, Model& out, std::string& error);

} // namespace vredit
