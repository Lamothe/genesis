#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"

namespace vredit {

// Loads glTF 2.0 binary (.glb). POSITION is required per primitive; NORMAL is
// generated (smoothed) when absent; COLOR_0 (float or normalised u8/u16) feeds
// vertex colours; TEXCOORD_0 feeds UVs. Materials contribute baseColorFactor
// (colour and alpha), baseColorTexture (embedded PNG/JPEG images, sampled with
// the texture's wrap modes) and alphaMode/alphaCutoff. Node TRS or column-major
// matrices are preserved; multi-primitive meshes become a wrapper node with one
// child per primitive. Index/offset/count fields are read as checked
// non-negative integers so hostile files cannot wrap the size arithmetic.
bool loadGlb(std::vector<std::uint8_t> const& bytes,
             Model& out,
             std::string& error,
             bool withImages = true);
bool loadGlbFile(char const* path,
                 Model& out,
                 std::string& error,
                 bool withImages = true);

} // namespace vredit
