#pragma once

#include <cstddef>
#include <vector>

#include "mesh.h"

namespace vredit {

// Computes the glTF joint matrices that skin a mesh in its own node-local
// space. jointWorlds must be in ModelSkin::joints order (the scene's global
// transform for each joint node); meshWorld is the global transform of the
// node the skin belongs to. The result is
//
//   inverse(meshWorld) * jointWorld * inverseBindMatrix
//
// so drawing the deformed vertices with meshWorld reproduces the spec's
// world-space result jointWorld * inverseBindMatrix. Returns false for a
// singular mesh transform, mismatched data or non-finite matrices, leaving
// `out` untouched.
bool computeSkinMatrices(ModelSkin const& skin,
                         Mat4 const& meshWorld,
                         std::vector<Mat4> const& jointWorlds,
                         std::vector<Mat4>& out);

// Deforms a mesh's positions and normals with four-influence linear blend
// skinning. `rest` supplies colour/UV/index data; `out` must have the same
// topology (it is replaced by a copy of rest otherwise). Vertices whose skin
// influences are absent, zero or out of range stay at their rest position.
// Normals use each joint's inverse-transpose (the correct transform under
// non-uniform joint scale) and are normalised after blending. When
// `boundsOut` is non-null it receives the deformed position bounds, fused into
// the same pass (callers would otherwise stream the vertices again). Returns
// false for non-finite output (including a non-finite bind-pose vertex) or an
// unsupported joint count; out may then be partially overwritten (callers must
// discard it).
bool deformSkinnedMesh(MeshData const& rest,
                       std::vector<SkinVertexWeight> const& weights,
                       std::vector<Mat4> const& jointMatrices,
                       MeshData& out,
                       Aabb* boundsOut = nullptr);

// Exact comparison used to skip CPU skinning and GPU uploads when a clip's
// sampled pose has not changed (e.g. a static key or a paused animation).
bool sameSkinMatrices(std::vector<Mat4> const& a,
                      std::vector<Mat4> const& b,
                      float epsilon = 1e-6F);

// Every non-zero influence must name a joint inside the skin. A file whose
// weights reference a joint the skin does not own cannot be skinned faithfully
// (the deformer would silently drop the influence and renormalise the rest), so
// callers treat such a primitive as rigid. Pairing, not renormalisation, is what
// this checks: zero-weight slots may name any joint.
bool skinWeightsMatchSkin(std::vector<SkinVertexWeight> const& weights,
                          ModelSkin const& skin);

} // namespace vredit