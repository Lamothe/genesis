#pragma once

#include <string_view>
#include <vector>

#include "handles.h"
#include "mesh.h"
#include "vecmath.h"

namespace vredit {

// CPU-side geometry for the viewport: a unit cube (registered as mesh 0), the
// floor grid and axis lines, gizmo handle lines, and per-item draws of
// renderer-registered meshes.

using RendererVertex = MeshVertex; // single vertex format for the pipeline

struct DrawItem {
  MeshHandle mesh{0};         // renderer-registered mesh; 0 = built-in cube
  TextureHandle texture{0};   // renderer-registered texture; 0 = white fallback
  TextureHandle normalTexture{0}; // normal-map slot (used when normalMapped)
  bool normalMapped{false};
  float normalScale{1.0F};    // glTF normalTexture.scale
  Mat4 model;
  Vec3 colour{1.0F, 1.0F, 1.0F};
  float alphaFactor{1.0F};
  float alphaCutoff{0.5F};
  AlphaMode alpha{AlphaMode::Opaque};
  bool lit{true};
  float highlight{0.0F}; // 5 = selected (thin warm shader rim),
                         // 3 + charge = delete ramp (red; charge 0..1)
                         // 0 = plain
};

void appendCube(std::vector<RendererVertex>& out);
void appendGrid(std::vector<RendererVertex>& out, float extent, float step);

// Controller pointer: beam from the aim origin along the ray, with a small
// cross marking the tip so the aimed point is visible on plain surfaces.
void appendRayBeam(std::vector<RendererVertex>& out, Ray const& ray, float length, Vec3 colour);

// Line-font text into an overlay line list (the VR mode HUD): uppercase
// stroke glyphs on a unit cell (baseline y = 0, cap height y = 1), drawn
// through `place` — text occupies its XY plane and faces +Z toward the
// viewer — at `height` metres. Unknown characters render as spaces. No
// font files, no textures; the strokes are ordinary line vertices.
void appendTextLines(std::vector<RendererVertex>& out, std::string_view text, Mat4 const& place,
                     float height, Vec3 colour);

} // namespace vredit
