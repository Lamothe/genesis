#pragma once

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
  float highlight{0.0F};      // 1 = selected: warm tint in the shader
};

void appendCube(std::vector<RendererVertex>& out);
void appendGrid(std::vector<RendererVertex>& out, float extent, float step);

// Adds handle lines as line-list vertices using standard gizmo colours.
void appendHandleLines(std::vector<RendererVertex>& out,
                       std::vector<HandleGeom> const& geometry);

// 12 edges of an axis-aligned box (line list), e.g. a selection outline.
void appendBoundsLines(std::vector<RendererVertex>& out, Aabb const& bounds, Vec3 colour);

// Controller pointer: beam from the aim origin along the ray, with a small
// cross marking the tip so the aimed point is visible on plain surfaces.
void appendRayBeam(std::vector<RendererVertex>& out, Ray const& ray, float length, Vec3 colour);

Vec3 handleColour(GizmoHandle handle);

} // namespace vredit
