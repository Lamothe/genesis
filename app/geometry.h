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
  MeshHandle mesh{0}; // renderer-registered mesh; 0 = built-in cube
  Mat4 model;
  Vec3 colour{1.0F, 1.0F, 1.0F};
  bool lit{true};
};

void appendCube(std::vector<RendererVertex>& out);
void appendGrid(std::vector<RendererVertex>& out, float extent, float step);

// Adds handle lines as line-list vertices using standard gizmo colours.
void appendHandleLines(std::vector<RendererVertex>& out,
                       std::vector<HandleGeom> const& geometry);

Vec3 handleColour(GizmoHandle handle);

} // namespace vredit
