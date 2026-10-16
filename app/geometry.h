#pragma once

#include <vector>

#include "handles.h"
#include "vecmath.h"

namespace vredit {

// CPU-side geometry for the viewport: a unit cube (per-node draws), the floor
// grid and axis lines, and gizmo handle lines with per-handle colours.

struct RendererVertex {
  Vec3 pos;
  Vec3 normal;
  Vec3 colour;
};

struct DrawItem {
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
