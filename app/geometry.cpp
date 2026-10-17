#include "geometry.h"

namespace vredit {

namespace {

void appendTriangle(std::vector<RendererVertex>& out,
                    Vec3 a,
                    Vec3 b,
                    Vec3 c,
                    Vec3 normal) {
  out.push_back({a, normal, {1.0F, 1.0F, 1.0F}, {0.0F, 0.0F}});
  out.push_back({b, normal, {1.0F, 1.0F, 1.0F}, {0.0F, 0.0F}});
  out.push_back({c, normal, {1.0F, 1.0F, 1.0F}, {0.0F, 0.0F}});
}

void appendQuad(std::vector<RendererVertex>& out,
                Vec3 a,
                Vec3 b,
                Vec3 c,
                Vec3 d,
                Vec3 normal) {
  appendTriangle(out, a, b, c, normal);
  appendTriangle(out, a, c, d, normal);
}

} // namespace

void appendCube(std::vector<RendererVertex>& out) {
  constexpr float h = 0.5F;
  // Faces wound consistently; culling is disabled in the pipeline so winding
  // only affects normals, which are supplied per face.
  appendQuad(out, {h, -h, -h}, {h, h, -h}, {h, h, h}, {h, -h, h}, {1.0F, 0.0F, 0.0F});
  appendQuad(out, {-h, -h, h}, {-h, h, h}, {-h, h, -h}, {-h, -h, -h}, {-1.0F, 0.0F, 0.0F});
  appendQuad(out, {-h, h, -h}, {-h, h, h}, {h, h, h}, {h, h, -h}, {0.0F, 1.0F, 0.0F});
  appendQuad(out, {-h, -h, h}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {0.0F, -1.0F, 0.0F});
  appendQuad(out, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}, {0.0F, 0.0F, 1.0F});
  appendQuad(out, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {0.0F, 0.0F, -1.0F});
}

void appendGrid(std::vector<RendererVertex>& out, float extent, float step) {
  Vec3 const gridColour{0.28F, 0.30F, 0.34F};
  Vec3 const xColour{0.75F, 0.22F, 0.22F};
  Vec3 const zColour{0.22F, 0.38F, 0.80F};
  Vec3 const up{0.0F, 1.0F, 0.0F};
  for (float v = -extent; v <= extent + 0.5F * step; v += step) {
    bool xAxis = (v == 0.0F);
    bool zAxis = (v == 0.0F);
    out.push_back({{-extent, 0.0F, v}, up, xAxis ? xColour : gridColour, {0.0F, 0.0F}});
    out.push_back({{extent, 0.0F, v}, up, xAxis ? xColour : gridColour, {0.0F, 0.0F}});
    out.push_back({{v, 0.0F, -extent}, up, zAxis ? zColour : gridColour, {0.0F, 0.0F}});
    out.push_back({{v, 0.0F, extent}, up, zAxis ? zColour : gridColour, {0.0F, 0.0F}});
  }
}

Vec3 handleColour(GizmoHandle handle) {
  switch (handle) {
  case GizmoHandle::AxisX:
    return {0.90F, 0.22F, 0.22F};
  case GizmoHandle::AxisY:
    return {0.25F, 0.85F, 0.30F};
  case GizmoHandle::AxisZ:
    return {0.25F, 0.45F, 0.95F};
  case GizmoHandle::PlaneXY:
    return {0.25F, 0.75F, 0.85F};
  case GizmoHandle::PlaneYZ:
    return {0.85F, 0.55F, 0.25F};
  case GizmoHandle::PlaneZX:
    return {0.80F, 0.35F, 0.80F};
  default:
    return {0.90F, 0.90F, 0.90F};
  }
}

void appendHandleLines(std::vector<RendererVertex>& out,
                       std::vector<HandleGeom> const& geometry) {
  Vec3 const up{0.0F, 1.0F, 0.0F};
  for (HandleGeom const& geom : geometry) {
    Vec3 colour = handleColour(geom.handle);
    std::size_t count = geom.points.size();
    if (count < 2U) {
      continue;
    }
    for (std::size_t i = 1; i < count; ++i) {
      out.push_back({geom.points[i - 1U], up, colour, {0.0F, 0.0F}});
      out.push_back({geom.points[i], up, colour, {0.0F, 0.0F}});
    }
    if (geom.loop) {
      out.push_back({geom.points[count - 1U], up, colour, {0.0F, 0.0F}});
      out.push_back({geom.points[0], up, colour, {0.0F, 0.0F}});
    }
  }
}

void appendBoundsLines(std::vector<RendererVertex>& out, Aabb const& bounds, Vec3 colour) {
  Vec3 const up{0.0F, 1.0F, 0.0F};
  Vec3 c[8];
  for (int i = 0; i < 8; ++i) {
    c[i] = {(i & 1) != 0 ? bounds.max.x : bounds.min.x,
            (i & 2) != 0 ? bounds.max.y : bounds.min.y,
            (i & 4) != 0 ? bounds.max.z : bounds.min.z};
  }
  int const edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                            {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (auto const& edge : edges) {
    out.push_back({c[edge[0]], up, colour, {0.0F, 0.0F}});
    out.push_back({c[edge[1]], up, colour, {0.0F, 0.0F}});
  }
}

void appendRayBeam(std::vector<RendererVertex>& out, Ray const& ray, float length, Vec3 colour) {
  Vec3 const up{0.0F, 1.0F, 0.0F};
  Vec3 const dir = normalized(ray.direction);
  Vec3 const tip = ray.origin + dir * length;
  out.push_back({ray.origin, up, colour, {0.0F, 0.0F}});
  out.push_back({tip, up, colour, {0.0F, 0.0F}});
  // Tip cross: two short segments perpendicular to the beam, so the aimed
  // point stays readable when it lands on an untextured surface.
  float const arm = length * 0.03F;
  Vec3 side = cross(dir, up);
  if (dot(side, side) < 1e-8F) {
    side = cross(dir, Vec3{1.0F, 0.0F, 0.0F});
  }
  side = normalized(side);
  Vec3 const lift = normalized(cross(side, dir));
  out.push_back({tip - side * arm, up, colour, {0.0F, 0.0F}});
  out.push_back({tip + side * arm, up, colour, {0.0F, 0.0F}});
  out.push_back({tip - lift * arm, up, colour, {0.0F, 0.0F}});
  out.push_back({tip + lift * arm, up, colour, {0.0F, 0.0F}});
}

} // namespace vredit
