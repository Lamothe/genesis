#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "gizmo.h"
#include "vecmath.h"

namespace vredit {

// Gizmo handle geometry and screen-space hit resolution. buildHandleGeometry is
// the single source of truth: the renderer draws it, hit-testing projects the
// same geometry, so what is drawn is exactly what is picked.

using ProjectFn = std::function<Vec2(Vec3 const&)>;

// Projectors return NaN coordinates for points that cannot be projected (for
// example, points behind the camera). pickHandle skips any shape containing a
// non-finite projected point, so such geometry can never be picked.

struct HandleGeom {
  GizmoHandle handle{GizmoHandle::Free};
  bool loop{false};   // closed polyline (rings, quads, tip squares)
  bool filled{false}; // polygon interior is the hit region
  std::vector<Vec3> points;
};

struct HandleShape {
  GizmoHandle handle{GizmoHandle::Free};
  bool loop{false};
  bool filled{false};
  std::vector<Vec2> points;
};

// Builds the 3D handle geometry for a mode at `pivot`, oriented by
// `orientation`, with a world-space size (callers usually derive it from the
// viewport scale so handles keep a constant pixel size).
std::vector<HandleGeom> buildHandleGeometry(GizmoMode mode,
                                            Vec3 pivot,
                                            Quat orientation,
                                            float size);

// Projects the geometry for hit testing against a pointer position.
std::vector<HandleShape> buildHandleShapes(GizmoMode mode,
                                           Vec3 pivot,
                                           Quat orientation,
                                           float size,
                                           ProjectFn const& project);

// Picks the handle under `mouse`. Line-like shapes win over filled shapes;
// threshold is in the same pixel space as the projected points. Returns
// nullopt when nothing is within reach.
std::optional<GizmoHandle> pickHandle(std::vector<HandleShape> const& shapes,
                                      Vec2 mouse,
                                      float thresholdPx);

// Ray-space pick for controller pointers (VR): tests the same world-space
// geometry pickHandle projects, so what is drawn is what is grabbed here too.
// Line-like shapes win over filled shapes; `threshold` is world-space distance
// from the ray. Returns nullopt when nothing is within reach.
std::optional<GizmoHandle> pickHandleByRay(std::vector<HandleGeom> const& geometry,
                                           Ray const& ray,
                                           float threshold);

float pointSegmentDistance(Vec2 point, Vec2 a, Vec2 b);
bool pointInPolygon(std::vector<Vec2> const& polygon, Vec2 point);

// Distance from a half-line (origin + t*dir, t >= 0) to a segment [a, b].
// `dir` should be normalised.
float raySegmentDistance(Ray const& ray, Vec3 a, Vec3 b);

} // namespace vredit
