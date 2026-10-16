#include "handles.h"

#include <cmath>

namespace vredit {

namespace {

constexpr float kPi = 3.14159265359F;

} // namespace

float pointSegmentDistance(Vec2 point, Vec2 a, Vec2 b) {
  Vec2 ab{b.x - a.x, b.y - a.y};
  float lengthSq = ab.x * ab.x + ab.y * ab.y;
  if (lengthSq <= 1e-12F) {
    float dx = point.x - a.x;
    float dy = point.y - a.y;
    return std::sqrt(dx * dx + dy * dy);
  }
  float t = ((point.x - a.x) * ab.x + (point.y - a.y) * ab.y) / lengthSq;
  if (t < 0.0F) {
    t = 0.0F;
  } else if (t > 1.0F) {
    t = 1.0F;
  }
  float px = a.x + ab.x * t;
  float py = a.y + ab.y * t;
  float dx = point.x - px;
  float dy = point.y - py;
  return std::sqrt(dx * dx + dy * dy);
}

bool pointInPolygon(std::vector<Vec2> const& polygon, Vec2 point) {
  bool inside = false;
  std::size_t count = polygon.size();
  if (count < 3U) {
    return false;
  }
  for (std::size_t i = 0, j = count - 1U; i < count; j = i++) {
    Vec2 const& a = polygon[i];
    Vec2 const& b = polygon[j];
    bool crosses = (a.y > point.y) != (b.y > point.y);
    if (crosses) {
      float xAtY = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
      if (point.x < xAtY) {
        inside = !inside;
      }
    }
  }
  return inside;
}

std::vector<HandleGeom> buildHandleGeometry(GizmoMode mode,
                                            Vec3 pivot,
                                            Quat orientation,
                                            float size) {
  std::vector<HandleGeom> geoms;
  Vec3 axes[3] = {rotate(orientation, {1.0F, 0.0F, 0.0F}),
                  rotate(orientation, {0.0F, 1.0F, 0.0F}),
                  rotate(orientation, {0.0F, 0.0F, 1.0F})};
  GizmoHandle const axisHandles[3] = {GizmoHandle::AxisX, GizmoHandle::AxisY,
                                      GizmoHandle::AxisZ};
  GizmoHandle const planeHandles[3] = {GizmoHandle::PlaneXY, GizmoHandle::PlaneYZ,
                                       GizmoHandle::PlaneZX};

  if (mode == GizmoMode::Rotate) {
    int const ringSteps = 32;
    float const ringRadius = size * 0.8F;
    for (int a = 0; a < 3; ++a) {
      Vec3 u = axes[(a + 1) % 3];
      Vec3 v = axes[(a + 2) % 3];
      HandleGeom ring;
      ring.handle = axisHandles[a];
      ring.loop = true;
      for (int i = 0; i < ringSteps; ++i) {
        float angle = (2.0F * kPi * static_cast<float>(i)) / static_cast<float>(ringSteps);
        ring.points.push_back(pivot + (u * std::cos(angle) + v * std::sin(angle)) * ringRadius);
      }
      geoms.push_back(std::move(ring));
    }
    return geoms;
  }

  // Translate and Scale: axis lines from the pivot, plane quads in the corner
  // between each axis pair; Scale adds a filled tip square per axis.
  for (int a = 0; a < 3; ++a) {
    HandleGeom line;
    line.handle = axisHandles[a];
    line.points.push_back(pivot);
    line.points.push_back(pivot + axes[a] * size);
    geoms.push_back(std::move(line));

    if (mode == GizmoMode::Scale) {
      Vec3 tip = pivot + axes[a] * size;
      Vec3 u = axes[(a + 1) % 3] * (size * 0.06F);
      Vec3 v = axes[(a + 2) % 3] * (size * 0.06F);
      HandleGeom square;
      square.handle = axisHandles[a];
      square.loop = true;
      square.filled = true;
      square.points.push_back(tip - u - v);
      square.points.push_back(tip + u - v);
      square.points.push_back(tip + u + v);
      square.points.push_back(tip - u + v);
      geoms.push_back(std::move(square));
    }
  }

  float const inner = size * 0.25F;
  float const outer = size * 0.45F;
  for (int p = 0; p < 3; ++p) {
    Vec3 u = axes[p];
    Vec3 v = axes[(p + 1) % 3];
    HandleGeom quad;
    quad.handle = planeHandles[p];
    quad.loop = true;
    quad.filled = true;
    quad.points.push_back(pivot + u * inner + v * inner);
    quad.points.push_back(pivot + u * outer + v * inner);
    quad.points.push_back(pivot + u * outer + v * outer);
    quad.points.push_back(pivot + u * inner + v * outer);
    geoms.push_back(std::move(quad));
  }
  return geoms;
}

std::vector<HandleShape> buildHandleShapes(GizmoMode mode,
                                           Vec3 pivot,
                                           Quat orientation,
                                           float size,
                                           ProjectFn const& project) {
  std::vector<HandleShape> shapes;
  for (HandleGeom const& geom : buildHandleGeometry(mode, pivot, orientation, size)) {
    HandleShape shape;
    shape.handle = geom.handle;
    shape.loop = geom.loop;
    shape.filled = geom.filled;
    shape.points.reserve(geom.points.size());
    for (Vec3 const& point : geom.points) {
      shape.points.push_back(project(point));
    }
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

std::optional<GizmoHandle> pickHandle(std::vector<HandleShape> const& shapes,
                                      Vec2 mouse,
                                      float thresholdPx) {
  std::optional<GizmoHandle> lineHit;
  float lineHitDistance = thresholdPx;
  std::optional<GizmoHandle> filledHit;

  for (HandleShape const& shape : shapes) {
    std::size_t count = shape.points.size();
    if (count < 2U) {
      continue;
    }
    bool finite = true;
    for (Vec2 const& point : shape.points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        finite = false;
        break;
      }
    }
    if (!finite) {
      continue;
    }
    if (!shape.filled) {
      for (std::size_t i = 1; i < count; ++i) {
        float d = pointSegmentDistance(mouse, shape.points[i - 1U], shape.points[i]);
        if (d < lineHitDistance) {
          lineHitDistance = d;
          lineHit = shape.handle;
        }
      }
      if (shape.loop) {
        float d = pointSegmentDistance(mouse, shape.points[count - 1U], shape.points[0]);
        if (d < lineHitDistance) {
          lineHitDistance = d;
          lineHit = shape.handle;
        }
      }
    } else if (pointInPolygon(shape.points, mouse)) {
      if (!filledHit.has_value()) {
        filledHit = shape.handle;
      }
    }
  }

  // Line-like shapes (axes, rings) win over filled plane regions.
  if (lineHit.has_value()) {
    return lineHit;
  }
  return filledHit;
}

} // namespace vredit
