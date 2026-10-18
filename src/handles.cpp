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

namespace {

float clamp01(float value) {
  return value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
}

// Ray vs coplanar filled polygon: plane hit, then 2D point-in-polygon in the
// plane's own basis.
bool rayHitsPolygon(Ray const& ray, std::vector<Vec3> const& points, float* rayDistOut) {
  if (points.size() < 3U) {
    return false;
  }
  Vec3 normal{0.0F, 0.0F, 0.0F};
  for (std::size_t i = 2; i < points.size(); ++i) {
    normal = cross(points[1] - points[0], points[i] - points[0]);
    if (length(normal) > 1e-9F) {
      break;
    }
  }
  if (length(normal) <= 1e-9F) {
    return false;
  }
  normal = normalized(normal);
  float denom = dot(ray.direction, normal);
  if (std::fabs(denom) < 1e-8F) {
    return false;
  }
  float t = dot(points[0] - ray.origin, normal) / denom;
  if (t < 0.0F) {
    return false;
  }
  Vec3 hit = ray.origin + ray.direction * t;
  Vec3 basisU =
      normalized(cross(std::fabs(normal.x) < 0.9F ? Vec3{1.0F, 0.0F, 0.0F} : Vec3{0.0F, 1.0F, 0.0F},
                       normal));
  Vec3 basisV = cross(normal, basisU);
  std::vector<Vec2> flat;
  flat.reserve(points.size());
  for (Vec3 const& point : points) {
    Vec3 d = point - points[0];
    flat.push_back({dot(d, basisU), dot(d, basisV)});
  }
  Vec2 q{dot(hit - points[0], basisU), dot(hit - points[0], basisV)};
  if (!pointInPolygon(flat, q)) {
    return false;
  }
  if (rayDistOut != nullptr) {
    *rayDistOut = t;
  }
  return true;
}

} // namespace

float raySegmentDistance(Ray const& ray, Vec3 a, Vec3 b, float* rayDistOut) {
  // Closest points between the half-line (s >= 0) and the segment (t in [0,1])
  // solved directly in double precision: aiming ALONG a handle is the common
  // case and the float normal equations cancel catastrophically there.
  double ux = ray.direction.x;
  double uy = ray.direction.y;
  double uz = ray.direction.z;
  double vx = static_cast<double>(b.x) - a.x;
  double vy = static_cast<double>(b.y) - a.y;
  double vz = static_cast<double>(b.z) - a.z;
  double wx = static_cast<double>(ray.origin.x) - a.x;
  double wy = static_cast<double>(ray.origin.y) - a.y;
  double wz = static_cast<double>(ray.origin.z) - a.z;
  double uu = ux * ux + uy * uy + uz * uz;
  double uv = ux * vx + uy * vy + uz * vz;
  double vv = vx * vx + vy * vy + vz * vz;
  double uw = ux * wx + uy * wy + uz * wz;
  double vw = vx * wx + vy * wy + vz * wz;
  double const eps = 1e-12;

  double s = 0.0;
  double t = 0.0;
  if (vv <= eps) {
    // Degenerate segment: a point against the half-line.
    s = uu > eps ? (0.0 - uw) / uu : 0.0;
    s = s < 0.0 ? 0.0 : s;
  } else if (uu <= eps) {
    t = clamp01(static_cast<float>(vw / vv));
  } else {
    double denom = uu * vv - uv * uv;
    if (denom <= eps) {
      // Parallel: the distance is constant wherever the ray overlaps the
      // segment; probe the closest t at s = 0 and let the s clamp handle the
      // fully-behind case.
      t = clamp01(static_cast<float>(vw / vv));
      s = (t * uv - uw) / uu;
      if (s < 0.0) {
        s = 0.0;
      }
    } else {
      s = (uv * vw - uw * vv) / denom;
      if (s < 0.0) {
        s = 0.0;
      }
      t = (vw + s * uv) / vv;
      if (t < 0.0) {
        t = 0.0;
        s = -uw / uu;
        s = s < 0.0 ? 0.0 : s;
      } else if (t > 1.0) {
        t = 1.0;
        s = (uv - uw) / uu;
        s = s < 0.0 ? 0.0 : s;
      }
    }
  }

  double dx = wx + s * ux - t * vx;
  double dy = wy + s * uy - t * vy;
  double dz = wz + s * uz - t * vz;
  if (rayDistOut != nullptr) {
    *rayDistOut = static_cast<float>(s);
  }
  return static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));
}

std::optional<RayHandleHit> pickHandleByRay(std::vector<HandleGeom> const& geometry,
                                            Ray const& ray, float threshold) {
  std::optional<RayHandleHit> lineHit;
  float lineHitDistance = threshold;
  std::optional<RayHandleHit> filledHit;

  for (HandleGeom const& shape : geometry) {
    std::size_t count = shape.points.size();
    if (shape.filled) {
      float depth = 0.0F;
      if (!filledHit.has_value() && rayHitsPolygon(ray, shape.points, &depth)) {
        filledHit = RayHandleHit{shape.handle, depth};
      }
      continue;
    }
    if (count < 2U) {
      continue;
    }
    for (std::size_t i = 1; i < count; ++i) {
      float depth = 0.0F;
      float d = raySegmentDistance(ray, shape.points[i - 1U], shape.points[i], &depth);
      if (d < lineHitDistance) {
        lineHitDistance = d;
        lineHit = RayHandleHit{shape.handle, depth};
      }
    }
    if (shape.loop && count >= 3U) {
      float depth = 0.0F;
      float d = raySegmentDistance(ray, shape.points[count - 1U], shape.points[0], &depth);
      if (d < lineHitDistance) {
        lineHitDistance = d;
        lineHit = RayHandleHit{shape.handle, depth};
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
