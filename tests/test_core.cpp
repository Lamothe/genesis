// Core tests for the scene-editor maths: vecmath, scene graph, picking, gizmos.
// Compile and run via run-tests.sh (typed C++ only; no scripting).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "camera.h"
#include "gizmo.h"
#include "gltf.h"
#include "handles.h"
#include "json.h"
#include "mesh.h"
#include "obj.h"
#include "picking.h"
#include "scene.h"
#include "vecmath.h"

using namespace vredit;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, char const* expr, char const* file, int line) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s\n", file, line, expr);
  }
}

void checkNear(float a, float b, float eps, char const* expr, char const* file, int line) {
  ++g_checks;
  if (!(std::fabs(a - b) <= eps)) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s (%f vs %f)\n", file, line, expr, static_cast<double>(a),
                static_cast<double>(b));
  }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, eps) checkNear((a), (b), (eps), #a " ~= " #b, __FILE__, __LINE__)

bool near3(Vec3 a, Vec3 b, float eps = 1e-4F) {
  return std::fabs(a.x - b.x) <= eps && std::fabs(a.y - b.y) <= eps &&
         std::fabs(a.z - b.z) <= eps;
}

bool nearQuat(Quat a, Quat b, float eps = 1e-4F) {
  // q and -q are the same rotation, so compare |dot| against 1.
  float d = std::fabs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  return d >= 1.0F - eps;
}

void checkNear3(Vec3 a, Vec3 b, char const* expr, char const* file, int line, float eps = 1e-4F) {
  ++g_checks;
  if (!near3(a, b, eps)) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s ((%f,%f,%f) vs (%f,%f,%f))\n", file, line, expr,
                static_cast<double>(a.x), static_cast<double>(a.y), static_cast<double>(a.z),
                static_cast<double>(b.x), static_cast<double>(b.y), static_cast<double>(b.z));
  }
}

// Variadic so braced Vec3 literals survive macro argument splitting.
#define CHECK_NEAR3(...) checkNear3(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__)

bool nearIdentity(Mat4 const& m, float eps = 1e-3F) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      float expected = (c == r) ? 1.0F : 0.0F;
      if (std::fabs(m.m[c * 4 + r] - expected) > eps) {
        return false;
      }
    }
  }
  return true;
}

bool nearMat(Mat4 const& a, Mat4 const& b, float eps = 1e-3F) {
  for (int i = 0; i < 16; ++i) {
    if (std::fabs(a.m[i] - b.m[i]) > eps) {
      return false;
    }
  }
  return true;
}

float const kHalfPi = 1.57079632679F;
float const kQuarterTurn = kHalfPi;

void testVecBasics() {
  CHECK_NEAR3(cross({1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}), {0.0F, 0.0F, 1.0F});
  CHECK_NEAR(dot({1.0F, 2.0F, 3.0F}, {4.0F, -5.0F, 6.0F}), 12.0F, 1e-6F);
  CHECK_NEAR(length(normalized(Vec3{3.0F, 4.0F, 0.0F})), 1.0F, 1e-6F);
}

void testQuatMatchesMatrix() {
  Quat q = quatFromAxisAngle({0.0F, 0.0F, 1.0F}, kQuarterTurn);
  CHECK_NEAR3(rotate(q, {1.0F, 0.0F, 0.0F}), {0.0F, 1.0F, 0.0F});
  CHECK_NEAR3(transformVector(rotationMatrix(q), {1.0F, 0.0F, 0.0F}), {0.0F, 1.0F, 0.0F});

  // Composition: (a * b) applies b first.
  Quat qx = quatFromAxisAngle({1.0F, 0.0F, 0.0F}, kQuarterTurn);
  Quat composed = q * qx;
  CHECK_NEAR3(rotate(composed, {0.0F, 0.0F, 1.0F}), {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(rotate(q, rotate(qx, {0.0F, 0.0F, 1.0F})), rotate(composed, {0.0F, 0.0F, 1.0F}));
  CHECK_NEAR3(transformVector(rotationMatrix(composed), {0.0F, 0.0F, 1.0F}),
              rotate(composed, {0.0F, 0.0F, 1.0F}));
}

void testMatrixInverseAndDecompose() {
  Quat r = quatFromAxisAngle(normalized(Vec3{1.0F, 1.0F, 0.0F}), 0.7F);
  Mat4 m = trs({1.0F, 2.0F, 3.0F}, r, {2.0F, 1.0F, 0.5F});
  Mat4 inv;
  CHECK(invertAffine(m, inv));
  CHECK(nearIdentity(mul(m, inv)));
  CHECK(nearIdentity(mul(inv, m)));

  Vec3 t;
  Quat rq;
  Vec3 s;
  decompose(m, t, rq, s);
  CHECK_NEAR3(t, {1.0F, 2.0F, 3.0F});
  CHECK_NEAR3(s, {2.0F, 1.0F, 0.5F});
  CHECK(nearMat(trs(t, rq, s), m));
}

void testSceneHierarchy() {
  vredit::Scene scene;
  vredit::NodeId p = scene.create();
  vredit::NodeId c = scene.create(p);
  vredit::NodeId g = scene.create(c);
  vredit::Transform pt;
  pt.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(p, pt);
  vredit::Transform ct;
  ct.translation = {0.0F, 2.0F, 0.0F};
  scene.setLocal(c, ct);
  vredit::Transform gt;
  gt.translation = {0.0F, 0.0F, 3.0F};
  scene.setLocal(g, gt);

  CHECK(scene.valid(p) && scene.valid(c) && scene.valid(g));
  CHECK(scene.parent(c) == p);
  CHECK_NEAR3(scene.worldPosition(c), {1.0F, 2.0F, 0.0F});
  CHECK_NEAR3(scene.worldPosition(g), {1.0F, 2.0F, 3.0F});

  pt.translation = {1.0F, 0.0F, 5.0F};
  scene.setLocal(p, pt);
  CHECK_NEAR3(scene.worldPosition(g), {1.0F, 2.0F, 8.0F});
}

void testReparentKeepsLocal() {
  vredit::Scene scene;
  vredit::NodeId a = scene.create();
  vredit::NodeId b = scene.create();
  vredit::NodeId c = scene.create(a);
  vredit::Transform bt;
  bt.translation = {0.0F, 10.0F, 0.0F};
  scene.setLocal(b, bt);
  vredit::Transform ct;
  ct.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(c, ct);
  CHECK_NEAR3(scene.worldPosition(c), {1.0F, 0.0F, 0.0F});

  scene.setParent(c, b);
  CHECK(scene.parent(c) == b);
  CHECK_NEAR3(scene.local(c).translation, {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(scene.worldPosition(c), {1.0F, 10.0F, 0.0F});

  // Cycle rejection: b cannot become a child of its own descendant.
  scene.setParent(b, c);
  CHECK(scene.parent(b) == vredit::kInvalidNode);
}

void testDestroySubtree() {
  vredit::Scene scene;
  vredit::NodeId a = scene.create();
  vredit::NodeId c = scene.create(a);
  vredit::NodeId b = scene.create();
  scene.destroy(a);
  CHECK(!scene.valid(a));
  CHECK(!scene.valid(c));
  CHECK(scene.valid(b));
}

void testEpochSemantics() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);
  std::uint64_t e1 = scene.epoch();

  // Reads do not bump.
  (void)scene.world(n);
  (void)scene.worldPosition(n);
  CHECK(scene.epoch() == e1);

  // Exactly unchanged writes do not bump.
  scene.setLocal(n, t);
  CHECK(scene.epoch() == e1);

  // Changed writes do.
  t.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);
  CHECK(scene.epoch() == e1 + 1U);
}

void testRayAabb() {
  vredit::Aabb box{{0.0F, 0.0F, 0.0F}, {2.0F, 2.0F, 2.0F}};
  float t = 0.0F;
  CHECK(vredit::rayAabb({{-1.0F, 1.0F, 1.0F}, {1.0F, 0.0F, 0.0F}}, box, t));
  CHECK_NEAR(t, 1.0F, 1e-5F);
  CHECK(!vredit::rayAabb({{-1.0F, 3.0F, 1.0F}, {1.0F, 0.0F, 0.0F}}, box, t));
  CHECK(vredit::rayAabb({{1.0F, 1.0F, 1.0F}, {1.0F, 0.0F, 0.0F}}, box, t));
  CHECK_NEAR(t, 0.0F, 1e-5F);
  // Parallel to a slab and outside it.
  CHECK(!vredit::rayAabb({{-1.0F, 1.0F, 1.0F}, {0.0F, 1.0F, 0.0F}}, box, t));
  // Behind the origin.
  CHECK(!vredit::rayAabb({{-1.0F, 1.0F, 1.0F}, {-1.0F, 0.0F, 0.0F}}, box, t));
}

void testRayTriangle() {
  Vec3 a{0.0F, 0.0F, 0.0F};
  Vec3 b{1.0F, 0.0F, 0.0F};
  Vec3 c{0.0F, 1.0F, 0.0F};
  float t = 0.0F;
  CHECK(vredit::rayTriangle({{0.25F, 0.25F, 1.0F}, {0.0F, 0.0F, -1.0F}}, a, b, c, t));
  CHECK_NEAR(t, 1.0F, 1e-5F);
  // Two-sided: hit from behind.
  CHECK(vredit::rayTriangle({{0.25F, 0.25F, -1.0F}, {0.0F, 0.0F, 1.0F}}, a, b, c, t));
  CHECK_NEAR(t, 1.0F, 1e-5F);
  // Outside the triangle (u + v > 1).
  CHECK(!vredit::rayTriangle({{0.9F, 0.9F, 1.0F}, {0.0F, 0.0F, -1.0F}}, a, b, c, t));
  // Parallel to the triangle plane.
  CHECK(!vredit::rayTriangle({{2.0F, 2.0F, 1.0F}, {1.0F, 0.0F, 0.0F}}, a, b, c, t));
}

void testPickClosest() {
  std::vector<std::pair<vredit::NodeId, vredit::Aabb>> candidates;
  candidates.push_back({7, {{5.0F, 0.0F, 0.0F}, {7.0F, 2.0F, 2.0F}}});
  candidates.push_back({3, {{1.0F, 0.0F, 0.0F}, {3.0F, 2.0F, 2.0F}}});
  auto hit = vredit::pickClosest({{0.0F, 1.0F, 1.0F}, {1.0F, 0.0F, 0.0F}}, candidates);
  CHECK(hit.has_value());
  CHECK(hit->node == 3);
  CHECK_NEAR(hit->t, 1.0F, 1e-5F);

  auto miss = vredit::pickClosest({{0.0F, 9.0F, 1.0F}, {1.0F, 0.0F, 0.0F}}, candidates);
  CHECK(!miss.has_value());
}

void testGizmoTranslateWorld() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  settings.snapStep = 0.5F;
  CHECK(drag.begin(scene, {n}, {2.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{6.2F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), 4.0F, 1e-5F);
  CHECK_NEAR3(scene.worldPosition(n), {6.0F, 0.0F, 0.0F});
  drag.end();
  CHECK(!drag.active());
}

void testGizmoTranslateRotatedParent() {
  vredit::Scene scene;
  vredit::NodeId p = scene.create();
  vredit::NodeId c = scene.create(p);
  vredit::Transform pt;
  pt.rotation = quatFromAxisAngle({0.0F, 0.0F, 1.0F}, kHalfPi);
  scene.setLocal(p, pt);
  vredit::Transform ct;
  ct.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(c, ct);
  CHECK_NEAR3(scene.worldPosition(c), {0.0F, 1.0F, 0.0F});

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  CHECK(drag.begin(scene, {c}, {0.0F, 1.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{0.0F, 1.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{1.0F, 1.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR3(scene.worldPosition(c), {1.0F, 1.0F, 0.0F});
  // The parent's rotation must not be baked into the child's local transform.
  CHECK(nearQuat(scene.local(c).rotation, ct.rotation));
  CHECK_NEAR3(scene.local(c).scale, {1.0F, 1.0F, 1.0F});
}

void testGizmoRotateSnapped() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Rotate;
  settings.snapStep = kHalfPi;
  CHECK(drag.begin(scene, {n}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisZ,
                   {{1.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  // Drag to ~80 degrees; snap must round to 90.
  float c80 = std::cos(1.3962634F);
  float s80 = std::sin(1.3962634F);
  drag.update({{c80, s80, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), kHalfPi, 1e-4F);
  CHECK_NEAR3(scene.worldPosition(n), {0.0F, 1.0F, 0.0F});
}

void testGizmoScaleAxis() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Scale;
  settings.snapStep = 0.25F;
  CHECK(drag.begin(scene, {n}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{4.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), 2.0F, 1e-5F);
  CHECK_NEAR3(scene.worldPosition(n), {4.0F, 0.0F, 0.0F});
  CHECK_NEAR3(scene.local(n).scale, {2.0F, 1.0F, 1.0F});
}

void testGizmoLocalSpaceRotate() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {1.0F, 0.0F, 0.0F};
  t.rotation = quatFromAxisAngle({0.0F, 0.0F, 1.0F}, kHalfPi);
  scene.setLocal(n, t);

  // Local space: gizmo X axis = node world X = world +Y. Rotating about it keeps
  // the node position (it sits on the pivot) and swings its world Z axis to world X.
  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Rotate;
  settings.space = vredit::GizmoSpace::Local;
  CHECK(drag.begin(scene, {n}, {1.0F, 0.0F, 0.0F}, t.rotation, vredit::GizmoHandle::AxisX,
                   {{1.0F, 2.0F, 2.0F}, {0.0F, -2.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{2.0F, 2.0F, 0.0F}, {0.0F, -1.0F, 0.0F}});
  CHECK_NEAR(drag.cumulative(), kHalfPi, 1e-4F);
  CHECK_NEAR3(scene.worldPosition(n), {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(transformVector(scene.world(n), {0.0F, 0.0F, 1.0F}), {1.0F, 0.0F, 0.0F});
}

void testGizmoDiscontinuityEpoch() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  CHECK(drag.begin(scene, {n}, {2.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  vredit::Ray dragRay{{4.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}};
  drag.update(dragRay);
  std::uint64_t afterFirst = scene.epoch();
  drag.update(dragRay);
  // Identical update produces identical transforms: no spurious discontinuity.
  CHECK(scene.epoch() == afterFirst);
}

void testGizmoMultiSelection() {
  vredit::Scene scene;
  vredit::NodeId a = scene.create();
  vredit::NodeId b = scene.create();
  vredit::Transform at;
  at.translation = {0.0F, 0.0F, 0.0F};
  scene.setLocal(a, at);
  vredit::Transform bt;
  bt.translation = {3.0F, 0.0F, 0.0F};
  scene.setLocal(b, bt);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  settings.snapStep = 0.5F;
  CHECK(drag.begin(scene, {a, b}, {1.5F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisY,
                   {{1.5F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{1.5F, 2.2F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR3(scene.worldPosition(a), {0.0F, 2.0F, 0.0F});
  CHECK_NEAR3(scene.worldPosition(b), {3.0F, 2.0F, 0.0F});
}

void testDecomposeMirror() {
  Mat4 m = trs({}, Quat{}, {-1.0F, 1.0F, 1.0F});
  Vec3 t;
  Quat r;
  Vec3 s;
  decompose(m, t, r, s);
  CHECK(nearMat(trs(t, r, s), m));
  CHECK_NEAR3(s, {-1.0F, 1.0F, 1.0F});

  // Mirrored and rotated.
  Quat rz = quatFromAxisAngle({0.0F, 0.0F, 1.0F}, kHalfPi);
  Mat4 m2 = trs({1.0F, 2.0F, 3.0F}, rz, {-2.0F, 1.0F, 0.5F});
  decompose(m2, t, r, s);
  CHECK(nearMat(trs(t, r, s), m2));
}

void testGizmoMirrorPreserved() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.scale = {-1.0F, 1.0F, 1.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  CHECK(drag.begin(scene, {n}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{0.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{1.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR3(scene.worldPosition(n), {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(scene.local(n).scale, {-1.0F, 1.0F, 1.0F});
}

void testGizmoFreeTranslateSnapStaysInPlane() {
  Vec3 n = normalized(Vec3{1.0F, 1.0F, 1.0F});
  vredit::Scene scene;
  vredit::NodeId node = scene.create();

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  settings.snapStep = 0.1F;
  CHECK(drag.begin(scene, {node}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::Free,
                   {n * 5.0F, -n}, n, settings));
  Vec3 hit{0.16F, -0.12F, -0.04F};
  drag.update({hit + n * 5.0F, -n});

  Vec3 pos = scene.worldPosition(node);
  CHECK_NEAR(dot(pos, n), 0.0F, 1e-5F); // exactly in the drag plane
  // Snapped along the plane basis (0.2u - 0.1v), not per world component.
  CHECK_NEAR3(pos, {0.163299F, -0.152360F, -0.010939F});
}

void testGizmoRotateUnwrap() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Rotate;
  settings.snapStep = 2.5F;
  CHECK(drag.begin(scene, {n}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisZ,
                   {{1.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{std::cos(3.0F), std::sin(3.0F), 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), 2.5F, 1e-4F);

  // Cross the +/-pi seam: the raw angle wraps to -3.1 rad but the drag turned
  // continuously to ~3.18 rad; snapping must stay continuous (still 2.5 rad).
  drag.update({{std::cos(-3.1F), std::sin(-3.1F), 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), 2.5F, 1e-4F);
  CHECK_NEAR3(scene.worldPosition(n), {std::cos(2.5F), std::sin(2.5F), 0.0F});
}

void testGizmoScaleClamp() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Scale;
  CHECK(drag.begin(scene, {n}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{0.001F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), 0.01F, 1e-5F);
  CHECK_NEAR3(scene.worldPosition(n), {0.02F, 0.0F, 0.0F});
}

void testGizmoBeginFailureClean() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t;
  t.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(n, t);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  // Valid drag first, so stale state would be observable after a failed begin.
  CHECK(drag.begin(scene, {n}, {2.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{4.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR(drag.cumulative(), 2.0F, 1e-5F);
  drag.end();

  // Pointer ray parallel to the drag plane: begin must fail and leave no state.
  CHECK(!drag.begin(scene, {n}, {2.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                    {{2.0F, 0.0F, 5.0F}, {1.0F, 0.0F, 0.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  CHECK(!drag.active());
  CHECK_NEAR(drag.cumulative(), 0.0F, 1e-6F);

  // View direction along the handle axis: the drag plane degenerates.
  CHECK(!drag.begin(scene, {n}, {2.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                    {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {1.0F, 0.0F, 0.0F}, settings));
  CHECK(!drag.active());

  // A good begin still works afterwards.
  CHECK(drag.begin(scene, {n}, {2.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  CHECK(drag.active());
}

void testGizmoPlaneHandle() {
  vredit::Scene scene;
  vredit::NodeId node = scene.create();

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  CHECK(drag.begin(scene, {node}, {0.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::PlaneXY,
                   {{2.0F, 3.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{2.3F, 3.4F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR3(scene.worldPosition(node), {0.3F, 0.4F, 0.0F});
}

void testEpochStructuralChanges() {
  vredit::Scene scene;
  vredit::NodeId p = scene.create();
  vredit::NodeId c = scene.create(p);
  vredit::NodeId c2 = scene.create(p);
  CHECK(scene.valid(c2));

  std::uint64_t e = scene.epoch();
  scene.setParent(c, vredit::kInvalidNode);
  CHECK(scene.epoch() == e + 1U);

  std::uint64_t e2 = scene.epoch();
  scene.destroy(c);
  CHECK(scene.epoch() == e2 + 1U);

  std::uint64_t e3 = scene.epoch();
  scene.destroy(p); // subtree: p and c2
  CHECK(scene.epoch() == e3 + 2U);
}

void testCameraFraming() {
  Camera camera;
  camera.target = {0.0F, 0.0F, 0.0F};
  camera.yaw = 0.0F;
  camera.pitch = 0.0F;
  camera.distance = 5.0F;

  CHECK_NEAR3(cameraPosition(camera), {0.0F, 0.0F, 5.0F});
  CHECK_NEAR3(cameraForward(camera), {0.0F, 0.0F, -1.0F});
  CHECK_NEAR3(transformPoint(cameraView(camera), camera.target), {0.0F, 0.0F, -5.0F});

  Vec2 viewport{800.0F, 600.0F};
  float aspect = viewport.x / viewport.y;
  Ray centre = cameraRay(camera, aspect, {400.0F, 300.0F}, viewport);
  CHECK_NEAR3(centre.origin, {0.0F, 0.0F, 5.0F});
  CHECK_NEAR3(centre.direction, {0.0F, 0.0F, -1.0F});

  Ray right = cameraRay(camera, aspect, {600.0F, 300.0F}, viewport);
  CHECK(right.direction.x > 0.0F);
  Ray below = cameraRay(camera, aspect, {400.0F, 450.0F}, viewport);
  CHECK(below.direction.y < 0.0F); // pixel Y grows downward
  Ray above = cameraRay(camera, aspect, {400.0F, 150.0F}, viewport);
  CHECK(above.direction.y > 0.0F);

  // At the right edge the ray's x/-z ratio is (aspect / (1/tan(fovY/2))).
  Ray edge = cameraRay(camera, aspect, {800.0F, 300.0F}, viewport);
  CHECK_NEAR(edge.direction.x / -edge.direction.z, 0.7283687F, 1e-4F);
}

void testCameraManipulation() {
  Camera camera;
  camera.target = {0.0F, 0.0F, 0.0F};
  camera.yaw = 0.0F;
  camera.pitch = 0.0F;
  camera.distance = 5.0F;

  cameraOrbit(camera, kHalfPi, 0.0F);
  CHECK_NEAR3(cameraPosition(camera), {5.0F, 0.0F, 0.0F});

  cameraOrbit(camera, -kHalfPi, 0.0F);
  float before = camera.distance;
  cameraDolly(camera, 1.0F);
  CHECK(camera.distance < before);
  CHECK(camera.distance > 4.0F);

  cameraPan(camera, 10.0F, 0.0F, 600.0F);
  // Dragging right moves the world right: the target moves left along +X.
  CHECK(camera.target.x < 0.0F);
  CHECK_NEAR(camera.target.y, 0.0F, 1e-6F);
}

void testSegmentAndPolygonPicking() {
  CHECK_NEAR(pointSegmentDistance({0.0F, 1.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}), 1.0F, 1e-5F);
  CHECK_NEAR(pointSegmentDistance({0.5F, 1.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}), 1.0F, 1e-5F);
  CHECK_NEAR(pointSegmentDistance({2.0F, 0.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}), 1.0F, 1e-5F);

  std::vector<Vec2> square{{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}};
  CHECK(pointInPolygon(square, {0.5F, 0.5F}));
  CHECK(!pointInPolygon(square, {1.5F, 0.5F}));
  CHECK(!pointInPolygon(square, {0.5F, 1.5F}));
}

void testHandlePicking() {
  ProjectFn project = [](Vec3 const& p) { return Vec2{p.x, p.y}; };

  auto shapes = buildHandleShapes(GizmoMode::Translate, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F,
                                  project);
  CHECK(pickHandle(shapes, {0.5F, 0.05F}, 0.1F) == GizmoHandle::AxisX);
  CHECK(pickHandle(shapes, {0.05F, 0.5F}, 0.1F) == GizmoHandle::AxisY);
  // Plane quad band lives between 0.25 and 0.45 along each axis.
  CHECK(pickHandle(shapes, {0.35F, 0.35F}, 0.1F) == GizmoHandle::PlaneXY);
  CHECK(!pickHandle(shapes, {2.0F, 2.0F}, 0.1F).has_value());

  // Rotate: the Z ring lies in the XY plane; pick a point on the ring away
  // from the other rings' projections.
  auto rings = buildHandleShapes(GizmoMode::Rotate, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F, project);
  float onRing = 0.8F * 0.70710678F;
  CHECK(pickHandle(rings, {onRing, onRing}, 0.1F) == GizmoHandle::AxisZ);

  // Scale: axis lines carry the pick near the tips.
  auto scaled = buildHandleShapes(GizmoMode::Scale, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F, project);
  CHECK(pickHandle(scaled, {1.0F, 0.02F}, 0.1F) == GizmoHandle::AxisX);
}

void testHandlePickingBehindCamera() {
  // A projector that rejects points "behind the camera" (NaN): the X axis is
  // unpickable, the Y axis still works.
  ProjectFn projectBehind = [](Vec3 const& p) {
    if (p.x > 0.9F) {
      float nan = std::numeric_limits<float>::quiet_NaN();
      return Vec2{nan, nan};
    }
    return Vec2{p.x, p.y};
  };
  auto shapes =
      buildHandleShapes(GizmoMode::Translate, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F, projectBehind);
  CHECK(!pickHandle(shapes, {0.5F, 0.05F}, 0.1F).has_value());
  CHECK(pickHandle(shapes, {0.05F, 0.5F}, 0.1F) == GizmoHandle::AxisY);
}

void testJsonParser() {
  JsonDocument doc;
  std::string error;
  CHECK(doc.parse(R"({"a": [1, 2.5, true, null, "x\n"], "b": {"c": "d"}, "e": "\u0041", "f": -1.5e2})",
                  error));
  JsonDocument::NodeId root = doc.root();
  CHECK(doc.type(root) == JsonDocument::Type::Object);
  CHECK(doc.size(root) == 4U);

  JsonDocument::NodeId a = doc.member(root, "a");
  CHECK(doc.type(a) == JsonDocument::Type::Array);
  CHECK(doc.size(a) == 5U);
  CHECK_NEAR(doc.asNumber(doc.element(a, 0)), 1.0, 1e-12);
  CHECK_NEAR(doc.asNumber(doc.element(a, 1)), 2.5, 1e-12);
  CHECK(doc.asBool(doc.element(a, 2)));
  CHECK(doc.isNull(doc.element(a, 3)));
  CHECK(doc.asString(doc.element(a, 4)) == "x\n");
  CHECK(doc.asString(doc.member(doc.member(root, "b"), "c")) == "d");
  CHECK(doc.asString(doc.member(root, "e")) == "A");
  CHECK_NEAR(doc.asNumber(doc.member(root, "f")), -150.0, 1e-9);
  CHECK(doc.member(root, "missing") == JsonDocument::kInvalid);

  CHECK(!doc.parse("[1,]", error));       // trailing comma
  CHECK(!doc.parse("{\"a\" 1}", error));  // missing colon
  CHECK(!doc.parse("123 x", error));      // trailing garbage
  CHECK(!doc.parse("\"unterminated", error));
  CHECK(!doc.parse("", error));
  CHECK(!doc.parse("tru", error));
  std::string deep(300, '[');
  deep += std::string(300, ']');
  CHECK(!doc.parse(deep, error)); // nesting cap: clean error, no stack overflow
  // Parser must remain usable after failure.
  CHECK(doc.parse("[7]", error));
  CHECK(doc.size(doc.root()) == 1U);
}

void testMeshBounds() {
  MeshData mesh;
  mesh.vertices.push_back({{-1.0F, 2.0F, -3.0F}, {}, {}});
  mesh.vertices.push_back({{4.0F, -5.0F, 6.0F}, {}, {}});
  Aabb box = meshBounds(mesh);
  CHECK_NEAR3(box.min, {-1.0F, -5.0F, -3.0F});
  CHECK_NEAR3(box.max, {4.0F, 2.0F, 6.0F});
}

namespace fixtures {

void pushU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  std::uint8_t raw[4];
  std::memcpy(raw, &value, 4);
  bytes.insert(bytes.end(), raw, raw + 4);
}

std::vector<std::uint8_t> buildGlb(std::string const& json, std::vector<std::uint8_t> const& bin) {
  std::vector<std::uint8_t> bytes;
  std::string jsonPadded = json;
  while (jsonPadded.size() % 4U != 0) {
    jsonPadded.push_back(' ');
  }
  std::vector<std::uint8_t> binPadded = bin;
  while (binPadded.size() % 4U != 0) {
    binPadded.push_back(0);
  }
  std::uint32_t total =
      12U + 8U + static_cast<std::uint32_t>(jsonPadded.size()) + 8U +
      static_cast<std::uint32_t>(binPadded.size());
  pushU32(bytes, 0x46546C67U);
  pushU32(bytes, 2U);
  pushU32(bytes, total);
  pushU32(bytes, static_cast<std::uint32_t>(jsonPadded.size()));
  pushU32(bytes, 0x4E4F534AU);
  bytes.insert(bytes.end(), jsonPadded.begin(), jsonPadded.end());
  pushU32(bytes, static_cast<std::uint32_t>(binPadded.size()));
  pushU32(bytes, 0x004E4942U);
  bytes.insert(bytes.end(), binPadded.begin(), binPadded.end());
  return bytes;
}

bool writeFile(char const* path, std::vector<std::uint8_t> const& bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  file.write(reinterpret_cast<char const*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(file);
}

std::string const kTriangleJson =
    R"({"asset": {"version": "2.0"},)"
    R"( "scenes": [{"nodes": [0]}], "scene": 0,)"
    R"( "nodes": [{"name": "parent", "translation": [1, 2, 3], "children": [1]},)"
    R"(             {"name": "child", "mesh": 0, "scale": [2, 2, 2]}],)"
    R"( "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],)"
    R"( "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.25, 0.125, 1]}}],)"
    R"( "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},)"
    R"(                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],)"
    R"( "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},)"
    R"(                 {"buffer": 0, "byteOffset": 36, "byteLength": 6}],)"
    R"( "buffers": [{"byteLength": 44}]})";

std::vector<std::uint8_t> triangleBin(std::uint32_t lastVertexIndex) {
  std::vector<std::uint8_t> bin;
  float const positions[9] = {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  std::uint8_t const* raw = reinterpret_cast<std::uint8_t const*>(positions);
  bin.insert(bin.end(), raw, raw + 36);
  std::uint16_t indices[3] = {0U, 1U, static_cast<std::uint16_t>(lastVertexIndex)};
  std::uint8_t const* rawIndices = reinterpret_cast<std::uint8_t const*>(indices);
  bin.insert(bin.end(), rawIndices, rawIndices + 6);
  return bin;
}

} // namespace fixtures

void testGlbLoader() {
  using namespace fixtures;
  Model model;
  std::string error;

  std::vector<std::uint8_t> bytes = buildGlb(kTriangleJson, triangleBin(2));
  CHECK(loadGlb(bytes, model, error));

  CHECK(model.meshes.size() == 1U);
  CHECK_NEAR3(model.meshes[0].colour, {0.5F, 0.25F, 0.125F});
  CHECK(model.meshes[0].data.vertices.size() == 3U);
  CHECK(model.meshes[0].data.indices.size() == 3U);
  CHECK(model.meshes[0].data.indices[2] == 2U);
  CHECK_NEAR3(model.meshes[0].data.vertices[1].pos, {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(model.meshes[0].data.vertices[0].normal, {0.0F, 0.0F, 1.0F}); // generated

  CHECK(model.nodes.size() == 2U);
  CHECK(model.nodes[0].name == "parent");
  CHECK_NEAR3(model.nodes[0].local.translation, {1.0F, 2.0F, 3.0F});
  CHECK(model.nodes[0].parent == -1);
  CHECK(model.nodes[0].mesh == -1);
  CHECK(model.nodes[1].name == "child");
  CHECK(model.nodes[1].parent == 0);
  CHECK(model.nodes[1].mesh == 0);
  CHECK_NEAR3(model.nodes[1].local.scale, {2.0F, 2.0F, 2.0F});
  CHECK(model.roots.size() == 1U);
  CHECK(model.roots[0] == 0);

  // Failure cases.
  CHECK(!loadGlb({1, 2, 3}, model, error));                    // too small
  CHECK(!loadGlb(buildGlb("not json", {}), model, error));     // bad JSON
  CHECK(!loadGlb(buildGlb(kTriangleJson, triangleBin(9)), model, error)); // index out of range
  std::vector<std::uint8_t> noMagic = buildGlb(kTriangleJson, triangleBin(2));
  noMagic[0] = 0;
  CHECK(!loadGlb(noMagic, model, error));

  // End-to-end fixture for the renderer smoke run (best effort).
  writeFile(".build/test-model.glb", buildGlb(kTriangleJson, triangleBin(2)));
}

void testGlbInterleavedAccessors() {
  using namespace fixtures;
  // One bufferView with byteStride 24: [pos(12) normal(12)] x3, then indices.
  std::vector<std::uint8_t> bin;
  for (int i = 0; i < 3; ++i) {
    float pos[3] = {static_cast<float>(i), 0.0F, 0.0F};
    float normal[3] = {0.0F, 0.0F, 1.0F};
    std::uint8_t const* rawPos = reinterpret_cast<std::uint8_t const*>(pos);
    std::uint8_t const* rawNormal = reinterpret_cast<std::uint8_t const*>(normal);
    bin.insert(bin.end(), rawPos, rawPos + 12);
    bin.insert(bin.end(), rawNormal, rawNormal + 12);
  }
  std::uint16_t indices[3] = {0U, 1U, 2U};
  std::uint8_t const* rawIndices = reinterpret_cast<std::uint8_t const*>(indices);
  bin.insert(bin.end(), rawIndices, rawIndices + 6);

  std::string const json =
      R"({"asset": {"version": "2.0"},)"
      R"( "scenes": [{"nodes": [0]}], "scene": 0,)"
      R"( "nodes": [{"name": "strip", "mesh": 0}],)"
      R"( "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}]}],)"
      R"( "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},)"
      R"(                {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC3"},)"
      R"(                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],)"
      R"( "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 72, "byteStride": 24},)"
      R"(                 {"buffer": 0, "byteOffset": 72, "byteLength": 6}],)"
      R"( "buffers": [{"byteLength": 80}]})";

  Model model;
  std::string error;
  CHECK(loadGlb(buildGlb(json, bin), model, error));
  CHECK(model.meshes.size() == 1U);
  CHECK(model.meshes[0].data.vertices.size() == 3U);
  CHECK_NEAR3(model.meshes[0].data.vertices[2].pos, {2.0F, 0.0F, 0.0F});
  CHECK_NEAR3(model.meshes[0].data.vertices[1].normal, {0.0F, 0.0F, 1.0F}); // strided read
}

void testGlbSizeWrapRejected() {
  using namespace fixtures;
  Model model;
  std::string error;

  // 2^63 byteOffset + 2^63 byteLength wraps to 0 in unchecked size_t maths and
  // would pass a naive bounds check; it must be rejected.
  std::string const wrapBufferView =
      R"({"asset": {"version": "2.0"}, "scenes": [{"nodes": []}], "scene": 0,)"
      R"( "nodes": [], "meshes": [],)"
      R"( "bufferViews": [{"buffer": 0, "byteOffset": 9223372036854775808, )"
      R"("byteLength": 9223372036854775808}],)"
      R"( "buffers": [{"byteLength": 4}]})";
  CHECK(!loadGlb(buildGlb(wrapBufferView, {0, 0, 0, 0}), model, error));

  // A hostile byteStride makes the classic (count-1)*stride+size span wrap to a
  // small value; it must be rejected before any read.
  std::string const wrapStride =
      R"({"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "scene": 0,)"
      R"( "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],)"
      R"( "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],)"
      R"( "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12, )"
      R"("byteStride": 9223372036854775808}],)"
      R"( "buffers": [{"byteLength": 12}]})";
  std::vector<std::uint8_t> bin(12, 0);
  CHECK(!loadGlb(buildGlb(wrapStride, bin), model, error));

  // Negative or fractional counts and indices are rejected as bad integers.
  std::string const negativeCount =
      R"({"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "scene": 0,)"
      R"( "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],)"
      R"( "accessors": [{"bufferView": 0, "componentType": 5126, "count": -1, "type": "VEC3"}],)"
      R"( "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12}],)"
      R"( "buffers": [{"byteLength": 12}]})";
  CHECK(!loadGlb(buildGlb(negativeCount, bin), model, error));
}

void testObjLoader() {
  MtlColours materials;
  CHECK(parseMtl("newmtl Red\nKd 1 0 0\nnewmtl Blue\nKd 0 0 1\n", materials));
  CHECK(materials.size() == 2U);

  std::string text =
      "# comment\n"
      "mtllib ignored.mtl\n"
      "o First\n"
      "usemtl Red\n"
      "v 0 0 0\n"
      "v 1 0 0\n"
      "v 1 1 0\n"
      "v 0 1 0\n"
      "vn 0 0 1\n"
      "f 1//1 2//1 3//1\n"
      "f 1//1 3//1 4//1\n"
      "usemtl Blue\n"
      "f 1//1 4//1 2//1\n"
      "o Second\n"
      "v 2 0 0\n"
      "v 3 0 0\n"
      "v 3 1 0\n"
      "f -3 -2 -1\n";

  Model model;
  std::string error;
  CHECK(loadObj(text, materials, model, error));

  CHECK(model.nodes.size() == 3U);  // First, material split, Second
  CHECK(model.meshes.size() == 3U);
  CHECK(model.nodes[0].name == "First");
  CHECK(model.roots.size() == 3U);
  CHECK_NEAR3(model.meshes[0].colour, {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(model.meshes[1].colour, {0.0F, 0.0F, 1.0F});
  CHECK(model.meshes[0].data.vertices.size() == 6U); // two triangles
  CHECK(model.meshes[1].data.vertices.size() == 3U);
  CHECK(model.meshes[2].data.vertices.size() == 3U);
  CHECK_NEAR3(model.meshes[0].data.vertices[0].normal, {0.0F, 0.0F, 1.0F});
  CHECK_NEAR3(model.meshes[1].data.vertices[0].normal, {0.0F, 0.0F, 1.0F}); // file normal
  CHECK_NEAR3(model.meshes[2].data.vertices[0].pos, {2.0F, 0.0F, 0.0F});    // negative index
  CHECK_NEAR3(model.meshes[2].data.vertices[0].normal, {0.0F, 0.0F, 1.0F}); // generated flat

  CHECK(!loadObj("f 1 2 3\n", materials, model, error));           // no vertices
  CHECK(!loadObj("v 0 0 0\nf 1 2 3\n", materials, model, error));  // out of range
  CHECK(!loadObj("v x y z\n", materials, model, error));           // bad number
  CHECK(!loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1x 2 3\n", materials, model, error)); // strict tokens
  CHECK(!loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n", materials, model, error));  // 0 is invalid
  CHECK(!loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1//1 2//1 3//1\n", materials, model,
                 error)); // no vn defined
  CHECK(model.nodes.empty()); // failed loads leave a clean model
  CHECK(loadObj("  v 0 0 0\n  v 1 0 0\n  v 0 1 0\n  f 1 2 3\n", materials, model, error));
  CHECK(loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nl 1 2\nf 1 2 3\n", materials, model, error));

  // End-to-end fixture for the renderer smoke run (best effort).
  std::vector<std::uint8_t> objBytes(text.begin(), text.end());
  fixtures::writeFile(".build/test-model.obj", objBytes);
}

} // namespace

int main() {
  testVecBasics();
  testQuatMatchesMatrix();
  testMatrixInverseAndDecompose();
  testSceneHierarchy();
  testReparentKeepsLocal();
  testDestroySubtree();
  testEpochSemantics();
  testRayAabb();
  testRayTriangle();
  testPickClosest();
  testGizmoTranslateWorld();
  testGizmoTranslateRotatedParent();
  testGizmoRotateSnapped();
  testGizmoScaleAxis();
  testGizmoLocalSpaceRotate();
  testGizmoDiscontinuityEpoch();
  testGizmoMultiSelection();
  testDecomposeMirror();
  testGizmoMirrorPreserved();
  testGizmoFreeTranslateSnapStaysInPlane();
  testGizmoRotateUnwrap();
  testGizmoScaleClamp();
  testGizmoBeginFailureClean();
  testGizmoPlaneHandle();
  testEpochStructuralChanges();
  testCameraFraming();
  testCameraManipulation();
  testSegmentAndPolygonPicking();
  testHandlePicking();
  testHandlePickingBehindCamera();
  testJsonParser();
  testMeshBounds();
  testGlbLoader();
  testGlbInterleavedAccessors();
  testGlbSizeWrapRejected();
  testObjLoader();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
