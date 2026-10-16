// Core tests for the scene-editor maths: vecmath, scene graph, picking, gizmos.
// Compile and run via run-tests.sh (typed C++ only; no scripting).

#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <vector>

#include "camera.h"
#include "gizmo.h"
#include "handles.h"
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

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
