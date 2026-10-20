// Core tests for the scene-editor maths: vecmath, scene graph, picking, gizmos.
// Compile and run via run-tests.sh (typed C++ only; no scripting).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "camera.h"
#include "editor_state.h"
#include "gizmo.h"
#include "gltf.h"
#include "handles.h"
#include "image.h"
#include "json.h"
#include "mesh.h"
#include "obj.h"
#include "picking.h"
#include "scene.h"
#include "undo.h"
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

void checkNear2(Vec2 a, Vec2 b, char const* expr, char const* file, int line, float eps = 1e-4F) {
  ++g_checks;
  if (std::fabs(a.x - b.x) > eps || std::fabs(a.y - b.y) > eps) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s ((%f,%f) vs (%f,%f))\n", file, line, expr,
                static_cast<double>(a.x), static_cast<double>(a.y), static_cast<double>(b.x),
                static_cast<double>(b.y));
  }
}

#define CHECK_NEAR2(...) checkNear2(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__)

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

void testPickClosestSurface() {
  // Regression: box entry distance is not surface distance. A large box whose
  // entry is closer (t=1) but whose visible surface is far (t=8) must lose to
  // a small object whose surface is genuinely in front (t=4). The old
  // bounds-only pick returned the large one; surface picking must not.
  auto farGeometry = std::make_shared<vredit::PickGeometry>();
  farGeometry->positions = {{-1.0F, -1.0F, 8.0F},
                           {1.0F, -1.0F, 8.0F},
                           {0.0F, 1.0F, 8.0F}};
  farGeometry->indices = {0, 1, 2};

  auto nearGeometry = std::make_shared<vredit::PickGeometry>();
  nearGeometry->positions = {{-0.5F, -0.5F, 4.0F},
                            {0.5F, -0.5F, 4.0F},
                            {0.0F, 0.5F, 4.0F}};
  nearGeometry->indices = {0, 1, 2};

  vredit::PickCandidate large;
  large.node = 1;
  large.worldBounds = {{-2.0F, -2.0F, 1.0F}, {2.0F, 2.0F, 9.0F}};
  large.geometry = farGeometry;

  vredit::PickCandidate small;
  small.node = 2;
  small.worldBounds = {{-0.5F, -0.5F, 3.9F}, {0.5F, 0.5F, 4.1F}};
  small.geometry = nearGeometry;

  std::vector<vredit::PickCandidate> both{large, small};
  vredit::Ray ray{{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}};

  auto hit = vredit::pickClosestSurface(ray, both);
  CHECK(hit.has_value());
  CHECK(hit->node == 2); // the near SURFACE wins, not the near box entry
  CHECK_NEAR(hit->t, 4.0F, 1e-5F);

  // Order must not matter: same result with the near candidate listed first.
  std::vector<vredit::PickCandidate> reversed{small, large};
  auto hit2 = vredit::pickClosestSurface(ray, reversed);
  CHECK(hit2.has_value());
  CHECK(hit2->node == 2);

  // A box-shaped candidate (null geometry = "the bounds are the shape") still
  // picks at its bounds entry: clicking a visible cube face must select it.
  vredit::PickCandidate boxOnly;
  boxOnly.node = 3;
  boxOnly.worldBounds = {{-0.5F, -0.5F, 2.0F}, {0.5F, 0.5F, 3.0F}};
  std::vector<vredit::PickCandidate> withBox{large, boxOnly};
  auto hit3 = vredit::pickClosestSurface(ray, withBox);
  CHECK(hit3.has_value());
  CHECK(hit3->node == 3); // bounds entry t=2 beats the large surface at t=8
  CHECK_NEAR(hit3->t, 2.0F, 1e-5F);

  // But a mesh candidate whose triangles the ray misses is NOT a hit even
  // though the ray passes through its bounds: the bounding box is not part
  // of the click target, so empty space next to a model selects nothing.
  std::vector<vredit::PickCandidate> meshOnly{large};
  vredit::Ray missRay{{3.0F, 3.0F, 0.0F}, {0.0F, 0.0F, 1.0F}}; // inside the bounds, off-triangle
  CHECK(!vredit::pickClosestSurface(missRay, meshOnly).has_value());

  // Ray misses every surface inside the large box: empty volume must not pick.
  vredit::Ray offSide{{0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 1.0F}};
  std::vector<vredit::PickCandidate> onlyLarge{large};
  auto miss = vredit::pickClosestSurface(offSide, onlyLarge);
  CHECK(!miss.has_value());
}

void testSceneSnapshotRestore() {
  vredit::Scene scene;
  vredit::NodeId a = scene.create();
  vredit::NodeId b = scene.create();
  vredit::NodeId child = scene.create(b);
  vredit::Transform ta;
  ta.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(a, ta);
  vredit::Transform tb;
  tb.translation = {0.0F, 2.0F, 0.0F};
  scene.setLocal(b, tb);

  vredit::SceneSnapshot snapshot = scene.snapshot();
  CHECK(snapshot.nodes.size() == 3U);
  CHECK(snapshot.nodes[0].id == a);
  CHECK(snapshot.nodes[1].id == b);
  CHECK(snapshot.nodes[2].parent == b);

  vredit::Transform moved = ta;
  moved.translation = {9.0F, 9.0F, 9.0F};
  scene.setLocal(a, moved);
  scene.setLocal(b, moved);
  CHECK(!vredit::sameSnapshot(snapshot, scene.snapshot()));

  scene.restore(snapshot);
  CHECK(vredit::sameSnapshot(snapshot, scene.snapshot()));
  CHECK_NEAR(scene.local(a).translation.x, 1.0F, 1e-5F);
  CHECK_NEAR(scene.local(b).translation.y, 2.0F, 1e-5F);

  // Restoring after destroying nodes RESURRECTS them with their original ids
  // (this is what makes delete undoable), hierarchy included.
  scene.destroy(b); // subtree: b + child
  CHECK(!scene.valid(b));
  CHECK(!scene.valid(child));
  scene.setLocal(a, moved);
  scene.restore(snapshot);
  CHECK(scene.valid(b));
  CHECK(scene.valid(child));
  CHECK(scene.parent(child) == b);
  CHECK_NEAR(scene.local(a).translation.x, 1.0F, 1e-5F);
  CHECK_NEAR(scene.local(b).translation.y, 2.0F, 1e-5F);
  CHECK(vredit::sameSnapshot(snapshot, scene.snapshot()));

  // Nodes created since the snapshot go away again on restore.
  vredit::NodeId extra = scene.create();
  scene.restore(snapshot);
  CHECK(!scene.valid(extra));
  CHECK(vredit::sameSnapshot(snapshot, scene.snapshot()));

  // A parent destroyed without its recorded child: the child is detached to
  // root rather than left dangling on a dead slot.
  vredit::SceneSnapshot partial;
  partial.nodes.push_back(vredit::NodeRecord{child, vredit::kInvalidNode, {}});
  scene.restore(partial);
  CHECK(scene.valid(child));
  CHECK(scene.parent(child) == vredit::kInvalidNode);
  CHECK(!scene.valid(a));
  CHECK(!scene.valid(b));

  // Restore trusts captured states (acyclic); ids keep their identity across
  // any number of restore cycles.
  scene.restore(snapshot);
  CHECK(scene.valid(a) && scene.valid(b) && scene.valid(child));
  CHECK(vredit::sameSnapshot(snapshot, scene.snapshot()));
}

void testUndoHistory() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t0;
  t0.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(n, t0);

  vredit::UndoHistory<vredit::SceneSnapshot> undo;
  auto apply = [&scene](vredit::SceneSnapshot& s) { scene.restore(s); };
  auto undoStep = [&]() {
    vredit::SceneSnapshot current = scene.snapshot();
    bool did = undo.undo(current);
    if (did) {
      apply(current);
    }
    return did;
  };
  auto redoStep = [&]() {
    vredit::SceneSnapshot current = scene.snapshot();
    bool did = undo.redo(current);
    if (did) {
      apply(current);
    }
    return did;
  };
  CHECK(!undo.canUndo());
  CHECK(!undo.canRedo());

  // One drag: begin before the change, commit after; exactly one undo step.
  undo.begin(scene.snapshot());
  vredit::Transform t1 = t0;
  t1.translation = {5.0F, 0.0F, 0.0F};
  scene.setLocal(n, t1);
  undo.commit(scene.snapshot());
  CHECK(undo.canUndo());
  CHECK(!undo.canRedo());

  CHECK(undoStep());
  CHECK_NEAR(scene.local(n).translation.x, 1.0F, 1e-5F);
  CHECK(undo.canRedo());
  CHECK(!undo.canUndo());

  CHECK(redoStep());
  CHECK_NEAR(scene.local(n).translation.x, 5.0F, 1e-5F);
  CHECK(undo.canUndo());

  // A commit that changed nothing records no step.
  undo.begin(scene.snapshot());
  undo.commit(scene.snapshot());
  CHECK(undoStep()); // only the first drag is undone...
  CHECK_NEAR(scene.local(n).translation.x, 1.0F, 1e-5F);
  CHECK(!undo.canUndo()); // ...and nothing more

  // A new edit after undo clears the redo branch.
  redoStep();
  CHECK_NEAR(scene.local(n).translation.x, 5.0F, 1e-5F);
  undo.begin(scene.snapshot());
  vredit::Transform t2 = t1;
  t2.translation = {7.0F, 0.0F, 0.0F};
  scene.setLocal(n, t2);
  undo.commit(scene.snapshot());
  CHECK(!undo.canRedo());
  CHECK(undoStep());
  CHECK_NEAR(scene.local(n).translation.x, 5.0F, 1e-5F);

  // Regression: begin() with an edit still open (a drag whose release was
  // lost) must commit that edit, not overwrite it — otherwise its transforms
  // are silently lost from history.
  undo.clear();
  undo.begin(scene.snapshot()); // base: x=5
  vredit::Transform t3 = t1;
  t3.translation = {8.0F, 0.0F, 0.0F};
  scene.setLocal(n, t3); // dangling edit, never committed
  undo.begin(scene.snapshot()); // new drag starts at x=8
  vredit::Transform t4 = t1;
  t4.translation = {9.0F, 0.0F, 0.0F};
  scene.setLocal(n, t4);
  undo.commit(scene.snapshot());
  CHECK(undoStep());
  CHECK_NEAR(scene.local(n).translation.x, 8.0F, 1e-5F); // second drag undone
  CHECK(undoStep());
  CHECK_NEAR(scene.local(n).translation.x, 5.0F, 1e-5F); // dangling edit too
}

void testCancelPendingEdit() {
  vredit::Scene scene;
  vredit::NodeId n = scene.create();
  vredit::Transform t0;
  t0.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(n, t0);

  vredit::UndoHistory<vredit::SceneSnapshot> undo;
  {
    vredit::SceneSnapshot current = scene.snapshot();
    CHECK(!undo.cancel(current)); // nothing pending
  }

  // Build one step and take it back, so a redo branch exists.
  undo.begin(scene.snapshot());
  vredit::Transform t1 = t0;
  t1.translation = {5.0F, 0.0F, 0.0F};
  scene.setLocal(n, t1);
  undo.commit(scene.snapshot());
  {
    vredit::SceneSnapshot current = scene.snapshot();
    CHECK(undo.undo(current));
    scene.restore(current);
  }
  CHECK(undo.canRedo());

  // A live grab, cancelled: the pre-drag state returns and no step lands on
  // either stack.
  undo.begin(scene.snapshot());
  vredit::Transform grab = t0;
  grab.translation = {99.0F, 0.0F, 0.0F};
  scene.setLocal(n, grab);
  {
    vredit::SceneSnapshot current = scene.snapshot();
    CHECK(undo.cancel(current));
    scene.restore(current);
  }
  CHECK_NEAR(scene.local(n).translation.x, 1.0F, 1e-5F);
  CHECK(!undo.canUndo()); // the cancelled grab is not undoable...
  CHECK(undo.canRedo());  // ...and it did not eat the existing redo branch

  // Redo replays the ORIGINAL edit, not the cancelled grab.
  {
    vredit::SceneSnapshot current = scene.snapshot();
    CHECK(undo.redo(current));
    scene.restore(current);
  }
  CHECK_NEAR(scene.local(n).translation.x, 5.0F, 1e-5F);
}

void testExpandSelection() {
  // One model file = one group: a plain pick selects every node of the group,
  // a secondary pick just the single part.
  std::vector<std::pair<vredit::NodeId, std::uint32_t>> groups{
      {10, 1U}, {11, 1U}, {12, 1U}, {20, 2U}};
  auto whole = vredit::expandSelection(groups, 11, false);
  CHECK(whole.size() == 3U);
  CHECK(whole[0] == 10U && whole[1] == 11U && whole[2] == 12U);
  auto part = vredit::expandSelection(groups, 11, true);
  CHECK(part.size() == 1U);
  CHECK(part[0] == 11U);
  // Group 0 (ungrouped) selects individually; unknown nodes select themselves.
  std::vector<std::pair<vredit::NodeId, std::uint32_t>> loose{{5, 0U}, {6, 0U}};
  auto single = vredit::expandSelection(loose, 5, false);
  CHECK(single.size() == 1U);
  CHECK(single[0] == 5U);
  auto unknown = vredit::expandSelection(groups, 99, false);
  CHECK(unknown.size() == 1U);
  CHECK(unknown[0] == 99U);
}

void testBoundsHelpers() {
  // transformAabb must be the exact box of the transformed corners (rotation
  // included), and the centre/union helpers the obvious things.
  vredit::Aabb box{{-1.0F, -2.0F, -3.0F}, {1.0F, 2.0F, 3.0F}};
  vredit::Mat4 translated = vredit::translationMatrix({10.0F, 0.0F, 0.0F});
  vredit::Aabb moved = vredit::transformAabb(translated, box);
  CHECK_NEAR3(moved.min, {9.0F, -2.0F, -3.0F});
  CHECK_NEAR3(moved.max, {11.0F, 2.0F, 3.0F});
  CHECK_NEAR3(vredit::aabbCentre(moved), {10.0F, 0.0F, 0.0F});

  // 90 degrees about Z: x and y swap extents.
  vredit::Mat4 rotated = vredit::rotationMatrix(
      vredit::quatFromAxisAngle({0.0F, 0.0F, 1.0F}, kHalfPi));
  vredit::Aabb spun = vredit::transformAabb(rotated, box);
  CHECK_NEAR3(spun.min, {-2.0F, -1.0F, -3.0F});
  CHECK_NEAR3(spun.max, {2.0F, 1.0F, 3.0F});

  vredit::Aabb u = vredit::unite(moved, spun);
  CHECK_NEAR3(u.min, {-2.0F, -2.0F, -3.0F});
  CHECK_NEAR3(u.max, {11.0F, 2.0F, 3.0F});
}

void testGizmoParentChildSelectionRides() {
  // Rigid-group regression: selecting both a parent and its child and moving
  // them must advance the child's world position by exactly one delta. The
  // child rides along with its parent instead of receiving the delta twice.
  vredit::Scene scene;
  vredit::NodeId parent = scene.create();
  vredit::Transform tp;
  tp.translation = {0.0F, 0.0F, 0.0F};
  scene.setLocal(parent, tp);
  vredit::NodeId child = scene.create(parent);
  vredit::Transform tc;
  tc.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(child, tc);

  vredit::GizmoDrag drag;
  vredit::GizmoSettings settings;
  settings.mode = vredit::GizmoMode::Translate;
  CHECK(drag.begin(scene, {parent, child}, {1.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                   {{1.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  drag.update({{3.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}); // +2 along X
  CHECK_NEAR3(scene.worldPosition(parent), {2.0F, 0.0F, 0.0F});
  CHECK_NEAR3(scene.worldPosition(child), {4.0F, 0.0F, 0.0F}); // 2 (parent) + 2 (local)
  drag.end();

  // Selecting only the child still moves it independently of the parent.
  vredit::GizmoDrag partDrag;
  CHECK(partDrag.begin(scene, {child}, {4.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisY,
                       {{4.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F}, settings));
  partDrag.update({{4.0F, 1.5F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR3(scene.worldPosition(child), {4.0F, 1.5F, 0.0F});
  CHECK_NEAR3(scene.worldPosition(parent), {2.0F, 0.0F, 0.0F}); // parent untouched

  // Rotate rides along too: parent+child rotate about the pivot by exactly
  // one angle, so the child stays rigidly attached to its parent.
  vredit::Scene rotScene;
  vredit::NodeId rotParent = rotScene.create();
  vredit::Transform rp;
  rp.translation = {1.0F, 0.0F, 0.0F};
  rotScene.setLocal(rotParent, rp);
  vredit::NodeId rotChild = rotScene.create(rotParent);
  vredit::Transform rc;
  rc.translation = {1.0F, 0.0F, 0.0F};
  rotScene.setLocal(rotChild, rc);
  vredit::GizmoDrag rotDrag;
  vredit::GizmoSettings rotSettings;
  rotSettings.mode = vredit::GizmoMode::Rotate;
  // The start ray must land off-pivot: the angle needs a reference vector.
  CHECK(rotDrag.begin(rotScene, {rotParent, rotChild}, {1.0F, 0.0F, 0.0F}, {},
                      vredit::GizmoHandle::AxisZ,
                      {{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F},
                      rotSettings));
  // Rotate the drag plane point 90 degrees around the pivot on the Z axis.
  rotDrag.update({{1.0F, 1.0F, 5.0F}, {0.0F, 0.0F, -1.0F}});
  CHECK_NEAR3(rotScene.worldPosition(rotParent), {1.0F, 0.0F, 0.0F}); // on the pivot
  CHECK_NEAR3(rotScene.worldPosition(rotChild), {1.0F, 1.0F, 0.0F});  // one quarter-turn
  rotDrag.end();

  // Three-level chain with the MIDDLE node unselected: the grandchild still
  // rides along with its selected grandparent through the unselected parent.
  vredit::Scene chain;
  vredit::NodeId grand = chain.create();
  vredit::NodeId middle = chain.create(grand);
  vredit::Transform mid;
  mid.translation = {1.0F, 0.0F, 0.0F};
  chain.setLocal(middle, mid);
  vredit::NodeId leaf = chain.create(middle);
  vredit::Transform lf;
  lf.translation = {1.0F, 0.0F, 0.0F};
  chain.setLocal(leaf, lf);
  vredit::GizmoDrag chainDrag;
  CHECK(chainDrag.begin(chain, {grand, leaf}, {1.0F, 0.0F, 0.0F}, {}, vredit::GizmoHandle::AxisX,
                        {{1.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}, {0.0F, 0.0F, -1.0F},
                        settings));
  chainDrag.update({{2.0F, 0.0F, 5.0F}, {0.0F, 0.0F, -1.0F}}); // +1 along X
  CHECK_NEAR3(chain.worldPosition(grand), {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(chain.worldPosition(leaf), {3.0F, 0.0F, 0.0F}); // one delta, not two
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
  // Uniform scale ONLY: every handle — axis or free — drives one factor on
  // all three axes ("I never, ever want scaling on just one axis").
  struct Case {
    vredit::GizmoHandle handle;
    vredit::Vec3 view;
    vredit::Ray start;
    vredit::Ray update;
  };
  vredit::Vec3 const downZ{0.0F, 0.0F, -1.0F};
  vredit::Vec3 const downY{0.0F, -1.0F, 0.0F};
  Case const cases[] = {
      {vredit::GizmoHandle::AxisX, downZ, {{2.0F, 0.0F, 5.0F}, downZ}, {{4.0F, 0.0F, 5.0F}, downZ}},
      {vredit::GizmoHandle::AxisY, downZ, {{0.0F, 2.0F, 5.0F}, downZ}, {{0.0F, 4.0F, 5.0F}, downZ}},
      // Z points at a -Z camera (degenerate), so this case views from above.
      {vredit::GizmoHandle::AxisZ, downY, {{2.0F, 5.0F, 2.0F}, downY}, {{2.0F, 5.0F, 4.0F}, downY}},
      {vredit::GizmoHandle::Free, downZ, {{2.0F, 0.0F, 5.0F}, downZ}, {{4.0F, 0.0F, 5.0F}, downZ}},
  };
  for (Case const& c : cases) {
    vredit::Scene scene;
    vredit::NodeId n = scene.create();
    vredit::Transform t;
    t.translation = {2.0F, 0.0F, 0.0F};
    scene.setLocal(n, t);

    vredit::GizmoDrag drag;
    vredit::GizmoSettings settings;
    settings.mode = vredit::GizmoMode::Scale;
    settings.snapStep = 0.25F;
    CHECK(drag.begin(scene, {n}, {0.0F, 0.0F, 0.0F}, {}, c.handle, c.start, c.view, settings));
    drag.update(c.update);
    CHECK_NEAR(drag.cumulative(), 2.0F, 1e-5F);
    CHECK_NEAR3(scene.local(n).scale, {2.0F, 2.0F, 2.0F});
    // Scaling about the pivot (origin) also doubles the offset.
    CHECK_NEAR3(scene.worldPosition(n), {4.0F, 0.0F, 0.0F});
  }
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
  camera.position = {0.0F, 0.0F, 5.0F};
  camera.yaw = 0.0F;
  camera.pitch = 0.0F;

  // First-person basis: yaw 0 looks toward -Z.
  CHECK_NEAR3(cameraForward(camera), {0.0F, 0.0F, -1.0F});
  CHECK_NEAR3(cameraRight(camera), {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(cameraUp(camera), {0.0F, 1.0F, 0.0F});
  CHECK_NEAR3(transformPoint(cameraView(camera), {0.0F, 0.0F, 0.0F}), {0.0F, 0.0F, -5.0F});

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
  camera.position = {1.0F, 2.0F, 3.0F};
  camera.yaw = 0.0F;
  camera.pitch = 0.0F;

  // Mouse look turns in place.
  cameraLook(camera, kHalfPi, 0.0F);
  CHECK_NEAR3(cameraForward(camera), {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(cameraRight(camera), {0.0F, 0.0F, 1.0F}); // facing +X, right is +Z
  CHECK_NEAR3(camera.position, {1.0F, 2.0F, 3.0F});

  // Pitch clamps short of gimbal flip.
  cameraLook(camera, 0.0F, 3.0F);
  CHECK(camera.pitch < 1.5707964F);
  cameraLook(camera, 0.0F, -3.0F);
  CHECK(camera.pitch > -1.5707964F);

  // Back to the yaw-0 basis for the movement expectations below.
  cameraLook(camera, -kHalfPi, -camera.pitch);
  CHECK_NEAR3(cameraForward(camera), {0.0F, 0.0F, -1.0F});

  // Fly: local x = right, y = up, z = forward.
  cameraFly(camera, {2.0F, 0.0F, 0.0F});
  CHECK_NEAR3(camera.position, {3.0F, 2.0F, 3.0F});
  cameraFly(camera, {0.0F, 1.5F, 0.0F});
  CHECK_NEAR3(camera.position, {3.0F, 3.5F, 3.0F});
  cameraFly(camera, {0.0F, 0.0F, -4.0F});
  CHECK_NEAR3(camera.position, {3.0F, 3.5F, 7.0F}); // backward is +Z at yaw 0

  // Fly-where-you-look (VR): forward follows the head's PITCH — looking down
  // and pushing the stick forward descends (that is the height control) — and
  // the signs match cameraFly's forward-positive convention.
  Camera lookCam;
  lookCam.pitch = 0.0F; // the expectations below assume a level rig
  Vec3 level = cameraFlyLook(lookCam, {0.0F, 0.0F, -1.0F}, {0.0F, 0.0F, 2.0F});
  CHECK_NEAR3(level, {0.0F, 0.0F, -2.0F}); // stick forward = world forward
  Vec3 pitched = cameraFlyLook(lookCam, {0.0F, -0.5F, -0.8660254F}, {0.0F, 0.0F, 2.0F});
  CHECK_NEAR3(pitched, {0.0F, -1.0F, -1.7320508F}); // forward 1.732, down 1
  Vec3 strafe = cameraFlyLook(lookCam, {0.0F, -0.5F, -0.8660254F}, {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(strafe, {1.0F, 0.0F, 0.0F}); // strafe stays level

  // Wheel dolly moves along the view direction.
  cameraDolly(camera, 1.0F);
  CHECK_NEAR3(camera.position, {3.0F, 3.5F, 6.0F});

  // Pan: dragging right moves the camera left relative to the world.
  cameraPan(camera, 10.0F, 0.0F, 600.0F);
  CHECK(camera.position.x < 3.0F);
}

void testGravityAlign() {
  using vredit::gravityAlign;
  using vredit::Quat;
  using vredit::quatFromAxisAngle;
  using vredit::rotate;
  using vredit::Vec3;

  // A pure roll about the forward axis is removed entirely.
  Quat const rolled = quatFromAxisAngle({0.0F, 0.0F, -1.0F}, 0.5F);
  Vec3 const flatUp = rotate(gravityAlign(rolled), {0.0F, 1.0F, 0.0F});
  CHECK_NEAR(flatUp.x, 0.0F, 1e-5F);
  CHECK_NEAR(flatUp.y, 1.0F, 1e-5F);
  CHECK_NEAR(flatUp.z, 0.0F, 1e-5F);

  // A full head orientation (yaw + pitch + roll): the look direction is
  // preserved EXACTLY (the correction rotates about it) and the result has
  // no roll — its right axis is horizontal.
  Quat const head = quatFromAxisAngle({0.0F, 1.0F, 0.0F}, 0.7F) *
                    quatFromAxisAngle({1.0F, 0.0F, 0.0F}, -0.4F) *
                    quatFromAxisAngle({0.0F, 0.0F, -1.0F}, 0.3F);
  Quat const aligned = gravityAlign(head);
  CHECK_NEAR(rotate(aligned, {0.0F, 0.0F, -1.0F}).x, rotate(head, {0.0F, 0.0F, -1.0F}).x, 1e-6F);
  CHECK_NEAR(rotate(aligned, {0.0F, 0.0F, -1.0F}).y, rotate(head, {0.0F, 0.0F, -1.0F}).y, 1e-6F);
  CHECK_NEAR(rotate(aligned, {0.0F, 0.0F, -1.0F}).z, rotate(head, {0.0F, 0.0F, -1.0F}).z, 1e-6F);
  CHECK_NEAR(rotate(aligned, {1.0F, 0.0F, 0.0F}).y, 0.0F, 1e-5F);

  // A zero-roll head is returned unchanged.
  Quat const level = quatFromAxisAngle({0.0F, 1.0F, 0.0F}, 0.7F) *
                     quatFromAxisAngle({1.0F, 0.0F, 0.0F}, -0.4F);
  CHECK_NEAR(rotate(gravityAlign(level), {0.0F, 1.0F, 0.0F}).x, rotate(level, {0.0F, 1.0F, 0.0F}).x,
             1e-6F);
  CHECK_NEAR(rotate(gravityAlign(level), {0.0F, 1.0F, 0.0F}).y, rotate(level, {0.0F, 1.0F, 0.0F}).y,
             1e-6F);

  // The shell's composition: the correction preserves the forward exactly and
  // rigidly rotates eye offsets (length preserved) about the head centre.
  Quat const correction = aligned * conjugate(normalized(head));
  Vec3 const offset{0.032F, 0.0F, 0.0F};
  Vec3 const rotatedOffset = rotate(correction, offset);
  CHECK_NEAR(dot(rotatedOffset, rotatedOffset), dot(offset, offset), 1e-6F);

  // Idempotent, and a degenerate look direction stays finite.
  CHECK_NEAR3(rotate(gravityAlign(aligned), {0.0F, 1.0F, 0.0F}),
              rotate(aligned, {0.0F, 1.0F, 0.0F}));
  Quat const lookingUp = quatFromAxisAngle({1.0F, 0.0F, 0.0F}, -1.5707964F);
  Vec3 const v = rotate(gravityAlign(lookingUp), {0.0F, 0.0F, -1.0F});
  CHECK(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z));

  // Near vertical the correction fades out (it would act as a yaw and let
  // tracking noise spin the world): a rolled head 0.5 degrees off vertical
  // comes back essentially unchanged.
  Quat const nearVertical =
      quatFromAxisAngle({1.0F, 0.0F, 0.0F}, -1.5707964F + 0.0087F) *
      quatFromAxisAngle({0.0F, 0.0F, -1.0F}, 0.3F);
  Quat const faded = gravityAlign(nearVertical);
  CHECK_NEAR(rotate(faded, {0.0F, 1.0F, 0.0F}).x, rotate(nearVertical, {0.0F, 1.0F, 0.0F}).x,
             1e-3F);
  CHECK_NEAR(rotate(faded, {0.0F, 1.0F, 0.0F}).y, rotate(nearVertical, {0.0F, 1.0F, 0.0F}).y,
             1e-3F);

  // Phantom roll: a PITCHED rig composed with a YAWED head produces roll in
  // the total view even though each part is individually roll-free — that is
  // what tilted the world as the user turned. The app's fix strips the roll
  // from the composition (gravityAlign(cameraOrientation(rig) * eyeOrient));
  // this test pins that the result is level and that the test detects roll.
  Camera rigCam;
  rigCam.yaw = 0.8F;
  rigCam.pitch = 0.5F;
  Quat const rigQuat = cameraOrientation(rigCam);
  // cameraOrientation must match cameraForward's convention exactly (a
  // positive yaw turns toward +X): this is the sign bug the review caught.
  CHECK_NEAR3(rotate(rigQuat, {0.0F, 0.0F, -1.0F}), cameraForward(rigCam));
  Quat const headPose = quatFromAxisAngle({0.0F, 1.0F, 0.0F}, 1.0F) *
                        quatFromAxisAngle({1.0F, 0.0F, 0.0F}, 0.2F);
  Quat const total = rigQuat * headPose;
  // World up in eye space = the view rotation's first row (column-major
  // m[0], m[4], m[8]... no: the eye-right row is m[0]/m[4]/m[8]; the
  // world-up x in eye space is m[4]); roll shows up as a non-zero x there.
  Mat4 const rolledView = cameraViewFromPose({0.0F, 0.0F, 0.0F}, total);
  CHECK(std::abs(rolledView.m[4]) > 0.05F); // the un-fixed composition IS rolled
  Mat4 const levelView = cameraViewFromPose({0.0F, 0.0F, 0.0F}, gravityAlign(total));
  CHECK_NEAR(levelView.m[4], 0.0F, 1e-4F);  // the fix leaves the horizon level
  // The look direction is preserved through the lock.
  CHECK_NEAR3(rotate(gravityAlign(total), {0.0F, 0.0F, -1.0F}),
              rotate(total, {0.0F, 0.0F, -1.0F}));
}

void testCameraLookPivot() {
  Camera camera;
  camera.position = {1.0F, 2.0F, 3.0F};
  camera.yaw = 0.0F;
  camera.pitch = 0.0F;

  // A turn around a local pivot keeps that pivot's world position fixed —
  // this is what makes VR smooth-turn spin in place instead of orbiting the
  // tracking origin (the head is offset from it).
  Vec3 const pivotLocal{0.4F, -0.2F, -1.5F};
  Mat4 beforeWorld{};
  CHECK(vredit::invertAffine(cameraView(camera), beforeWorld));
  Vec3 const pivotWorld = vredit::transformPoint(beforeWorld, pivotLocal);

  cameraLookPivot(camera, pivotLocal, kHalfPi, 0.0F);

  Mat4 afterWorld{};
  CHECK(vredit::invertAffine(cameraView(camera), afterWorld));
  CHECK_NEAR3(vredit::transformPoint(afterWorld, pivotLocal), pivotWorld);
  // The orientation still turned; only the position compensated.
  CHECK_NEAR3(cameraForward(camera), {1.0F, 0.0F, 0.0F});

  // With the pivot at the camera origin the compensation is zero: plain look.
  Camera plain;
  plain.position = {1.0F, 2.0F, 3.0F};
  plain.pitch = 0.0F;
  cameraLookPivot(plain, {0.0F, 0.0F, 0.0F}, kHalfPi, 0.0F);
  CHECK_NEAR3(plain.position, {1.0F, 2.0F, 3.0F});
  CHECK_NEAR3(cameraForward(plain), {1.0F, 0.0F, 0.0F});

  // Pitching around the pivot keeps it fixed too.
  Mat4 midWorld{};
  CHECK(vredit::invertAffine(cameraView(camera), midWorld));
  Vec3 const midPivot = vredit::transformPoint(midWorld, pivotLocal);
  cameraLookPivot(camera, pivotLocal, 0.0F, 0.7F);
  Mat4 pitchedWorld{};
  CHECK(vredit::invertAffine(cameraView(camera), pitchedWorld));
  CHECK_NEAR3(vredit::transformPoint(pitchedWorld, pivotLocal), midPivot);

  // Combined turns never roll the horizon: the camera's right vector stays
  // horizontal at any yaw and pitch ("not level with the ground as they
  // turn" cannot come back), while the stick pitch does accumulate.
  Camera roll;
  roll.pitch = 0.0F;
  for (int i = 0; i < 7; ++i) {
    cameraLookPivot(roll, {0.4F, 0.3F, -0.2F}, 0.37F, -0.21F);
    CHECK_NEAR(vredit::cameraRight(roll).y, 0.0F, 1e-5F);
  }
  CHECK(roll.pitch < 0.0F);
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

void testStickMotion() {
  // Stick-driven move: horizontal only — models never leave the floor.
  vredit::Scene scene;
  vredit::NodeId parent = scene.create();
  vredit::NodeId child = scene.create(parent);
  vredit::NodeVisual vp;
  vp.id = parent;
  vredit::NodeVisual vc;
  vc.id = child;
  std::vector<vredit::NodeVisual> visuals{vp, vc};

  vredit::moveSelectionHorizontal(scene, {parent, child}, {2.0F, 5.0F, -3.0F});
  CHECK_NEAR3(scene.worldPosition(parent), {2.0F, 0.0F, -3.0F});
  // The selected child rides along ONCE: its local transform is untouched
  // (no double-move through the hierarchy).
  CHECK_NEAR3(scene.local(child).translation, {0.0F, 0.0F, 0.0F});
  CHECK_NEAR3(scene.worldPosition(child), {2.0F, 0.0F, -3.0F});

  // The y component of the requested delta is ignored entirely.
  vredit::moveSelectionHorizontal(scene, {parent}, {0.0F, 100.0F, 0.0F});
  CHECK_NEAR(scene.worldPosition(parent).y, 0.0F, 1e-6F);

  // Stick-driven rotate: turntable around world up through the selection's
  // bounds centre; height is preserved, the model never tips over.
  vredit::Transform t;
  t.translation = {2.0F, 1.0F, 0.0F};
  scene.setLocal(parent, t);
  vredit::rotateSelectionYaw(scene, visuals, {parent}, kHalfPi);
  CHECK_NEAR(scene.worldPosition(parent).y, 1.0F, 1e-5F);
  // A quarter turn around its own centre keeps the centre fixed.
  CHECK_NEAR3(scene.worldPosition(parent), {2.0F, 1.0F, 0.0F});
  vredit::Transform childStart;
  childStart.translation = {1.0F, 0.0F, 0.0F};
  scene.setLocal(child, childStart);
  vredit::rotateSelectionYaw(scene, visuals, {parent, child}, kHalfPi);
  // Selected child rides along once: both turn about the shared bounds
  // centre ((2,1,-0.5): the two unit cubes at (2,1,0) and (2,1,-1)).
  CHECK_NEAR3(scene.worldPosition(child), {1.5F, 1.0F, -0.5F});
  CHECK_NEAR3(scene.worldPosition(parent), {2.5F, 1.0F, -0.5F});
  CHECK_NEAR3(scene.local(child).translation, {1.0F, 0.0F, 0.0F});
}

void testScaleSelectionUniform() {
  // Uniform scale primitive: grow/shrink a model about its bounds centre.
  // Points at (2,0,0) and (2,0,-1) give a known centre (2,0,-0.5).
  vredit::Scene scene;
  vredit::NodeId parent = scene.create();
  vredit::NodeId child = scene.create(parent);
  vredit::NodeVisual vp;
  vp.id = parent;
  vredit::NodeVisual vc;
  vc.id = child;
  std::vector<vredit::NodeVisual> visuals{vp, vc};

  vredit::Transform parentStart;
  parentStart.translation = {2.0F, 0.0F, 0.0F};
  scene.setLocal(parent, parentStart);
  vredit::Transform childStart;
  childStart.translation = {0.0F, 0.0F, -1.0F};
  scene.setLocal(child, childStart);

  CHECK(vredit::scaleSelectionUniform(scene, visuals, {parent, child}, 2.0F));
  // The bounds centre stays put and both points spread 2x from it.
  CHECK_NEAR3(scene.worldPosition(parent), {2.0F, 0.0F, 0.5F});
  // The selected child RIDES: local untouched, world scaled with the parent.
  CHECK_NEAR3(scene.local(child).translation, {0.0F, 0.0F, -1.0F});
  CHECK_NEAR3(scene.worldPosition(child), {2.0F, 0.0F, -1.5F});

  // UNIFORM: every basis axis of the world matrix grows identically (the
  // columns are the basis vectors in this column-major Mat4).
  vredit::Mat4 const world = scene.world(parent);
  CHECK_NEAR(vredit::length(vredit::Vec3{world.m[0], world.m[1], world.m[2]}), 2.0F, 1e-5F);
  CHECK_NEAR(vredit::length(vredit::Vec3{world.m[4], world.m[5], world.m[6]}), 2.0F, 1e-5F);
  CHECK_NEAR(vredit::length(vredit::Vec3{world.m[8], world.m[9], world.m[10]}), 2.0F, 1e-5F);

  // Guards: zero/negative factors and an empty selection change nothing.
  vredit::Vec3 const before = scene.worldPosition(parent);
  CHECK(!vredit::scaleSelectionUniform(scene, visuals, {parent}, 0.0F));
  CHECK(!vredit::scaleSelectionUniform(scene, visuals, {parent}, -1.0F));
  CHECK(!vredit::scaleSelectionUniform(scene, visuals, {}, 1.5F));
  CHECK_NEAR3(scene.worldPosition(parent), before);
}

void testEditorStateOps() {
  using vredit::EditorSnapshot;
  using vredit::ModelGroups;
  using vredit::NodeVisual;

  vredit::Scene scene;
  std::vector<NodeVisual> visuals;
  ModelGroups groups;
  std::vector<vredit::NodeId> selection;

  // One model file = one group: a root with two parts.
  vredit::NodeId root = scene.create();
  vredit::NodeId partA = scene.create(root);
  vredit::NodeId partB = scene.create(root);
  NodeVisual vr;
  vr.id = root;
  vr.colour = {0.9F, 0.1F, 0.1F};
  NodeVisual va;
  va.id = partA;
  NodeVisual vb;
  vb.id = partB;
  visuals = {vr, va, vb};
  groups = {{root, 1U}, {partA, 1U}, {partB, 1U}};

  // Capture/restore round-trips the FULL state: structure, visuals, groups.
  EditorSnapshot snapshot = vredit::captureState(scene, visuals, groups, selection);
  CHECK(snapshot.visuals.size() == 3U);
  CHECK(snapshot.groups.size() == 3U);
  CHECK(vredit::sameSnapshot(snapshot, vredit::captureState(scene, visuals, groups, selection)));

  // Deleting a part: the subtree root goes, its visuals and group entries
  // with it; the rest of the model survives.
  selection = {partA};
  vredit::deleteSubtrees(scene, visuals, groups, selection, {partA});
  CHECK(!scene.valid(partA));
  CHECK(scene.valid(root) && scene.valid(partB));
  CHECK(visuals.size() == 2U);
  CHECK(groups.size() == 2U);
  CHECK(selection.empty()); // the deleted id left the selection

  // Undo restores it exactly — same ids, same visuals, same colours.
  vredit::restoreState(snapshot, scene, visuals, groups, selection);
  CHECK(scene.valid(partA));
  CHECK(visuals.size() == 3U);
  CHECK_NEAR(visuals[0].colour.x, 0.9F, 1e-6F);
  CHECK(vredit::sameSnapshot(snapshot, vredit::captureState(scene, visuals, groups, selection)));

  // Whole-model delete = the same rule over the whole selection: roots that
  // are descendants of other roots collapse, so {root, partA, partB} is one
  // subtree delete, not three.
  selection = {root, partA, partB};
  vredit::deleteSubtrees(scene, visuals, groups, selection, {root, partA, partB});
  CHECK(visuals.empty());
  CHECK(groups.empty());
  CHECK(selection.empty());
  vredit::restoreState(snapshot, scene, visuals, groups, selection);
  CHECK(vredit::sameSnapshot(snapshot, vredit::captureState(scene, visuals, groups, selection)));

  // Spawn placement: the loaded subtree lands ON THE FLOOR centred at target.
  vredit::Transform tall;
  tall.translation = {0.0F, 5.0F, 0.0F}; // parts float above the origin
  scene.setLocal(root, tall);
  vredit::placeOnFloor(scene, visuals, {root}, {3.0F, 0.0F, -2.0F});
  vredit::Aabb bounds{};
  CHECK(vredit::nodesBounds(scene, visuals, {root, partA, partB}, bounds));
  CHECK_NEAR(bounds.min.y, 0.0F, 1e-5F);   // standing on the floor
  CHECK_NEAR(vredit::aabbCentre(bounds).x, 3.0F, 1e-4F);
  CHECK_NEAR(vredit::aabbCentre(bounds).z, -2.0F, 1e-4F);
}

void testHandlePicking() {
  ProjectFn project = [](Vec3 const& p) { return Vec2{p.x, p.y}; };

  auto shapes = buildHandleShapes(GizmoMode::Translate, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F,
                                  project);
  CHECK(pickHandle(shapes, {0.5F, 0.05F}, 0.1F) == GizmoHandle::AxisX);
  CHECK(pickHandle(shapes, {0.05F, 0.5F}, 0.1F) == GizmoHandle::AxisY);
  // The planar squares are gone (user request): the band between the axes
  // picks nothing.
  CHECK(!pickHandle(shapes, {0.35F, 0.35F}, 0.1F).has_value());
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
  mesh.vertices.push_back({{-1.0F, 2.0F, -3.0F}, {}, {}, {}});
  mesh.vertices.push_back({{4.0F, -5.0F, 6.0F}, {}, {}, {}});
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

void testImageDecode() {
  // Fixtures are generated by run-tests.sh with ImageMagick; without them these
  // checks are skipped.
  auto load = [](char const* path, Image& image, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      return false;
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    return decodeImage(bytes.data(), bytes.size(), image, error);
  };

  Image image;
  std::string error;
  if (!load(".build/fixtures/red8.png", image, error)) {
    std::printf("  (image fixture tests skipped: .build/fixtures not present)\n");
    return;
  }

  CHECK(image.width == 8U && image.height == 8U);
  CHECK(image.rgba[0] == 255 && image.rgba[1] == 0 && image.rgba[2] == 0 && image.rgba[3] == 255);

  CHECK(load(".build/fixtures/blue16.png", image, error)); // 16-bit depth -> high byte
  CHECK(image.rgba[0] == 32 && image.rgba[1] == 64 && image.rgba[2] == 128);

  CHECK(load(".build/fixtures/grey8.png", image, error));
  CHECK(image.rgba[0] == 128 && image.rgba[1] == 128 && image.rgba[2] == 128);

  CHECK(load(".build/fixtures/alpha.png", image, error)); // straight alpha preserved
  CHECK(image.rgba[0] == 255 && image.rgba[1] == 0);
  CHECK(std::abs(static_cast<int>(image.rgba[3]) - 128) <= 2);

  CHECK(load(".build/fixtures/palette.png", image, error)); // palette colour type
  CHECK(image.rgba[0] == 255 && image.rgba[1] == 0 && image.rgba[2] == 0);

  CHECK(load(".build/fixtures/checker2.png", image, error));
  CHECK(image.rgba[0] == 255 && image.rgba[1] == 0);       // (0,0) red
  CHECK(image.rgba[4] == 0 && image.rgba[6] == 255);       // (1,0) blue
  CHECK(image.rgba[8] == 0 && image.rgba[10] == 255);      // (0,1) blue
  CHECK(image.rgba[12] == 255 && image.rgba[14] == 0);     // (1,1) red

  // Baseline JPEG: solid colours survive within tolerance.
  CHECK(load(".build/fixtures/blue.jpg", image, error));
  CHECK(image.width == 8U && image.height == 8U);
  CHECK(std::abs(static_cast<int>(image.rgba[0]) - 48) <= 10);
  CHECK(std::abs(static_cast<int>(image.rgba[1]) - 96) <= 10);
  CHECK(std::abs(static_cast<int>(image.rgba[2]) - 160) <= 10);

  CHECK(load(".build/fixtures/grey.jpg", image, error));
  CHECK(std::abs(static_cast<int>(image.rgba[0]) - 64) <= 10);
  CHECK(std::abs(static_cast<int>(image.rgba[2]) - 64) <= 10);

  CHECK(load(".build/fixtures/two.jpg", image, error)); // halves: red | blue
  CHECK(image.width == 16U && image.height == 8U);
  CHECK(image.rgba[0] > 150 && image.rgba[2] < 100);  // red side
  CHECK(image.rgba[(15U * 4U) + 2U] > 150);           // blue side

  // Progressive JPEG is rejected cleanly (not silently corrupted).
  CHECK(!load(".build/fixtures/progressive.jpg", image, error));
  CHECK(error.find("progressive") != std::string::npos);

  // Unknown formats are rejected.
  std::vector<std::uint8_t> junk{0x00, 0x01, 0x02, 0x03};
  CHECK(!decodeImage(junk.data(), junk.size(), image, error));
  std::vector<std::uint8_t> truncatedPng{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00};
  CHECK(!decodeImage(truncatedPng.data(), truncatedPng.size(), image, error));
}

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

void testDecodeSizeCaps() {
  // Production textures exceed the old 64M-pixel decode cap (a 12288^2 normal
  // map is ~151M pixels); absurd declarations must still be rejected up front.
  auto load = [](char const* path, Image& image, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      return false;
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    return decodeImage(bytes.data(), bytes.size(), image, error);
  };

  Image image;
  std::string error;
  if (!load(".build/fixtures/big9k.png", image, error)) {
    std::printf("  (size cap test skipped: .build/fixtures not present)\n");
    return;
  }
  CHECK(image.width == 9000U && image.height == 9000U); // 81M pixels: allowed
  CHECK(!load(".build/fixtures/huge20k.png", image, error));
  CHECK(error.find("bad dimensions") != std::string::npos); // 400M pixels: rejected
}

void testGlbTextureAndUv() {
  using namespace fixtures;
  // Reads the checkerboard PNG fixture and embeds it as an image bufferView.
  std::ifstream checker(".build/fixtures/checker2.png", std::ios::binary);
  if (!checker) {
    std::printf("  (glb texture fixture test skipped: .build/fixtures not present)\n");
    return;
  }
  std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(checker)),
                                std::istreambuf_iterator<char>());
  std::ifstream normalUpFile(".build/fixtures/normal-up.png", std::ios::binary);
  std::vector<std::uint8_t> normalUp((std::istreambuf_iterator<char>(normalUpFile)),
                                     std::istreambuf_iterator<char>());

  // BIN: positions (36) + texcoords (24) + indices (6) + PNG bytes + normal PNG.
  std::vector<std::uint8_t> bin;
  float const positions[9] = {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  float const uvs[6] = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F};
  std::uint8_t const* rawPos = reinterpret_cast<std::uint8_t const*>(positions);
  std::uint8_t const* rawUv = reinterpret_cast<std::uint8_t const*>(uvs);
  bin.insert(bin.end(), rawPos, rawPos + 36);
  bin.insert(bin.end(), rawUv, rawUv + 24);
  std::uint16_t indices[3] = {0U, 1U, 2U};
  std::uint8_t const* rawIndices = reinterpret_cast<std::uint8_t const*>(indices);
  bin.insert(bin.end(), rawIndices, rawIndices + 6);
  std::size_t pngOffset = bin.size();
  bin.insert(bin.end(), png.begin(), png.end());
  std::size_t normalOffset = bin.size();
  bin.insert(bin.end(), normalUp.begin(), normalUp.end());

  std::string json =
      R"({"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "scene": 0,)"
      R"( "nodes": [{"name": "tex", "mesh": 0}],)"
      R"( "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, )"
      R"("indices": 2, "material": 0}]}],)"
      R"( "materials": [{"alphaMode": "MASK", "alphaCutoff": 0.4, )"
      R"("pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},)"
      R"( "normalTexture": {"index": 1, "scale": 0.7}}],)"
      R"( "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}],)"
      R"( "samplers": [{"wrapS": 33071, "wrapT": 10497}],)"
      R"( "images": [{"bufferView": 2, "mimeType": "image/png"},)"
      R"(             {"bufferView": 3, "mimeType": "image/png"}],)"
      R"( "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},)"
      R"(                {"bufferView": 0, "byteOffset": 36, "componentType": 5126, "count": 3, "type": "VEC2"},)"
      R"(                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],)"
      R"( "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 60},)"
      R"(                 {"buffer": 0, "byteOffset": 60, "byteLength": 6},)"
      R"(                 {"buffer": 0, "byteOffset": )"
      + std::to_string(pngOffset) + R"(, "byteLength": )" + std::to_string(png.size()) + R"(},)"
      R"(                 {"buffer": 0, "byteOffset": )"
      + std::to_string(normalOffset) + R"(, "byteLength": )" + std::to_string(normalUp.size()) +
      R"(}],)"
      R"( "buffers": [{"byteLength": )" + std::to_string(bin.size()) + R"(}]})";

  Model model;
  std::string error;
  CHECK(loadGlb(buildGlb(json, bin), model, error));
  CHECK(model.textures.size() == 2U);
  CHECK(model.textures[0].image.width == 2U && model.textures[0].image.height == 2U);
  CHECK(!model.textures[0].repeatU); // wrapS 33071 = clamp
  CHECK(model.textures[0].repeatV);
  CHECK(model.meshes.size() == 1U);
  CHECK(model.meshes[0].texture == 0);
  CHECK(model.meshes[0].normalTexture == 1); // material.normalTexture index
  CHECK_NEAR(model.meshes[0].normalScale, 0.7F, 1e-6F);
  CHECK(model.meshes[0].alpha == AlphaMode::Mask);
  CHECK_NEAR(model.meshes[0].alphaCutoff, 0.4F, 1e-6F);
  CHECK_NEAR(model.meshes[0].alphaFactor, 1.0F, 1e-6F);
  CHECK_NEAR2(model.meshes[0].data.vertices[1].uv, {1.0F, 0.0F});
  CHECK_NEAR2(model.meshes[0].data.vertices[2].uv, {0.0F, 1.0F});
  CHECK(model.meshes[0].data.vertices[0].normal.z > 0.9F); // generated normal (+Z)

  // End-to-end fixture for the renderer smoke run (best effort). The normal
  // map is a flat +Y tangent normal: normal-mapped shading must brighten the
  // scene vs flat normals (the smoke toggles it and compares luminance).
  writeFile(".build/test-textured.glb", buildGlb(json, bin));
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
  MtlMaterials materials;
  CHECK(parseMtl("newmtl Red\nKd 1 0 0\nmap_Kd red.png\nnewmtl Blue\nKd 0 0 1\nd 0.5\n",
                 materials));
  CHECK(materials.size() == 2U);
  CHECK(materials[0].mapKd == "red.png");
  CHECK_NEAR(materials[1].dissolve, 0.5F, 1e-6F);

  materials[0].texture = 0;
  std::vector<ModelTexture> textures(1);
  textures[0].image.width = 2;
  textures[0].image.height = 2;
  textures[0].image.rgba.assign(16U, 255U);

  std::string text =
      "# comment\n"
      "mtllib ignored.mtl\n"
      "o First\n"
      "usemtl Red\n"
      "v 0 0 0\n"
      "v 1 0 0\n"
      "v 1 1 0\n"
      "v 0 1 0\n"
      "vt 0 0\n"
      "vt 1 0\n"
      "vt 1 1\n"
      "vt 0 1\n"
      "vn 0 0 1\n"
      "f 1/1/1 2/2/1 3/3/1\n"
      "f 1/1/1 3/3/1 4/4/1\n"
      "usemtl Blue\n"
      "f 1//1 4//1 2//1\n"
      "o Second\n"
      "v 2 0 0\n"
      "v 3 0 0\n"
      "v 3 1 0\n"
      "f -3 -2 -1\n";

  Model model;
  std::string error;
  bool loaded = loadObj(text, materials, textures, model, error);
  if (!loaded) {
    std::fprintf(stderr, "(debug) loadObj failed: %s\n", error.c_str());
  }
  CHECK(loaded);

  CHECK(model.nodes.size() == 3U);  // First, material split, Second
  CHECK(model.meshes.size() == 3U);
  CHECK(model.textures.size() == 1U); // carried through from the argument
  CHECK(model.nodes[0].name == "First");
  CHECK(model.roots.size() == 3U);
  CHECK_NEAR3(model.meshes[0].colour, {1.0F, 0.0F, 0.0F});
  CHECK_NEAR3(model.meshes[1].colour, {0.0F, 0.0F, 1.0F});
  CHECK(model.meshes[0].texture == 0); // from usemtl Red
  CHECK(model.meshes[1].texture == -1);
  CHECK(model.meshes[1].alpha == AlphaMode::Blend); // d 0.5
  CHECK_NEAR(model.meshes[1].alphaFactor, 0.5F, 1e-6F);
  CHECK(model.meshes[0].data.vertices.size() == 6U); // two triangles
  CHECK(model.meshes[1].data.vertices.size() == 3U);
  CHECK(model.meshes[2].data.vertices.size() == 3U);
  CHECK_NEAR3(model.meshes[0].data.vertices[0].normal, {0.0F, 0.0F, 1.0F});
  CHECK_NEAR3(model.meshes[1].data.vertices[0].normal, {0.0F, 0.0F, 1.0F}); // file normal
  CHECK_NEAR2(model.meshes[0].data.vertices[1].uv, {1.0F, 0.0F});           // vt carried
  CHECK_NEAR2(model.meshes[0].data.vertices[2].uv, {1.0F, 1.0F});
  CHECK_NEAR2(model.meshes[2].data.vertices[0].uv, {0.0F, 0.0F}); // no vt on face
  CHECK_NEAR3(model.meshes[2].data.vertices[0].pos, {2.0F, 0.0F, 0.0F});    // negative index
  CHECK_NEAR3(model.meshes[2].data.vertices[0].normal, {0.0F, 0.0F, 1.0F}); // generated flat

  CHECK(!loadObj("f 1 2 3\n", materials, textures, model, error));           // no vertices
  CHECK(!loadObj("v 0 0 0\nf 1 2 3\n", materials, textures, model, error));  // out of range
  CHECK(!loadObj("v x y z\n", materials, textures, model, error));           // bad number
  CHECK(!loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1x 2 3\n", materials, textures, model,
                 error)); // strict tokens
  CHECK(!loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n", materials, textures, model,
                 error)); // 0 is invalid
  CHECK(!loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1//1 2//1 3//1\n", materials, textures, model,
                 error)); // no vn defined
  CHECK(!loadObj("v 0 0 0\nvt 0 0\nv 1 0 0\nv 0 1 0\nf 1/2 2/1 3/1\n", materials, textures, model,
                 error)); // vt index out of range
  CHECK(model.nodes.empty()); // failed loads leave a clean model
  CHECK(loadObj("  v 0 0 0\n  v 1 0 0\n  v 0 1 0\n  f 1 2 3\n", materials, textures, model,
                error));
  CHECK(loadObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nl 1 2\nf 1 2 3\n", materials, textures, model,
                error));

  // End-to-end fixtures for the renderer smoke run (best effort): an OBJ with
  // a sibling .mtl and checkerboard map_Kd texture.
  std::vector<std::uint8_t> objBytes(text.begin(), text.end());
  fixtures::writeFile(".build/test-model.obj", objBytes);
  std::string const texObj =
      "mtllib test-texture.mtl\n"
      "o TexQuad\n"
      "usemtl Textured\n"
      "v -1 0 -1\n"
      "v 1 0 -1\n"
      "v 1 0 1\n"
      "v -1 0 1\n"
      "vt 0 0\n"
      "vt 1 0\n"
      "vt 1 1\n"
      "vt 0 1\n"
      "vn 0 1 0\n"
      "f 1/1/1 2/2/1 3/3/1\n"
      "f 1/1/1 3/3/1 4/4/1\n";
  std::string const texMtl = "newmtl Textured\nKd 1 1 1\nmap_Kd test-texture.png\n";
  fixtures::writeFile(".build/test-texture.obj", std::vector<std::uint8_t>(texObj.begin(), texObj.end()));
  fixtures::writeFile(".build/test-texture.mtl", std::vector<std::uint8_t>(texMtl.begin(), texMtl.end()));
  std::ifstream checker(".build/fixtures/checker2.png", std::ios::binary);
  if (checker) {
    std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(checker)),
                                  std::istreambuf_iterator<char>());
    fixtures::writeFile(".build/test-texture.png", png);
  }

  // map_Bump binds a normal map through the same texture list (option flags
  // like -bm are skipped; the last token is the path).
  MtlMaterials bumpMats;
  CHECK(parseMtl("newmtl B\nKd 1 1 1\nmap_Bump -bm 0.5 normal-up.png\n", bumpMats));
  CHECK(bumpMats.size() == 1U);
  CHECK(bumpMats[0].mapBump == "normal-up.png");
  CHECK(bumpMats[0].mapKd.empty());
  CHECK_NEAR(bumpMats[0].normalScale, 0.5F, 1e-6F); // -bm value captured
  MtlMaterials kdMats;
  CHECK(parseMtl("newmtl K\nmap_Kd -s 1 1 1 tex.png\n", kdMats));
  CHECK(kdMats.size() == 1U && kdMats[0].mapKd == "tex.png"); // options skipped
  ModelTexture flatNormal;
  flatNormal.image.width = 1;
  flatNormal.image.height = 1;
  flatNormal.image.rgba = {128U, 128U, 255U, 255U};
  std::vector<ModelTexture> bumpTextures{flatNormal};
  bumpMats[0].normalTexture = 0;
  Model bumpModel;
  CHECK(loadObj("o Q\nusemtl B\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", bumpMats, bumpTextures,
                bumpModel, error));
  CHECK(bumpModel.meshes.size() == 1U);
  CHECK(bumpModel.meshes[0].normalTexture == 0);
}

} // namespace

void testUpscaleToPot() {
  // NPOT images upscale to power-of-two (the Vulkan NPOT-upload workaround);
  // POT images pass through with identical dimensions.
  Image solid;
  solid.width = 3;
  solid.height = 5;
  solid.rgba.assign(3U * 5U * 4U, 200U);
  Image up = upscaleToPot(solid);
  CHECK(up.width == 4U && up.height == 8U);
  CHECK(up.rgba.size() == 4U * 8U * 4U);
  CHECK(up.rgba[0] == 200U && up.rgba.back() == 200U); // solid stays solid

  // A 2x1 image is POT in both axes: the no-op passthrough check.
  Image edge;
  edge.width = 2;
  edge.height = 1;
  edge.rgba = {0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U};
  Image upEdge = upscaleToPot(edge);
  CHECK(upEdge.width == 2U && upEdge.height == 1U);
  Image ramp;
  ramp.width = 2;
  ramp.height = 2;
  ramp.rgba = {0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U,
               0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U};
  Image upRamp = upscaleToPot(ramp); // 2x2 is POT: no-op passthrough
  CHECK(upRamp.width == 2U && upRamp.height == 2U);
  CHECK(upRamp.rgba == ramp.rgba);

  // Non-trivial bilinear output: a 3x1 edge (3 is NPOT -> 4 wide) with
  // values 0, 255, 255 interpolates at the boundary.
  Image line;
  line.width = 3;
  line.height = 1;
  line.rgba = {0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U, 255U, 255U, 255U, 255U};
  Image upLine = upscaleToPot(line);
  CHECK(upLine.width == 4U && upLine.height == 1U);
  CHECK(upLine.rgba[0] == 0U);   // left edge clamps to the first texel
  CHECK(upLine.rgba[12] == 255U); // right edge clamps to the last
  CHECK(upLine.rgba[4] > 0U && upLine.rgba[4] < 255U); // interior interpolates

  Image empty;
  CHECK(upscaleToPot(empty).rgba.empty());
}

void testMipChain() {
  // 4x4 checkerboard: the first derived level averages each 2x2 block, then
  // the mean. Level 0 (the base) is not part of the output.
  Image base;
  base.width = 4;
  base.height = 4;
  base.rgba.assign(4U * 4U * 4U, 0U);
  for (std::uint32_t y = 0; y < 4; ++y) {
    for (std::uint32_t x = 0; x < 4; ++x) {
      std::uint8_t v = ((x + y) % 2U) == 0U ? 255U : 0U;
      std::size_t p = (static_cast<std::size_t>(y) * 4 + x) * 4U;
      base.rgba[p] = v;
      base.rgba[p + 1U] = v;
      base.rgba[p + 2U] = v;
      base.rgba[p + 3U] = 255U;
    }
  }
  auto levels = buildMipLevels(base);
  CHECK(levels.size() == 2U); // 2x2 -> 1x1 below the 4x4 base
  CHECK(levels[0].width == 2 && levels[0].height == 2);
  CHECK(levels[1].width == 1 && levels[1].height == 1);
  // Every 2x2 block of the checkerboard averages to half.
  CHECK(levels[0].rgba[0] == 128U && levels[0].rgba[4] == 128U);
  CHECK(levels[0].rgba[8] == 128U && levels[0].rgba[12] == 128U);
  CHECK(levels[1].rgba[0] == 128U);
  CHECK(levels[1].rgba[3] == 255U); // alpha is preserved

  // Odd dimensions floor-divide (Vulkan mip extents) and never over-read. The
  // chain also respects the Vulkan level cap (floor(log2(max))+1 total):
  // 3x5 -> 1x2 -> 1x1 only.
  Image odd;
  odd.width = 3;
  odd.height = 5;
  odd.rgba.assign(3U * 5U * 4U, 200U);
  auto oddLevels = buildMipLevels(odd);
  CHECK(oddLevels.size() == 2U); // 1x2 -> 1x1 below the 3x5 base
  CHECK(oddLevels[0].width == 1 && oddLevels[0].height == 2);
  CHECK(oddLevels[0].rgba[0] == 200U);
  CHECK(oddLevels[1].width == 1 && oddLevels[1].height == 1);

  // Non-power-of-two sizes stop under the Vulkan level cap (this is the
  // 12288^2 case in miniature).
  Image oddSquare;
  oddSquare.width = 9;
  oddSquare.height = 9;
  oddSquare.rgba.assign(9U * 9U * 4U, 100U);
  auto cappedLevels = buildMipLevels(oddSquare);
  CHECK(cappedLevels.size() == 3U); // 4x4 -> 2x2 -> 1x1 below the 9x9 base
  CHECK(cappedLevels[0].width == 4 && cappedLevels[0].height == 4);
  CHECK(cappedLevels[2].width == 1 && cappedLevels[2].height == 1);

  // Degenerate inputs yield nothing.
  Image one;
  one.width = 1;
  one.height = 1;
  one.rgba = {1U, 2U, 3U, 4U};
  CHECK(buildMipLevels(one).empty()); // the level cap allows no derived levels
  CHECK(buildMipLevels(Image{}).empty());
}

void testHandlePickingByRay() {
  // Segment distances from a half-line, including behind-origin clamping.
  Ray downX{{2.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F}};
  CHECK_NEAR(raySegmentDistance(downX, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}), 0.0F, 1e-5F);
  Ray offByHalf{{2.0F, 0.5F, 0.0F}, {-1.0F, 0.0F, 0.0F}};
  CHECK_NEAR(raySegmentDistance(offByHalf, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}), 0.5F, 1e-5F);
  Ray awayFromOrigin{{2.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}};
  CHECK_NEAR(raySegmentDistance(awayFromOrigin, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}), 1.0F,
             1e-5F); // segment behind the ray: distance from the origin

  // Parallel ray and segment: the distance is the line offset, not an endpoint
  // artefact (this is the "aiming along an axis" case).
  Ray parallel{{0.0F, 0.5F, 0.0F}, {1.0F, 0.0F, 0.0F}};
  CHECK_NEAR(raySegmentDistance(parallel, {5.0F, 0.0F, 0.0F}, {6.0F, 0.0F, 0.0F}), 0.5F, 1e-5F);
  CHECK(raySegmentDistance(parallel, {-6.0F, 0.0F, 0.0F}, {-5.0F, 0.0F, 0.0F}) > 4.9F);

  // Real gizmo geometry (translate mode, unit size, identity orientation).
  // Rays are aimed away from the pivot region where the axes share points.
  auto geometry = buildHandleGeometry(GizmoMode::Translate, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F);
  Ray ontoX{{0.7F, 3.0F, 0.0F}, {0.0F, -1.0F, 0.0F}};
  auto hit = pickHandleByRay(geometry, ontoX, 0.1F);
  CHECK(hit.has_value());
  CHECK(hit.has_value() && hit->handle == GizmoHandle::AxisX);
  Ray ontoY{{5.0F, 0.5F, 0.0F}, {-1.0F, 0.0F, 0.0F}};
  hit = pickHandleByRay(geometry, ontoY, 0.1F);
  CHECK(hit.has_value() && hit->handle == GizmoHandle::AxisY);
  Ray away{{0.0F, 5.0F, 5.0F}, {0.0F, 1.0F, 0.0F}};
  CHECK(!pickHandleByRay(geometry, away, 0.1F).has_value());
  Ray awayAlongX{{2.5F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}};
  CHECK(!pickHandleByRay(geometry, awayAlongX, 0.1F).has_value());

  // Rotate rings: the ring for axis a is the circle perpendicular to that axis
  // at radius 0.8 * size. Aim through a rim point unique to the AxisX ring.
  auto rings = buildHandleGeometry(GizmoMode::Rotate, {0.0F, 0.0F, 0.0F}, Quat{}, 1.0F);
  float rim = 0.8F * 0.70710678F;
  Ray throughRim{{0.0F, rim, 3.0F}, {0.0F, 0.0F, -1.0F}};
  hit = pickHandleByRay(rings, throughRim, 0.1F);
  CHECK(hit.has_value() && hit->handle == GizmoHandle::AxisX);
  // The diagonal misses every ring (each ring stays 0.46 from it).
  float diag = 0.57735027F;
  Ray diagonal{{1.0F, 1.0F, 1.0F}, {-diag, -diag, -diag}};
  CHECK(!pickHandleByRay(rings, diagonal, 0.1F).has_value());

  // Filled polygons and the line-wins-over-filled priority, with synthetic
  // geometry (decoupled from the builder's exact extents).
  HandleGeom square;
  square.handle = GizmoHandle::PlaneXY;
  square.filled = true;
  square.loop = true;
  square.points = {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
  Ray throughSquare{{0.5F, 0.5F, 3.0F}, {0.0F, 0.0F, -1.0F}};
  hit = pickHandleByRay({square}, throughSquare, 0.1F);
  CHECK(hit.has_value() && hit->handle == GizmoHandle::PlaneXY);
  Ray besideSquare{{1.5F, 0.5F, 3.0F}, {0.0F, 0.0F, -1.0F}};
  CHECK(!pickHandleByRay({square}, besideSquare, 0.1F).has_value());
  Ray behindSquare{{0.5F, 0.5F, 3.0F}, {0.0F, 0.0F, 1.0F}};
  CHECK(!pickHandleByRay({square}, behindSquare, 0.1F).has_value()); // plane hit behind origin

  HandleGeom axisLine;
  axisLine.handle = GizmoHandle::AxisY;
  axisLine.points = {{0.0F, 0.0F, 0.0F}, {0.0F, 2.0F, 0.0F}};
  Ray throughBoth{{0.05F, 0.5F, 3.0F}, {0.0F, 0.0F, -1.0F}}; // near the line, over the square
  hit = pickHandleByRay({square, axisLine}, throughBoth, 0.1F);
  CHECK(hit.has_value() && hit->handle == GizmoHandle::AxisY); // line-like wins

  // Depth reporting: the returned distance is the along-ray distance to the
  // hit, so callers can depth-order handles against a model surface.
  float hitDepth = hit.has_value() ? hit->distance : -1.0F;
  CHECK_NEAR(hitDepth, 3.0F, 1e-4F); // the line is at z = 0, ray from z = 3
  // Same hit approached obliquely from farther away: the depth grows with the
  // travel distance while the geometry under the ray is unchanged.
  Vec3 deepDir = normalized(Vec3{0.0F, -3.0F, 2.0F});
  Ray ontoXDeep{{0.7F, 3.0F, -2.0F}, deepDir};
  auto deepHit = pickHandleByRay(geometry, ontoXDeep, 0.1F);
  CHECK(deepHit.has_value() && deepHit->handle == GizmoHandle::AxisX);
  float deepDepth = deepHit.has_value() ? deepHit->distance : -1.0F;
  CHECK_NEAR(deepDepth, 3.6055513F, 1e-4F); // |(0, 3, -2)| to (0.7, 0, 0)
  float lineDepth = -1.0F;
  CHECK_NEAR(raySegmentDistance(parallel, {5.0F, 0.0F, 0.0F}, {6.0F, 0.0F, 0.0F}, &lineDepth),
             0.5F, 1e-5F);
  CHECK_NEAR(lineDepth, 5.0F, 1e-4F); // closest approach sits at the segment start
}

void testXrMath() {
  // Symmetric tangents must equal cameraProjection at aspect 1.
  float a = 0.9F;
  float nearZ = 0.1F;
  float farZ = 20.0F;
  Mat4 fromTangents =
      cameraProjectionFromTangents(-a, a, a, -a, nearZ, farZ);
  Camera camera;
  camera.fovY = 2.0F * std::atan(a);
  camera.nearZ = nearZ;
  camera.farZ = farZ;
  Mat4 fromCamera = cameraProjection(camera, 1.0F);
  for (int i = 0; i < 16; ++i) {
    CHECK_NEAR(fromTangents.m[i], fromCamera.m[i], 1e-5F);
  }

  // Asymmetric frustum: view-space corners map to NDC edges (clip Y down).
  float tanLeft = -0.5F;
  float tanRight = 1.0F;
  float tanUp = 0.25F;
  float tanDown = -1.0F;
  Mat4 proj = cameraProjectionFromTangents(tanLeft, tanRight, tanUp, tanDown, nearZ, farZ);
  Vec3 topRight = transformPoint(proj, {tanRight, tanUp, -1.0F});
  Vec3 bottomLeft = transformPoint(proj, {tanLeft, tanDown, -1.0F});
  CHECK_NEAR(topRight.x, 1.0F, 1e-5F);
  CHECK_NEAR(topRight.y, -1.0F, 1e-5F); // +Y up maps to NDC -1 in Vulkan
  CHECK_NEAR(bottomLeft.x, -1.0F, 1e-5F);
  CHECK_NEAR(bottomLeft.y, 1.0F, 1e-5F);
  CHECK_NEAR(transformPoint(proj, {0.0F, 0.0F, -nearZ}).z, 0.0F, 1e-5F);
  // transformPoint returns clip coordinates before the perspective divide;
  // w_clip = farZ at view z = -farZ, so z/w = z_clip / farZ.
  Vec3 farPoint = transformPoint(proj, {0.0F, 0.0F, -farZ});
  CHECK_NEAR(farPoint.z / farZ, 1.0F, 1e-3F);

  // Identity pose: view is just the translation inverse.
  Mat4 view = cameraViewFromPose({1.0F, 2.0F, 3.0F}, {0.0F, 0.0F, 0.0F, 1.0F});
  CHECK_NEAR3(transformPoint(view, {1.0F, 2.0F, 3.0F}), {0.0F, 0.0F, 0.0F}); // eye at origin
  CHECK_NEAR3(transformPoint(view, {0.0F, 0.0F, 0.0F}), {-1.0F, -2.0F, -3.0F});
  CHECK_NEAR3(transformPoint(view, {2.0F, 2.0F, 3.0F}), {1.0F, 0.0F, 0.0F});

  // Yaw +90deg: world +X lies along view +Z (view forward is -Z).
  Quat yaw90 = quatFromAxisAngle({0.0F, 1.0F, 0.0F}, 1.5707964F);
  Mat4 yawView = cameraViewFromPose({0.0F, 0.0F, 0.0F}, yaw90);
  CHECK_NEAR3(transformPoint(yawView, {1.0F, 0.0F, 0.0F}), {0.0F, 0.0F, 1.0F});
  CHECK_NEAR3(transformPoint(yawView, {-1.0F, 0.0F, 0.0F}), {0.0F, 0.0F, -1.0F});

  // Rotated pose: the eye position itself still maps to the view origin.
  Vec3 eye{0.7F, -1.2F, 3.4F};
  Mat4 rotated = cameraViewFromPose(eye, yaw90);
  CHECK_NEAR3(transformPoint(rotated, eye), {0.0F, 0.0F, 0.0F});
}

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
  testPickClosestSurface();
  testSceneSnapshotRestore();
  testUndoHistory();
  testCancelPendingEdit();
  testExpandSelection();
  testBoundsHelpers();
  testGizmoParentChildSelectionRides();
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
  testCameraLookPivot();
  testGravityAlign();
  testXrMath();
  testUpscaleToPot();
  testMipChain();
  testSegmentAndPolygonPicking();
  testStickMotion();
  testScaleSelectionUniform();
  testEditorStateOps();
  testHandlePicking();
  testHandlePickingBehindCamera();
  testHandlePickingByRay();
  testJsonParser();
  testMeshBounds();
  testImageDecode();
  testDecodeSizeCaps();
  testGlbLoader();
  testGlbInterleavedAccessors();
  testGlbTextureAndUv();
  testGlbSizeWrapRejected();
  testObjLoader();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
