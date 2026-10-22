#include "editor_state.h"

#include <algorithm>
#include <cmath>

#include "gizmo.h"

namespace vredit {

namespace {

bool sameVec3(Vec3 const& a, Vec3 const& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool sameAabb(Aabb const& a, Aabb const& b) {
  return sameVec3(a.min, b.min) && sameVec3(a.max, b.max);
}

} // namespace

bool sameVisual(NodeVisual const& a, NodeVisual const& b) {
  return a.id == b.id && a.mesh == b.mesh && a.texture == b.texture &&
         a.normalTexture == b.normalTexture && a.normalMapped == b.normalMapped &&
         a.normalScale == b.normalScale && sameAabb(a.localBounds, b.localBounds) &&
         a.pickGeometry.get() == b.pickGeometry.get() && sameVec3(a.colour, b.colour) &&
         a.alphaFactor == b.alphaFactor && a.alpha == b.alpha && a.alphaCutoff == b.alphaCutoff;
}

bool sameSnapshot(EditorSnapshot const& a, EditorSnapshot const& b) {
  if (!sameSnapshot(a.scene, b.scene) || a.visuals.size() != b.visuals.size() ||
      a.groups != b.groups || a.selection != b.selection) {
    return false;
  }
  for (std::size_t i = 0; i < a.visuals.size(); ++i) {
    if (!sameVisual(a.visuals[i], b.visuals[i])) {
      return false;
    }
  }
  return true;
}

EditorSnapshot captureState(Scene const& scene,
                            std::vector<NodeVisual> const& visuals,
                            ModelGroups const& groups,
                            std::vector<NodeId> const& selection) {
  EditorSnapshot snapshot;
  snapshot.scene = scene.snapshot();
  snapshot.visuals = visuals;
  snapshot.groups = groups;
  snapshot.selection = selection;
  return snapshot;
}

void restoreState(EditorSnapshot const& snapshot,
                  Scene& scene,
                  std::vector<NodeVisual>& visuals,
                  ModelGroups& groups,
                  std::vector<NodeId>& selection) {
  scene.restore(snapshot.scene);
  visuals = snapshot.visuals;
  groups = snapshot.groups;
  selection = snapshot.selection;
}

bool nodesBounds(Scene& scene,
                 std::vector<NodeVisual> const& visuals,
                 std::vector<NodeId> const& ids,
                 Aabb& boundsOut) {
  bool any = false;
  Aabb bounds{};
  for (NodeId id : ids) {
    if (!scene.valid(id)) {
      continue;
    }
    Aabb contribution;
    bool found = false;
    for (NodeVisual const& visual : visuals) {
      if (visual.id == id) {
        contribution = transformAabb(scene.world(id), visual.localBounds);
        found = true;
        break;
      }
    }
    if (!found) {
      Vec3 point = scene.worldPosition(id);
      contribution = {point, point};
    }
    bounds = any ? unite(bounds, contribution) : contribution;
    any = true;
  }
  boundsOut = bounds;
  return any;
}

void deleteSubtrees(Scene& scene,
                    std::vector<NodeVisual>& visuals,
                    ModelGroups& groups,
                    std::vector<NodeId>& selection,
                    std::vector<NodeId> const& roots) {
  // Collapse descendants into their topmost selected ancestor: destroying a
  // parent already removes the child's subtree, so the child root is noise.
  std::vector<NodeId> topmost;
  for (NodeId id : roots) {
    if (!scene.valid(id)) {
      continue;
    }
    bool underAnotherRoot = false;
    for (NodeId walker = scene.parent(id); walker != kInvalidNode; walker = scene.parent(walker)) {
      if (std::find(roots.begin(), roots.end(), walker) != roots.end()) {
        underAnotherRoot = true;
        break;
      }
    }
    if (!underAnotherRoot && std::find(topmost.begin(), topmost.end(), id) == topmost.end()) {
      topmost.push_back(id);
    }
  }
  for (NodeId id : topmost) {
    scene.destroy(id); // whole subtree
  }
  // Prune everything that belonged to the destroyed nodes.
  visuals.erase(std::remove_if(visuals.begin(), visuals.end(),
                               [&scene](NodeVisual const& visual) {
                                 return !scene.valid(visual.id);
                               }),
                visuals.end());
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [&scene](std::pair<NodeId, std::uint32_t> const& entry) {
                                return !scene.valid(entry.first);
                              }),
               groups.end());
  selection.erase(std::remove_if(selection.begin(), selection.end(),
                                 [&scene](NodeId id) { return !scene.valid(id); }),
                  selection.end());
}

void placeOnFloor(Scene& scene,
                  std::vector<NodeVisual> const& visuals,
                  std::vector<NodeId> const& roots,
                  Vec3 const& target) {
  // Collect the whole loaded subtree: every alive node rooted at one of
  // `roots` (walking the parent links up from each node).
  std::vector<NodeId> nodes;
  SceneSnapshot alive = scene.snapshot();
  for (NodeRecord const& record : alive.nodes) {
    for (NodeId walker = record.id; walker != kInvalidNode; walker = scene.parent(walker)) {
      if (std::find(roots.begin(), roots.end(), walker) != roots.end()) {
        nodes.push_back(record.id);
        break;
      }
    }
  }

  Aabb bounds{};
  if (!nodesBounds(scene, visuals, nodes, bounds)) {
    return;
  }
  Vec3 const centre = aabbCentre(bounds);
  Vec3 const delta{target.x - centre.x, -bounds.min.y, target.z - centre.z};
  for (NodeId root : roots) {
    if (!scene.valid(root)) {
      continue;
    }
    Vec3 localDelta = delta;
    NodeId parent = scene.parent(root);
    if (parent != kInvalidNode) {
      Mat4 parentWorld{};
      if (invertAffine(scene.world(parent), parentWorld)) {
        localDelta = transformVector(parentWorld, delta);
      }
    }
    Transform local = scene.local(root);
    local.translation = local.translation + localDelta;
    scene.setLocal(root, local);
  }
}

void moveSelectionHorizontal(Scene& scene,
                             std::vector<NodeId> const& selection,
                             Vec3 const& worldDelta) {
  std::vector<DragStart> starts;
  captureDragStarts(scene, selection, starts);
  Vec3 const delta{worldDelta.x, 0.0F, worldDelta.z};
  applyDragStarts(scene, starts, translationMatrix(delta));
}

void moveSelectionVertical(Scene& scene,
                           std::vector<NodeId> const& selection,
                           float worldDeltaY) {
  if (worldDeltaY == 0.0F) {
    return;
  }
  std::vector<DragStart> starts;
  captureDragStarts(scene, selection, starts);
  applyDragStarts(scene, starts, translationMatrix({0.0F, worldDeltaY, 0.0F}));
}

// The set of nodes a selection's edit actually moves: each selected node plus
// its whole subtree (a selected child rides along with a selected ancestor
// rather than receiving the delta twice — but its descendants still move).
// Pivot and floor bounds MUST cover this whole set: a part selection with a
// large child below it otherwise tips through the floor undetected (the
// drop's bounds covered only the selected node's own box).
bool movingSetBounds(Scene& scene,
                     std::vector<NodeVisual> const& visuals,
                     std::vector<NodeId> const& selection,
                     Aabb& boundsOut) {
  std::vector<NodeId> moving;
  for (NodeId id : selection) {
    if (!scene.valid(id)) {
      continue;
    }
    moving.push_back(id);
    for (NodeId const d : scene.descendants(id)) {
      moving.push_back(d);
    }
  }
  return nodesBounds(scene, visuals, moving, boundsOut);
}

void viewerRotationAxes(Vec3 const& viewRightFlat, Vec3& yawAxisOut, Vec3& tipAxisOut) {
  // Spin: the viewer's level vertical (the world up — the views are horizon
  // locked, so this is "up" to the viewer however they lean).
  yawAxisOut = Vec3{0.0F, 1.0F, 0.0F};
  // Tip: the viewer's level right axis, PRE-SIGNED for the stickRotate*
  // seams: a positive Y deflection lifts the model's face upward (its top
  // edge leans back — one rotation, seen from each end): "the model should
  // rotate in the direction of the stick".
  tipAxisOut = Vec3{-viewRightFlat.x, -viewRightFlat.y, -viewRightFlat.z};
}

bool stickRotateYaw(Scene& scene,
                    std::vector<NodeId> const& selection,
                    Vec3 const& pivot,
                    Vec3 const& yawAxis,
                    float deflection,
                    float rateRadiansPerSec,
                    float dt) {
  float const angle = deflection * rateRadiansPerSec * dt;
  if (angle == 0.0F) {
    return false;
  }
  rotateSelectionAbout(scene, selection, pivot, yawAxis, angle);
  return true;
}

bool stickRotateTip(Scene& scene,
                    std::vector<NodeId> const& selection,
                    Vec3 const& pivot,
                    Vec3 const& tipAxis,
                    float deflection,
                    float rateRadiansPerSec,
                    float dt) {
  float const angle = deflection * rateRadiansPerSec * dt;
  if (angle == 0.0F) {
    return false;
  }
  rotateSelectionAbout(scene, selection, pivot, tipAxis, angle);
  return true;
}

void rotateSelectionAxis(Scene& scene,
                         std::vector<NodeVisual> const& visuals,
                         std::vector<NodeId> const& selection,
                         Vec3 const& worldAxis,
                         float radians) {
  Aabb bounds{};
  if (!movingSetBounds(scene, visuals, selection, bounds)) {
    return;
  }
  rotateSelectionAbout(scene, selection, aabbCentre(bounds), worldAxis, radians);
}

void rotateSelectionAbout(Scene& scene,
                          std::vector<NodeId> const& selection,
                          Vec3 const& pivot,
                          Vec3 const& worldAxis,
                          float radians) {
  if (length(worldAxis) < 1e-6F || radians == 0.0F) {
    return;
  }
  Vec3 const axis = normalized(worldAxis);
  float const half = radians * 0.5F;
  float const s = std::sin(half);
  Quat const q{axis.x * s, axis.y * s, axis.z * s, std::cos(half)};
  std::vector<DragStart> starts;
  captureDragStarts(scene, selection, starts);
  Mat4 const delta =
      mul(translationMatrix(pivot), mul(rotationMatrix(q), translationMatrix(-pivot)));
  applyDragStarts(scene, starts, delta);
}

void rotateSelectionYaw(Scene& scene,
                        std::vector<NodeVisual> const& visuals,
                        std::vector<NodeId> const& selection,
                        float radians) {
  rotateSelectionAxis(scene, visuals, selection, Vec3{0.0F, 1.0F, 0.0F}, radians);
}

void dropSelectionToFloor(Scene& scene,
                          std::vector<NodeVisual> const& visuals,
                          std::vector<NodeId> const& selection) {
  Aabb bounds{};
  if (!movingSetBounds(scene, visuals, selection, bounds)) {
    return;
  }
  float const lift = -bounds.min.y;
  if (std::fabs(lift) < 1e-6F) {
    return;
  }
  std::vector<DragStart> starts;
  captureDragStarts(scene, selection, starts);
  applyDragStarts(scene, starts, translationMatrix(Vec3{0.0F, lift, 0.0F}));
}

bool scaleSelectionUniform(Scene& scene,
                           std::vector<NodeVisual> const& visuals,
                           std::vector<NodeId> const& selection,
                           float factor) {
  if (!(factor > 0.0F)) {
    return false;
  }
  Aabb bounds{};
  if (!nodesBounds(scene, visuals, selection, bounds)) {
    return false;
  }
  Vec3 const pivot = aabbCentre(bounds);
  std::vector<DragStart> starts;
  captureDragStarts(scene, selection, starts);
  Mat4 const delta = mul(translationMatrix(pivot),
                         mul(scaleMatrix(Vec3{factor, factor, factor}),
                             translationMatrix(-pivot)));
  applyDragStarts(scene, starts, delta);
  return true;
}

Vec3 floorAimPoint(Ray const& aimRay) {
  if (aimRay.direction.y < -1e-4F) {
    float const t = -aimRay.origin.y / aimRay.direction.y;
    if (t > 0.0F) {
      return aimRay.origin + aimRay.direction * t;
    }
  }
  // Level or upward aim: 2.5 m ahead on the floor.
  Vec3 forward{aimRay.direction.x, 0.0F, aimRay.direction.z};
  if (dot(forward, forward) < 1e-8F) {
    forward = {0.0F, 0.0F, -1.0F};
  }
  forward = normalized(forward);
  Vec3 const ahead = aimRay.origin + forward * 2.5F;
  return {ahead.x, 0.0F, ahead.z};
}

} // namespace vredit
