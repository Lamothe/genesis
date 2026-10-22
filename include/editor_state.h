#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "mesh.h"
#include "picking.h"
#include "scene.h"
#include "vecmath.h"

namespace vredit {

// One drawn node: mesh/texture bindings and picking shape. Renderer handles
// are plain integers (include/mesh.h), so this type is core-testable.
struct NodeVisual {
  NodeId id{kInvalidNode};
  MeshHandle mesh{0};
  TextureHandle texture{0};
  TextureHandle normalTexture{0};
  bool normalMapped{false};
  float normalScale{1.0F};
  Aabb localBounds{{-0.5F, -0.5F, -0.5F}, {0.5F, 0.5F, 0.5F}};
  // Exact picking triangles (node-local); shared between visuals that loaded
  // the same mesh. Null for box-shaped visuals (demo cubes), where the bounds
  // are the shape.
  std::shared_ptr<const PickGeometry> pickGeometry;
  Vec3 colour{1.0F, 1.0F, 1.0F};
  float alphaFactor{1.0F};
  AlphaMode alpha{AlphaMode::Opaque};
  float alphaCutoff{0.5F};
};

// Model grouping: every node created for one loaded model shares a group id,
// so a plain click selects the whole model and a secondary click (shift/
// ctrl, or grip+trigger in VR) selects a single part. Group 0 = ungrouped.

// The editor's two input modes (the menu button toggles them): move = the
// left stick scales (X) and changes height (Y); rotate = it sweeps the
// model's face toward the viewer's right (X) and lifts/lowers the face (Y)
// — a trackball about the VIEWER's axes. NOTHING rotates outside rotate —
// that rule lives in the app's mode switch, driven by this enum.
enum class EditMode : std::uint8_t { move, rotate };
using ModelGroups = std::vector<std::pair<NodeId, std::uint32_t>>;

// Full editor state for undo/redo: scene structure + visuals + groups +
// selection. The earlier transforms-only history could not resurrect a
// deleted model; this can, because nodes keep their ids across restore.
struct EditorSnapshot {
  SceneSnapshot scene;
  std::vector<NodeVisual> visuals;
  ModelGroups groups;
  std::vector<NodeId> selection;
};

bool sameVisual(NodeVisual const& a, NodeVisual const& b);
bool sameSnapshot(EditorSnapshot const& a, EditorSnapshot const& b);

EditorSnapshot captureState(Scene const& scene,
                            std::vector<NodeVisual> const& visuals,
                            ModelGroups const& groups,
                            std::vector<NodeId> const& selection);
void restoreState(EditorSnapshot const& snapshot,
                  Scene& scene,
                  std::vector<NodeVisual>& visuals,
                  ModelGroups& groups,
                  std::vector<NodeId>& selection);

// World-space bounds of the given nodes: visual bounds transformed by the
// node's world matrix; nodes without a visual contribute their position.
bool nodesBounds(Scene& scene,
                 std::vector<NodeVisual> const& visuals,
                 std::vector<NodeId> const& ids,
                 Aabb& boundsOut);

// The one delete rule: removes the SUBTREES rooted at the given nodes.
// Roots that are descendants of other roots collapse into their topmost
// ancestor, so a whole-model selection and a single-part selection share one
// code path (a model's nodes are its group's roots' subtrees). Matching
// visuals and group entries are dropped, and deleted ids leave the
// selection. Bumps the scene epoch — a discontinuity for the neural seam.
void deleteSubtrees(Scene& scene,
                    std::vector<NodeVisual>& visuals,
                    ModelGroups& groups,
                    std::vector<NodeId>& selection,
                    std::vector<NodeId> const& roots);

// Spawning placement: translates the freshly loaded subtree so its bounds sit
// ON THE FLOOR (bounds min y = 0) centred at `target` in x/z. `roots` are the
// loaded top-level nodes; their local translations absorb the delta (in the
// parent's space, so it works under any parent).
void placeOnFloor(Scene& scene,
                  std::vector<NodeVisual> const& visuals,
                  std::vector<NodeId> const& roots,
                  Vec3 const& target);

// Stick-driven editing (VR): the right stick translates the selection
// horizontally ("models stay stuck to the floor" — y is never touched) in
// both modes; the left stick is mode-driven (the app's EditMode): move =
// X uniform scale / Y height, rotate = X sweeps the face toward the
// viewer's right about the viewer's vertical, Y lifts/lowers the face about
// the viewer's level right (trackball; stickRotateYaw/stickRotateTip hold
// the signs) — after which dropSelectionToFloor re-rests it on the ground.
// All are drift-free (built on the gizmo's capture/apply machinery) and
// respect the no-double-move
// rule: a selected child rides along with its selected ancestor instead of
// receiving the delta twice.
void moveSelectionHorizontal(Scene& scene,
                             std::vector<NodeId> const& selection,
                             Vec3 const& worldDelta);
// Explicit height control (stick mode): moves the selection along the world
// vertical with NO floor rule — the floor clamp bounds *movement* (walking
// models around), while lifting is a deliberate, requested operation ("a way
// to scale and move vertically"). Same ride-along and drift rules.
void moveSelectionVertical(Scene& scene,
                           std::vector<NodeId> const& selection,
                           float worldDeltaY);
void rotateSelectionYaw(Scene& scene,
                        std::vector<NodeVisual> const& visuals,
                        std::vector<NodeId> const& selection,
                        float radians);
// The stick-edit rotation axes: STRICTLY VIEWER-relative ("the y-axis
// (up/down) should only ever rotate the model around the axis that stretches
// to the left and right of the viewer... Same thing with the x-axis"): spin
// about the viewer's level vertical, tip about the viewer's level right
// axis — never the model's own frame, no roll. The tip axis is PRE-SIGNED
// for the stickRotate* seams: a positive stick deflection rotates the model
// in the direction of the stick (trackball) — stick right sweeps the face
// toward the viewer's right, stick up lifts the face upward (its top edge
// leans back).
void viewerRotationAxes(Vec3 const& viewRightFlat, Vec3& yawAxisOut, Vec3& tipAxisOut);

// One stick rotation step about the caller-latched axes (the two above).
// TRACKBALL-signed: a POSITIVE deflection rotates the model in the
// direction of the stick — stickRotateYaw sweeps the model's face toward
// the VIEWER's right (about the viewer's vertical), stickRotateTip lifts
// the face upward (its top edge leans back; about the pre-signed viewer's
// right). The signs live HERE so tests pin the user-facing direction.
// Returns true if anything rotated.
bool stickRotateYaw(Scene& scene,
                    std::vector<NodeId> const& selection,
                    Vec3 const& pivot,
                    Vec3 const& yawAxis,
                    float deflection,
                    float rateRadiansPerSec,
                    float dt);
bool stickRotateTip(Scene& scene,
                    std::vector<NodeId> const& selection,
                    Vec3 const& pivot,
                    Vec3 const& tipAxis,
                    float deflection,
                    float rateRadiansPerSec,
                    float dt);

// Rotates the selection `radians` around `worldAxis` (need not be unit)
// through the MOVING SET's bounds centre (the selection's subtrees — see
// movingSetBounds). The generic form rotateSelectionYaw is built on.
void rotateSelectionAxis(Scene& scene,
                         std::vector<NodeVisual> const& visuals,
                         std::vector<NodeId> const& selection,
                         Vec3 const& worldAxis,
                         float radians);
// Rotates about an EXPLICIT pivot — the stick gestures latch their pivot at
// gesture start and use this, so a per-frame re-centre cannot walk asymmetric
// models sideways as they tip.
void rotateSelectionAbout(Scene& scene,
                          std::vector<NodeId> const& selection,
                          Vec3 const& pivot,
                          Vec3 const& worldAxis,
                          float radians);
// Bounds of everything a selection's rigid edit moves: each selected node
// plus its whole subtree (a selected child rides along with its ancestor but
// its descendants still move). The pivot source for rotation and the bounds
// source for the floor rule.
bool movingSetBounds(Scene& scene,
                     std::vector<NodeVisual> const& visuals,
                     std::vector<NodeId> const& selection,
                     Aabb& boundsOut);
// Lifts the selection so its bounds' lowest point rests exactly on the floor
// (y = 0). Tipping rotations swing the bounding box below ground; this puts
// the model back "stuck to the floor" after every tip. No-op when already
// resting.
void dropSelectionToFloor(Scene& scene,
                          std::vector<NodeVisual> const& visuals,
                          std::vector<NodeId> const& selection);

// Uniformly scales the selection about its bounds centre — the ONLY kind of
// scaling in the editor; per-axis scaling does not exist. `factor` is
// multiplicative (1.05 = 5% larger). Called by the move-mode left stick X
// (VR and desktop arrows) and the carousel's stage/place sizing. Same
// parent-vs-descendant rules as the other stick helpers.
bool scaleSelectionUniform(Scene& scene,
                           std::vector<NodeVisual> const& visuals,
                           std::vector<NodeId> const& selection,
                           float factor);

// Where an aim ray means "put it here": the point where the beam meets the
// floor (y = 0). A ray that never reaches the floor (level or up-aimed)
// falls back to 2.5 m ahead of its origin, projected onto the floor.
Vec3 floorAimPoint(Ray const& aimRay);

} // namespace vredit
