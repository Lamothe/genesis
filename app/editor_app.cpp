// vr-editor: desktop shell over the scene-editor core — viewport, camera,
// selection, gizmo handles. Typed C++ only; the shell is a thin layer over
// vredit::Scene/GizmoDrag and VkRenderer.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <xcb/xcb.h>

#include "xr_shell.h"

#include "camera.h"
#include "editor_state.h"
#include "gizmo.h"
#include "gltf.h"
#include "handles.h"
#include "mesh.h"
#include "obj.h"
#include "picking.h"
#include "scene.h"
#include "undo.h"
#include "vk_renderer.h"

using namespace vredit;

namespace {

constexpr float kLookSpeed = 0.005F;     // radians per pixel of mouse look
constexpr float kFlySpeed = 1.8F;        // metres/second (boost covers distance)
constexpr float kFlyBoost = 3.0F;
constexpr float kMinEyeHeight = 0.15F;   // the camera never passes below the grid (y = 0)
constexpr float kWheelMetres = 1.0F;     // metres per wheel notch
constexpr float kGizmoPixels = 120.0F;
constexpr float kPickThresholdPx = 8.0F;
constexpr float kVrPickThresholdPx = 24.0F; // aim rays jitter: 3x the mouse
// Stick-driven editing rates (VR): the right stick slides the selection, the
// left stick turns it. Grip held runs both at quarter speed (fine control).
constexpr float kStickDeadZone = 0.12F; // a gesture STARTS past this deflection
constexpr float kStickEndZone = 0.06F;  // ...and ENDS below this (hysteresis, so one
                                        // physical gesture is one undo step)
constexpr float kStickMoveMps = 1.5F;    // metres per second at full deflection
constexpr float kStickTurnRadps = 1.75F; // ~100 degrees per second at full
// Hold-X delete: the delete fires when the hold reaches full charge.
constexpr double kDeleteChargeS = 0.6;
constexpr float kSnapTranslate = 0.25F;
constexpr float kSnapRotate = 0.2617994F; // 15 degrees
constexpr float kSnapScale = 0.25F;

// Free-flight movement with the floor constraint: y = 0 is the ground grid and
// the camera never passes below it (any axis: even forward flight can descend).
void clampAboveGround(Camera& camera) {
  camera.position.y = std::fmax(camera.position.y, kMinEyeHeight);
}

void flyCamera(Camera& camera, Vec3 const& localDelta) {
  cameraFly(camera, localDelta);
  clampAboveGround(camera);
}

void dollyCamera(Camera& camera, float metres) {
  cameraDolly(camera, metres);
  clampAboveGround(camera);
}

void panCamera(Camera& camera, float dx, float dy, float viewportHeight) {
  cameraPan(camera, dx, dy, viewportHeight);
  clampAboveGround(camera);
}

// NodeVisual lives in the core now (include/editor_state.h) so delete/undo
// state is unit-testable; the app uses it directly.
using vredit::NodeVisual;

struct GallerySlot {
  std::string file;
  std::vector<vredit::NodeId> roots; // top-level nodes of the preview load
  std::vector<vredit::NodeId> nodes; // the preview's whole node set (hover)
};

struct App {
  Scene scene;
  std::vector<NodeVisual> visuals;
  std::vector<NodeId> selection;
  // Model grouping: every node created for one loaded model shares a group id,
  // so a plain click selects the whole model and a secondary click (shift/
  // ctrl, or grip+trigger in VR) selects a single part. Group 0 = ungrouped.
  std::vector<std::pair<NodeId, std::uint32_t>> modelGroups;
  std::uint32_t nextModelGroup{1};
  // Full-state undo: structure + visuals + groups + selection, so spawning and
  // deleting models are undoable exactly like transform drags.
  vredit::UndoHistory<vredit::EditorSnapshot> undo;
  Camera camera;
  // Backdrop = the ground grid + the tinted clear colour. Toggled off
  // ("show the model without the background"), the scene is a plain void.
  bool showBackdrop{false}; // passthrough/void is the default; B or grip+menu adds the grid
  GizmoMode mode{GizmoMode::Translate};
  GizmoSpace space{GizmoSpace::World};
  GizmoDrag drag;
  bool dragging{false};
  // Per-frame pointer hover: the hovered gizmo HANDLE brightens (model hover
  // tints were explicitly rejected: the only model tint is on the selection).
  std::optional<GizmoHandle> hoverHandle;
  bool looking{false};
  bool panning{false};
  bool running{true};
  bool snap{false};
  bool partSelect{false}; // shift/ctrl (desktop) or grip (VR): select one part
  // Model library for spawn-on-the-fly (N on desktop, A in VR): the directory
  // is rescanned on every spawn, so a file dropped in mid-session is spawnable
  // without a restart.
  std::string libraryDir;
  // The model gallery (spawn key): small geometry-only previews of the library
  // in an arc; the aimed one brightens, the trigger spawns the full model at
  // the aim point (one undo step). Previews never touch undo history.
  bool galleryActive{false};
  std::vector<GallerySlot> gallerySlots;
  std::vector<vredit::NodeId> galleryHoverNodes; // warm-tinted in the frame
  // Stick-driven editing (VR): one undo step per stick gesture — begun on the
  // first deflection past the dead zone, committed when the sticks settle.
  bool stickSession{false};
  // The stick gesture's frame, latched at gesture start: the pivot (the
  // moving set's bounds centre), the pre-signed tip axis and the fine rate.
  // Latching keeps one gesture one coherent step — a live head pose or grip
  // toggle mid-gesture must not steer the tip, change its rate, or walk
  // asymmetric models sideways via a re-centred pivot.
  Vec3 stickPivot{};
  Vec3 stickTipAxis{0.0F, 0.0F, -1.0F};
  float stickFineRate{1.0F};
  // Re-arm latch: after an undo/redo the sticks must return to neutral
  // before a new gesture may start, or the same frame's drift would
  // cancel-and-restart and swallow the undo.
  bool stickArmed{true};
  // Menu tap/hold (VR): tap cycles the gizmo mode (fires on release), hold
  // toggles the backdrop once at 0.4 s and cancels the tap.
  float menuCharge{0.0F};
  bool menuHoldLatched{false};
  bool menuTapped{false};
  bool menuPrevHeld{false};
  // Hold-X delete (VR): the selection's tint ramps red over kDeleteChargeS
  // and the delete fires at full charge; release before that cancels.
  bool deleteHolding{false};
  std::chrono::steady_clock::time_point deleteHoldStart{};
  float deleteCharge{0.0F};
  // Free-flight state (WASD/QE held keys + Shift boost).
  bool flyForward{false};
  bool flyBack{false};
  bool flyLeft{false};
  bool flyRight{false};
  bool flyUp{false};
  bool flyDown{false};
  bool flyBoost{false};
  std::uint8_t panButton{0};
  Vec2 lastMouse{};
  std::uint32_t width{1280};
  std::uint32_t height{800};
};

struct XcbWindow {
  xcb_connection_t* connection{nullptr};
  xcb_window_t window{0};
  xcb_intern_atom_reply_t* wmProtocols{nullptr};
  xcb_intern_atom_reply_t* wmDelete{nullptr};
  xcb_keycode_t minKeycode{0};
  xcb_keycode_t maxKeycode{0};
  std::uint8_t keysymsPerKeycode{0};
  xcb_keysym_t* keysyms{nullptr};
  xcb_get_keyboard_mapping_reply_t* keymapReply{nullptr};
};

void addCubeVisual(App& app, NodeId id, Vec3 colour) {
  NodeVisual visual;
  visual.id = id;
  visual.colour = colour;
  app.visuals.push_back(visual);
}

void buildDemoScene(App& app) {
  // Two single-part models and one two-part model (the parent/child assembly),
  // so model-vs-part selection is exercised by the demo scene itself.
  std::uint32_t const groupA = app.nextModelGroup++;
  std::uint32_t const groupB = app.nextModelGroup++;
  std::uint32_t const groupAssembly = app.nextModelGroup++;

  NodeId a = app.scene.create();
  Transform ta;
  ta.translation = {1.2F, 0.5F, 0.0F};
  app.scene.setLocal(a, ta);
  addCubeVisual(app, a, {0.85F, 0.25F, 0.20F});
  app.modelGroups.push_back({a, groupA});

  NodeId b = app.scene.create();
  Transform tb;
  tb.translation = {-1.8F, 0.35F, 1.2F};
  tb.scale = {0.7F, 0.7F, 0.7F};
  app.scene.setLocal(b, tb);
  addCubeVisual(app, b, {0.25F, 0.45F, 0.85F});
  app.modelGroups.push_back({b, groupB});

  NodeId parent = app.scene.create();
  Transform tp;
  tp.translation = {0.0F, 0.5F, -2.4F};
  tp.rotation = quatFromAxisAngle({0.0F, 1.0F, 0.0F}, 0.7853982F);
  app.scene.setLocal(parent, tp);
  addCubeVisual(app, parent, {0.35F, 0.75F, 0.35F});
  app.modelGroups.push_back({parent, groupAssembly});

  NodeId child = app.scene.create(parent);
  Transform tc;
  tc.translation = {0.0F, 1.4F, 0.0F};
  tc.scale = {0.55F, 0.55F, 0.55F};
  app.scene.setLocal(child, tc);
  addCubeVisual(app, child, {0.85F, 0.75F, 0.25F});
  app.modelGroups.push_back({child, groupAssembly});
}

// Adds a loaded model's meshes and node hierarchy to the scene and visuals.
// Node-local picking triangles derived from loaded mesh data, shared per mesh
// so meshes referenced by several nodes cost one copy.
std::shared_ptr<const vredit::PickGeometry> makePickGeometry(MeshData const& data) {
  auto geometry = std::make_shared<vredit::PickGeometry>();
  geometry->positions.reserve(data.vertices.size());
  for (MeshVertex const& vertex : data.vertices) {
    geometry->positions.push_back(vertex.pos);
  }
  geometry->indices = data.indices;
  return geometry;
}

// The insertion itself: copies mesh/pick data and allocates, so it CAN throw
// on a hostile/huge model. Never let that terminate the process (this runs on
// arbitrary user files during on-the-fly spawning).
bool addModelToSceneImpl(App& app,
                         VkRenderer& renderer,
                         Model const& model,
                         std::vector<NodeId>* rootsOut = nullptr) {
  std::vector<MeshHandle> meshHandles(model.meshes.size(), 0U);
  std::vector<std::shared_ptr<const vredit::PickGeometry>> pickGeometries(model.meshes.size());
  for (std::size_t m = 0; m < model.meshes.size(); ++m) {
    meshHandles[m] = renderer.addMesh(model.meshes[m].data);
    if (meshHandles[m] == kInvalidMesh) {
      std::printf("mesh registration failed for mesh %zu\n", m);
      return false;
    }
    pickGeometries[m] = makePickGeometry(model.meshes[m].data);
  }
  std::vector<TextureHandle> textureHandles(model.textures.size(), 0U);
  for (std::size_t t = 0; t < model.textures.size(); ++t) {
    Image const& image = model.textures[t].image;
    if (image.width == 0 || image.height == 0) {
      // An empty image (the geometry-only preview loads, or a skipped decode)
      // means UNTEXTURED — handle 0 binds nothing. It is not a failure.
      textureHandles[t] = 0U;
      continue;
    }
    textureHandles[t] = renderer.addTexture(model.textures[t].image, model.textures[t].repeatU,
                                            model.textures[t].repeatV);
    if (textureHandles[t] == kInvalidTexture) {
      std::printf("texture registration failed for texture %zu\n", t);
      return false;
    }
  }
  std::vector<NodeId> handles(model.nodes.size(), kInvalidNode);
  std::uint32_t const modelGroup = app.nextModelGroup++;
  for (std::size_t n = 0; n < model.nodes.size(); ++n) {
    ModelNode const& node = model.nodes[n];
    NodeId parent = node.parent >= 0 ? handles[static_cast<std::size_t>(node.parent)] : kInvalidNode;
    NodeId id = app.scene.create(parent);
    app.scene.setLocal(id, node.local);
    handles[n] = id;
    if (rootsOut != nullptr && node.parent < 0) {
      rootsOut->push_back(id); // top-level nodes: the spawn placement roots
    }
    app.modelGroups.push_back({id, modelGroup}); // one file = one model
    if (node.mesh >= 0 && static_cast<std::size_t>(node.mesh) < model.meshes.size()) {
      ModelMesh const& mesh = model.meshes[static_cast<std::size_t>(node.mesh)];
      NodeVisual visual;
      visual.id = id;
      visual.mesh = meshHandles[static_cast<std::size_t>(node.mesh)];
      visual.localBounds = meshBounds(mesh.data);
      visual.pickGeometry = pickGeometries[static_cast<std::size_t>(node.mesh)];
      visual.colour = mesh.colour;
      visual.alphaFactor = mesh.alphaFactor;
      visual.alpha = mesh.alpha;
      visual.alphaCutoff = mesh.alphaCutoff;
      if (mesh.texture >= 0 && static_cast<std::size_t>(mesh.texture) < textureHandles.size()) {
        visual.texture = textureHandles[static_cast<std::size_t>(mesh.texture)];
      }
      if (mesh.normalTexture >= 0 &&
          static_cast<std::size_t>(mesh.normalTexture) < textureHandles.size() &&
          model.textures[static_cast<std::size_t>(mesh.normalTexture)].image.width > 0U) {
        // An empty normal image (geometry-only previews) means FLAT: binding
        // the white fallback as a normal map would distort the lighting.
        visual.normalTexture = textureHandles[static_cast<std::size_t>(mesh.normalTexture)];
        visual.normalMapped = true;
        visual.normalScale = mesh.normalScale;
      }
      app.visuals.push_back(visual);
    }
  }
  return true;
}

bool addModelToScene(App& app,
                     VkRenderer& renderer,
                     Model const& model,
                     std::vector<NodeId>* rootsOut = nullptr) {
  try {
    return addModelToSceneImpl(app, renderer, model, rootsOut);
  } catch (std::exception const& e) {
    std::printf("model insertion failed (%s) — model not added\n", e.what());
    return false;
  }
}

// Frames the camera on the combined world bounds of all visuals.
void frameScene(App& app) {
  bool first = true;
  Aabb bounds{};
  for (NodeVisual const& visual : app.visuals) {
    Mat4 world = app.scene.world(visual.id);
    for (int corner = 0; corner < 8; ++corner) {
      Vec3 local{((corner & 1) != 0) ? visual.localBounds.max.x : visual.localBounds.min.x,
                 ((corner & 2) != 0) ? visual.localBounds.max.y : visual.localBounds.min.y,
                 ((corner & 4) != 0) ? visual.localBounds.max.z : visual.localBounds.min.z};
      Vec3 point = transformPoint(world, local);
      if (first) {
        bounds = {point, point};
        first = false;
      } else {
        bounds.min = {std::fmin(bounds.min.x, point.x), std::fmin(bounds.min.y, point.y),
                      std::fmin(bounds.min.z, point.z)};
        bounds.max = {std::fmax(bounds.max.x, point.x), std::fmax(bounds.max.y, point.y),
                      std::fmax(bounds.max.z, point.z)};
      }
    }
  }
  if (first) {
    return;
  }
  Vec3 centre = (bounds.min + bounds.max) * 0.5F;
  float radius = length(bounds.max - bounds.min) * 0.5F;
  float distance = std::fmax(2.2F * radius, 1.5F);
  app.camera.yaw = 0.6F;
  app.camera.pitch = -0.3F;
  app.camera.position = centre - cameraForward(app.camera) * distance;
  clampAboveGround(app.camera); // framing must never start below the grid
}

Ray appRay(App const& app, Vec2 pixel) {
  Vec2 viewport{static_cast<float>(app.width), static_cast<float>(app.height)};
  return cameraRay(app.camera, viewport.x / viewport.y, pixel, viewport);
}

// Combined world bounds of the selection: visuals contribute their exact
// transformed bounds, hierarchy-only nodes (no mesh) their origin point.
bool selectionBounds(App& app, Aabb& boundsOut) {
  bool any = false;
  Aabb bounds{};
  for (NodeId id : app.selection) {
    if (!app.scene.valid(id)) {
      continue;
    }
    Aabb contribution;
    bool found = false;
    for (NodeVisual const& visual : app.visuals) {
      if (visual.id == id) {
        contribution = transformAabb(app.scene.world(id), visual.localBounds);
        found = true;
        break;
      }
    }
    if (!found) {
      Vec3 point = app.scene.worldPosition(id);
      contribution = {point, point};
    }
    bounds = any ? unite(bounds, contribution) : contribution;
    any = true;
  }
  boundsOut = bounds;
  return any;
}

bool gizmoFrame(App& app, Vec3& pivot, Quat& orientation) {
  if (app.selection.empty()) {
    return false;
  }
  // Handles sit at the centre of the selection's bounds, not at the node
  // origin (which is often at the model's feet).
  Aabb bounds{};
  pivot = selectionBounds(app, bounds) ? aabbCentre(bounds)
                                       : app.scene.worldPosition(app.selection[0]);
  orientation = Quat{};
  if (app.space == GizmoSpace::Local) {
    Vec3 t;
    Vec3 s;
    decompose(app.scene.world(app.selection[0]), t, orientation, s);
  }
  return true;
}

float gizmoSize(App& app, Vec3 pivot) {
  float distance = length(app.camera.position - pivot);
  float worldPerPixel =
      2.0F * distance * std::tan(app.camera.fovY * 0.5F) / static_cast<float>(app.height);
  float size = kGizmoPixels * worldPerPixel;
  // Scale with the selection so the handles stay reachable past the silhouette
  // of a large model (the gizmo sits at the bounds centre — inside the body),
  // bounded at 2x so the desktop feel stays familiar.
  Aabb bounds{};
  if (selectionBounds(app, bounds)) {
    float radius = 0.5F * length(bounds.max - bounds.min);
    size = std::fmax(size, std::fmin(2.0F * size, 0.75F * radius));
  }
  return size;
}

// Body pick for a pointer ray: exact surface only (closest surface still wins
// through gaps). The bounding box is not a click target — a ray that misses
// the triangles hits nothing, so empty space never selects a model. Returns
// the hit with its along-ray distance so callers can depth-order bodies
// against gizmo handles.
std::optional<vredit::PickHit> pickNode(App& app, Ray const& ray) {
  std::vector<vredit::PickCandidate> candidates;
  for (NodeVisual const& visual : app.visuals) {
    Mat4 world = app.scene.world(visual.id);
    Aabb box{};
    bool first = true;
    for (int corner = 0; corner < 8; ++corner) {
      Vec3 local{((corner & 1) != 0) ? visual.localBounds.max.x : visual.localBounds.min.x,
                 ((corner & 2) != 0) ? visual.localBounds.max.y : visual.localBounds.min.y,
                 ((corner & 4) != 0) ? visual.localBounds.max.z : visual.localBounds.min.z};
      Vec3 point = transformPoint(world, local);
      if (first) {
        box = {point, point};
        first = false;
      } else {
        box.min = {std::fmin(box.min.x, point.x), std::fmin(box.min.y, point.y),
                   std::fmin(box.min.z, point.z)};
        box.max = {std::fmax(box.max.x, point.x), std::fmax(box.max.y, point.y),
                   std::fmax(box.max.z, point.z)};
      }
    }
    vredit::PickCandidate candidate;
    candidate.node = visual.id;
    candidate.worldBounds = box;
    candidate.world = world;
    candidate.geometry = visual.pickGeometry;
    candidates.push_back(std::move(candidate));
  }
  // The bounding box is NOT a click target: a press on empty space must
  // select nothing, even inside a model's bounds (a generous box fallback
  // made "click the air next to her" select the model).
  return vredit::pickClosestSurface(ray, candidates);
}

// Cheap bounds-only body pick (per-frame hover feedback).
// What a press on this ray would act on, resolved by depth: a gizmo handle
// wins only when it is nearer along the ray than the model body. The handles
// sit at the selection's bounds centre — inside the body — so without depth
// ordering every press near a model's middle stole a handle drag instead of
// the body grab (one axis of movement = "very slowly", and AxisY/Free could
// lift models off the floor). Hover feedback reports this same resolution,
// so what lights up is what a press will do.
struct PointerTarget {
  std::optional<GizmoHandle> handle; // set when a handle wins
  NodeId node{kInvalidNode};         // set when a body wins
};

PointerTarget resolvePointerTarget(App& app, Ray const& ray, float thresholdPx) {
  PointerTarget target;
  // The body uses the press's own pick (exact surface first, bounds fallback):
  // hover and press must agree on what is under the pointer, and the exact
  // pick is what keeps "closest surface through gaps" honest. It costs
  // milliseconds on a dense model — the price of "what lights up is what a
  // press grabs".
  auto body = pickNode(app, ray);
  Vec3 pivot;
  Quat orientation;
  std::optional<vredit::RayHandleHit> handleHit;
  if (gizmoFrame(app, pivot, orientation)) {
    float size = gizmoSize(app, pivot);
    auto geometry = buildHandleGeometry(app.mode, pivot, orientation, size);
    handleHit = vredit::pickHandleByRay(geometry, ray, size * (thresholdPx / kGizmoPixels));
  }
  // Depth order: a handle wins only when it is in front of the body hit. The
  // handles sit at the bounds centre — inside the body — so pressing a model's
  // middle grabs the body (the primary action) and handles are used where they
  // protrude past the silhouette, which the selection-scaled gizmo makes
  // generous. When only the bounds fallback hit, the body still wins ties:
  // pointing into an empty box region should grab the model, not a buried
  // handle.
  bool const handleWins =
      handleHit.has_value() && (!body.has_value() || handleHit->distance < body->t);
  if (handleWins) {
    target.handle = handleHit->handle;
  } else if (body.has_value()) {
    target.node = body->node;
  }
  return target;
}

// The floor rule ("models stay stuck to the floor") applies to whole models:
// tipping a PART must not lift it out of its model by flooring through the
// part's own bounds. True when every model group touched by the selection is
// FULLY selected (a part selection is false; an empty selection is false).
bool wholeModelSelection(App const& app) {
  if (app.selection.empty()) {
    return false;
  }
  for (auto const& [node, group] : app.modelGroups) {
    if (group == 0U) {
      continue; // group 0 = ungrouped loose nodes: each is its own "model"
    }
    if (std::find(app.selection.begin(), app.selection.end(), node) == app.selection.end()) {
      continue;
    }
    // A selected node: its whole group must be selected.
    for (auto const& [other, otherGroup] : app.modelGroups) {
      if (otherGroup != group) {
        continue;
      }
      if (std::find(app.selection.begin(), app.selection.end(), other) == app.selection.end()) {
        return false;
      }
    }
  }
  return true;
}

// Refresh the per-frame hover state (the hovered gizmo handle brightens).
// Suppressed during a live drag so the highlight does not chase the pointer
// mid-edit. Model hover tints were explicitly rejected.
void updateHover(App& app, Ray const& ray, float thresholdPx, bool partSelect) {
  app.hoverHandle.reset();
  if (app.dragging || app.stickSession || app.galleryActive) {
    return; // the gallery runs its own preview hover instead
  }
  PointerTarget target = resolvePointerTarget(app, ray, thresholdPx);
  if (target.handle.has_value()) {
    app.hoverHandle = target.handle;
  } else {
    (void)partSelect; // models are not highlighted on hover
  }
}

float snapStepFor(App const& app) {
  if (!app.snap) {
    return 0.0F;
  }
  switch (app.mode) {
  case GizmoMode::Rotate:
    return kSnapRotate;
  case GizmoMode::Scale:
    return kSnapScale;
  default:
    return kSnapTranslate;
  }
}

// Full editor state for undo/redo: structure + visuals + groups + selection.
vredit::EditorSnapshot snapshot(App const& app) {
  return vredit::captureState(app.scene, app.visuals, app.modelGroups, app.selection);
}

void applySnapshot(App& app, vredit::EditorSnapshot const& state) {
  vredit::restoreState(state, app.scene, app.visuals, app.modelGroups, app.selection);
}

// Defined below; pointerPress commits any live edit before starting anew.
bool endDrag(App& app);
bool endEdits(App& app);
void galleryClose(App& app);
bool galleryPick(App& app, VkRenderer& renderer, Ray const& aimRay);

// Pointer interaction shared by the mouse and the VR controller ray. The
// caller resolves the handle under the pointer (screen-space for the mouse,
// ray-space for VR); a handle press starts a drag, anything else SELECTS.
// partSelect (shift/ctrl click, grip+trigger in VR) limits the selection to
// the single picked part instead of the whole model. A body press never moves
// anything — movement and rotation are the sticks' job in VR (controller aim
// is too jittery for accurate placement) and the gizmo's job on desktop.
void pointerPress(App& app,
                  Ray const& ray,
                  Vec3 const& viewDir,
                  PointerTarget const& target,
                  bool partSelect) {
  // While the model gallery is open it owns the interaction: previews are
  // transient UI and clicking elsewhere must not touch the world.
  if (app.galleryActive) {
    return;
  }
  // A press while a live edit is running means its release was never seen
  // (pointer left the window) or a stick gesture is mid-flight: commit it as
  // one undo step before starting over. endEdits covers BOTH engines (drag
  // and stick) — ending only the drag here left the stick session fused with
  // the drag, clobbering transforms and losing movement from history.
  endEdits(app);
  if (target.handle.has_value()) {
    Vec3 pivot;
    Quat orientation;
    if (gizmoFrame(app, pivot, orientation)) {
      GizmoSettings settings;
      settings.mode = app.mode;
      settings.space = app.space;
      settings.snapStep = snapStepFor(app);
      bool started =
          app.drag.begin(app.scene, app.selection, pivot, orientation, *target.handle, ray, viewDir,
                         settings);
      if (started) {
        // Snapshot after begin() (it captures state, never mutates) and before
        // the first update() — undo restores exactly this pre-drag state.
        app.undo.begin(snapshot(app));
      }
      app.dragging = started;
    }
    return;
  }
  if (target.node == kInvalidNode) {
    app.selection.clear();
    return;
  }
  app.selection = vredit::expandSelection(app.modelGroups, target.node, partSelect);
}

void pointerMove(App& app, Ray const& ray) {
  if (app.dragging) {
    app.drag.update(ray);
  }
}

// Ends an active drag and records it as one undo step (no-op when it moved
// nothing). Drag-only: the trigger release path uses this so a release never
// cuts a stick gesture short (see endEdits for the both-engines version).
bool endDrag(App& app) { // returns true when an undo step was recorded
  if (!app.dragging) {
    return false;
  }
  app.drag.end();
  app.dragging = false;
  return app.undo.commit(snapshot(app));
}

// Cancels an active edit (handle drag or stick gesture) without touching the
// history: restores its pre-edit state and drops the pending undo entry, so
// both the undo and redo stacks stay exactly as they were. Any undo/redo
// keypress during a live edit does this and nothing else — going through
// commit() instead would clear an existing redo branch and make the cancelled
// edit itself redoable.
void cancelLiveEdit(App& app) {
  app.drag.end();
  app.dragging = false;
  app.stickSession = false;
  // Disarm the sticks exactly like the apply paths: a held stick would
  // otherwise restart the cancelled gesture in the SAME frame, re-applying
  // the drift and swallowing the undo/redo press that cancelled it.
  app.stickArmed = false;
  if (app.galleryActive) {
    galleryClose(app); // previews leave; no history was taken for them
    return;
  }
  vredit::EditorSnapshot state = snapshot(app);
  if (app.undo.cancel(state)) {
    applySnapshot(app, state);
  }
}

void pointerRelease(App& app) {
  endDrag(app);
}

// Commits any live edit — handle drag or stick gesture — as its own undo
// step. Used where a new action begins (spawn, delete, gizmo-mode switches),
// as opposed to endDrag (drag-only) which the trigger release path uses so a
// release never cuts a stick gesture short.
bool endEdits(App& app) {
  bool const wasActive = app.dragging || app.stickSession;
  app.drag.end();
  app.dragging = false;
  app.stickSession = false;
  return wasActive ? app.undo.commit(snapshot(app)) : false;
}

// The one delete rule (see deleteSubtrees): removes the selected models or
// parts — subtrees rooted at the selection — as one undo step.
bool deleteSelection(App& app) {
  if (app.selection.empty() || app.galleryActive) {
    // While the gallery is open, delete must not fire: the previews are
    // transient UI and the gallery owns the session. Nothing else mutates.
    return false;
  }
  endEdits(app);
  app.undo.begin(snapshot(app));
  std::vector<NodeId> const targets = app.selection;
  vredit::deleteSubtrees(app.scene, app.visuals, app.modelGroups, app.selection, targets);
  return app.undo.commit(snapshot(app));
}

// Lowercased 4-character filename suffix for extension checks. The library
// scan and the loader entry point share this so "Model.GLB" can never pass
// one and fail the other (that mismatch was the "stopped loading models"
// report: case-sensitive loader vs case-insensitive scan).
std::string lowerSuffix(std::string const& name) {
  std::string ext = name.size() >= 4U ? name.substr(name.size() - 4U) : std::string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext;
}

// The spawn library: every .glb/.obj under the folder. Rescanned on every
// spawn, so a file dropped in mid-session is spawnable without a restart.
std::vector<std::string> scanLibrary(std::string const& dir) {
  std::vector<std::string> out;
  std::error_code ec;
  std::filesystem::recursive_directory_iterator it(dir, ec);
  std::filesystem::recursive_directory_iterator const end;
  for (; !ec && it != end; it.increment(ec)) {
    // A separate error_code: one unreadable entry must not abort the scan
    // (the iteration error_code would end the loop with no diagnostic).
    std::error_code statusEc;
    if (!it->is_regular_file(statusEc)) {
      continue;
    }
    std::string const name = it->path().filename().string();
    std::string const ext = lowerSuffix(name);
    bool const glb = ext == ".glb";
    bool const obj = ext == ".obj";
    if (glb || obj) {
      out.push_back(it->path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool loadModelFile(std::string const& path,
                   Model& model,
                   std::string& error,
                   bool withImages = true) {
  // Extension matching is case-INSENSITIVE (see lowerSuffix).
  try {
    std::string const ext = lowerSuffix(path);
    if (ext == ".glb") {
      return loadGlbFile(path.c_str(), model, error, withImages);
    }
    if (ext == ".obj") {
      return loadObjFile(path.c_str(), model, error, withImages);
    }
    error = "unsupported model type (want .glb or .obj)";
    return false;
  } catch (std::exception const& e) {
    // A loader must never take the process down: treat allocation/format
    // surprises as a failed load (this path is exercised by on-the-fly spawns
    // of arbitrary user files).
    error = std::string("load failed (") + e.what() + ")";
    return false;
  }
}

// The model gallery ("I want a model selector"). The spawn key opens a
// gallery: small, geometry-only preview copies of every library model laid
// out in an arc in front of the viewer. The aimed preview brightens; the
// trigger (click on desktop) spawns the FULL textured model on the floor at
// the aim point as one undo step and closes the gallery. The spawn key or
// undo (B) closes without spawning. Previews decode no images (megabytes of
// geometry, not gigabytes of textures) and live only while the gallery is
// open — the pick closes it BEFORE its undo step, so history never contains
// a preview.

// The preview slot under the aim ray, or -1. World models do not count: the
// gallery picks only among its own previews, never a placed model behind them.
int gallerySlotAt(App& app, Ray const& aimRay) {
  auto hit = pickNode(app, aimRay);
  if (!hit.has_value()) {
    return -1;
  }
  for (std::size_t s = 0; s < app.gallerySlots.size(); ++s) {
    std::vector<vredit::NodeId> const& nodes = app.gallerySlots[s].nodes;
    if (std::find(nodes.begin(), nodes.end(), hit->node) != nodes.end()) {
      return static_cast<int>(s);
    }
  }
  return -1;
}

// Removes every preview and closes the gallery (no history impact: previews
// are transient UI, and the pick path closes BEFORE taking its undo step).
void galleryClose(App& app) {
  std::vector<vredit::NodeId> all;
  for (GallerySlot const& slot : app.gallerySlots) {
    all.insert(all.end(), slot.roots.begin(), slot.roots.end());
  }
  if (!all.empty()) {
    vredit::deleteSubtrees(app.scene, app.visuals, app.modelGroups, app.selection, all);
  }
  app.gallerySlots.clear();
  app.galleryHoverNodes.clear();
  app.galleryActive = false;
  std::printf("[gallery] closed\n");
}

// Opens the gallery: every library model becomes a small preview in an arc in
// front of the viewer (a second row when there are more than 8). Files that
// fail to load are skipped with a message — one bad file cannot block others.
void galleryOpen(App& app,
                 VkRenderer& renderer,
                 Vec3 const& viewPos,
                 Vec3 const& viewForward) {
  if (app.galleryActive) {
    return;
  }
  // A live drag or stick gesture must not keep mutating the world behind the
  // gallery (and its pending edit would tangle with the pick's): close it as
  // its own undo step first.
  endEdits(app);
  std::vector<std::string> files = scanLibrary(app.libraryDir);
  if (files.empty()) {
    std::printf("[gallery] library folder is empty: %s\n", app.libraryDir.c_str());
    return;
  }
  // Set before any load: a failed load still leaves a closeable gallery.
  app.galleryActive = true;
  Vec3 centre{viewPos.x, 0.0F, viewPos.z};
  Vec3 fwd{viewForward.x, 0.0F, viewForward.z};
  if (dot(fwd, fwd) < 1e-6F) {
    fwd = {0.0F, 0.0F, -1.0F};
  }
  fwd = normalized(fwd);
  Vec3 const right{fwd.z, 0.0F, -fwd.x};
  constexpr std::size_t kPerRow = 8;
  for (std::size_t i = 0; i < files.size(); ++i) {
    Model model;
    std::string error;
    std::vector<vredit::NodeId> roots;
    vredit::EditorSnapshot beforeFile = snapshot(app);
    if (!loadModelFile(files[i], model, error, /*withImages=*/false) ||
        !addModelToScene(app, renderer, model, &roots) || roots.empty()) {
      // A hostile model can throw MID-insert (addModelToScene catches), and a
      // scene with no top-level nodes loads "successfully" with zero roots:
      // both leave junk in the scene. Restore the pre-file state so no preview
      // can leak into the scene or later undo snapshots.
      if (!roots.empty() || !app.visuals.empty()) {
        applySnapshot(app, beforeFile);
      }
      std::printf("[gallery] cannot show %s: %s\n", files[i].c_str(),
                  error.empty() ? "could not add to scene" : error.c_str());
      continue;
    }
    GallerySlot slot;
    slot.file = files[i];
    slot.roots = roots;
    slot.nodes = vredit::expandSelection(app.modelGroups, roots[0], false);
    // Rows of up to 8: arc angles from -62 to +62 degrees, each further row
    // 0.9 m deeper and staggered by half a column so rows never coincide.
    // Previews are scaled to ~0.4 m and sit ON the floor like everything else.
    std::size_t const row = i / kPerRow;
    std::size_t const col = i % kPerRow;
    std::size_t const rowCount =
        std::min(kPerRow, files.size() - row * kPerRow);
    float const spacing = rowCount > 1 ? 2.18F / static_cast<float>(rowCount - 1) : 2.18F;
    float t = rowCount > 1 ? static_cast<float>(col) / static_cast<float>(rowCount - 1) : 0.5F;
    if (row % 2 == 1) {
      t += 0.5F * spacing / 2.18F; // stagger back rows by half a column
    }
    float const angle = (t - 0.5F) * 2.18F; // +/-62.5 degrees
    float const radius = 2.0F + 0.9F * static_cast<float>(row);
    Vec3 const slotPos = centre + (fwd * std::cos(angle) + right * std::sin(angle)) * radius;
    // Scale-to-fit must see the WHOLE model (meshless wrapper roots make
    // root-only bounds degenerate — the preview would stay full size).
    vredit::Aabb bounds{};
    if (vredit::nodesBounds(app.scene, app.visuals, slot.nodes, bounds)) {
      float const maxExtent =
          std::fmax(bounds.max.x - bounds.min.x,
                    std::fmax(bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z));
      if (maxExtent > 1e-4F) {
        vredit::scaleSelectionUniform(app.scene, app.visuals, roots, 0.4F / maxExtent);
      }
    }
    vredit::placeOnFloor(app.scene, app.visuals, roots, slotPos);
    app.gallerySlots.push_back(std::move(slot));
    std::printf("[gallery] %s\n", files[i].c_str());
  }
  if (app.gallerySlots.empty()) {
    std::printf("[gallery] no loadable model in %s\n", app.libraryDir.c_str());
    galleryClose(app);
    return;
  }
  std::printf("[gallery] %zu models — the trigger (click) on a preview spawns it\n",
              app.gallerySlots.size());
}

// The preview under the aim: its node set tints warm in the frame — the
// selector's "you are pointing at THIS one" feedback.
void galleryHover(App& app, Ray const& aimRay) {
  app.galleryHoverNodes.clear();
  if (!app.galleryActive) {
    return;
  }
  int const slot = gallerySlotAt(app, aimRay);
  if (slot >= 0) {
    app.galleryHoverNodes = app.gallerySlots[static_cast<std::size_t>(slot)].nodes;
  }
}

// The trigger while the gallery is open: spawn the hovered preview's model
// (full textures) on the floor at the aim point as ONE undo step and close.
// No hovered preview = no spawn (never pick the wrong thing by accident).
bool galleryPick(App& app, VkRenderer& renderer, Ray const& aimRay) {
  if (!app.galleryActive) {
    return false;
  }
  // Belt and braces: no live edit should exist while the gallery is open, but
  // if one does, commit it as its own step before the spawn's.
  endEdits(app);
  int const slot = gallerySlotAt(app, aimRay);
  if (slot < 0) {
    return false;
  }
  std::string const file = app.gallerySlots[static_cast<std::size_t>(slot)].file;
  // Close FIRST: the previews leave the scene before the undo step is taken,
  // so history contains the spawn alone — never a preview.
  galleryClose(app);
  app.undo.begin(snapshot(app));
  Model model;
  std::string error;
  std::vector<vredit::NodeId> roots;
  if (!loadModelFile(file, model, error) || !addModelToScene(app, renderer, model, &roots)) {
    std::printf("[gallery] cannot open %s: %s\n", file.c_str(),
                error.empty() ? "could not add to scene" : error.c_str());
    vredit::EditorSnapshot rollback = snapshot(app);
    if (app.undo.cancel(rollback)) {
      applySnapshot(app, rollback);
    }
    return false;
  }
  vredit::placeOnFloor(app.scene, app.visuals, roots, vredit::floorAimPoint(aimRay));
  app.undo.commit(snapshot(app));
  std::printf("[gallery] spawned %s\n", file.c_str());
  return true;
}

// "Show the model without the background": the grid goes and the clear
// becomes a plain void — in VR that void is the real world (alpha 0 lets the
// passthrough cameras show through), on desktop it is just black. This is the
// DEFAULT state (passthrough first): toggleBackdrop brings the grid back.
void applyBackdrop(App& app, VkRenderer& renderer, XrShell* shell) {
  if (app.showBackdrop) {
    renderer.setClearColour(0.08F, 0.09F, 0.11F, 1.0F);
    if (shell != nullptr) {
      shell->setPassthrough(false);
    }
  } else {
    // Alpha 0 lets the passthrough cameras show through; without runtime
    // support (or an alpha channel) keep an opaque void instead of black.
    bool const transparent = shell == nullptr || shell->passthroughSupported();
    renderer.setClearColour(0.0F, 0.0F, 0.0F, transparent ? 0.0F : 1.0F);
    if (shell != nullptr) {
      shell->setPassthrough(transparent);
    }
  }
}

void toggleBackdrop(App& app, VkRenderer& renderer, XrShell* shell) {
  app.showBackdrop = !app.showBackdrop;
  applyBackdrop(app, renderer, shell);
}

void assembleFrame(App& app,
                    std::vector<DrawItem>& items,
                    std::vector<RendererVertex>& depthLines,
                    std::vector<RendererVertex>& overlayLines) {
  if (app.showBackdrop) {
    appendGrid(depthLines, 8.0F, 1.0F);
  }
  auto isSelected = [&app](NodeId id) {
    return std::find(app.selection.begin(), app.selection.end(), id) != app.selection.end();
  };
  auto isGalleryHovered = [&app](NodeId id) {
    return std::find(app.galleryHoverNodes.begin(), app.galleryHoverNodes.end(), id) !=
           app.galleryHoverNodes.end();
  };
  auto appendItem = [&app, &items, &isSelected, &isGalleryHovered](NodeVisual const& visual) {
    DrawItem item;
    item.mesh = visual.mesh;
    item.texture = visual.texture;
    item.normalTexture = visual.normalTexture;
    item.normalMapped = visual.normalMapped;
    item.normalScale = visual.normalScale;
    item.model = app.scene.world(visual.id);
    item.colour = visual.colour;
    item.alphaFactor = visual.alphaFactor;
    item.alphaCutoff = visual.alphaCutoff;
    item.alpha = visual.alpha;
    item.lit = true;
    // 1 = selected (warm) — the only tint the user wants: the selection is
    // also visible from the gizmo sitting at the selection's centre. While
    // the hold-X delete charges, the selection ramps red (3 + charge). The
    // gallery's aimed preview gets the same warm tint (the selector's
    // "you are pointing at THIS one" feedback).
    if (isSelected(visual.id) && app.deleteCharge > 0.0F) {
      item.highlight = 3.0F + app.deleteCharge;
    } else {
      item.highlight = (isSelected(visual.id) || isGalleryHovered(visual.id)) ? 1.0F : 0.0F;
    }
    items.push_back(item);
  };
  // Opaque and mask-cutout items first, blended items last (painter's order).
  for (NodeVisual const& visual : app.visuals) {
    if (visual.alpha != AlphaMode::Blend) {
      appendItem(visual);
    }
  }
  for (NodeVisual const& visual : app.visuals) {
    if (visual.alpha == AlphaMode::Blend) {
      appendItem(visual);
    }
  }
  Vec3 pivot;
  Quat orientation;
  if (gizmoFrame(app, pivot, orientation)) {
    auto geometry = buildHandleGeometry(app.mode, pivot, orientation, gizmoSize(app, pivot));
    appendHandleLines(overlayLines, geometry, app.hoverHandle);
  }
}

bool openWindow(XcbWindow& window, std::uint32_t width, std::uint32_t height, char const* title) {
  window.connection = xcb_connect(nullptr, nullptr);
  if (xcb_connection_has_error(window.connection) != 0) {
    std::printf("[xcb] cannot connect to the display\n");
    return false;
  }
  xcb_screen_t* screen = xcb_setup_roots_iterator(xcb_get_setup(window.connection)).data;
  window.window = xcb_generate_id(window.connection);
  std::uint32_t values[2] = {
      screen->black_pixel,
      XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_BUTTON_PRESS | XCB_EVENT_MASK_BUTTON_RELEASE |
          XCB_EVENT_MASK_POINTER_MOTION | XCB_EVENT_MASK_KEY_PRESS | XCB_EVENT_MASK_KEY_RELEASE |
          XCB_EVENT_MASK_STRUCTURE_NOTIFY};
  xcb_create_window(window.connection, XCB_COPY_FROM_PARENT, window.window, screen->root, 0, 0,
                    width, height, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual,
                    XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK, values);
  xcb_change_property(window.connection, XCB_PROP_MODE_REPLACE, window.window, XCB_ATOM_WM_NAME,
                      XCB_ATOM_STRING, 8, static_cast<std::uint32_t>(std::strlen(title)), title);

  window.wmProtocols = xcb_intern_atom_reply(
      window.connection, xcb_intern_atom(window.connection, 0, 12, "WM_PROTOCOLS"), nullptr);
  window.wmDelete = xcb_intern_atom_reply(
      window.connection, xcb_intern_atom(window.connection, 0, 16, "WM_DELETE_WINDOW"), nullptr);
  if (window.wmProtocols != nullptr && window.wmDelete != nullptr) {
    xcb_change_property(window.connection, XCB_PROP_MODE_REPLACE, window.window,
                        window.wmProtocols->atom, XCB_ATOM_ATOM, 32, 1, &window.wmDelete->atom);
  }

  xcb_setup_t const* setup = xcb_get_setup(window.connection);
  window.minKeycode = setup->min_keycode;
  window.maxKeycode = setup->max_keycode;
  std::uint8_t keycodeCount = setup->max_keycode - setup->min_keycode + 1;
  window.keymapReply = xcb_get_keyboard_mapping_reply(
      window.connection,
      xcb_get_keyboard_mapping(window.connection, setup->min_keycode, keycodeCount), nullptr);
  if (window.keymapReply != nullptr) {
    window.keysymsPerKeycode = window.keymapReply->keysyms_per_keycode;
    window.keysyms = xcb_get_keyboard_mapping_keysyms(window.keymapReply);
  }

  xcb_map_window(window.connection, window.window);
  xcb_flush(window.connection);
  return true;
}

void closeWindow(XcbWindow& window) {
  if (window.keymapReply != nullptr) {
    std::free(window.keymapReply);
  }
  if (window.wmProtocols != nullptr) {
    std::free(window.wmProtocols);
  }
  if (window.wmDelete != nullptr) {
    std::free(window.wmDelete);
  }
  if (window.connection != nullptr) {
    xcb_disconnect(window.connection);
  }
}

xcb_keysym_t keysymFor(XcbWindow const& window, xcb_keycode_t keycode) {
  if (window.keysyms == nullptr || window.keysymsPerKeycode == 0U ||
      keycode < window.minKeycode || keycode > window.maxKeycode) {
    return 0;
  }
  std::size_t index =
      static_cast<std::size_t>(keycode - window.minKeycode) * window.keysymsPerKeycode;
  return window.keysyms[index];
}

void handleEvent(App& app,
                 XcbWindow& window,
                 xcb_generic_event_t* event,
                 VkRenderer& renderer) {
  switch (event->response_type & ~0x80U) {
  case XCB_BUTTON_PRESS: {
    auto* press = reinterpret_cast<xcb_button_press_event_t*>(event);
    Vec2 mouse{static_cast<float>(press->event_x), static_cast<float>(press->event_y)};
    app.lastMouse = mouse;
    app.snap = (press->state & XCB_MOD_MASK_CONTROL) != 0U;
    app.partSelect = (press->state & (XCB_MOD_MASK_SHIFT | XCB_MOD_MASK_CONTROL)) != 0U;
    if (app.width == 0U || app.height == 0U) {
      break; // minimised: pointer maths would divide by zero
    }
    if (press->detail == 1U) {
      Ray mouseRay = appRay(app, mouse);
      if (app.galleryActive) {
        // The gallery owns clicks: a click on a preview spawns that model.
        (void)galleryPick(app, renderer, mouseRay);
        break;
      }
      bool const partSelect = (press->state & (XCB_MOD_MASK_SHIFT | XCB_MOD_MASK_CONTROL)) != 0U;
      PointerTarget target = resolvePointerTarget(app, mouseRay, kPickThresholdPx);
      pointerPress(app, mouseRay, cameraForward(app.camera), target, partSelect);
    } else if (press->detail == 3U) {
      if ((press->state & XCB_MOD_MASK_SHIFT) != 0U) {
        app.panning = true;
        app.panButton = 3U;
      } else {
        app.looking = true;
      }
    } else if (press->detail == 2U) {
      app.panning = true;
      app.panButton = 2U;
    } else if (press->detail == 4U) {
      dollyCamera(app.camera, kWheelMetres);
    } else if (press->detail == 5U) {
      dollyCamera(app.camera, -kWheelMetres);
    }
    break;
  }
  case XCB_BUTTON_RELEASE: {
    auto* release = reinterpret_cast<xcb_button_release_event_t*>(event);
    if (release->detail == 1U) {
      pointerRelease(app);
    }
    if (release->detail == 3U) {
      app.looking = false;
    }
    if (app.panning && release->detail == app.panButton) {
      app.panning = false;
      app.panButton = 0U;
    }
    break;
  }
  case XCB_MOTION_NOTIFY: {
    auto* motion = reinterpret_cast<xcb_motion_notify_event_t*>(event);
    Vec2 mouse{static_cast<float>(motion->event_x), static_cast<float>(motion->event_y)};
    Vec2 delta{mouse.x - app.lastMouse.x, mouse.y - app.lastMouse.y};
    app.lastMouse = mouse;
    app.snap = (motion->state & XCB_MOD_MASK_CONTROL) != 0U;
    app.partSelect = (motion->state & (XCB_MOD_MASK_SHIFT | XCB_MOD_MASK_CONTROL)) != 0U;
    if (app.width == 0U || app.height == 0U) {
      endDrag(app); // minimised mid-drag: commit what was done, stop tracking
      app.looking = false;
      app.panning = false;
      app.panButton = 0U;
      break; // pointer maths would divide by zero
    }
    if (app.dragging) {
      pointerMove(app, appRay(app, mouse));
    } else if (app.looking) {
      cameraLook(app.camera, delta.x * kLookSpeed, -delta.y * kLookSpeed);
    } else if (app.panning) {
      panCamera(app.camera, delta.x, delta.y, static_cast<float>(app.height));
    }
    break;
  }
  case XCB_KEY_PRESS:
  case XCB_KEY_RELEASE: {
    bool pressed = (event->response_type & ~0x80U) == XCB_KEY_PRESS;
    auto* key = reinterpret_cast<xcb_key_press_event_t*>(event);
    xcb_keysym_t keysym = keysymFor(window, key->detail);
    // Free-flight movement keys are tracked on press/release and applied per
    // frame. Both cases are matched because this lookup reads the unshifted
    // keysym column only, and layouts differ.
    bool movementKey = true;
    switch (keysym) {
    case 'w': case 'W': app.flyForward = pressed; break;
    case 's': case 'S': app.flyBack = pressed; break;
    case 'a': case 'A': app.flyLeft = pressed; break;
    case 'd': case 'D': app.flyRight = pressed; break;
    case 'q': case 'Q': app.flyDown = pressed; break;
    case 'e': case 'E': app.flyUp = pressed; break;
    case 0xFFE1: case 0xFFE2: app.flyBoost = pressed; break; // Shift L/R
    default: movementKey = false; break;
    }
    if (movementKey) {
      break;
    }
    if (!pressed) {
      break;
    }
    // Only keys that change gizmo state cancel an active drag; modifier presses
    // (e.g. Ctrl for snap) must not.
    bool stateChanged = false;
    if ((keysym == 'z' || keysym == 'Z') && (key->state & XCB_MOD_MASK_CONTROL) != 0U) {
      // Undo/redo during a live edit cancels that edit first and stops there,
      // leaving both stacks untouched (see cancelLiveEdit).
      if (app.dragging || app.stickSession || app.galleryActive) {
        cancelLiveEdit(app);
        break;
      }
      bool const redo = (key->state & XCB_MOD_MASK_SHIFT) != 0U;
      vredit::EditorSnapshot state = snapshot(app);
      if (redo ? app.undo.redo(state) : app.undo.undo(state)) {
        applySnapshot(app, state);
        app.stickArmed = false; // sticks must re-centre before a new gesture
      }
    } else if ((keysym == 'y' || keysym == 'Y') &&
               (key->state & XCB_MOD_MASK_CONTROL) != 0U) {
      if (app.dragging || app.stickSession || app.galleryActive) {
        cancelLiveEdit(app);
        break;
      }
      vredit::EditorSnapshot state = snapshot(app);
      if (app.undo.redo(state)) {
        applySnapshot(app, state);
        app.stickArmed = false; // sticks must re-centre before a new gesture
      }
    } else if (keysym == 0xFF1BU) {
      app.running = false;
      stateChanged = true;
    } else if (keysym == 0xFFFFU || keysym == 0xFF08U) {
      // Delete / Backspace: remove the selection (model or part).
      deleteSelection(app);
    } else if (keysym == 'n' || keysym == 'N') {
      // The model gallery: N opens it (previews in an arc), N closes again.
      // A click on a preview spawns that model at the mouse aim point.
      if (app.galleryActive) {
        galleryClose(app);
      } else {
        galleryOpen(app, renderer, app.camera.position, cameraForward(app.camera));
      }
    } else if (keysym == '[' || keysym == ']') {
      // The gallery shows every model at once; there is nothing to cycle.
    } else if (keysym == '1' || keysym == 'g' || keysym == 'G') {
      app.mode = GizmoMode::Translate;
      stateChanged = true;
    } else if (keysym == '2' || keysym == 'r' || keysym == 'R') {
      app.mode = GizmoMode::Rotate;
      stateChanged = true;
    } else if (keysym == '3') {
      app.mode = GizmoMode::Scale;
      stateChanged = true;
    } else if (keysym == 'l' || keysym == 'L') {
      app.space = (app.space == GizmoSpace::World) ? GizmoSpace::Local : GizmoSpace::World;
      std::printf("gizmo space: %s\n", app.space == GizmoSpace::World ? "world" : "local");
      stateChanged = true;
    } else if (keysym == 'b' || keysym == 'B') {
      toggleBackdrop(app, renderer, nullptr);
    }
    if (stateChanged) {
      endEdits(app);
    }
    break;
  }
  case XCB_CONFIGURE_NOTIFY: {
    auto* configure = reinterpret_cast<xcb_configure_notify_event_t*>(event);
    app.width = configure->width;
    app.height = configure->height;
    if (!renderer.resize(app.width, app.height)) {
      std::printf("[vk] resize failed; exiting\n");
      app.running = false;
    }
    break;
  }
  case XCB_MAP_NOTIFY: {
    // Restoring a minimised window: some WMs send no ConfigureNotify at the
    // previous geometry, which would leave the renderer at zero extent (frozen
    // window). Re-request the resize when the two disagree.
    if (app.width > 0U && app.height > 0U && renderer.width() == 0U) {
      renderer.resize(app.width, app.height);
    }
    break;
  }
  case XCB_CLIENT_MESSAGE: {
    auto* message = reinterpret_cast<xcb_client_message_event_t*>(event);
    if (window.wmDelete != nullptr && message->data.data32[0] == window.wmDelete->atom) {
      app.running = false;
    }
    break;
  }
  default:
    break;
  }
}

bool writePpm(char const* path,
              std::vector<std::uint8_t> const& rgba,
              std::uint32_t width,
              std::uint32_t height) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  file << "P6\n" << width << " " << height << "\n255\n";
  for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
    char rgb[3] = {static_cast<char>(rgba[i * 4U]), static_cast<char>(rgba[i * 4U + 1U]),
                   static_cast<char>(rgba[i * 4U + 2U])};
    file.write(rgb, 3);
  }
  return static_cast<bool>(file);
}

bool loadModels(App& app, VkRenderer& renderer, std::vector<char const*> const& paths) {
  for (char const* path : paths) {
    Model model;
    std::string error;
    if (!loadModelFile(path, model, error)) {
      std::printf("failed to load %s: %s\n", path, error.c_str());
      return false;
    }
    if (!addModelToScene(app, renderer, model)) {
      return false;
    }
    std::printf("loaded %s: %zu meshes, %zu nodes\n", path, model.meshes.size(),
                model.nodes.size());
  }
  return true;
}

int runSmokeOffscreen(char const* shaderDir, std::vector<char const*> const& modelPaths,
                      bool textured) {
  App app;
  if (modelPaths.empty()) {
    buildDemoScene(app);
    app.selection.push_back(app.visuals[0].id); // exercise the gizmo overlay path
  }

  VkRenderer renderer;
  VkRenderer::Config config;
  config.windowed = false;
  config.width = 640;
  config.height = 480;
  config.shaderDir = shaderDir;
  if (!renderer.init(config)) {
    std::printf("SMOKE FAIL: renderer init\n");
    return 1;
  }
  if (!modelPaths.empty()) {
    if (!loadModels(app, renderer, modelPaths)) {
      std::printf("SMOKE FAIL: model loading\n");
      return 1;
    }
    frameScene(app);
  }

  std::vector<DrawItem> items;
  std::vector<RendererVertex> depthLines;
  std::vector<RendererVertex> overlayLines;
  assembleFrame(app, items, depthLines, overlayLines);
  if (!renderer.draw(app.camera, items, depthLines, overlayLines)) {
    std::printf("SMOKE FAIL: draw\n");
    return 1;
  }
  std::vector<std::uint8_t> pixels;
  if (!renderer.readback(pixels)) {
    std::printf("SMOKE FAIL: readback\n");
    return 1;
  }

  int covered = 0;
  int reddish = 0;
  int bluish = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(config.width) * config.height; ++i) {
    int r = pixels[i * 4U];
    int g = pixels[i * 4U + 1U];
    int b = pixels[i * 4U + 2U];
    if (std::abs(r - 20) > 4 || std::abs(g - 23) > 4 || std::abs(b - 28) > 4) {
      ++covered;
    }
    if (r > 40 && r * 4 > g * 5 && r * 4 > b * 5) {
      ++reddish;
    }
    if (b > 40 && b * 4 > g * 5 && b * 4 > r * 5) {
      ++bluish;
    }
  }
  writePpm(".build/smoke.ppm", pixels, config.width, config.height);
  std::printf("SMOKE pixels: covered=%d reddish=%d bluish=%d (of %d) -> .build/smoke.ppm\n", covered,
              reddish, bluish, config.width * config.height);

  // Structural edit round-trip: select everything, DELETE it (the geometry
  // leaves the image), UNDO (it returns pixel-exact). This is the end-to-end
  // proof of the full-state snapshot path: scene + visuals + groups through
  // deleteSubtrees and the undo history.
  {
    auto coveredOf = [&config](std::vector<std::uint8_t> const& px) {
      int count = 0;
      for (std::size_t i = 0; i < static_cast<std::size_t>(config.width) * config.height; ++i) {
        int r = px[i * 4U];
        int g = px[i * 4U + 1U];
        int b = px[i * 4U + 2U];
        if (std::abs(r - 20) > 4 || std::abs(g - 23) > 4 || std::abs(b - 28) > 4) {
          ++count;
        }
      }
      return count;
    };
    std::vector<NodeId> const savedSelection = app.selection;
    app.selection.clear();
    for (NodeVisual const& visual : app.visuals) {
      app.selection.push_back(visual.id);
    }
    // Re-render as the round-trip baseline: the selection state (and its warm
    // tint) must match the post-undo image or the comparison is meaningless.
    items.clear();
    depthLines.clear();
    overlayLines.clear();
    assembleFrame(app, items, depthLines, overlayLines);
    if (!renderer.draw(app.camera, items, depthLines, overlayLines) ||
        !renderer.readback(pixels)) {
      std::printf("SMOKE FAIL: draw for round-trip baseline\n");
      return 1;
    }
    int const itemsBefore = static_cast<int>(items.size());
    int const coveredBefore = coveredOf(pixels);
    if (!deleteSelection(app)) {
      std::printf("SMOKE FAIL: delete recorded no undo step\n");
      return 1;
    }
    items.clear();
    depthLines.clear();
    overlayLines.clear();
    assembleFrame(app, items, depthLines, overlayLines);
    if (!renderer.draw(app.camera, items, depthLines, overlayLines) ||
        !renderer.readback(pixels)) {
      std::printf("SMOKE FAIL: draw after delete\n");
      return 1;
    }
    int const coveredGone = coveredOf(pixels);
    if (coveredGone >= coveredBefore || !items.empty()) {
      std::printf("SMOKE FAIL: delete left geometry behind (%d -> %d pixels, %zu items)\n",
                  coveredBefore, coveredGone, items.size());
      return 1;
    }
    vredit::EditorSnapshot state = snapshot(app);
    if (!app.undo.undo(state)) {
      std::printf("SMOKE FAIL: undo after delete\n");
      return 1;
    }
    applySnapshot(app, state);
    items.clear();
    depthLines.clear();
    overlayLines.clear();
    assembleFrame(app, items, depthLines, overlayLines);
    if (!renderer.draw(app.camera, items, depthLines, overlayLines) ||
        !renderer.readback(pixels)) {
      std::printf("SMOKE FAIL: draw after undo\n");
      return 1;
    }
    int const coveredRestored = coveredOf(pixels);
    if (coveredRestored != coveredBefore ||
        static_cast<int>(items.size()) != itemsBefore) {
      std::printf("SMOKE FAIL: undo did not restore exactly (%d vs %d pixels, %zu vs %d items)\n",
                  coveredRestored, coveredBefore, items.size(), itemsBefore);
      return 1;
    }
    std::printf("SMOKE structural round-trip: %d -> %d -> %d pixels OK\n", coveredBefore,
                coveredGone, coveredRestored);
    // The round-trip works on the "select everything" state; hand the item
    // list back exactly as the textured checks found it (their brightness
    // deltas are sensitive to the selection tint).
    app.selection = savedSelection;
    items.clear();
    depthLines.clear();
    overlayLines.clear();
    assembleFrame(app, items, depthLines, overlayLines);
  }
  if (textured) {
    // Normal-map TBN check: flat normals vs the fixture's +Y map vs a diagonal
    // (+X+Y+Z) map registered here. +Y must brighten (bitangent axis), and the
    // diagonal must brighten further — it fails if any tangent axis is
    // sign-flipped, which the pure +Y map cannot detect (the light uses abs).
    bool anyNormal = false;
    for (DrawItem const& item : items) {
      anyNormal = anyNormal || item.normalMapped;
    }
    Image triImage;
    std::string triError;
    std::ifstream triFile(".build/fixtures/normal-tri.png", std::ios::binary);
    std::vector<std::uint8_t> triBytes((std::istreambuf_iterator<char>(triFile)),
                                       std::istreambuf_iterator<char>());
    if (!anyNormal || triBytes.empty() ||
        !decodeImage(triBytes.data(), triBytes.size(), triImage, triError)) {
      std::printf("SMOKE FAIL: normal-map fixtures missing\n");
      return 1;
    }
    TextureHandle triTexture = renderer.addTexture(triImage, true, true);
    if (triTexture == kInvalidTexture) {
      std::printf("SMOKE FAIL: normal-tri texture registration\n");
      return 1;
    }

    std::vector<DrawItem> flatItems = items;
    std::vector<DrawItem> triItems = items;
    for (std::size_t i = 0; i < items.size(); ++i) {
      flatItems[i].normalMapped = false;
      if (triItems[i].normalMapped) {
        triItems[i].normalTexture = triTexture;
      }
    }
    auto luminanceSum = [&renderer, &app](std::vector<DrawItem> const& drawItems,
                                          std::vector<RendererVertex> const& depthLines,
                                          std::vector<RendererVertex> const& overlayLines,
                                          double& sumOut) {
      if (!renderer.draw(app.camera, drawItems, depthLines, overlayLines)) {
        return false;
      }
      std::vector<std::uint8_t> pixels;
      if (!renderer.readback(pixels)) {
        return false;
      }
      sumOut = 0.0;
      for (std::size_t i = 0; i < pixels.size() / 4U; ++i) {
        for (std::size_t c = 0; c < 3U; ++c) {
          sumOut += pixels[i * 4U + c];
        }
      }
      return true;
    };
    double flatLum = 0.0;
    double upLum = 0.0;
    double triLum = 0.0;
    if (!luminanceSum(flatItems, depthLines, overlayLines, flatLum) ||
        !luminanceSum(items, depthLines, overlayLines, upLum) ||
        !luminanceSum(triItems, depthLines, overlayLines, triLum)) {
      std::printf("SMOKE FAIL: normal-toggle draws\n");
      return 1;
    }
    std::printf("SMOKE normal map: flat=%.0f up=%.0f tri=%.0f\n", flatLum, upLum, triLum);
    if (!(upLum > flatLum * 1.002)) {
      std::printf("SMOKE FAIL: +Y normal map did not brighten the lighting\n");
      return 1;
    }
    if (!(triLum > upLum * 1.002)) {
      std::printf("SMOKE FAIL: diagonal normal map did not brighten further (tangent axis "
                  "sign-flipped?)\n");
      return 1;
    }
  }
  int minCovered = modelPaths.empty() ? 2000 : 100;
  int minReddish = modelPaths.empty() ? 200 : 0;
  int minBluish = 0;
  if (textured) {
    // The textured fixtures are red/blue checkerboards: both colours must
    // survive sampling to prove UVs and textures reach the fragment shader.
    minReddish = 50;
    minBluish = 50;
  }
  if (covered < minCovered || reddish < minReddish || bluish < minBluish) {
    std::printf("SMOKE FAIL: pixel assertions\n");
    return 1;
  }
  std::printf("SMOKE PASS (offscreen) on %s\n", renderer.deviceName());
  renderer.shutdown();
  return 0;
}

int runSmokeWindow(char const* shaderDir, int frames, std::vector<char const*> const& modelPaths) {
  App app;
  if (modelPaths.empty()) {
    buildDemoScene(app);
  }
  XcbWindow window;
  if (!openWindow(window, app.width, app.height, "vr-editor (smoke)")) {
    return 2;
  }
  VkRenderer renderer;
  VkRenderer::Config config;
  config.windowed = true;
  config.xcbConnection = window.connection;
  config.xcbWindow = window.window;
  config.width = app.width;
  config.height = app.height;
  config.shaderDir = shaderDir;
  if (!renderer.init(config)) {
    renderer.shutdown(); // surface outlives the XCB connection
    closeWindow(window);
    return 1;
  }
  if (!modelPaths.empty()) {
    if (!loadModels(app, renderer, modelPaths)) {
      std::printf("SMOKE FAIL: model loading\n");
      renderer.shutdown();
      closeWindow(window);
      return 1;
    }
    frameScene(app);
  }
  for (int frame = 0; frame < frames && app.running; ++frame) {
    xcb_generic_event_t* event = nullptr;
    while ((event = xcb_poll_for_event(window.connection)) != nullptr) {
      handleEvent(app, window, event, renderer);
      std::free(event);
    }
    cameraLook(app.camera, 0.01F, 0.0F);
    std::vector<DrawItem> items;
    std::vector<RendererVertex> depthLines;
    std::vector<RendererVertex> overlayLines;
    assembleFrame(app, items, depthLines, overlayLines);
    if (!renderer.draw(app.camera, items, depthLines, overlayLines)) {
      std::printf("SMOKE FAIL: draw\n");
      renderer.shutdown();
      closeWindow(window);
      return 1;
    }
  }
  std::printf("SMOKE PASS (windowed, %d frames) on %s\n", frames, renderer.deviceName());
  renderer.shutdown();
  closeWindow(window);
  return 0;
}

constexpr float kXrTurnSpeed = 2.2F; // radians/second at full stick deflection

// Perf probe: renders the loaded scene at the headset's per-eye resolution
// (2889x3026 on WiVRn/Quest 3) offscreen and times the draws. Compares the
// serialised per-eye cost against the 90 Hz frame budget to decide whether
// choppiness is GPU-bound (needs render scale) or sync-bound (needs frame
// pipelining) before touching the XR loop.
int runPerfSmoke(char const* shaderDir, std::vector<char const*> const& modelPaths) {
  constexpr std::uint32_t kEyeWidth = 2889U;
  constexpr std::uint32_t kEyeHeight = 3026U;
  constexpr int kProbeFrames = 60;

  App app;
  app.width = kEyeWidth;
  app.height = kEyeHeight;
  VkRenderer renderer;
  VkRenderer::Config config;
  config.windowed = false;
  config.width = kEyeWidth;
  config.height = kEyeHeight;
  config.shaderDir = shaderDir;
  if (!renderer.init(config)) {
    std::printf("SMOKE FAIL: renderer init\n");
    return 1;
  }
  if (!modelPaths.empty()) {
    if (!loadModels(app, renderer, modelPaths)) {
      std::printf("SMOKE FAIL: model loading\n");
      return 1;
    }
    frameScene(app);
  } else {
    buildDemoScene(app);
  }

  std::vector<DrawItem> items;
  std::vector<RendererVertex> depthLines;
  std::vector<RendererVertex> overlayLines;
  assembleFrame(app, items, depthLines, overlayLines);

  // Warm up (compiles pipelines, faults in memory), then time the draws.
  for (int warm = 0; warm < 3; ++warm) {
    if (!renderer.draw(app.camera, items, depthLines, overlayLines)) {
      std::printf("SMOKE FAIL: draw\n");
      return 1;
    }
  }
  double totalMs = 0.0;
  double maxMs = 0.0;
  for (int frame = 0; frame < kProbeFrames; ++frame) {
    auto start = std::chrono::steady_clock::now();
    if (!renderer.draw(app.camera, items, depthLines, overlayLines)) {
      std::printf("SMOKE FAIL: draw\n");
      return 1;
    }
    double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    totalMs += ms;
    maxMs = ms > maxMs ? ms : maxMs;
  }
  std::vector<std::uint8_t> pixels;
  if (!renderer.readback(pixels)) {
    std::printf("SMOKE FAIL: readback\n");
    return 1;
  }

  double avgMs = totalMs / kProbeFrames;
  double budgetMs = 1000.0 / 90.0;
  std::printf("SMOKE PERF: %ux%u per eye, %d draws: avg %.2f ms, max %.2f ms\n", kEyeWidth,
              kEyeHeight, kProbeFrames, avgMs, maxMs);
  std::printf("SMOKE PERF: two serialised eyes ~ %.2f ms/frame vs 90 Hz budget %.2f ms (%s)\n",
              avgMs * 2.0, budgetMs, avgMs * 2.0 < budgetMs ? "GPU has headroom" : "GPU-bound");
  renderer.shutdown();
  return 0;
}

// Interactive or smoke VR session through the OpenXR shell. Returns 0 when no
// runtime/headset is available (nothing to test against), 2 on setup failure.
int runXr(char const* shaderDir,
          std::vector<char const*> const& modelPaths,
          bool smoke,
          std::string const& libraryDir) {
  XrShell shell;
  XrShell::InitResult initResult = shell.init();
  if (initResult == XrShell::InitResult::NoRuntime ||
      initResult == XrShell::InitResult::NoHeadset) {
    std::printf(smoke ? "SMOKE SKIP: no OpenXR runtime/headset\n"
                      : "no OpenXR runtime/headset; install WiVRn and configure the active "
                        "OpenXR runtime, then retry\n");
    return 0;
  }
  if (initResult != XrShell::InitResult::Ready) {
    std::printf(smoke ? "SMOKE FAIL: xr init\n" : "xr init failed\n");
    return 1;
  }

  App app;
  app.width = shell.eyeWidth();
  app.height = shell.eyeHeight();
  app.libraryDir = libraryDir;

  VkRenderer renderer;
  VkRenderer::Config config;
  config.windowed = false;
  config.width = shell.eyeWidth();
  config.height = shell.eyeHeight();
  config.shaderDir = shaderDir;
  if (!renderer.initExternal(config, shell.vulkanInstance(), shell.physicalDevice(),
                             shell.device(), shell.queueFamily(), shell.swapchainFormat())) {
    std::printf(smoke ? "SMOKE FAIL: renderer init\n" : "renderer init failed\n");
    renderer.shutdown(); // release any partial resources before the device dies
    shell.shutdown();
    return 2;
  }
  // Passthrough is the default view: the real world with models drawn over it.
  applyBackdrop(app, renderer, &shell);
  if (!modelPaths.empty()) {
    if (!loadModels(app, renderer, modelPaths)) {
      std::printf(smoke ? "SMOKE FAIL: model loading\n" : "model loading failed\n");
      renderer.shutdown();
      shell.shutdown();
      return 2;
    }
    frameScene(app);
  } else if (smoke) {
    // The demo cubes are a SMOKE fixture ("get rid of those stupid boxes at
    // the start"): interactive VR starts with an empty world — the model
    // gallery (A) is how models get in.
    buildDemoScene(app);
    app.selection.push_back(app.visuals[0].id);
  }

  // The headset drives the eye views; locomotion is a rig transform composed
  // with the head pose (per-eye viewProj = proj*viewHead * rigView). The rig
  // must be a gravity-aligned frame — yaw + position only. An inherited pitch
  // (the desktop framing camera looks down at the model) tilts the entire
  // world off the headset's gravity: the ground stops being level and the
  // tilt direction sweeps as the head turns.
  Camera rig = app.camera;
  rig.pitch = 0.0F;

  bool running = true;
  int framesRendered = 0;
  int skipsInARow = 0;
  double frameMsTotal = 0.0;
  double frameMsMax = 0.0;
  std::chrono::steady_clock::time_point frameStart = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point lastFrameTime = frameStart;
  while (running) {
    XrFrameInput input;
    frameStart = std::chrono::steady_clock::now();
    XrShell::FrameStatus status = shell.beginFrame(input);
    if (status == XrShell::FrameStatus::Exit) {
      break;
    }
    if (status == XrShell::FrameStatus::Pump) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5)); // session still starting
      continue;
    }
    if (status == XrShell::FrameStatus::Skipped) {
      // Frame already begun+ended; go straight to the next one. A persistent
      // skip streak means the session is broken — fail the smoke rather than
      // spin forever (interactive mode stays user-escapable).
      if (smoke && ++skipsInARow >= 600) {
        std::printf("[xr] %d consecutive skipped frames; giving up\n", skipsInARow);
        break;
      }
      continue;
    }
    skipsInARow = 0;

    // Stick rates and the delete ramp need frame timing.
    auto const now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - lastFrameTime).count();
    lastFrameTime = now;
    dt = dt > 0.1F ? 0.1F : dt;

    // No flight controls: passthrough means you walk in the real world with
    // the models placed in it — the sticks' flight/turn bindings were removed
    // at the user's request ("I can now move in the real world"). Grip is the
    // gizmo snap modifier.
    app.snap = input.gripHeld > 0.5F;

    // NO grip chords on face buttons — the grip sensor spikes when squeezing
    // the controller to press one, which made this dispatch silently take the
    // wrong branch (the reported "can only rotate horizontally": the mode
    // cycle was unreachable behind a grip+menu backdrop chord). Tap/hold on
    // ONE button instead: TAP (< 0.4 s) = gizmo mode cycle (on release),
    // HOLD = backdrop toggle (fires once at threshold, the tap is cancelled —
    // the same pattern as hold-X delete).
    if (input.menuPressed || (input.menuHeld && !app.menuPrevHeld)) {
      // The app-side rising edge backs up the shell's edge: a press swallowed
      // by a skipped frame must still initialise the tap/hold state, or a
      // later hold resumes from a stale charge (and can fire the backdrop
      // instead of the mode cycle).
      app.menuCharge = 0.0F;
      app.menuHoldLatched = false;
      app.menuTapped = true;
    }
    app.menuPrevHeld = input.menuHeld;
    if (input.menuHeld) {
      app.menuCharge += dt;
      if (!app.menuHoldLatched && app.menuCharge >= 0.4) {
        app.menuHoldLatched = true;
        app.menuTapped = false;
        toggleBackdrop(app, renderer, &shell);
      }
    } else if (app.menuTapped) {
      app.menuTapped = false;
      app.mode = app.mode == GizmoMode::Translate ? GizmoMode::Rotate
                 : app.mode == GizmoMode::Rotate  ? GizmoMode::Scale
                                                  : GizmoMode::Translate;
      endEdits(app);
    }
    bool cancelledLiveEdit = false;
    if (input.undoPressed) {
      if (app.dragging || app.stickSession || app.galleryActive) {
        cancelLiveEdit(app);
        cancelledLiveEdit = true;
      } else {
        vredit::EditorSnapshot state = snapshot(app);
        if (app.undo.undo(state)) {
          applySnapshot(app, state);
          app.stickArmed = false; // sticks must re-centre before a new gesture
        }
      }
    }
    // Same guard as the desktop keys: one cancel per frame leaves both stacks
    // untouched even if both arrive together.
    if (input.redoPressed && !cancelledLiveEdit) {
      if (app.dragging || app.stickSession || app.galleryActive) {
        cancelLiveEdit(app);
      } else {
        vredit::EditorSnapshot state = snapshot(app);
        if (app.undo.redo(state)) {
          applySnapshot(app, state);
          app.stickArmed = false; // sticks must re-centre before a new gesture
        }
      }
    }

    // Head and pointer in world space (rig-transformed).
    Mat4 rigView = cameraView(rig);
    Mat4 rigWorld{};
    if (!invertAffine(rigView, rigWorld)) {
      shell.endFrame(); // frame is begun: pair the begin before bailing
      running = false;
      break;
    }
    Vec3 headWorld = transformPoint(rigWorld, input.headPosition);
    Vec3 headForward =
        normalized(transformVector(rigWorld, rotate(input.headOrient, {0.0F, 0.0F, -1.0F})));
    Ray worldRay{transformPoint(rigWorld, input.pointerRay.origin),
                 normalized(transformVector(rigWorld, input.pointerRay.direction))};

    // The right controller's aim ray drives the same state machine as the
    // mouse; the head pose sizes the gizmo. A body press only SELECTS —
    // movement and rotation are the sticks' job (controller aim is too
    // jittery for accurate placement).
    app.camera.position = headWorld;
    if (input.selectPressed) {
      if (app.galleryActive) {
        // The gallery owns the trigger: a press on a preview spawns that model
        // at the aim point; a press on empty space does nothing.
        (void)galleryPick(app, renderer, worldRay);
      } else {
        // Grip + trigger selects a single part; trigger alone the whole model.
        bool const partSelect = input.gripHeld > 0.5F;
        PointerTarget target = resolvePointerTarget(app, worldRay, kVrPickThresholdPx);
        pointerPress(app, worldRay, headForward, target, partSelect);
      }
    } else if (input.selectHeld && !app.galleryActive) {
      pointerMove(app, worldRay);
    }
    if (input.selectReleased) {
      pointerRelease(app);
    }
    updateHover(app, worldRay, kVrPickThresholdPx, input.gripHeld > 0.5F);

    // Stick-driven editing (user request: "right stick to move, left stick to
    // rotate"): the right stick slides the selection across the floor
    // relative to the head's yaw (pushing forward walks it away from you),
    // the left stick rotates on all three axes (spin on X, tip/roll on Y —
    // grip selects roll). One undo step per gesture.
    //
    // The gesture's frame is latched at start (pivot, pre-signed tip axis,
    // fine rate): turning your head or toggling grip mid-gesture must not
    // steer the tip plane, change its rate mid-step, or walk asymmetric
    // models sideways through a per-frame re-centred pivot.
    Vec3 headForwardFlat{headForward.x, 0.0F, headForward.z};
    headForwardFlat = dot(headForwardFlat, headForwardFlat) > 1e-8F
                          ? normalized(headForwardFlat)
                          : Vec3{0.0F, 0.0F, -1.0F};
    Vec3 const headRightFlat = cross(headForwardFlat, Vec3{0.0F, 1.0F, 0.0F});
    float const stickThreshold = app.stickSession ? kStickEndZone : kStickDeadZone;
    // Dead-zoned stick VALUES (the gate alone is not enough): sub-threshold
    // drift must never edit — a move gesture used to keep the session open
    // while left-Y drift silently tipped the model at degrees per second.
    auto stickValue = [](float v) {
      float const a = std::fabs(v);
      if (a <= kStickDeadZone) {
        return 0.0F;
      }
      return (v > 0.0F ? 1.0F : -1.0F) * (a - kStickDeadZone) / (1.0F - kStickDeadZone);
    };
    float const rightX = stickValue(input.rightStick.x);
    float const rightY = stickValue(input.rightStick.y);
    float const leftX = stickValue(input.leftStick.x);
    float const leftY = stickValue(input.leftStick.y);
    // The right stick moves (X/Z). The left stick rotates on ALL THREE axes
    // ("I still can't rotate other than horizontally"): X spins the model on
    // the floor, Y tips it — grip free: pitch (stick up tips the top away),
    // grip held: roll (stick up rolls the top to your right).
    bool const sticksActive = std::fabs(input.rightStick.x) > stickThreshold ||
                              std::fabs(input.rightStick.y) > stickThreshold ||
                              std::fabs(input.leftStick.x) > stickThreshold ||
                              std::fabs(input.leftStick.y) > stickThreshold;
    bool const gestureAlive = sticksActive;
    // After an undo/redo the sticks must return to neutral before a new
    // gesture may start, or the same frame's drift cancels-and-restarts and
    // swallows the undo.
    if (!sticksActive) {
      app.stickArmed = true;
    }
    if (sticksActive && !app.stickSession && !app.dragging && !app.galleryActive &&
        !app.selection.empty() && app.stickArmed) {
      app.undo.begin(snapshot(app));
      app.stickSession = true;
      app.stickFineRate = input.gripHeld > 0.5F ? 0.25F : 1.0F;
      // Pre-signed tip axis: stick up tips the top AWAY (negative about the
      // viewer's right) grip-free; grip + up rolls the top to the viewer's
      // RIGHT (positive about the view axis).
      app.stickTipAxis = input.gripHeld > 0.5F
                             ? headForwardFlat
                             : Vec3{-headRightFlat.x, -headRightFlat.y, -headRightFlat.z};
      vredit::Aabb bounds{};
      if (vredit::movingSetBounds(app.scene, app.visuals, app.selection, bounds)) {
        app.stickPivot = vredit::aabbCentre(bounds);
      } else if (!app.selection.empty()) {
        app.stickPivot = app.scene.worldPosition(app.selection.front());
      }
    }
    if (app.stickSession && !app.dragging && !app.galleryActive) {
      if (gestureAlive && !app.selection.empty()) {
        Vec3 const moveDelta = (headRightFlat * rightX + headForwardFlat * rightY) *
                               (kStickMoveMps * app.stickFineRate * dt);
        // Guard the call, not just the maths: a zero delta would still
        // round-trip every local transform through matrix decompose (float
        // noise, epoch churn, junk in the undo snapshot).
        if (moveDelta.x != 0.0F || moveDelta.y != 0.0F || moveDelta.z != 0.0F) {
          vredit::moveSelectionHorizontal(app.scene, app.selection, moveDelta);
        }
        // Stick right = the model turns to its right: a clockwise turn seen
        // from above, which is negative around +Y in right-handed maths.
        float const yawDelta = -leftX * kStickTurnRadps * app.stickFineRate * dt;
        if (yawDelta != 0.0F) {
          vredit::rotateSelectionAbout(app.scene, app.selection, app.stickPivot,
                                       Vec3{0.0F, 1.0F, 0.0F}, yawDelta);
        }
        if (std::fabs(leftY) > 1e-4F) {
          vredit::rotateSelectionAbout(app.scene, app.selection, app.stickPivot, app.stickTipAxis,
                                       leftY * kStickTurnRadps * app.stickFineRate * dt);
          // The floor rule is about MODELS ("models stay stuck to the
          // floor"): tipping a PART must not lift it out of its model by
          // flooring through the part's own bounds.
          if (wholeModelSelection(app)) {
            vredit::dropSelectionToFloor(app.scene, app.visuals, app.selection);
          }
        }
      } else {
        app.undo.commit(snapshot(app));
        app.stickSession = false;
      }
    }

    // Hold-X delete: the selection's tint ramps red over kDeleteChargeS and
    // the delete fires at full charge; release before that cancels. Hold-to-
    // confirm on purpose — face buttons on Quest controllers are wobbly.
    if (input.deleteHeld && !app.selection.empty() && !app.galleryActive) {
      if (!app.deleteHolding) {
        app.deleteHolding = true;
        app.deleteHoldStart = std::chrono::steady_clock::now();
      }
      double const held =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - app.deleteHoldStart)
              .count();
      app.deleteCharge = static_cast<float>(std::min(held / kDeleteChargeS, 1.0));
      if (app.deleteCharge >= 1.0F) {
        deleteSelection(app);
        app.deleteHolding = false;
        app.deleteCharge = 0.0F;
      }
    } else {
      app.deleteHolding = false;
      app.deleteCharge = 0.0F;
    }
    // The model gallery: the spawn key opens it (small previews of every
    // library model in an arc) or closes it. NO grip chord here on purpose —
    // the grip sensor spikes when squeezing the controller to press A, and
    // the old "grip+A cancels" chord silently swallowed EVERY press (the
    // reported "nothing appears when I press A"). Cancelling is undo (B) — a
    // single, stable button the user already has.
    if (input.spawnPressed) {
      if (app.galleryActive) {
        galleryClose(app);
      } else {
        galleryOpen(app, renderer, headWorld, headForward);
      }
    }
    if (app.galleryActive) {
      // The gallery owns the pointer: the aimed preview brightens and the
      // trigger spawns it at the aim point (handled at the trigger press).
      galleryHover(app, worldRay);
    }

    std::vector<DrawItem> items;
    std::vector<RendererVertex> depthLines;
    std::vector<RendererVertex> overlayLines;
    assembleFrame(app, items, depthLines, overlayLines);
    // Controller pointers: one beam per tracked hand, always on top so the
    // aimed point is visible even against the model. Left is cool, right (the
    // manipulation pointer) is warm.
    for (int hand = 0; hand < 2; ++hand) {
      if (!input.aimValid[hand]) {
        continue;
      }
      Ray aimWorld{transformPoint(rigWorld, input.aimRays[hand].origin),
                   normalized(transformVector(rigWorld, input.aimRays[hand].direction))};
      appendRayBeam(overlayLines, aimWorld, 10.0F,
                    hand == 0 ? Vec3{0.45F, 0.75F, 1.0F} : Vec3{1.0F, 0.82F, 0.35F});
    }

    bool drew = true;
    // Compose the rig with the head as ONE basis and horizon-lock the whole:
    // a pitched rig multiplied by a yawed head view produces phantom roll
    // (the world tilts as you turn), and the roll has to be stripped from the
    // composition, not from the parts. One correction serves both eyes so the
    // stereo pair stays rigid relative to each other.
    Quat const headTotal = cameraOrientation(rig) * input.headOrient;
    Quat const horizonLock = gravityAlign(headTotal) * conjugate(normalized(headTotal));
    for (int eye = 0; eye < 2; ++eye) {
      Vec3 const eyeWorld = transformPoint(rigWorld, input.eyePos[eye]);
      Quat const totalOrient = horizonLock * (cameraOrientation(rig) * input.eyeOrient[eye]);
      Mat4 const view = cameraViewFromPose(eyeWorld, totalOrient);
      Mat4 viewProj = mul(input.eyeProj[eye], view);
      if (!renderer.drawToView(shell.eyeView(eye), shell.eyeImage(eye), viewProj, items,
                               depthLines, overlayLines)) {
        std::printf("[xr] eye draw failed\n");
        drew = false;
        break;
      }
    }
    if (!shell.endFrame() || !drew) {
      running = false;
    }
    double frameMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart)
            .count();
    frameMsTotal += frameMs;
    frameMsMax = frameMs > frameMsMax ? frameMs : frameMsMax;
    ++framesRendered;
    if (smoke && framesRendered >= 60) {
      std::printf("SMOKE PASS (xr, %d frames)\n", framesRendered);
      break;
    }
  }

  XrShell::FrameStats const& stats = shell.frameStats();
  if (framesRendered > 0) {
    std::printf("[xr] frame time: avg %.2f ms, max %.2f ms over %d frames\n",
                frameMsTotal / framesRendered, frameMsMax, framesRendered);
    std::printf("[xr] frames begun %llu: zero-layer %llu (viewFail %llu, actionFail %llu, "
                "acquireFail %llu)\n",
                static_cast<unsigned long long>(stats.begun),
                static_cast<unsigned long long>(stats.zeroLayer),
                static_cast<unsigned long long>(stats.viewFail),
                static_cast<unsigned long long>(stats.actionFail),
                static_cast<unsigned long long>(stats.acquireFail));
    std::printf("[xr] rendered anyway: stalePose %llu (reused last good views), "
                "renderFalse %llu (runtime hint ignored)\n",
                static_cast<unsigned long long>(stats.stalePose),
                static_cast<unsigned long long>(stats.renderFalse));
    std::printf("[xr] controller sticks (right = move, left = rotate): "
                "left X %.2f Y %.2f, right X %.2f Y %.2f (peaks)\n",
                stats.leftXMax, stats.leftYMax, stats.rightXMax, stats.rightYMax);
  }

  renderer.shutdown();
  shell.shutdown();
  if (smoke && framesRendered < 60) {
    std::printf("SMOKE FAIL: only %d xr frames rendered\n", framesRendered);
    return 1;
  }
  return 0;
}

// Applies held movement keys once per frame (free flight).
void updateFlight(App& app, float dt) {
  Vec3 local{};
  if (app.flyForward) {
    local.z += 1.0F;
  }
  if (app.flyBack) {
    local.z -= 1.0F;
  }
  if (app.flyRight) {
    local.x += 1.0F;
  }
  if (app.flyLeft) {
    local.x -= 1.0F;
  }
  if (app.flyUp) {
    local.y += 1.0F;
  }
  if (app.flyDown) {
    local.y -= 1.0F;
  }
  if (local.x == 0.0F && local.y == 0.0F && local.z == 0.0F) {
    return;
  }
  float speed = kFlySpeed * (app.flyBoost ? kFlyBoost : 1.0F) * dt;
  flyCamera(app.camera, local * speed);
}

int runInteractive(char const* shaderDir,
                   std::vector<char const*> const& modelPaths,
                   std::string const& libraryDir) {
  App app;
  app.libraryDir = libraryDir;
  // No demo scene in the interactive editor ("get rid of those stupid boxes
  // at the start"): an empty world is the correct start — the model gallery
  // (A/N) is how models get in. The smoke modes keep the cubes as their
  // pixel-assertion fixture.
  XcbWindow window;
  if (!openWindow(window, app.width, app.height, "vr-editor")) {
    return 1;
  }
  VkRenderer renderer;
  VkRenderer::Config config;
  config.windowed = true;
  config.xcbConnection = window.connection;
  config.xcbWindow = window.window;
  config.width = app.width;
  config.height = app.height;
  config.shaderDir = shaderDir;
  if (!renderer.init(config)) {
    renderer.shutdown(); // surface outlives the XCB connection
    closeWindow(window);
    return 1;
  }
  // Passthrough is the default view in VR; on desktop that same state is a
  // plain void (menu hold in VR, B on desktop, toggles the grid backdrop back).
  applyBackdrop(app, renderer, nullptr);
  if (!modelPaths.empty()) {
    if (!loadModels(app, renderer, modelPaths)) {
      renderer.shutdown();
      closeWindow(window);
      return 1;
    }
    frameScene(app);
  }
  std::printf(
      "vr-editor on %s\n"
      "  flight: WASD move   Q/E down/up   Shift boost   RMB look   MMB or Shift+RMB pan\n"
      "  wheel: fly forward/back   LMB: select / drag gizmo\n"
      "  1 translate   2 rotate   3 scale   L world/local space   Ctrl snap   Esc quit\n",
      renderer.deviceName());

  auto previous = std::chrono::steady_clock::now();
  while (app.running) {
    xcb_generic_event_t* event = nullptr;
    while ((event = xcb_poll_for_event(window.connection)) != nullptr) {
      handleEvent(app, window, event, renderer);
      std::free(event);
    }
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - previous).count();
    previous = now;
    if (dt > 0.1F) {
      dt = 0.1F; // tab stalls must not teleport the camera
    }
    updateFlight(app, dt);
    // The gallery's preview under the mouse brightens (the same "you are
    // pointing at THIS one" feedback as the controller aim in VR).
    if (app.galleryActive) {
      galleryHover(app, appRay(app, app.lastMouse));
    }
    if (app.width == 0U || app.height == 0U) {
      std::this_thread::sleep_for(std::chrono::milliseconds(8)); // minimised: no busy spin
      continue;
    }
    updateHover(app, appRay(app, app.lastMouse), kPickThresholdPx, app.partSelect);
    std::vector<DrawItem> items;
    std::vector<RendererVertex> depthLines;
    std::vector<RendererVertex> overlayLines;
    assembleFrame(app, items, depthLines, overlayLines);
    if (!renderer.draw(app.camera, items, depthLines, overlayLines)) {
      std::printf("draw failed; exiting\n");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(8)); // pace to ~120 Hz
  }

  renderer.shutdown();
  closeWindow(window);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  enum class Mode {
    Interactive,
    SmokeOffscreen,
    SmokeTextured,
    SmokeWindow,
    SmokePerf,
    Xr,
    SmokeXr
  };
  Mode mode = Mode::Interactive;
  char const* shaderDir = ".build/shaders";
  int frames = 30;
  std::vector<char const*> modelPaths;
  std::string libraryDir; // spawn library: --library DIR (default below)

  for (int i = 1; i < argc; ++i) {
    char const* arg = argv[i];
    if (std::strcmp(arg, "--shaders") == 0 && i + 1 < argc) {
      shaderDir = argv[++i];
    } else if (std::strcmp(arg, "--library") == 0 && i + 1 < argc) {
      libraryDir = argv[++i];
    } else if (std::strcmp(arg, "--smoke-offscreen") == 0) {
      mode = Mode::SmokeOffscreen;
    } else if (std::strcmp(arg, "--smoke-textured") == 0) {
      mode = Mode::SmokeTextured;
    } else if (std::strcmp(arg, "--smoke-perf") == 0) {
      mode = Mode::SmokePerf;
    } else if (std::strcmp(arg, "--smoke-window") == 0) {
      mode = Mode::SmokeWindow;
      if (i + 1 < argc) {
        // Only consume the frame count when the next argument is all digits, so
        // model files are never mistaken for it.
        char const* next = argv[i + 1];
        std::size_t length = std::strlen(next);
        bool allDigits = length > 0U;
        for (std::size_t c = 0; c < length; ++c) {
          if (next[c] < '0' || next[c] > '9') {
            allDigits = false;
            break;
          }
        }
        if (allDigits) {
          frames = std::atoi(next);
          ++i;
        }
      }
    } else if (std::strcmp(arg, "--xr") == 0) {
      mode = Mode::Xr;
    } else if (std::strcmp(arg, "--smoke-xr") == 0) {
      mode = Mode::SmokeXr;
    } else if (std::strcmp(arg, "--help") == 0) {
      std::printf(
          "usage: vr-editor [model.glb|model.obj ...] [--shaders DIR] [--library DIR]\n"
          "                [--xr | --smoke-offscreen | --smoke-textured |\n"
          "                 --smoke-perf | --smoke-xr | --smoke-window [FRAMES]]\n"
          "keys: WASD/QE fly (RMB look), 1/2/3 move/rotate/scale, L space,\n"
          "      N model gallery (click preview spawns; N closes), Delete remove,\n"
          "      B backdrop, Ctrl-Z/Y undo/redo, shift+click part, Esc quit\n"
          "vr:   trigger select (grip+trigger part), right stick move on the floor,\n"
          "      left stick rotate (X spin, Y tip; grip+Y roll), grip = fine,\n"
          "      A model gallery (trigger on preview spawns),\n"
          "      hold X delete, B/Y undo/redo, menu tap = mode, menu hold = backdrop\n");
      return 0;
    } else if (arg[0] == '-') {
      std::printf("unknown option: %s\n", arg);
      return 1;
    } else {
      modelPaths.push_back(arg);
    }
  }

  // Spawn library default: ~/Downloads/VR when it exists (the model
  // collection), else the working directory. Rescanned on every spawn.
  if (libraryDir.empty()) {
    char const* home = std::getenv("HOME");
    std::string const candidate =
        home != nullptr ? std::string(home) + "/Downloads/VR" : std::string();
    std::error_code ec;
    libraryDir = (!candidate.empty() && std::filesystem::is_directory(candidate, ec)) ? candidate
                                                                                     : ".";
  }

  switch (mode) {
  case Mode::SmokeOffscreen:
    return runSmokeOffscreen(shaderDir, modelPaths, false);
  case Mode::SmokeTextured: {
    // Fixed fixtures written by the test binary (checkerboard GLB + OBJ).
    std::vector<char const*> fixtures{".build/test-textured.glb", ".build/test-texture.obj"};
    return runSmokeOffscreen(shaderDir, fixtures, true);
  }
  case Mode::Xr:
    return runXr(shaderDir, modelPaths, false, libraryDir);
  case Mode::SmokeXr:
    return runXr(shaderDir, modelPaths, true, libraryDir);
  case Mode::SmokeWindow:
    return runSmokeWindow(shaderDir, frames, modelPaths);
  case Mode::SmokePerf:
    return runPerfSmoke(shaderDir, modelPaths);
  default:
    return runInteractive(shaderDir, modelPaths, libraryDir);
  }
}
