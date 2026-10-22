// vr-editor: desktop shell over the scene-editor core — viewport, camera,
// selection, stick/arrow editing. Typed C++ only; the shell is a thin layer
// over vredit::Scene and VkRenderer.

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
#include <map>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <array>

#include <xcb/xcb.h>

#include "xr_shell.h"

#include "camera.h"
#include "carousel.h"
#include "editor_state.h"
#include "gizmo.h"
#include "gltf.h"
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
constexpr float kPickThresholdPx = 8.0F;
constexpr float kVrPickThresholdPx = 24.0F; // aim rays jitter: 3x the mouse
// Stick-driven editing rates (VR): the right stick slides the selection, the
// left stick turns it. Grip held runs both at quarter speed (fine control).
constexpr float kStickDeadZone = 0.12F; // a gesture STARTS past this deflection
constexpr float kStickEndZone = 0.06F;  // ...and ENDS below this (hysteresis, so one
                                        // physical gesture is one undo step)
constexpr float kStickMoveMps = 1.5F;    // metres per second at full deflection
constexpr float kStickTurnRadps = 1.75F; // ~100 degrees per second at full
constexpr float kStickHeightMps = 1.0F;  // move mode: metres/second up/down
constexpr float kStickScalePerS = 1.0F;  // move mode: e-fold growth per second
// Hold-X delete: the delete fires when the hold reaches full charge.
constexpr double kDeleteChargeS = 0.6;

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

// Carousel load state (see the carousel section below for the worker loop).
struct AsyncLoad {
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  bool quit{false};
  bool busy{false}; // a request is outstanding (main thread only)
  std::uint64_t reqGen{0};
  std::string reqPath;
  bool reqPending{false};
  std::uint64_t doneGen{0};
  Model doneModel;
  std::string doneError;
  bool doneReady{false};

  // Self-stopping: a joinable std::thread at destruction is std::terminate,
  // and every exit path must leave the worker clean.
  ~AsyncLoad() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      quit = true;
    }
    cv.notify_all();
    if (thread.joinable()) {
      thread.join();
    }
  }
};

// A carousel ring slot: one library model staged under a WRAPPER node whose
// transform is the ring pose (set absolutely every frame — no delta drift).
struct CarouselSlot {
  int index{-1}; // library index, or -1 when empty
  std::uint64_t gen{0}; // load generation (stale results are dropped)
  bool hasModel{false};
  bool failed{false};       // terminal: this file will not load (never retried)
  Model model;
  vredit::NodeId wrapper{vredit::kInvalidNode};
  float fit{1.0F};
  Vec3 centre{0.0F, 0.0F, 0.0F}; // model bounds centre before fitting
};

constexpr std::size_t kSlotCount = 5; // centre ± 2
constexpr float kPreviewExtent = 0.6F;

// ---- GPU mesh cache (content-keyed) ----
// The renderer has no removal API, so re-staging the same model (preview and
// placement) would register its buffers twice. Every registration goes
// through here instead: identical mesh content maps to one GPU mesh for the
// process lifetime (bounded by the library's unique models, like the texture
// registry's content dedupe).
inline std::uint64_t hashBytes(std::uint64_t hash, void const* data, std::size_t size) {
  auto const* p = static_cast<unsigned char const*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ULL; // FNV-1a
  }
  return hash;
}

inline std::uint64_t meshContentHash(MeshData const& data) {
  std::uint64_t h = 1469598103934665603ULL;
  h = hashBytes(h, data.vertices.data(), data.vertices.size() * sizeof(MeshVertex));
  h = hashBytes(h, data.indices.data(), data.indices.size() * sizeof(std::uint32_t));
  return h;
}

struct App;
inline MeshHandle cachedMesh(App& app, VkRenderer& renderer, MeshData const& data);

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
  // Editing mode (menu tap toggles; desktop 1/2). move = left stick X
  // scales, Y = height; rotate = X spins, Y tips (grip: roll). The right
  // stick is planar floor-move in BOTH modes; nothing rotates in move.
  EditMode mode{EditMode::move};
  bool looking{false};
  bool panning{false};
  bool running{true};
  // Model library for spawn-on-the-fly (N on desktop, A in VR): the directory
  // is rescanned on every spawn, so a file dropped in mid-session is spawnable
  // without a restart.
  std::string libraryDir;
  // The model carousel: a RING of library models around the viewer (see
  // carousel.h) — the next/previous models are visible at the sides and
  // switching rotates the ring instantly; decoding happens on a worker
  // thread so a huge model never blocks the switch. The centred model is
  // placed at the aim point (one undo step).
  bool galleryActive{false};
  std::vector<std::string> galleryFiles; // the library (rescanned on open)
  vredit::CarouselRing ring;
  std::array<CarouselSlot, 5> carousel{}; // centre ± 2 ring slots
  std::uint64_t carouselGen{1};           // load generation (stale = dropped)
  AsyncLoad worker{};                     // decode thread (started with the app)
  std::map<std::uint64_t, MeshHandle> meshCache; // content hash -> GPU mesh
  // The ring is STATIONARY in the world: latched at open (eye position and
  // level heading) so turning your head does not spin the carousel around
  // you. "The carousel turns with the users head which is disorientating.
  // It should be stationary."
  bool ringLatched{false};
  Vec3 ringOrigin{0.0F, 0.0F, 0.0F};
  Vec3 ringForward{0.0F, 0.0F, -1.0F};
  vredit::EditorSnapshot galleryBefore{}; // undo "before" for the placement
  double galleryCycleCooldown{0.0}; // stick-held cycling: one step per 0.25 s
  // Stick-driven editing (VR): one undo step per stick gesture — begun on the
  // first deflection past the dead zone, committed when the sticks settle.
  bool stickSession{false};
  // The stick gesture's frame, latched at gesture start: the pivot (the
  // moving set's bounds centre), the pre-signed tip axis and the fine rate.
  // Latching keeps one gesture one coherent step — a live head pose or grip
  // toggle mid-gesture must not steer the tip, change its rate, or walk
  // asymmetric models sideways via a re-centred pivot.
  Vec3 stickPivot{};
  Vec3 stickYawAxis{0.0F, 1.0F, 0.0F};
  Vec3 stickTipAxis{0.0F, 0.0F, -1.0F};
  Vec3 stickRollAxis{0.0F, 0.0F, -1.0F};
  float stickFineRate{1.0F};
  // Latched with the frame: whether this gesture's Y roll (grip/Shift held
  // at start). Read live it would flip tip->roll mid-gesture and interleave
  // two axes in one undo step.
  bool stickRollMode{false};
  // Re-arm latch: after an undo/redo the sticks must return to neutral
  // before a new gesture may start, or the same frame's drift would
  // cancel-and-restart and swallow the undo.
  bool stickArmed{true};
  // Menu tap/hold (VR): tap cycles the stick mode (fires on release), hold
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
  // Arrow keys mirror the sticks on desktop (the coloured handles are gone):
  // left/right spins, up/down drives the mode's Y (height / tip / scale).
  bool arrowUp{false};
  bool arrowDown{false};
  bool arrowLeft{false};
  bool arrowRight{false};
  // Frames with no arrow deflected. The session commits only after 2 quiet
  // frames: X11 auto-repeat delivers a Release+Press pair per repeat, and
  // when the pair straddles a frame drain an immediate commit would split
  // one physical hold into many undo steps.
  int arrowQuietFrames{0};
  bool flyDown{false};
  bool flyBoost{false};
  std::uint8_t panButton{0};
  Vec2 lastMouse{};
  std::uint32_t width{1280};
  std::uint32_t height{800};
};

inline MeshHandle cachedMesh(App& app, VkRenderer& renderer, MeshData const& data) {
  std::uint64_t const key = meshContentHash(data);
  auto const it = app.meshCache.find(key);
  if (it != app.meshCache.end()) {
    return it->second;
  }
  MeshHandle const handle = renderer.addMesh(data);
  if (handle != kInvalidMesh) {
    app.meshCache.emplace(key, handle);
  }
  return handle;
}

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
                         std::vector<NodeId>* rootsOut = nullptr,
                         bool withTextures = true) {
  std::vector<MeshHandle> meshHandles(model.meshes.size(), 0U);
  std::vector<std::shared_ptr<const vredit::PickGeometry>> pickGeometries(model.meshes.size());
  for (std::size_t m = 0; m < model.meshes.size(); ++m) {
    meshHandles[m] = cachedMesh(app, renderer, model.meshes[m].data);
    if (meshHandles[m] == kInvalidMesh) {
      std::printf("mesh registration failed for mesh %zu\n", m);
      return false;
    }
    pickGeometries[m] = makePickGeometry(model.meshes[m].data);
  }
  std::vector<TextureHandle> textureHandles(model.textures.size(), 0U);
  for (std::size_t t = 0; withTextures && t < model.textures.size(); ++t) {
    Image const& image = model.textures[t].image;
    if (image.width == 0 || image.height == 0) {
      // An empty image (a skipped or failed decode) means UNTEXTURED —
      // handle 0 binds nothing. It is not a failure.
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
  std::vector<NodeId> createdRoots; // top-level nodes created so far (rollback)
  try {
  for (std::size_t n = 0; n < model.nodes.size(); ++n) {
    ModelNode const& node = model.nodes[n];
    NodeId parent = node.parent >= 0 ? handles[static_cast<std::size_t>(node.parent)] : kInvalidNode;
    NodeId id = app.scene.create(parent);
    app.scene.setLocal(id, node.local);
    handles[n] = id;
    if (rootsOut != nullptr && node.parent < 0) {
      rootsOut->push_back(id); // top-level nodes: the spawn placement roots
    }
    if (node.parent < 0) {
      createdRoots.push_back(id);
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
        // An empty normal image (a skipped decode) means FLAT: binding
        // the white fallback as a normal map would distort the lighting.
        visual.normalTexture = textureHandles[static_cast<std::size_t>(mesh.normalTexture)];
        visual.normalMapped = true;
        visual.normalScale = mesh.normalScale;
      }
      app.visuals.push_back(visual);
    }
  }
  } catch (...) {
    // A hostile model can throw MID-insert: the nodes created so far must
    // not linger as floating geometry (and must not pollute undo snapshots).
    vredit::deleteSubtrees(app.scene, app.visuals, app.modelGroups, app.selection,
                           createdRoots);
    if (rootsOut != nullptr) {
      rootsOut->clear();
    }
    std::printf("model insertion failed mid-way — rolled back\n");
    return false;
  }
  return true;
}

bool addModelToScene(App& app,
                     VkRenderer& renderer,
                     Model const& model,
                     std::vector<NodeId>* rootsOut = nullptr,
                     bool withTextures = true) {
  try {
    return addModelToSceneImpl(app, renderer, model, rootsOut, withTextures);
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

// Body pick for a pointer ray: exact surface only (closest surface still wins
// through gaps). The bounding box is not a click target — a ray that misses
// the triangles hits nothing, so empty space never selects a model. Returns
// the hit with its along-ray distance.
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

// What a press on this ray would act on: the model/part under the pointer
// (exact surface pick). The handles are gone; a press only ever SELECTS.
struct PointerTarget {
  NodeId node{kInvalidNode}; // the picked model/part (exact surface)
};

PointerTarget resolvePointerTarget(App& app, Ray const& ray) {
  PointerTarget target;
  // The press's own pick (exact surface first): the bounding box is not a
  // click target — pressing empty space selects nothing.
  auto body = pickNode(app, ray);
  if (body.has_value()) {
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

// Per-frame hover was handle feedback; the handles are gone and model hover
// tints were explicitly rejected, so there is nothing to update.

// Full editor state for undo/redo: structure + visuals + groups + selection.
vredit::EditorSnapshot snapshot(App const& app) {
  return vredit::captureState(app.scene, app.visuals, app.modelGroups, app.selection);
}

void applySnapshot(App& app, vredit::EditorSnapshot const& state) {
  vredit::restoreState(state, app.scene, app.visuals, app.modelGroups, app.selection);
}

// Defined below; pointerPress commits any live edit before starting anew.
bool endEdits(App& app);
void galleryClose(App& app);
bool galleryPick(App& app, VkRenderer& renderer, Ray const& aimRay);

// Pointer interaction shared by the mouse and the VR controller ray: a
// press only SELECTS ("I never want to see them again" — the coloured axis
// handles are gone): movement, rotation, height and scale are all the
// thumbsticks' job in VR (controller aim is too jittery for accurate
// placement) and the arrow keys' on desktop.
// partSelect (shift/ctrl click, grip+trigger in VR) limits the selection to
// the single picked part instead of the whole model.
void pointerPress(App& app, PointerTarget const& target, bool partSelect) {
  // While the model gallery is open it owns the interaction: previews are
  // transient UI and clicking elsewhere must not touch the world.
  if (app.galleryActive) {
    return;
  }
  // A press while a live stick gesture is running means its release was
  // never seen: commit it as one undo step before starting over.
  endEdits(app);
  if (target.node == kInvalidNode) {
    app.selection.clear();
    return;
  }
  app.selection = vredit::expandSelection(app.modelGroups, target.node, partSelect);
}

// Cancels a live stick gesture without touching the history: restores its
// pre-edit state and drops the pending undo entry, so both the undo and redo
// stacks stay exactly as they were. Any undo/redo keypress during a live
// edit does this and nothing else — going through commit() instead would
// clear an existing redo branch and make the cancelled edit itself redoable.
void cancelLiveEdit(App& app) {
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

// Commits a live stick gesture as its own undo step. Used where a new
// action begins (spawn, delete, mode switches). Also disarms the input: a
// still-held key/stick must return to neutral before a new gesture starts,
// or the next frame would split one physical hold into two undo steps.
bool endEdits(App& app) {
  bool const wasActive = app.stickSession;
  app.stickSession = false;
  app.stickArmed = false;
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

// The model carousel: a RING of library models around the viewer ("I
// expected that the models would rotate in/out of view with a preview of the
// next/previous model visible"). The centred slot is dead ahead at eye
// height; the neighbours sit visible at the sides as GEOMETRY-ONLY previews
// ("could be lower poly"). Switching is instant — the ring animates on the
// frame of the step while the model behind it decodes on a worker thread
// (switching must never block on a 3 GB decode). Previews stay geometry-only
// for the whole time the carousel is open; textures load exactly once, when
// the trigger (click) places the centred model at NATURAL size on the floor
// at the aim point as one undo step (undo removes it entirely — the
// pre-open state is the "before"); the spawn key or undo (B) closes without
// placing.

// ---- asynchronous model loading (decode OFF the render thread) ----
// One request at a time (the nearest not-yet-loaded ring slot wants a file);
// newer requests supersede older ones by generation, and stale results are
// dropped on arrival. Decode (the slow part) runs on the worker; every
// Vulkan operation stays on the main thread. (AsyncLoad is declared above
// App; the worker functions live here.)

void asyncStart(AsyncLoad& a) {
  a.thread = std::thread([&a] {
    for (;;) {
      std::uint64_t gen = 0;
      std::string path;
      {
        std::unique_lock<std::mutex> lock(a.mutex);
        a.cv.wait(lock, [&a] { return a.quit || a.reqPending; });
        if (a.quit) {
          return; // drop any pending request: decoding it at shutdown is
                  // pure wasted work (and can block exit for seconds)
        }
        a.reqPending = false;
        gen = a.reqGen;
        path = a.reqPath;
      }
      Model model;
      std::string error;
      // GEOMETRY ONLY: previews are staged untextured and the model is
      // dropped after staging, so decoding images here was pure RAM/CPU
      // waste (the exact transient spike the release-after-staging fix is
      // about). Placement re-decodes the one chosen file WITH images.
      if (!loadModelFile(path, model, error, /*withImages=*/false)) {
        model = Model{}; // never stage a partial decode from an exception
        if (error.empty()) {
          error = "could not load";
        }
      }
      {
        std::lock_guard<std::mutex> lock(a.mutex);
        if (gen >= a.doneGen) {
          a.doneGen = gen;
          a.doneModel = std::move(model);
          a.doneError = error;
          a.doneReady = true;
        }
      }
    }
  });
}

void asyncRequest(AsyncLoad& a, std::uint64_t gen, std::string const& path) {
  {
    std::lock_guard<std::mutex> lock(a.mutex);
    a.reqGen = gen;
    a.reqPath = path;
    a.reqPending = true;
  }
  a.busy = true;
  a.cv.notify_all();
}

// ---- ring slots ----
// A slot holds one library model staged under a WRAPPER node (CarouselSlot
// is declared above App). `slotAttach` stages a model, `slotPose` writes the
// absolute ring pose every frame. The preview is scaled to fit 0.6 m about
// its bounds centre; placement re-adds the model at natural size, so the
// wrapper's scale never leaks into the placed instance.

void slotEvict(App& app, CarouselSlot& slot) {
  if (slot.wrapper != vredit::kInvalidNode) {
    std::vector<vredit::NodeId> const roots{slot.wrapper};
    vredit::deleteSubtrees(app.scene, app.visuals, app.modelGroups, app.selection, roots);
    slot.wrapper = vredit::kInvalidNode;
  }
  slot = CarouselSlot{};
}

// Stages `model` in `slot` under a fresh wrapper. Geometry always; textures
// only when asked (neighbours stay geometry-only — the ring shows them at
// the sides as cheap previews). The fit scale and bounds centre come from
// the WHOLE model (meshless wrapper roots make root-only bounds degenerate).
bool slotAttach(App& app, VkRenderer& renderer, CarouselSlot& slot, Model const& model,
                bool withTextures) {
  slot.wrapper = app.scene.create();
  std::vector<vredit::NodeId> roots;
  if (slot.wrapper == vredit::kInvalidNode) {
    std::printf("[gallery] slotAttach: wrapper create failed\n");
    return false;
  }
  if (!addModelToScene(app, renderer, model, &roots, withTextures)) {
    std::printf("[gallery] slotAttach: addModelToScene failed (textures=%d)\n",
                withTextures ? 1 : 0);
    app.scene.destroy(slot.wrapper);
    slot.wrapper = vredit::kInvalidNode;
    return false;
  }
  if (roots.empty()) {
    std::printf("[gallery] slotAttach: no roots (textures=%d)\n", withTextures ? 1 : 0);
    app.scene.destroy(slot.wrapper);
    slot.wrapper = vredit::kInvalidNode;
    return false;
  }
  for (vredit::NodeId const root : roots) {
    app.scene.setParent(root, slot.wrapper);
  }
  std::vector<vredit::NodeId> allNodes;
  for (vredit::NodeId const root : roots) {
    std::vector<vredit::NodeId> nodes = vredit::expandSelection(app.modelGroups, root, false);
    allNodes.insert(allNodes.end(), nodes.begin(), nodes.end());
  }
  vredit::Aabb bounds{};
  if (vredit::nodesBounds(app.scene, app.visuals, allNodes, bounds)) {
    Vec3 const size = bounds.max - bounds.min;
    float const extent = std::fmax(size.x, std::fmax(size.y, size.z));
    if (extent > 1e-4F) {
      slot.fit = kPreviewExtent / extent;
    }
    slot.centre = (bounds.min + bounds.max) * 0.5F;
  }
  slot.hasModel = true;
  return true;
}

// The absolute ring pose for a slot: T(slotPos) * R(yaw) * S(fit) * T(-centre),
// i.e. the model's bounds centre lands at slotPos at fit scale, facing the
// viewer. Composed exactly (translation = slotPos - R*(centre*fit)).
void slotPose(App& app, CarouselSlot& slot, vredit::CarouselRing const& ring, Vec3 const& eye,
              Vec3 const& forward) {
  if (slot.wrapper == vredit::kInvalidNode) {
    return;
  }
  vredit::CarouselRing::SlotPose const pose = ring.slotPose(slot.index, eye, forward);
  Quat const yaw = quatFromAxisAngle(Vec3{0.0F, 1.0F, 0.0F}, pose.yaw);
  Vec3 const rotatedCentre = rotate(yaw, slot.centre * slot.fit);
  vredit::Transform t;
  t.translation = pose.position - rotatedCentre;
  t.rotation = yaw;
  t.scale = Vec3{slot.fit, slot.fit, slot.fit};
  app.scene.setLocal(slot.wrapper, t);
}

void galleryClose(App& app);

// Removes every staged preview (never from history: previews are transient
// UI and leave before any undo step is taken) and cancels any in-flight load.
void galleryEvict(App& app) {
  for (std::size_t i = 0; i < kSlotCount; ++i) {
    slotEvict(app, app.carousel[i]);
  }
  // Bump the request generation: a load in flight is now stale and its
  // result is dropped on arrival. `busy` is deliberately NOT cleared here —
  // a decode genuinely in progress must finish before the next request (its
  // result clears busy when the drain consumes it).
  app.carouselGen += 2;
}

void galleryClose(App& app) {
  if (!app.galleryActive) {
    return;
  }
  galleryEvict(app);
  app.galleryActive = false;
  std::printf("[gallery] closed\n");
}

// The ring offset -> library index, wrapped.
int galleryWrapIndex(App const& app, int offset) {
  int const count = static_cast<int>(app.galleryFiles.size());
  if (count <= 0) {
    return -1;
  }
  return ((app.ring.index + offset) % count + count) % count;
}

// Asks the worker for the CLOSEST ring slot with no model yet (centre first,
// then the visible neighbours). One decode runs at a time; a newer request
// supersedes the old (stale results are dropped by generation), so cycling
// quickly just re-targets the worker.
void galleryRequestMissing(App& app) {
  if (!app.galleryActive || app.galleryFiles.empty()) {
    return;
  }
  if (app.worker.busy) {
    return; // strictly one decode at a time — overwriting the pending job
            // would strand its slot "in flight" forever
  }
  for (int offset : {0, 1, -1, 2, -2}) {
    int const index = galleryWrapIndex(app, offset);
    bool have = false;
    for (CarouselSlot const& slot : app.carousel) {
      if (slot.index == index && (slot.hasModel || slot.failed)) {
        have = true; // loaded, or terminally failed ("never retried")
        break;
      }
    }
    if (have) {
      continue;
    }
    // Nothing is decoding, so a slot claiming this index with no model is
    // STALE (its job was dropped) — clear it and start fresh. Slots that
    // hold a decoded model are never stale and must not be destroyed.
    CarouselSlot* target = nullptr;
    for (CarouselSlot& slot : app.carousel) {
      if (slot.index == index && !slot.hasModel && !slot.failed &&
          slot.model.nodes.empty()) {
        slotEvict(app, slot);
        break;
      }
    }
    for (CarouselSlot& slot : app.carousel) {
      if (!slot.hasModel && slot.wrapper == vredit::kInvalidNode && slot.index < 0) {
        target = &slot;
        break;
      }
    }
    if (target == nullptr) {
      return; // all slots busy
    }
    target->index = index;
    target->gen = ++app.carouselGen;
    asyncRequest(app.worker, target->gen, app.galleryFiles[static_cast<std::size_t>(index)]);
    return;
  }
}

// The per-frame ring driver: drains the worker (attaches finished loads as
// geometry-only previews), requests what is missing, and writes every slot's
// absolute ring pose in the LATCHED world frame (the ring is stationary —
// it does not follow the viewer). `dt` eases the ring.
void galleryTick(App& app, VkRenderer& renderer, float dt, Vec3 const& eye,
                 Vec3 const& forward) {
  if (!app.galleryActive) {
    return;
  }
  app.ring.tick(dt);
  // The ring is world-stationary: latch the viewer's eye and level heading
  // on the first tick after open; every pose uses the latched frame.
  if (!app.ringLatched) {
    app.ringLatched = true;
    app.ringOrigin = eye;
    Vec3 f = forward;
    if (f.x == 0.0F && f.z == 0.0F) {
      f = {0.0F, 0.0F, -1.0F};
    }
    app.ringForward = normalized(Vec3{f.x, 0.0F, f.z});
    std::printf("[gallery] ring latched (stationary)\n");
  }
  // Evict slots that fell outside the visible window (centre ± 2), using the
  // MINIMAL wrapped distance — a slot 15 is adjacent to 0 in a 16-library.
  int const librarySize = static_cast<int>(app.galleryFiles.size());
  for (CarouselSlot& slot : app.carousel) {
    if (slot.index < 0 || librarySize <= 0) {
      continue;
    }
    int d = ((slot.index - app.ring.index) % librarySize + librarySize) % librarySize;
    if (d > librarySize / 2) {
      d -= librarySize;
    }
    if (std::abs(d) > 2) {
      slotEvict(app, slot);
    }
  }
  // Drain finished loads (decode done on the worker; GPU work here).
  {
    bool ready = false;
    std::uint64_t gen = 0;
    Model model;
    std::string error;
    {
      std::lock_guard<std::mutex> lock(app.worker.mutex);
      if (app.worker.doneReady) {
        ready = true;
        gen = app.worker.doneGen;
        model = std::move(app.worker.doneModel);
        error = app.worker.doneError;
        app.worker.doneReady = false;
        app.worker.busy = false; // the job is consumed; the next may start
      }
    }
    if (ready) {
      CarouselSlot* target = nullptr;
      for (CarouselSlot& slot : app.carousel) {
        if (slot.gen == gen && !slot.hasModel) {
          target = &slot;
          break;
        }
      }
      if (target != nullptr) {
        // Stage the geometry-only preview, then RELEASE the decoded model:
        // holding five decoded models (with their image buffers) is what ran
        // the process out of memory. Placement re-decodes the one chosen
        // file with images — exactly one decode at a time.
        target->model = std::move(model);
        if (slotAttach(app, renderer, *target, target->model, /*withTextures=*/false)) {
          std::printf("[gallery] staged %s\n",
                      app.galleryFiles[static_cast<std::size_t>(target->index)].c_str());
          target->model = Model{}; // RAM released — previews stay low-poly
        } else {
          std::printf("[gallery] cannot show %s: %s\n",
                      app.galleryFiles[static_cast<std::size_t>(target->index)].c_str(),
                      error.empty() ? "could not add to scene" : error.c_str());
          // TERMINAL: mark the slot failed instead of evicting it — evicting
          // would make requestMissing re-decode this file in an endless loop.
          target->failed = true;
          target->model = Model{};
          if (target->wrapper != vredit::kInvalidNode) {
            app.scene.destroy(target->wrapper);
            target->wrapper = vredit::kInvalidNode;
          }
        }
      }
    }
  }
  // Previews stay GEOMETRY-ONLY for the whole time the carousel is open
  // ("Leave the models as low poly in the carousel, I don't need the
  // high-def version"): no textured promotion and no per-switch hitch.
  // Textures load exactly once, when a model is PLACED — galleryPick
  // re-decodes the one chosen file with images.
  galleryRequestMissing(app);
  // Absolute ring poses in the LATCHED world frame (stationary: the ring
  // does not follow the viewer's head or feet).
  for (CarouselSlot& slot : app.carousel) {
    slotPose(app, slot, app.ring, app.ringOrigin, app.ringForward);
  }
}

// Opens the carousel over the whole library (rescanned every open: drop a
// file in and it is there). The centre slot decodes first; the neighbours
// fill in behind it.
void galleryOpen(App& app, VkRenderer& renderer) {
  (void)renderer; // loads attach in galleryTick (GPU work stays on the frame)
  if (app.galleryActive) {
    return;
  }
  // A live stick gesture must not keep mutating the world behind the
  // carousel (and its pending edit would tangle with the placement's): close
  // it as its own undo step first.
  endEdits(app);
  app.galleryFiles = scanLibrary(app.libraryDir);
  if (app.galleryFiles.empty()) {
    std::printf("[gallery] library folder is empty: %s\n", app.libraryDir.c_str());
    return;
  }
  app.ring = vredit::CarouselRing{};
  app.ring.setCount(static_cast<int>(app.galleryFiles.size()));
  app.ringLatched = false; // latched on the first tick (stationary ring)
  // The pre-open state is the undo "before" for the eventual placement:
  // undoing a placement restores the world WITHOUT the model (never a
  // floating preview). Nothing else mutates while the carousel is open.
  app.galleryBefore = snapshot(app);
  app.galleryActive = true;
  if (!app.worker.thread.joinable()) {
    asyncStart(app.worker); // decode thread, lazily started (no-op after)
  }
  galleryRequestMissing(app);
  std::printf("[gallery] %zu models — stick X or [ ] rotates the ring, "
              "the trigger (click) places\n",
              app.galleryFiles.size());
}

// Rotates the ring (stick X in VR, [ ] on desktop). Instant: the ring
// animates this frame; the model behind the slot decodes on the worker.
void galleryCycle(App& app, VkRenderer& renderer, int dir) {
  (void)renderer;
  if (!app.galleryActive || app.galleryFiles.empty()) {
    return;
  }
  app.ring.step(dir);
  galleryRequestMissing(app);
}

// The trigger (click) while the carousel is open: place the CENTRED model on
// the floor at the aim point at NATURAL size as one undo step. The undo
// "before" is the PRE-OPEN state, so undoing removes the model entirely — it
// never returns as a floating preview. The placement closes the carousel.
bool galleryPick(App& app, VkRenderer& renderer, Ray const& aimRay) {
  if (!app.galleryActive || app.galleryFiles.empty()) {
    return false;
  }
  int const index = app.ring.index;
  CarouselSlot* centre = nullptr;
  for (CarouselSlot& slot : app.carousel) {
    if (slot.index == index && slot.hasModel) {
      centre = &slot;
      break;
    }
  }
  if (centre == nullptr) {
    std::printf("[gallery] still loading — try again in a moment\n");
    return false;
  }
  // Belt and braces: no live edit should exist while the carousel is open,
  // but if one does, commit it as its own step before the placement's.
  endEdits(app);
  std::string const file = app.galleryFiles[static_cast<std::size_t>(index)];
  // Place a NATURAL-SIZE, TEXTURED copy (the preview's wrapper never leaks
  // into it): decode the one chosen file WITH images and stage it fresh —
  // the slots hold only geometry (RAM), so this is the single texture load.
  Model model;
  std::string loadError;
  if (!loadModelFile(file, model, loadError, /*withImages=*/true)) {
    model = Model{};
    if (loadError.empty()) {
      loadError = "could not load";
    }
    std::printf("[gallery] cannot place %s: %s\n", file.c_str(), loadError.c_str());
    return false; // nothing mutated — the carousel stays open to try again
  }
  app.undo.begin(app.galleryBefore);
  galleryEvict(app);
  std::vector<vredit::NodeId> roots;
  if (!addModelToScene(app, renderer, model, &roots, /*withTextures=*/true) || roots.empty()) {
    // Roll back to the pre-open state; identical snapshots record no step.
    applySnapshot(app, app.galleryBefore);
    (void)app.undo.commit(snapshot(app));
    std::printf("[gallery] cannot place %s\n", file.c_str());
    return false;
  }
  vredit::placeOnFloor(app.scene, app.visuals, roots, vredit::floorAimPoint(aimRay));
  app.selection =
      vredit::expandSelection(app.modelGroups, roots[0], /*partSelect=*/false);
  app.undo.commit(snapshot(app));
  galleryClose(app);
  std::printf("[gallery] placed %s\n", file.c_str());
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
                   std::vector<RendererVertex>& overlayLines,
                   Vec3 const* hudPos = nullptr,
                   Quat const* hudOrient = nullptr) {
  if (app.showBackdrop) {
    appendGrid(depthLines, 8.0F, 1.0F);
  }
  auto isSelected = [&app](NodeId id) {
    return std::find(app.selection.begin(), app.selection.end(), id) != app.selection.end();
  };
  // Build the draw items: opaque and mask items first, blended last.
  auto appendItem = [&app, &items, &isSelected](NodeVisual const& visual) {
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
    // The selection gets a thin warm RIM in the shader (5) — the user
    // rejected both the fill tint and the floating diamond. While the hold-X
    // delete charges, the selection ramps red (3 + charge).
    if (isSelected(visual.id)) {
      item.highlight = app.deleteCharge > 0.0F ? 3.0F + app.deleteCharge : 5.0F;
    } else {
      item.highlight = 0.0F;
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
  // The mode HUD: the current mode's name in the line font, view-locked in
  // front of the camera/head (XR passes the head pose; desktop uses the
  // camera), so "which mode is this" is readable in the headset. No
  // filenames: the user said "I really don't need the file name in the
  // carousel".
  std::string hudText = app.mode == EditMode::move ? "MOVE" : "ROTATE";
  // No HUD over the carousel ("You still have the words MOVE in the
  // carousel"): the mode text is only for normal editing.
  if (!app.galleryActive) {
    Vec3 const hudEye = hudPos != nullptr ? *hudPos : app.camera.position;
    Quat const hudFace = hudOrient != nullptr ? *hudOrient : cameraOrientation(app.camera);
    Vec3 const hudFwd = rotate(hudFace, Vec3{0.0F, 0.0F, -1.0F});
    Vec3 const hudUp = rotate(hudFace, Vec3{0.0F, 1.0F, 0.0F});
    Vec3 const hudAt = hudEye + hudFwd * 0.55F + hudUp * 0.17F;
    appendTextLines(overlayLines, hudText, trs(hudAt, hudFace, Vec3{1.0F, 1.0F, 1.0F}),
                    0.028F, Vec3{1.0F, 0.85F, 0.4F});
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
          XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_FOCUS_CHANGE};
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
    if (app.width == 0U || app.height == 0U) {
      break; // minimised: pointer maths would divide by zero
    }
    if (press->detail == 1U) {
      Ray mouseRay = appRay(app, mouse);
      if (app.galleryActive) {
        // The carousel owns clicks: the click places the staged model at the
        // floor point under the pointer.
        (void)galleryPick(app, renderer, mouseRay);
        break;
      }
      bool const partSelect = (press->state & (XCB_MOD_MASK_SHIFT | XCB_MOD_MASK_CONTROL)) != 0U;
      PointerTarget target = resolvePointerTarget(app, mouseRay);
      pointerPress(app, target, partSelect);
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
  case XCB_FOCUS_OUT: {
    // Focus lost mid-hold: the release events will go to another window, so
    // the held-key flags would stay set forever (runaway scene mutation with
    // the arrows). Clear every held input and commit what was done.
    app.flyForward = app.flyBack = app.flyLeft = app.flyRight = false;
    app.flyUp = app.flyDown = app.flyBoost = false;
    app.arrowUp = app.arrowDown = app.arrowLeft = app.arrowRight = false;
    app.looking = false;
    app.panning = false;
    app.panButton = 0U;
    endEdits(app);
    break;
  }
  case XCB_BUTTON_RELEASE: {
    auto* release = reinterpret_cast<xcb_button_release_event_t*>(event);
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
    if (app.width == 0U || app.height == 0U) {
      endEdits(app); // minimised mid-gesture: commit what was done, stop tracking
      app.looking = false;
      app.panning = false;
      app.panButton = 0U;
      break; // pointer maths would divide by zero
    }
    if (app.looking) {
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
    case 0xFF52: app.arrowUp = pressed; break;    // Up
    case 0xFF54: app.arrowDown = pressed; break;  // Down
    case 0xFF51: app.arrowLeft = pressed; break;  // Left
    case 0xFF53: app.arrowRight = pressed; break; // Right
    case 0xFFE1: case 0xFFE2: app.flyBoost = pressed; break; // Shift L/R
    default: movementKey = false; break;
    }
    if (movementKey) {
      break;
    }
    if (!pressed) {
      break;
    }
    // Only keys that change editor state cancel a live gesture; modifier presses
    // (e.g. Ctrl for snap) must not.
    bool stateChanged = false;
    if ((keysym == 'z' || keysym == 'Z') && (key->state & XCB_MOD_MASK_CONTROL) != 0U) {
      // Undo/redo during a live edit cancels that edit first and stops there,
      // leaving both stacks untouched (see cancelLiveEdit).
      if (app.stickSession || app.galleryActive) {
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
      if (app.stickSession || app.galleryActive) {
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
      // The model carousel: N stages one library model in front of the
      // camera, N closes again. [ ] cycles; a click places it at the floor
      // point under the mouse.
      if (app.galleryActive) {
        galleryClose(app);
      } else {
        galleryOpen(app, renderer);
      }
    } else if (keysym == '[' || keysym == ']') {
      galleryCycle(app, renderer, keysym == ']' ? 1 : -1);
    } else if (keysym == '1' || keysym == 'g' || keysym == 'G') {
      app.mode = EditMode::move;
      stateChanged = true;
    } else if (keysym == '2' || keysym == 'r' || keysym == 'R') {
      app.mode = EditMode::rotate;
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
    app.selection.push_back(app.visuals[0].id); // exercise the selection highlight path
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
  // The model carousel end-to-end (its interactive paths are otherwise
  // unreachable in a smoke): open a RING over a library, rotate, place,
  // undo, redo. .build holds the fixture models (.glb + .obj). Loads decode
  // on a worker thread now, so the smoke pumps galleryTick until the slots
  // arrive (bounded). AFTER the tint-sensitive checks on purpose.
  std::size_t const baseVisuals = app.visuals.size();
  app.libraryDir = ".build";
  Vec3 const hudPos{0.0F, 1.5F, 4.0F};
  Vec3 const hudFwd{0.0F, 0.0F, -1.0F};
  auto pump = [&]() {
    for (int i = 0; i < 600; ++i) {
      galleryTick(app, renderer, 1.0F / 60.0F, hudPos, hudFwd);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  };
  auto staged = [&]() {
    int n = 0;
    for (CarouselSlot const& slot : app.carousel) {
      if (slot.hasModel) {
        ++n;
      }
    }
    return n;
  };
  galleryOpen(app, renderer);
  pump();
  if (!app.galleryActive || staged() == 0 || app.visuals.size() <= baseVisuals) {
    std::printf("SMOKE FAIL: carousel staged nothing\n");
    return 1;
  }
  int const centredBefore = app.ring.index;
  galleryCycle(app, renderer, 1);
  pump();
  if (app.ring.index == centredBefore || staged() == 0) {
    std::printf("SMOKE FAIL: carousel cycle did not rotate the ring\n");
    return 1;
  }
  // Wrapping: cycling back lands on the original slot.
  galleryCycle(app, renderer, -1);
  pump();
  if (app.ring.index != centredBefore) {
    std::printf("SMOKE FAIL: carousel ring did not wrap back\n");
    return 1;
  }
  Ray const placeAim{hudPos, normalized(Vec3{0.0F, -1.0F, -0.2F})};
  if (!galleryPick(app, renderer, placeAim) || app.galleryActive) {
    std::printf("SMOKE FAIL: carousel placement\n");
    return 1;
  }
  // Undo removes the placed model entirely (the pre-open "before" — never
  // a floating preview), and redo brings it back.
  vredit::EditorSnapshot carouselState = snapshot(app);
  if (!app.undo.undo(carouselState)) {
    std::printf("SMOKE FAIL: undo after placement\n");
    return 1;
  }
  applySnapshot(app, carouselState);
  if (app.visuals.size() != baseVisuals) {
    std::printf("SMOKE FAIL: undo left the placed model in the scene\n");
    return 1;
  }
  if (!app.undo.redo(carouselState)) {
    std::printf("SMOKE FAIL: redo after placement undo\n");
    return 1;
  }
  applySnapshot(app, carouselState);
  if (app.visuals.size() <= baseVisuals) {
    std::printf("SMOKE FAIL: redo did not restore the placed model\n");
    return 1;
  }
  std::printf("SMOKE carousel: ring open -> rotate -> place -> undo -> redo OK\n");
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
    // fine-control modifier (and selects roll in Rotate mode).

    // NO grip chords on face buttons — the grip sensor spikes when squeezing
    // the controller to press one, which made this dispatch silently take the
    // wrong branch (the reported "can only rotate horizontally": the mode
    // cycle was unreachable behind a grip+menu backdrop chord). Tap/hold on
    // ONE button instead: TAP (< 0.4 s) = stick mode cycle (on release),
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
      app.mode = app.mode == EditMode::move ? EditMode::rotate : EditMode::move;
      endEdits(app);
    }
    bool cancelledLiveEdit = false;
    if (input.undoPressed) {
      if (app.stickSession || app.galleryActive) {
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
      if (app.stickSession || app.galleryActive) {
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
    // mouse. A body press only SELECTS —
    // movement and rotation are the sticks' job (controller aim is too
    // jittery for accurate placement).
    app.camera.position = headWorld;
    if (input.selectPressed) {
      if (app.galleryActive) {
        // The carousel owns the trigger: it places the staged model at the
        // floor point under the aim ray.
        (void)galleryPick(app, renderer, worldRay);
      } else {
        // Grip + trigger selects a single part; trigger alone the whole model.
        bool const partSelect = input.gripHeld > 0.5F;
        PointerTarget target = resolvePointerTarget(app, worldRay);
        pointerPress(app, target, partSelect);
      }
    }

    // Stick-driven editing (the coloured handles are gone): the right stick
    // slides the selection across the floor relative to the head's yaw
    // (pushing forward walks it away from you) in BOTH modes; the left stick
    // is mode-driven (move: X scale, Y height; rotate: X spin, Y tip — grip
    // selects roll). One undo step per gesture.
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
    // The right stick moves (X/Z) in both modes. The left stick is
    // mode-driven (menu tap toggles): MOVE = X uniform scale, Y height;
    // ROTATE = X spin on the floor, Y tip — grip free: pitch (stick up tips
    // the top away), grip held: roll (stick up rolls the top to your right).
    bool const sticksActive = std::fabs(input.rightStick.x) > stickThreshold ||
                              std::fabs(input.rightStick.y) > stickThreshold ||
                              std::fabs(input.leftStick.x) > stickThreshold ||
                              std::fabs(input.leftStick.y) > stickThreshold;
    bool const gestureAlive = sticksActive;
    // While the carousel is open the sticks are ITS: the left stick's X
    // cycles the staged model (a flick = one step; held = one step per
    // 0.25 s). All editing stays blocked (the session gates below).
    galleryTick(app, renderer, dt, headWorld, headForward);
    if (app.galleryActive) {
      app.galleryCycleCooldown -= dt;
      if (std::fabs(input.leftStick.x) > 0.5F && app.galleryCycleCooldown <= 0.0) {
        galleryCycle(app, renderer, input.leftStick.x > 0.0F ? 1 : -1);
        app.galleryCycleCooldown = 0.25;
      }
    }
    // After an undo/redo the sticks must return to neutral before a new
    // gesture may start, or the same frame's drift cancels-and-restarts and
    // swallows the undo.
    if (!sticksActive) {
      app.stickArmed = true;
    }
    if (sticksActive && !app.stickSession && !app.galleryActive &&
        !app.selection.empty() && app.stickArmed) {
      app.undo.begin(snapshot(app));
      app.stickSession = true;
      app.stickFineRate = input.gripHeld > 0.5F ? 0.25F : 1.0F;
      app.stickRollMode = input.gripHeld > 0.5F; // grip+Y = roll (latched)
      // The rotation axes are the SELECTION'S OWN frame (perpendicular by
      // construction — the user requirement): spin about the model's up,
      // tip about the model's right, roll about the model's forward,
      // sign-aligned to the viewer at gesture start so "up tips away" keeps
      // its meaning. An unrotated model behaves exactly as before.
      Vec3 yawAxis{0.0F, 1.0F, 0.0F};
      Vec3 tipAxis{0.0F, 0.0F, -1.0F};
      Vec3 rollAxis{0.0F, 0.0F, -1.0F};
      if (vredit::selectionRotationAxes(app.scene, app.selection, headRightFlat,
                                        headForwardFlat, yawAxis, tipAxis, rollAxis)) {
        app.stickYawAxis = yawAxis;
        app.stickTipAxis = tipAxis;
        app.stickRollAxis = rollAxis;
      } else {
        // Degenerate selection frame: fall back to the viewer-aligned
        // defaults rather than silently reusing the PREVIOUS gesture's axes.
        app.stickYawAxis = yawAxis;
        app.stickTipAxis = tipAxis;
        app.stickRollAxis = rollAxis;
      }
      vredit::Aabb bounds{};
      if (vredit::movingSetBounds(app.scene, app.visuals, app.selection, bounds)) {
        app.stickPivot = vredit::aabbCentre(bounds);
      } else {
        app.stickPivot = app.scene.worldPosition(app.selection.front());
      }
    }
    if (app.stickSession && !app.galleryActive) {
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
        // from above, which is negative around the model's up. ROTATE MODE
        // ONLY — nothing rotates outside it (approved rule).
        // Move mode: left stick X = uniform scale (right = bigger), Y =
        // height (up = lift). Rotate mode: X = spin, Y = tip (grip latched
        // at gesture start selects roll).
        switch (app.mode) {
        case EditMode::move: {
          if (std::fabs(leftX) > 1e-4F) {
            float const factor =
                std::exp(leftX * kStickScalePerS * app.stickFineRate * dt);
            vredit::scaleSelectionUniform(app.scene, app.visuals, app.selection, factor);
          }
          if (std::fabs(leftY) > 1e-4F) {
            // Explicit height: no floor rule (lifting is deliberate).
            float const deltaY = leftY * kStickHeightMps * app.stickFineRate * dt;
            vredit::moveSelectionVertical(app.scene, app.selection, deltaY);
          }
          break;
        }
        case EditMode::rotate:
        default: {
          float const yawDelta = -leftX * kStickTurnRadps * app.stickFineRate * dt;
          if (yawDelta != 0.0F) {
            vredit::rotateSelectionAbout(app.scene, app.selection, app.stickPivot,
                                         app.stickYawAxis, yawDelta);
          }
          if (std::fabs(leftY) > 1e-4F) {
            // Grip selects roll at GESTURE start (latched); the axis follows
            // the model's frame (perpendicular to the spin axis).
            float const angle = leftY * kStickTurnRadps * app.stickFineRate * dt;
            Vec3 const axis = app.stickRollMode ? app.stickRollAxis : app.stickTipAxis;
            vredit::rotateSelectionAbout(app.scene, app.selection, app.stickPivot, axis, angle);
            // The floor rule is about MODELS ("models stay stuck to the
            // floor"): tipping a PART must not lift it out of its model by
            // flooring through the part's own bounds.
            if (wholeModelSelection(app)) {
              vredit::dropSelectionToFloor(app.scene, app.visuals, app.selection);
            }
          }
          break;
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
    // The model carousel: the spawn key opens it (one staged model in front
    // of the viewer) or closes it. NO grip chord here on purpose —
    // the grip sensor spikes when squeezing the controller to press A, and
    // the old "grip+A cancels" chord silently swallowed EVERY press (the
    // reported "nothing appears when I press A"). Cancelling is undo (B) — a
    // single, stable button the user already has.
    if (input.spawnPressed) {
      if (app.galleryActive) {
        galleryClose(app);
      } else {
        galleryOpen(app, renderer);
      }
    }
    if (app.galleryActive) {
      // The carousel owns the pointer: the trigger places the staged model
      // at the aim point (handled at the trigger press).
    }

    std::vector<DrawItem> items;
    std::vector<RendererVertex> depthLines;
    std::vector<RendererVertex> overlayLines;
    // The HUD faces the horizon-locked head — the same correction the eyes
    // get (gravityAlign(headTotal) is the corrected orientation exactly), so
    // the text stays level like everything else.
    Quat const hudHead = gravityAlign(cameraOrientation(rig) * input.headOrient);
    assembleFrame(app, items, depthLines, overlayLines, &headWorld, &hudHead);
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
      if (!renderer.drawToView(shell.eyeView(eye), shell.eyeImage(eye), viewProj, eyeWorld, items,
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

// Applies held arrow keys once per frame: the desktop mirror of the stick
// editing (the coloured handles are gone). Mode-driven like the sticks:
// move = X uniform scale, Y height; rotate = X spin about the selection's
// own up, Y tip (Shift = roll). Same one-undo-step-per-gesture lifecycle
// as the sticks: begun on the first deflection, committed when the arrows
// settle.
void updateArrowEdit(App& app, float dt) {
  float const axisX = app.arrowRight ? 1.0F : (app.arrowLeft ? -1.0F : 0.0F);
  float const axisY = app.arrowUp ? 1.0F : (app.arrowDown ? -1.0F : 0.0F);
  if (axisX == 0.0F && axisY == 0.0F) {
    // Debounced commit (2 quiet frames): X11 auto-repeat's Release+Press
    // pair must not split one physical hold into many undo steps.
    if (app.arrowQuietFrames < 2) {
      ++app.arrowQuietFrames;
      return;
    }
    if (app.stickSession && !app.galleryActive) {
      app.undo.commit(snapshot(app));
      app.stickSession = false;
    }
    app.stickArmed = true; // arrows re-centred: the next hold may start
    return;
  }
  app.arrowQuietFrames = 0;
  if (app.galleryActive || app.selection.empty()) {
    return;
  }
  Vec3 forward = cameraForward(app.camera);
  Vec3 forwardFlat{forward.x, 0.0F, forward.z};
  forwardFlat = dot(forwardFlat, forwardFlat) > 1e-8F ? normalized(forwardFlat)
                                                     : Vec3{0.0F, 0.0F, -1.0F};
  Vec3 const rightFlat = cross(forwardFlat, Vec3{0.0F, 1.0F, 0.0F});
  if (!app.stickSession) {
    // Armed latch: a cancelled edit must not restart from still-held keys in
    // the same frame (it would swallow the undo press that cancelled it).
    if (!app.stickArmed) {
      return;
    }
    app.undo.begin(snapshot(app));
    app.stickSession = true;
    app.stickFineRate = app.flyBoost ? 0.25F : 1.0F; // Shift = fine (latched)
    app.stickRollMode = app.flyBoost;                // Shift+Y = roll (latched)
    Vec3 yawAxis{0.0F, 1.0F, 0.0F};
    Vec3 tipAxis{0.0F, 0.0F, -1.0F};
    Vec3 rollAxis{0.0F, 0.0F, -1.0F};
    if (vredit::selectionRotationAxes(app.scene, app.selection, rightFlat, forwardFlat,
                                      yawAxis, tipAxis, rollAxis)) {
      app.stickYawAxis = yawAxis;
      app.stickTipAxis = tipAxis;
      app.stickRollAxis = rollAxis;
    } else {
      // Degenerate selection frame: fall back to the viewer-aligned defaults
      // rather than silently reusing the PREVIOUS gesture's axes.
      app.stickYawAxis = yawAxis;
      app.stickTipAxis = tipAxis;
      app.stickRollAxis = rollAxis;
    }
    vredit::Aabb bounds{};
    app.stickPivot = vredit::movingSetBounds(app.scene, app.visuals, app.selection, bounds)
                        ? vredit::aabbCentre(bounds)
                        : app.scene.worldPosition(app.selection.front());
  }
  // Same rules as the VR sticks: X scales in move mode (spin in rotate),
  // Y = height in move (tip/roll in rotate).
  if (axisX != 0.0F && app.mode == EditMode::rotate) {
    float const yawDelta = -axisX * kStickTurnRadps * app.stickFineRate * dt;
    vredit::rotateSelectionAbout(app.scene, app.selection, app.stickPivot, app.stickYawAxis,
                                 yawDelta);
  }
  if (axisX != 0.0F && app.mode == EditMode::move) {
    vredit::scaleSelectionUniform(
        app.scene, app.visuals, app.selection,
        std::exp(axisX * kStickScalePerS * app.stickFineRate * dt));
  }
  if (axisY != 0.0F) {
    switch (app.mode) {
    case EditMode::move:
      vredit::moveSelectionVertical(app.scene, app.selection,
                                    axisY * kStickHeightMps * app.stickFineRate * dt);
      break;
    case EditMode::rotate:
    default: {
      Vec3 const axis = app.stickRollMode ? app.stickRollAxis : app.stickTipAxis;
      vredit::rotateSelectionAbout(app.scene, app.selection, app.stickPivot, axis,
                                   axisY * kStickTurnRadps * app.stickFineRate * dt);
      if (wholeModelSelection(app)) {
        vredit::dropSelectionToFloor(app.scene, app.visuals, app.selection);
      }
      break;
    }
    }
  }
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
      "  wheel: fly forward/back   LMB: select (shift+click a part)\n"
      "  arrows edit (mode-driven): X = scale / spin   Y = height / tip (shift: roll)\n"
      "  1 move   2 rotate   N model carousel ([ ] cycle, click places)   Esc quit\n",
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
    updateArrowEdit(app, dt); // arrow keys mirror the sticks (edit the selection)
    galleryTick(app, renderer, dt, app.camera.position, cameraForward(app.camera));
    if (app.width == 0U || app.height == 0U) {
      std::this_thread::sleep_for(std::chrono::milliseconds(8)); // minimised: no busy spin
      continue;
    }
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
          "keys: WASD/QE fly (RMB look), arrows edit: left/right = X (scale in\n"
          "      move mode, spin in rotate), up/down = Y (height / tip; shift =\n"
          "      fine, and roll in rotate mode), 1 move / 2 rotate (menu toggle),\n"
          "      N model carousel ([ ] cycle, click places), Delete remove,\n"
          "      B backdrop, Ctrl-Z/Y undo/redo, shift+click part, Esc quit\n"
          "vr:   trigger select (grip+trigger part), right stick move on the floor,\n"
          "      left stick: X/Y = the mode (menu tap toggles MOVE scale+height /\n"
          "      ROTATE spin+tip, grip = fine and roll),\n"
          "      A model carousel (stick X cycles, trigger places),\n"
          "      hold X delete, B/Y undo/redo, menu hold = backdrop\n");
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
