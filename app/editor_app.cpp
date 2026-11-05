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

#include "animation.h"
#include "camera.h"
#include "carousel.h"
#include "editor_state.h"
#include "gizmo.h"
#include "gltf.h"
#include "hand_mesh.h"
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
// Stick-driven editing rates (VR): the left stick slides the selection, the
// right stick turns/scales it. Grip held runs both at quarter speed (fine
// control).
constexpr float kStickDeadZone = 0.12F; // a gesture STARTS past this deflection
constexpr float kStickEndZone = 0.06F;  // ...and ENDS below this (hysteresis, so one
                                        // physical gesture is one undo step)
constexpr float kStickMoveMps = 3.0F;    // metres per second at full deflection
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
  float halfHeight{0.0F};        // fitted vertical half-extent: the label hangs below it
};

constexpr std::size_t kSlotCount = 5; // centre ± 2
constexpr float kPreviewExtent = 0.6F;

// Carousel labels: each staged preview is NAMED underneath, so "which model is
// this" is answerable from the ring without placing it. The name is fitted
// into kLabelMaxWidth so the longest entries ("..._DIAMOND_EARRINGS") cannot
// run into their neighbours a slot away.
constexpr float kLabelHeight = 0.05F;
constexpr float kLabelMaxWidth = 0.8F;
constexpr float kLabelGap = 0.035F;

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
bool smokeHandEditing(App& app, VkRenderer& renderer);
inline MeshHandle cachedMesh(App& app, VkRenderer& renderer, MeshData const& data);

struct AnimationInstance {
  Model model; // nodes and clips only; mesh/image payload is not retained
  std::vector<NodeId> nodes;
  double elapsed{0};
  float duration{0};
};

struct App {

  bool xr{false};
  Scene scene;
  std::vector<AnimationInstance> animations;
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
  bool showBackdrop{false}; // passthrough/void is the default; B or menu-hold adds the grid
  // Editing mode (desktop 1/2 only: the VR menu tap is the carousel now, and
  // rotation comes back as gestures). The RIGHT stick is mode-driven: move =
  // X uniform scale, Y height; rotate = X spins about the scene's up axis,
  // Y lifts/lowers the face. The LEFT stick is planar floor-move in BOTH
  // modes; nothing rotates in move.
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
  std::map<std::uint64_t, TextureHandle> textureCache;
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
  bool handGrab{false};
  bool physicalGrab{false};
  HandContactCache handContactCache;
  HandPose grabStart{};
  HandPose grabFiltered{};
  HandPose scaleLeftFiltered{};
  bool handScaling{false};
  float scaleStartDistance{0};
  Vec3 grabPivot{};
  HandGrabView grabView{};
  std::vector<std::pair<NodeId, Mat4>> grabWorlds;
  std::vector<std::pair<NodeId, Transform>> grabLocals;
  std::vector<std::pair<NodeId, Mat4>> grabPoseStarts;
  HandPanel handPanel{};
  bool handPanelCarousel{false};
  int handHover{-1};
  int handTool{-1}; // captured until pinch release, never retarget mid-hold
  bool handUiOwned{false};
  bool stickSession{false};
  // The stick gesture's frame, latched at gesture start: the pivot (the
  // moving set's bounds centre), the pre-signed tip axis and the fine rate.
  // Latching keeps one gesture one coherent step — a live head pose or grip
  // toggle mid-gesture must not steer the tip, change its rate, or walk
  // asymmetric models sideways via a re-centred pivot.
  Vec3 stickPivot{};
  Vec3 stickYawAxis{0.0F, 1.0F, 0.0F};  // the scene's level vertical (X spin)
  Vec3 stickTipAxis{0.0F, 0.0F, -1.0F}; // the viewer's left-right line (Y tip)
  float stickFineRate{1.0F};
  // Re-arm latch: after an undo/redo the sticks must return to neutral
  // before a new gesture may start, or the same frame's drift would
  // cancel-and-restart and swallow the undo.
  bool stickArmed{true};
  // Menu tap/hold (VR): a tap opens/closes the model carousel; a hold toggles
  // the backdrop once at 0.4 s (cancelling the tap). Thumbstick clicks do
  // nothing (the user's binding).
  float menuCharge{0.0F};
  bool menuHoldLatched{false};
  bool menuTapped{false}; // a pending tap (fires the carousel on release)
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
  // left/right is X (scale in move, spin in rotate), up/down is Y (height in
  // move, tip in rotate).
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
    auto key = hashBytes(1469598103934665603ULL, &image.width, sizeof(image.width));
    key = hashBytes(key, &image.height, sizeof(image.height));
    key = hashBytes(key, image.rgba.data(), image.rgba.size());
    key = hashBytes(key, &model.textures[t].repeatU, sizeof(bool));
    key = hashBytes(key, &model.textures[t].repeatV, sizeof(bool));
    auto const cached = app.textureCache.find(key);
    if (cached != app.textureCache.end()) {
      textureHandles[t] = cached->second;
    } else {
      textureHandles[t] = renderer.addTexture(image, model.textures[t].repeatU, model.textures[t].repeatV);
      if (textureHandles[t] != kInvalidTexture) { app.textureCache.emplace(key, textureHandles[t]); }
    }

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
  auto const playable = std::find_if(model.clips.begin(), model.clips.end(),
      [](AnimationClip const& clip) { return !clip.channels.empty(); });
  if (withTextures && playable != model.clips.end()) {

    AnimationInstance instance;
    instance.model.nodes = model.nodes;
    instance.model.clips.push_back(*playable);
    instance.nodes = handles;
    for (auto const& channel : instance.model.clips.front().channels) {
      if (!channel.times.empty()) { instance.duration = std::fmax(instance.duration, channel.times.back()); }
    }
    app.animations.push_back(std::move(instance));
  }
  } catch (...) {
    vredit::deleteSubtrees(app.scene, app.visuals, app.modelGroups, app.selection, createdRoots);
    if (rootsOut != nullptr) { rootsOut->clear(); }
    std::printf("model insertion failed mid-way — rolled back\n");
    return false;
  }
  return true;

}

void tickAnimations(App& app, float dt) {
  app.animations.erase(std::remove_if(app.animations.begin(), app.animations.end(),
      [&](AnimationInstance const& instance) {
        for (auto id : instance.nodes) { if (app.scene.valid(id)) { return false; } }
        return !app.undo.anySnapshot([&](EditorSnapshot const& saved) {
          for (auto const& node : saved.scene.nodes) {
            if (std::find(instance.nodes.begin(), instance.nodes.end(), node.id) != instance.nodes.end()) {
              return true;
            }
          }
          return false;
        });
      }), app.animations.end());
  for (auto& instance : app.animations) {

    bool alive = false;
    for (auto id : instance.nodes) { alive = alive || app.scene.valid(id); }
    if (!alive) { continue; } // retained so delete/undo can restore playback
    instance.elapsed += std::fmax(0.0F, dt);
    float const time = instance.duration > 0 ?
        static_cast<float>(std::fmod(instance.elapsed, instance.duration)) : 0.0F;
    std::vector<Transform> pose;
    if (!evaluateAnimation(instance.model, instance.model.clips.front(), time, pose)) { continue; }
    for (auto const& channel : instance.model.clips.front().channels) {
      auto const index = channel.node;
      if (index >= instance.nodes.size() || index >= pose.size()) { continue; }
      Mat4 inverseRest;
      if (!invertAffine(toMatrix(instance.model.nodes[index].local), inverseRest)) { continue; }
      app.scene.setPoseOffset(instance.nodes[index], mul(inverseRest, toMatrix(pose[index])));
    }
  }
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
std::optional<vredit::PickHit> pickNode(App& app, Ray const& ray, NodeId subtree = kInvalidNode) {
  std::vector<vredit::PickCandidate> candidates;
  for (NodeVisual const& visual : app.visuals) {
    if (subtree != kInvalidNode) {
      auto node = visual.id;
      while (node != kInvalidNode && node != subtree) { node = app.scene.parent(node); }
      if (node != subtree) { continue; }
    }
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
  Vec3 point{};
};

PointerTarget resolvePointerTarget(App& app, Ray const& ray) {
  PointerTarget target;
  // The press's own pick (exact surface first): the bounding box is not a
  // click target — pressing empty space selects nothing.
  auto body = pickNode(app, ray);
  if (body.has_value()) {
    target.node = body->node;
    target.point = body->point;
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
  app.handGrab = false;
  app.physicalGrab = false;
  app.handScaling = false;
  app.handTool = -1;
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
  bool const wasActive = app.stickSession || app.handGrab || app.handTool >= 2;
  app.stickSession = false;
  app.handGrab = false;
  app.physicalGrab = false;
  app.handScaling = false;
  app.handTool = -1;
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
    slot.halfHeight = size.y * 0.5F * slot.fit;
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
  (void)aimRay;
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
  // The selection ray hits the preview, not a floor target. Place under the
  // stationary centre preview instead of extending shallow rays to infinity.
  tickAnimations(app, 0.0F);
  auto target = app.ring.slotPose(index, app.ringOrigin, app.ringForward).position;
  target.y = 0.0F;
  vredit::placeOnFloor(app.scene, app.visuals, roots, target);

  app.selection =
      vredit::expandSelection(app.modelGroups, roots[0], /*partSelect=*/false);
  app.undo.commit(snapshot(app));
  app.stickArmed = false;
  galleryClose(app);
  std::printf("[gallery] placed %s\n", file.c_str());
  return true;
}

// A model surface is the only placement target. Empty left/right space
// browses relative to the stationary ring, never the changing head heading.
int galleryPointerAction(App& app, Ray const& ray) {
  for (auto const& slot : app.carousel) {
    if (app.ring.atRest() && slot.index == app.ring.index && slot.hasModel &&
        pickNode(app, ray, slot.wrapper).has_value()) { return 2; }
  }
  return carouselPinchDirection(ray, app.ringOrigin, app.ringForward);
}

void galleryPointerPress(App& app, VkRenderer& renderer, Ray const& ray) {
  int const action = galleryPointerAction(app, ray);
  if (action == 2) { (void)galleryPick(app, renderer, ray); }
  else if (action != 0) { galleryCycle(app, renderer, action); }
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
  // front of the camera (desktop only — VR has no mode switch any more, so the
  // word would be a permanent lie in the headset; it comes back with the
  // gestures). The carousel names the MODELS instead (see below), so the mode
  // text is hidden there too.
  if (!app.xr && !app.galleryActive) {
    std::string hudText = app.mode == EditMode::move ? "MOVE" : "ROTATE";
    Vec3 const hudEye = app.camera.position;
    Quat const hudFace = cameraOrientation(app.camera);
    Vec3 const hudFwd = rotate(hudFace, Vec3{0.0F, 0.0F, -1.0F});
    Vec3 const hudUp = rotate(hudFace, Vec3{0.0F, 1.0F, 0.0F});
    Vec3 const hudAt = hudEye + hudFwd * 0.55F + hudUp * 0.17F;
    appendTextLines(overlayLines, hudText, trs(hudAt, hudFace, Vec3{1.0F, 1.0F, 1.0F}),
                    0.028F, Vec3{1.0F, 0.85F, 0.4F});
  }
  // The carousel's model names: one label per staged preview, centred on its
  // slot, hung just below the model's own feet (slot.halfHeight, not a fixed
  // box) and facing the viewer with the model. The centred slot is brighter,
  // so "the name in front of me" is unambiguous. Never drawn for a slot that
  // failed to load — there is nothing above the text to name.
  if (app.galleryActive) {
    for (CarouselSlot const& slot : app.carousel) {
      if (!slot.hasModel || slot.index < 0 ||
          static_cast<std::size_t>(slot.index) >= app.galleryFiles.size()) {
        continue;
      }
      std::string const name =
          std::filesystem::path(app.galleryFiles[static_cast<std::size_t>(slot.index)])
              .stem()
              .string();
      if (name.empty()) {
        continue;
      }
      vredit::CarouselRing::SlotPose const label = app.ring.labelPose(
          slot.index, app.ringOrigin, app.ringForward, slot.halfHeight, kLabelGap);
      // Long names shrink to fit the slot's width; short ones keep full size.
      float height = kLabelHeight;
      float const unitWidth = textLineWidth(name, 1.0F);
      if (unitWidth > 0.0F && unitWidth * height > kLabelMaxWidth) {
        height = kLabelMaxWidth / unitWidth;
      }
      Quat const face = quatFromAxisAngle(Vec3{0.0F, 1.0F, 0.0F}, label.yaw);
      // Centre the run: local +X is the viewer's right for this facing, so the
      // origin shifts half a width left and the pen starts at zero.
      Vec3 const right = rotate(face, Vec3{1.0F, 0.0F, 0.0F});
      Vec3 const at = label.position - right * (0.5F * textLineWidth(name, height));
      bool const centred = slot.index == app.ring.index;
      Vec3 const colour = centred ? Vec3{1.0F, 0.95F, 0.82F} : Vec3{0.62F, 0.68F, 0.78F};
      appendTextLines(overlayLines, name, trs(at, face, Vec3{1.0F, 1.0F, 1.0F}), height, colour);
      if (centred && app.ring.atRest()) {
        auto const mid = label.position + Vec3{0, -.035F, 0};
        Vec3 const gold{1.0F, .7F, .25F};
        auto const up = Vec3{0,1,0};
        for (auto const& point : {mid - right*.035F, mid + Vec3{0,.018F,0},
                                  mid + Vec3{0,.018F,0}, mid + right*.035F}) {
          overlayLines.push_back({point, up, gold, {0,0}});
        }
      }

    }
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
        galleryPointerPress(app, renderer, mouseRay);
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
      if (app.stickSession || app.handGrab || app.handTool >= 0 || app.galleryActive) {
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
      if (app.stickSession || app.handGrab || app.handTool >= 0 || app.galleryActive) {
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
  // EVERY PREVIEW IS NAMED. With the gallery open the overlay line list holds
  // nothing but the model labels (the mode HUD is suppressed there), so this
  // is a direct check of the production path: the labels exist, they sit
  // BELOW the previews they belong to (never over the models), and each one is
  // centred on its own slot rather than all stacking in the middle. Then the
  // frame is rendered so the placement is visible, not merely numbered.
  {
    items.clear();
    depthLines.clear();
    overlayLines.clear();
    assembleFrame(app, items, depthLines, overlayLines);
    int const labelVerts = static_cast<int>(overlayLines.size());
    if (labelVerts < 4 * staged()) {
      std::printf("SMOKE FAIL: carousel labels missing (%d vertices for %d slots)\n",
                  labelVerts, staged());
      return 1;
    }
    float minY = 1e9F;
    float maxY = -1e9F;
    float minX = 1e9F;
    float maxX = -1e9F;
    for (RendererVertex const& v : overlayLines) {
      minY = std::fmin(minY, v.pos.y);
      maxY = std::fmax(maxY, v.pos.y);
      minX = std::fmin(minX, v.pos.x);
      maxX = std::fmax(maxX, v.pos.x);
    }
    // The label BASELINE hangs below its preview; the glyphs themselves rise
    // one cap height from it, so a flat fixture can put its text top a few
    // centimetres above the preview's own centre. The band below allows that
    // while still catching a label drifting up over the models or to the ring
    // centre.
    if (maxY > hudPos.y + 0.1F || minY < hudPos.y - 1.0F || minX < -1.4F || maxX > 1.4F ||
        maxX - minX < 0.4F) {
      std::printf("SMOKE FAIL: carousel label placement out of range "
                  "(x %.2f..%.2f, y %.2f..%.2f)\n",
                  minX, maxX, minY, maxY);
      return 1;
    }
    // EVERY PREVIEW IS FITTED THE SAME. A model's stray armature (nodes far
    // outside the visible mesh) used to inflate the staged bounds: Maid drew
    // at 3.7 mm and Teen Kitty at a 172 m box, and their labels hung at chest
    // height instead of under the feet. Measure the box the viewer actually
    // gets — the wrapper's whole subtree — and require kPreviewExtent, with
    // the label baseline strictly below the geometry.
    for (CarouselSlot const& slot : app.carousel) {
      if (!slot.hasModel || slot.wrapper == vredit::kInvalidNode) {
        continue;
      }
      std::vector<vredit::NodeId> nodes = app.scene.descendants(slot.wrapper);
      nodes.push_back(slot.wrapper);
      vredit::Aabb box{};
      if (!vredit::nodesBounds(app.scene, app.visuals, nodes, box)) {
        std::printf("SMOKE FAIL: carousel slot %d has no bounds\n", slot.index);
        return 1;
      }
      Vec3 const size = box.max - box.min;
      float const maxDim = std::fmax(size.x, std::fmax(size.y, size.z));
      // The fit divides by the axis-aligned box of each mesh through its node
      // chain, so rotating the wrapper into its ring pose re-measures that box
      // and the drawn size lands anywhere in kPreviewExtent/sqrt(2) ..
      // kPreviewExtent*sqrt(2) for an awkwardly turned model. The band below
      // allows exactly that and nothing more: what it catches is the failure
      // seen in the real library, where a stray armature inflated the staged
      // box ~100x and the preview shrank to a speck (Maid drew at 3.7 mm).
      if (maxDim > kPreviewExtent * 1.45F || maxDim < kPreviewExtent * 0.69F) {
        std::printf("SMOKE FAIL: carousel preview %d draws at %.3f m (target %.3f)\n", slot.index,
                    maxDim, kPreviewExtent);
        return 1;
      }
      vredit::CarouselRing::SlotPose const label = app.ring.labelPose(
          slot.index, app.ringOrigin, app.ringForward, slot.halfHeight, kLabelGap);
      if (label.position.y >= box.min.y) {
        std::printf("SMOKE FAIL: carousel label %d sits at the model, not under it "
                    "(%.3f vs bottom %.3f)\n",
                    slot.index, label.position.y, box.min.y);
        return 1;
      }
    }
    // Stand where the ring was latched, looking along it, and render: the
    // picture proves what the numbers above only imply.
    app.camera.position = hudPos;
    app.camera.yaw = 0.0F;
    app.camera.pitch = -0.25F;
    if (!renderer.draw(app.camera, items, depthLines, overlayLines) ||
        !renderer.readback(pixels)) {
      std::printf("SMOKE FAIL: carousel label render\n");
      return 1;
    }
    writePpm(".build/carousel-labels.ppm", pixels, config.width, config.height);
    std::printf("SMOKE carousel labels: %d vertices over %d slots -> .build/carousel-labels.ppm\n",
                labelVerts, staged());
  }
  // Let the animation settle before testing a centred surface target.
  for (int i = 0; i < 120; ++i) { galleryTick(app, renderer, 1.0F / 60, hudPos, {0,0,-1}); }
  bool previewHit = false;
  for (auto const& slot : app.carousel) {
    if (slot.index != app.ring.index || !slot.hasModel) { continue; }
    for (auto const& visual : app.visuals) {
      auto node = visual.id;
      while (node != kInvalidNode && node != slot.wrapper) { node = app.scene.parent(node); }
      if (node != slot.wrapper || !visual.pickGeometry || visual.pickGeometry->indices.size() < 3) { continue; }
      auto const& geometry = *visual.pickGeometry;
      Vec3 centre{};
      for (int i = 0; i < 3; ++i) { centre = centre + geometry.positions[geometry.indices[i]]; }
      centre = transformPoint(app.scene.world(visual.id), centre / 3.0F);
      Ray const ray{hudPos, normalized(centre - hudPos)};
      if (galleryPointerAction(app, ray) == 2) {
        auto const settledAngle = app.ring.angle;
        app.ring.angle += .4F;
        if (galleryPointerAction(app, ray) == 2) {
          std::printf("SMOKE FAIL: moving preview cannot place\n"); return 1;
        }
        app.ring.angle = settledAngle;
        previewHit = true; break;
      }
    }
  }
  if (!previewHit || galleryPointerAction(app, {hudPos, {0,1,0}}) != 0) {
    std::printf("SMOKE FAIL: preview surface-only placement target\n"); return 1;
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
  if (galleryPointerAction(app, {hudPos, normalized(Vec3{-1,0,-1})}) != -1 ||
      galleryPointerAction(app, {hudPos, normalized(Vec3{1,0,-1})}) != 1) {
    std::printf("SMOKE FAIL: carousel side pinch zones\n"); return 1;
  }
  Ray const placeAim{hudPos, normalized(Vec3{0.0F, -1.0F, -0.2F})};
  if (!galleryPick(app, renderer, placeAim) || app.galleryActive) {
    std::printf("SMOKE FAIL: carousel placement\n");
    return 1;
  }
  Aabb placedBounds;
  auto const placement = app.ring.slotPose(app.ring.index, app.ringOrigin, app.ringForward).position;
  if (app.stickArmed || !movingSetBounds(app.scene, app.visuals, app.selection, placedBounds) ||
      std::fabs((placedBounds.min.x + placedBounds.max.x) * .5F - placement.x) > .001F ||
      std::fabs((placedBounds.min.z + placedBounds.max.z) * .5F - placement.z) > .001F ||
      std::fabs(placedBounds.min.y) > .001F) {
    std::printf("SMOKE FAIL: placement position or stick latch\n"); return 1;
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
  App handFixture;
  if (!smokeHandEditing(handFixture, renderer)) {
    return 1;
  }
  std::printf("SMOKE PASS (offscreen) on %s\n", renderer.deviceName());
  renderer.shutdown();
  return 0;
}

// Blend-order smoke: two overlapping translucent quads must composite
// identically whether the items arrive near-first or far-first, and whether
// their triangles share one mesh in the wrong order — the two Centre Stage
// drawing-order rules (blend items farthest-centre-first with a stable
// tie-break; back-to-front triangles within a blend mesh), proven through the
// real pipeline. Wrong input order must produce the SAME pixels as right
// input order; a missing sort flips the compositing and the bytes differ.
int runSmokeBlend(char const* shaderDir) {
  VkRenderer renderer;
  VkRenderer::Config config;
  config.windowed = false;
  config.width = 160;
  config.height = 120;
  config.shaderDir = shaderDir;
  if (!renderer.init(config)) {
    std::printf("SMOKE FAIL: renderer init\n");
    return 1;
  }
  renderer.setClearColour(0.0F, 0.0F, 0.0F, 1.0F);

  auto quadMesh = [](float z, Vec3 colour) {
    MeshData mesh;
    mesh.vertices = {
        {{-1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, colour, {0.0F, 0.0F}},
        {{1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, colour, {1.0F, 0.0F}},
        {{-1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, colour, {0.0F, 1.0F}},
        {{1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, colour, {1.0F, 1.0F}},
    };
    mesh.indices = {0, 1, 2, 2, 1, 3};
    return mesh;
  };
  MeshData const red = quadMesh(-3.0F, {1.0F, 0.0F, 0.0F});
  MeshData const green = quadMesh(-1.0F, {0.0F, 1.0F, 0.0F});
  MeshData both;
  both.vertices = red.vertices;
  both.vertices.insert(both.vertices.end(), green.vertices.begin(), green.vertices.end());
  both.indices = {4, 5, 6, 6, 5, 7, 0, 1, 2, 2, 1, 3}; // near quad first: wrong order

  MeshHandle const redHandle = renderer.addMesh(red);
  MeshHandle const greenHandle = renderer.addMesh(green);
  MeshHandle const bothHandle = renderer.addMesh(both);
  if (redHandle == kInvalidMesh || greenHandle == kInvalidMesh || bothHandle == kInvalidMesh) {
    std::printf("SMOKE FAIL: mesh registration\n");
    return 1;
  }

  Camera camera; // default yaw/pitch: looks down -Z at the quads
  camera.position = {0.0F, 0.0F, 2.0F};

  auto blendItem = [](MeshHandle mesh) {
    DrawItem item;
    item.mesh = mesh;
    item.alpha = AlphaMode::Blend;
    item.alphaFactor = 0.5F;
    item.lit = false;
    return item;
  };
  std::vector<RendererVertex> const noLines;
  std::vector<std::uint8_t> correct;
  std::vector<std::uint8_t> wrongItems;
  std::vector<std::uint8_t> wrongTriangles;
  auto render = [&](std::vector<DrawItem> const& items, std::vector<std::uint8_t>& pixels) {
    return renderer.draw(camera, items, noLines, noLines) && renderer.readback(pixels);
  };
  if (!render({blendItem(redHandle), blendItem(greenHandle)}, correct) ||
      !render({blendItem(greenHandle), blendItem(redHandle)}, wrongItems) ||
      !render({blendItem(bothHandle)}, wrongTriangles)) {
    std::printf("SMOKE FAIL: draw/readback\n");
    return 1;
  }

  bool const itemInvariant = correct == wrongItems;
  bool const triangleInvariant = correct == wrongTriangles;
  int const centre = (config.height / 2) * config.width + config.width / 2;
  int const er = correct[static_cast<std::size_t>(centre) * 4U];
  int const eg = correct[static_cast<std::size_t>(centre) * 4U + 1U];
  int const eb = correct[static_cast<std::size_t>(centre) * 4U + 2U];
  // Analytic compositing at the centre: green 0.5 over (red 0.5 over black)
  // = (0.25, 0.5, 0) -> about (64, 128, 0) in 8-bit.
  bool const analytic = std::abs(er - 64) <= 8 && std::abs(eg - 128) <= 8 && eb <= 8;
  writePpm(".build/smoke-blend.ppm", correct, config.width, config.height);
  std::printf(
      "SMOKE blend: centre=(%d,%d,%d) item-order-invariant=%s triangle-order-invariant=%s\n", er,
      eg, eb, itemInvariant ? "yes" : "no", triangleInvariant ? "yes" : "no");
  if (!itemInvariant || !triangleInvariant || !analytic) {
    std::printf("SMOKE FAIL (blend ordering)\n");
    return 1;
  }
  std::printf("SMOKE PASS (blend ordering) on %s\n", renderer.deviceName());
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
// Hand-only editing: capture world matrices of the topmost selected nodes.
// Children ride along once; converting through the current parent keeps part
// selections correct without applying a hierarchy transform twice.
// Only selected subtrees can supply contact. Built-in cubes really are boxes;
// imported meshes always use their triangles, never their bounding box.
bool selectedHandContact(App& app, XrFrameInput const& input, Mat4 const& rigWorld) {
  constexpr int joints[] = {0, 5, 10, 15, 20, 25}; // palm and fingertip joints
  constexpr float distance = .035F;
  static auto const cube = [] {
    auto g = std::make_shared<PickGeometry>();
    g->positions = {{-.5F,-.5F,-.5F},{.5F,-.5F,-.5F},{.5F,.5F,-.5F},{-.5F,.5F,-.5F},
                    {-.5F,-.5F,.5F},{.5F,-.5F,.5F},{.5F,.5F,.5F},{-.5F,.5F,.5F}};
    g->indices = {0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
                  3,2,6,3,6,7,0,3,7,0,7,4,1,5,6,1,6,2};
    return std::make_shared<HandContactMesh>(g);
  }();
  for (auto const& visual : app.visuals) {
    auto node = visual.id;
    while (node != kInvalidNode && std::find(app.selection.begin(), app.selection.end(), node) == app.selection.end()) {
      node = app.scene.parent(node);
    }
    if (node == kInvalidNode) { continue; }
    auto const world = app.scene.world(visual.id);
    // Reject distant nodes before building or traversing their BVH.
    Mat4 inverse;
    if (!invertAffine(world, inverse)) { continue; }
    float norm2 = 0;
    for (int col = 0; col < 3; ++col) {
      for (int row = 0; row < 3; ++row) { norm2 += inverse.m[col*4+row]*inverse.m[col*4+row]; }
    }
    if (!std::isfinite(norm2)) { continue; }
    std::shared_ptr<const HandContactMesh> mesh;
    for (int joint : joints) {
      if (!input.handJointValid[1][joint]) { continue; }
      auto const point = transformPoint(rigWorld, input.handJoints[1][joint]);
      if (!finiteHandPoint(point)) { continue; }
      auto const local = transformPoint(inverse, point);
      if (!finiteHandPoint(local) || handBoxDistanceSquared(local, visual.localBounds) > distance*distance*norm2) { continue; }
      if (!mesh) {
        if (visual.pickGeometry) {
          mesh = app.handContactCache.get(visual.pickGeometry);
        } else {
          mesh = cube;
        }
      }
      if (mesh->near(point, world, distance)) { return true; }
    }
  }
  return false;
}

void beginHandGrab(App& app, HandPose pose, std::optional<Vec3> point = std::nullopt, HandGrabView view = {}) {
  Aabb bounds;
  if (!pose.valid || !movingSetBounds(app.scene, app.visuals, app.selection, bounds)) {
    return;
  }
  endEdits(app);
  app.grabStart = pose;
  app.grabFiltered = pose;
  app.handScaling = false;
  app.grabPivot = point.value_or((bounds.min + bounds.max) * 0.5F);
  app.grabView = view;
  app.grabWorlds.clear();
  app.grabLocals.clear();
  app.grabPoseStarts.clear();
  for (auto id : app.selection) {
    bool child = false;
    for (auto parent = app.scene.parent(id); parent != kInvalidNode; parent = app.scene.parent(parent)) {
      if (std::find(app.selection.begin(), app.selection.end(), parent) != app.selection.end()) {
        child = true;
        break;
      }
    }
    if (!child) {
      app.grabWorlds.emplace_back(id, app.scene.world(id));
      app.grabLocals.emplace_back(id, app.scene.local(id));
      app.grabPoseStarts.emplace_back(id, app.scene.poseOffset(id));
    }
  }
  app.undo.begin(snapshot(app));
  app.handGrab = true;
}

void applyHandDelta(App& app, Mat4 const& delta) {
  bool identity = true;
  Mat4 const unit = Mat4{};
  for (int i = 0; i < 16; ++i) {
    if (std::fabs(delta.m[i] - unit.m[i]) > 1e-7F) { identity = false; break; }
  }
  if (identity) {
    for (auto const& [id, local] : app.grabLocals) { app.scene.setLocal(id, local); }
    return;
  }

  // A rotated, non-uniformly scaled parent can make the requested local
  // matrix sheared. TRS cannot express it: reject the whole sample rather
  // than silently distort one selected subtree.
  std::vector<std::pair<NodeId, Transform>> updates;
  for (std::size_t i = 0; i < app.grabWorlds.size(); ++i) {
    auto const& [id, initial] = app.grabWorlds[i];
    Mat4 inverseStartPose;
    if (!invertAffine(app.grabPoseStarts[i].second, inverseStartPose)) { return; }
    auto const animatedBaseline = mul(initial, mul(inverseStartPose, app.scene.poseOffset(id)));
    auto world = mul(delta, animatedBaseline);


    auto parent = app.scene.parent(id);
    if (parent != kInvalidNode) {
      Mat4 parentInverse;
      if (!invertAffine(app.scene.world(parent), parentInverse)) { return; }
      world = mul(parentInverse, world);
    }
    Mat4 poseInverse;
    if (!invertAffine(app.scene.poseOffset(id), poseInverse)) { return; }
    world = mul(world, poseInverse);
    auto const local = toTransform(world);

    auto const reconstructed = toMatrix(local);
    for (int i = 0; i < 16; ++i) {
      float const tolerance = 1e-4F * std::fmax(1.0F, std::fabs(world.m[i]));
      if (!std::isfinite(reconstructed.m[i]) || std::fabs(reconstructed.m[i] - world.m[i]) > tolerance) {
        return;
      }
    }
    updates.emplace_back(id, local);
  }
  for (auto const& [id, local] : updates) { app.scene.setLocal(id, local); }
}

void beginHandScale(App& app, HandPose left, HandPose right) {
  float const distance = length(left.position - right.position);
  if (distance < .04F) { return; }
  app.scaleLeftFiltered = left;
  app.grabFiltered = right;
  app.scaleStartDistance = distance;
  for (auto& [id, world] : app.grabWorlds) { world = app.scene.world(id); }
  for (auto& [id, local] : app.grabLocals) { local = app.scene.local(id); }
  for (auto& [id, pose] : app.grabPoseStarts) { pose = app.scene.poseOffset(id); }
  Aabb bounds;
  if (!movingSetBounds(app.scene, app.visuals, app.selection, bounds)) { return; }
  app.grabPivot = (bounds.min + bounds.max) * .5F;
  app.handScaling = true;
}

void updateHandScale(App& app, HandPose left, HandPose right, float dt) {
  app.scaleLeftFiltered = smoothHandPose(app.scaleLeftFiltered, left, dt);
  app.grabFiltered = smoothHandPose(app.grabFiltered, right, dt);
  float const ratio = handScaleRatio(app.scaleStartDistance,
      length(app.scaleLeftFiltered.position - app.grabFiltered.position));
  auto const delta = mul(translationMatrix(app.grabPivot),
      mul(scaleMatrix({ratio, ratio, ratio}), translationMatrix(-app.grabPivot)));
  applyHandDelta(app, delta);
}

void updateHandGrab(App& app, HandPose pose) {
  applyHandDelta(app, app.physicalGrab
      ? handGrabTransform(app.grabStart, pose, app.grabStart.position, Mat4{})
      : projectedHandGrabTransform(app.grabView, app.grabStart, pose, app.grabPivot, Mat4{}));
}

// Freeze a live controller edit while choosing a tool. Its snapshot stays
// pending so UNDO/REDO can cancel it without destroying an existing redo branch.
void prepareHandPanel(App& app) {
  if (app.handGrab || app.handTool >= 0) {
    endEdits(app);
  }
  app.stickArmed = false;
}

// History buttons must leave a pending edit intact for the shared cancellation
// path; committing here would discard an existing redo branch.
void prepareHandTool(App& app, int button) {
  if (button != 8 && button != 9) {
    endEdits(app);
  }
}

constexpr std::array<char const*, 12> kHandTools{
    "Models", "", "", "", "", "", "", "", "Undo", "Redo", "Delete", "Backdrop"};

void drawHandPanel(App const& app, std::vector<RendererVertex>& lines) {
  if (!app.handPanel.visible || app.galleryActive) {
    return;
  }
  auto const& panel = app.handPanel;
  constexpr int buttons[] = {0, 8, 9, 10, 11};
  auto point = [&](Vec3 local) { return panel.origin + rotate(panel.orientation, local); };
  for (int button : buttons) {
    auto const centre = handPanelButtonCentre(button);
    Vec3 const colour = button == app.handTool ? Vec3{1.0F, 0.55F, 0.25F} :
        button == app.handHover ? Vec3{1.0F, 0.9F, 0.4F} : Vec3{0.6F, 0.9F, 1.0F};
    Vec3 const corners[4] = {{centre.x-.105F, centre.y-.028F, 0},
        {centre.x+.105F, centre.y-.028F, 0}, {centre.x+.105F, centre.y+.028F, 0},
        {centre.x-.105F, centre.y+.028F, 0}};
    for (int edge = 0; edge < 4; ++edge) {
      lines.push_back({point(corners[edge]), {0,1,0}, colour, {0,0}});
      lines.push_back({point(corners[(edge+1)%4]), {0,1,0}, colour, {0,0}});
    }
    constexpr float height = .018F;
    auto at = centre - Vec3{.5F * textLineWidth(kHandTools[button], height), height*.5F, 0};
    appendTextLines(lines, kHandTools[button], trs(point(at), panel.orientation, {1,1,1}), height, colour);
  }
}

bool smokeHandEditing(App& app, VkRenderer& renderer) {
  auto check = [](bool passed, char const* what) {
    if (!passed) {
      std::printf("SMOKE FAIL: hand editing %s\n", what);
    }
    return passed;
  };
  auto nearVector = [](Vec3 a, Vec3 b) { return length(a - b) < 2e-4F; };
  auto nearMatrix = [](Mat4 const& a, Mat4 const& b) {
    for (int i = 0; i < 16; ++i) {
      if (std::fabs(a.m[i] - b.m[i]) > 2e-4F) {
        return false;
      }
    }
    return true;
  };
  buildDemoScene(app); // isolated fixture; renderer mesh 0 is the built-in cube
  auto const cube = app.visuals.front().id;
  app.selection = {cube};
  auto const before = snapshot(app);
  auto const initialWorld = app.scene.world(cube);
  HandPose const start{{.4F, 1.2F, -.3F}, quatFromAxisAngle({1, 2, -1}, .65F), true};
  auto const turn = quatFromAxisAngle({-1, 2, 3}, .9F);
  Vec3 const displacement{.3F, .4F, -.2F};
  HandPose const current{start.position + displacement, normalized(turn * start.orientation), true};
  // Physical contact is exact, selected-subtree-only, and independent of aim.
  XrFrameInput contactInput;
  contactInput.handJointValid[1][5] = true;
  auto const cubeWorld = app.scene.world(cube);
  contactInput.handJoints[1][5] = transformPoint(cubeWorld, {.5F, 0, 0});
  if (!check(selectedHandContact(app, contactInput, Mat4{}), "selected cube surface contact")) { return false; }
  contactInput.handJoints[1][5] = transformPoint(cubeWorld, {0, 0, 0});
  if (!check(!selectedHandContact(app, contactInput, Mat4{}), "cube interior is not contact")) { return false; }
  contactInput.handJoints[1][5] = transformPoint(cubeWorld, {.5F, 0, 0});
  app.selection.clear();
  if (!check(!selectedHandContact(app, contactInput, Mat4{}), "unselected cube cannot be grabbed")) { return false; }
  app.selection = {cube};
  beginHandGrab(app, start);
  app.physicalGrab = true;
  auto const physicalInitial = app.scene.world(cube);
  updateHandGrab(app, current);
  if (!check(nearMatrix(app.scene.world(cube), handGrabTransform(start, current, start.position, physicalInitial)),
             "physical grab anchored to palm")) { return false; }
  cancelLiveEdit(app);
  if (!check(sameSnapshot(snapshot(app), before), "physical grab cancel restores state")) { return false; }
  beginHandGrab(app, start);
  auto const pivot = app.grabPivot;
  updateHandGrab(app, start);
  if (!check(sameSnapshot(snapshot(app), before), "stationary grab preserves exact state")) { return false; }

  if (!check(app.handGrab && app.grabWorlds.size() == 1, "begin captures cube")) {
    return false;
  }
  updateHandGrab(app, current);
  auto const movedWorld = app.scene.world(cube);
  if (!check(nearVector(transformPoint(movedWorld, {}), pivot + displacement),
             "bounds pivot follows palm displacement") ||
      !check(nearVector(transformVector(movedWorld, {1, 0, 0}),
                        rotate(turn, transformVector(initialWorld, {1, 0, 0}))) &&
             nearVector(transformVector(movedWorld, {0, 1, 0}),
                        rotate(turn, transformVector(initialWorld, {0, 1, 0}))),
             "full rotation relative to nonidentity palm start")) {
    return false;
  }
  for (int repeat = 0; repeat < 32; ++repeat) {
    updateHandGrab(app, current);
  }
  if (!check(nearMatrix(app.scene.world(cube), movedWorld), "repeated update has no drift")) {
    return false;
  }
  if (!check(endEdits(app) && !app.handGrab, "release commits") ||
      !check(!endEdits(app), "release does not commit twice")) {
    return false;
  }
  auto const after = snapshot(app);
  auto state = snapshot(app);
  if (!check(app.undo.undo(state), "undo exists")) {
    return false;
  }
  applySnapshot(app, state);
  if (!check(sameSnapshot(snapshot(app), before) && !app.undo.canUndo(), "exactly one undo restores start") ||
      !check(app.undo.redo(state), "redo exists")) {
    return false;
  }
  applySnapshot(app, state);
  if (!check(sameSnapshot(snapshot(app), after) && !app.undo.canRedo(), "redo restores grab")) {
    return false;
  }

  // Motion changes editable placement without cancelling live node animation.
  app.scene.setPoseOffset(cube, translationMatrix({.3F,0,0}));
  auto const posedBefore = snapshot(app);
  auto const animatedBase = transformPoint(mul(toMatrix(app.scene.local(cube)), translationMatrix({.6F,0,0})), {});
  beginHandGrab(app, start);
  app.scene.setPoseOffset(cube, translationMatrix({.6F,0,0}));
  HandPose translated = start;
  translated.position.x += .2F;
  updateHandGrab(app, translated);
  if (!check(nearVector(app.scene.worldPosition(cube), animatedBase + Vec3{.2F,0,0}), "grab keeps live animation") ||
      !check(endEdits(app), "animated grab commits")) { return false; }
  applySnapshot(app, posedBefore);
  app.scene.setPoseOffset(cube, Mat4{});
  app.undo.clear();

  // Tiny initial separations must not overwrite the right-hand grab filter.
  beginHandGrab(app, start);
  auto const capturedFilter = app.grabFiltered;
  auto closeLeft = start;
  closeLeft.position.x += .01F;
  beginHandScale(app, closeLeft, start);
  if (!check(!app.handScaling && nearVector(app.grabFiltered.position, capturedFilter.position),
             "close scale acquisition preserves grab filter")) { return false; }
  endEdits(app);

  // Two-hand scale is cumulative, stable at rest, and one undo step.

  app.undo.clear();
  auto const scaleBefore = snapshot(app);
  HandPose const left{{-.2F, 1, 0}, {}, true};
  HandPose const right{{.2F, 1, 0}, {}, true};
  beginHandGrab(app, right);
  beginHandScale(app, left, right);
  updateHandScale(app, left, right, .016F);
  if (!check(app.handScaling && sameSnapshot(snapshot(app), scaleBefore), "stationary scale unchanged")) { return false; }
  auto stretched = left;
  stretched.position.x -= .4F;
  for (int i = 0; i < 60; ++i) { updateHandScale(app, stretched, right, .016F); }
  auto const scaleAfter = snapshot(app);
  if (!check(!sameSnapshot(scaleBefore, scaleAfter), "two hand scale changes model") ||
      !check(endEdits(app), "two hand scale commits")) { return false; }
  auto scaleState = snapshot(app);
  if (!check(app.undo.undo(scaleState) && sameSnapshot(scaleState, scaleBefore), "scale undo") ||
      !check(app.undo.redo(scaleState) && sameSnapshot(scaleState, scaleAfter), "scale redo")) { return false; }
  applySnapshot(app, scaleBefore);
  app.undo.clear();

  // Selecting both an ancestor and a child must transform the subtree once.

  auto const parent = app.scene.create();
  app.scene.setLocal(parent, {{.8F, .9F, -.4F}, quatFromAxisAngle({1, 1, 0}, .6F), {1.7F, 1.7F, 1.7F}});
  auto const child = app.scene.create(parent);
  app.scene.setLocal(child, {{.25F, .1F, -.2F}, quatFromAxisAngle({0, 0, 1}, .3F), {.4F, .4F, .4F}});
  addCubeVisual(app, child, {.6F, .8F, 1});
  app.selection = {child, parent}; // child first also exercises ancestor filtering
  app.undo.clear();
  auto const childLocal = app.scene.local(child);
  auto const childWorld = app.scene.world(child);
  beginHandGrab(app, start);
  auto const hierarchyPivot = app.grabPivot;
  auto expectedWorld = [&](Mat4 const& world, Vec3 centre) {
    return mul(translationMatrix(centre + displacement),
               mul(rotationMatrix(turn), mul(translationMatrix(-centre), world)));
  };
  if (!check(app.grabWorlds.size() == 1 && app.grabWorlds.front().first == parent,
             "ancestor and child captured once")) {
    return false;
  }
  updateHandGrab(app, current);
  if (!check(sameTransform(app.scene.local(child), childLocal) &&
             nearMatrix(app.scene.world(child), expectedWorld(childWorld, hierarchyPivot)),
             "ancestor moves child exactly once")) {
    return false;
  }
  cancelLiveEdit(app);
  if (!check(!app.handGrab &&
             nearMatrix(app.scene.world(child), childWorld) && !app.undo.canUndo(),
             "cancel clears grab and restores hierarchy without history")) {
    return false;
  }

  // Parent-only part selection includes the offset child's geometry in its
  // rotation pivot, including when the selected parent has no visual mesh.
  app.selection = {parent};
  Aabb movingBounds;
  if (!check(movingSetBounds(app.scene, app.visuals, app.selection, movingBounds),
             "parent-only moving bounds exist")) {
    return false;
  }
  beginHandGrab(app, start);
  auto const subtreePivot = (movingBounds.min + movingBounds.max) * .5F;
  if (!check(nearVector(app.grabPivot, subtreePivot), "parent-only pivot includes descendants")) {
    return false;
  }
  updateHandGrab(app, current);
  if (!check(nearMatrix(app.scene.world(child), expectedWorld(childWorld, subtreePivot)),
             "parent-only child rotates about moving subtree centre")) {
    return false;
  }
  cancelLiveEdit(app);

  // Child-only editing converts captured world back through the rotated,
  // uniformly scaled parent; the unselected parent must remain unchanged.
  app.selection = {child};
  auto const parentWorld = app.scene.world(parent);
  auto const childBefore = snapshot(app);
  beginHandGrab(app, start);
  auto const childPivot = app.grabPivot;
  updateHandGrab(app, current);
  if (!check(nearMatrix(app.scene.world(parent), parentWorld) &&
             nearMatrix(app.scene.world(child), expectedWorld(childWorld, childPivot)),
             "child-only world delta under rotated scaled parent")) {
    return false;
  }
  updateHandGrab(app, current);
  if (!check(nearMatrix(app.scene.world(child), expectedWorld(childWorld, childPivot)),
             "child-only repeat has no drift")) {
    return false;
  }
  cancelLiveEdit(app);
  if (!check(sameSnapshot(snapshot(app), childBefore), "child-only cancellation restores snapshot")) {
    return false;
  }
  // A sheared local result cannot be stored in TRS; reject without distortion.
  app.scene.setLocal(parent, {{.8F,.9F,-.4F}, quatFromAxisAngle({0,1,0}, .6F), {1,2,3}});
  auto const nonuniformBefore = snapshot(app);
  beginHandGrab(app, start);
  updateHandGrab(app, current);
  if (!check(sameSnapshot(snapshot(app), nonuniformBefore), "nonrepresentable shear rejected")) { return false; }
  endEdits(app);
  app.scene.setLocal(parent, {{.8F, .9F, -.4F}, quatFromAxisAngle({1, 1, 0}, .6F), {1.7F, 1.7F, 1.7F}});
  // At distant pivots, a stationary pose must not create a history entry.
  app.scene.setLocal(parent, {{10000, 10000, 10000}, quatFromAxisAngle({1, 2, 3}, .8F), {1,1,1}});
  auto const distantBefore = snapshot(app);
  beginHandGrab(app, start);
  updateHandGrab(app, start);
  if (!check(sameSnapshot(snapshot(app), distantBefore) && !endEdits(app), "distant stationary grab")) { return false; }
  app.scene.setLocal(parent, {{.8F, .9F, -.4F}, quatFromAxisAngle({1, 1, 0}, .6F), {1.7F, 1.7F, 1.7F}});
  HandPose invalid = current;


  invalid.valid = false;
  beginHandGrab(app, invalid);
  // Tracking-invalid samples are gated out by runXr, not passed to updateHandGrab.
  if (!check(!app.handGrab && sameSnapshot(snapshot(app), childBefore) &&
             nearMatrix(handGrabTransform(start, invalid, childPivot, childWorld), childWorld),
             "invalid tracking cannot start or transform a grab")) {
    return false;
  }

  // History tools cancel rather than commit live stick edits, preserving
  // the pre-existing redo branch for either UNDO or REDO.
  for (int button : {8, 9}) {
    app.undo.clear();
    auto const baseState = snapshot(app);
    app.undo.begin(baseState);
    moveSelectionVertical(app.scene, app.selection, .2F);
    app.undo.commit(snapshot(app));
    auto historyState = snapshot(app);
    app.undo.undo(historyState);
    applySnapshot(app, historyState);
    app.undo.begin(snapshot(app));
    app.stickSession = true;
    moveSelectionVertical(app.scene, app.selection, .1F);
    prepareHandPanel(app);
    if (!check(app.stickSession && app.undo.canRedo(), "panel opening preserves live history")) {
      return false;
    }
    prepareHandTool(app, button);
    if (!check(app.stickSession && app.undo.canRedo(), "history tool preserves live edit for cancellation")) {
      return false;
    }
    cancelLiveEdit(app);
    if (!check(sameSnapshot(snapshot(app), baseState) && app.undo.canRedo() && !app.undo.canUndo(),
               "history tool cancellation preserves redo branch")) {
      return false;
    }
  }

  // Rendering-level regression: VR must never inherit the desktop MOVE HUD.
  {
    App hud;
    std::vector<DrawItem> items;
    std::vector<RendererVertex> depth, overlay;
    assembleFrame(hud, items, depth, overlay);
    if (!check(!overlay.empty(), "desktop retains mode HUD")) { return false; }
    hud.xr = true;
    overlay.clear();
    assembleFrame(hud, items, depth, overlay);
    if (!check(overlay.empty(), "VR has no mode HUD")) { return false; }
    hud.galleryActive = true;
    assembleFrame(hud, items, depth, overlay);
    if (!check(overlay.empty(), "VR carousel has no mode HUD")) { return false; }
  }

  app.handPanel = {{0, .195F, 0}, quatFromAxisAngle({0, 1, 0}, .3F), true};
  auto const normal = rotate(app.handPanel.orientation, {0, 0, 1});
  for (int index : {0, 8, 9, 10, 11}) {
    auto const centre = app.handPanel.origin + rotate(app.handPanel.orientation, handPanelButtonCentre(index));
    if (!check(app.handPanel.hit({centre + normal, -normal}) == index,
               "panel button centre hit")) {
      return false;
    }
  }
  if (!check(app.handPanel.hit({app.handPanel.origin + normal, -normal}) == -1 &&
             app.handPanel.hit({app.handPanel.origin - normal, normal}) == -1,
             "panel gap and back face miss")) {
    return false;
  }
  app.handHover = 8;
  std::vector<RendererVertex> lines;
  drawHandPanel(app, lines);
  if (!check(!lines.empty() && lines.size() % 2 == 0, "panel produces line pairs")) {
    return false;
  }
  Camera camera;
  auto const centre = app.handPanel.origin + rotate(app.handPanel.orientation, {0, -.0975F, 0});
  camera.position = centre + normal * .85F;
  camera.yaw = -.3F;
  camera.pitch = 0;
  if (!check(renderer.draw(camera, {}, {}, {}), "panel blank draw")) {
    return false;
  }
  std::vector<std::uint8_t> blank;
  if (!check(renderer.readback(blank), "panel blank readback")) {
    return false;
  }
  if (!check(renderer.draw(camera, {}, {}, lines), "panel draw")) {
    return false;
  }
  std::vector<std::uint8_t> pixels;
  if (!check(renderer.readback(pixels), "panel readback succeeds")) {
    return false;
  }
  if (!check(!pixels.empty() && pixels.size() == blank.size(), "panel readback")) {
    return false;
  }
  std::size_t changed = 0;
  for (std::size_t i = 0; i < pixels.size(); i += 4) {
    if (pixels[i] != blank[i] || pixels[i + 1] != blank[i + 1] || pixels[i + 2] != blank[i + 2]) {
      ++changed;
    }
  }
  if (!check(changed > 300, "panel visible in rendered pixels") ||
      !check(writePpm(".build/hand-tools.ppm", pixels, renderer.width(), renderer.height()), "panel screenshot")) {
    return false;
  }
  app.galleryActive = true;
  lines.clear();
  drawHandPanel(app, lines);
  if (!check(lines.empty(), "carousel has no tool panel or browse buttons")) {
    return false;
  }
  app.galleryActive = false;
  app.handPanel.visible = false;
  lines.clear();
  drawHandPanel(app, lines);
  if (!check(lines.empty() && app.handPanel.hit({centre + normal, -normal}) == -1,
             "hidden panel neither draws nor hits")) {
    return false;
  }
  std::printf("SMOKE: hand editing grab/hierarchy/undo/cancel/panel checks passed (%zu panel pixels; .build/hand-tools.ppm)\n", changed);
  return true;
}

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
  app.xr = true;
  std::printf("[xr] Genesis: point-and-pinch carousel; palm-up left-pinch menu\n");
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
  bool handMenuOpen = false;
  HandGesture leftPinch;
  HandGesture rightPinch;
  HandGesture rightGrasp;
  HandGraspGate graspGate;
  bool palmUp = false;
  bool leftUiOwned = false;
  int jointSignature = -1;
  bool rightWasHand = false;
  HandSelectionLatch selectionLatch;
  NodeId recentHandNode = kInvalidNode;
  Vec3 recentHandLocalPoint{};
  int skipsInARow = 0;
  double frameMsTotal = 0.0;
  double frameMsMax = 0.0;
  std::chrono::steady_clock::time_point frameStart = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point lastFrameTime = frameStart;
  while (running) {
    app.handContactCache.prune();
    XrFrameInput input;
    frameStart = std::chrono::steady_clock::now();
    XrShell::FrameStatus status = shell.beginFrame(input);
    if (status == XrShell::FrameStatus::Exit) {
      break;
    }
    if (status == XrShell::FrameStatus::Pump || status == XrShell::FrameStatus::Skipped) {
      // Rendering must not hide a tracking interruption. End the edit and
      // require a released pinch before reacquisition can start another one.
      if (app.handGrab || app.handTool >= 0) {
        endEdits(app);
      }
      selectionLatch.reset();
      leftPinch.reset();
      rightPinch.reset();
      rightGrasp.reset();
      graspGate.reset();
      app.handUiOwned = true;
      app.handPanel.visible = false;
      palmUp = false;
      leftUiOwned = true;
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
    selectionLatch.advance(dt); // use elapsed time, not the animation timestep clamp
    dt = dt > 0.1F ? 0.1F : dt;

    tickAnimations(app, dt);

    // No flight controls: passthrough means you walk in the real world with
    // the models placed in it — the sticks' flight/turn bindings were removed
    // at the user's request ("I can now move in the real world"). Grip is the
    // fine-control modifier.

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

    // Only the hand profile can drive gestures: controller joints may be inferred.
    HandPose worldHands[2] = {input.hands[0], input.hands[1]};
    for (auto& hand : worldHands) {
      hand.position = transformPoint(rigWorld, hand.position);
      hand.orientation = normalized(cameraOrientation(rig) * hand.orientation);
    }
    int const jointsNow = (worldHands[0].valid ? 1 : 0) + (worldHands[1].valid ? 2 : 0);
    if (jointsNow != jointSignature) {
      std::printf("[hands] palm tracking: left %s, right %s\n",
          worldHands[0].valid ? "tracked" : "untracked", worldHands[1].valid ? "tracked" : "untracked");
      jointSignature = jointsNow;
    }
    bool const handRight = input.handActive[1];
    bool const handSignal = handPinchSignal(handRight, input.pinchActive[1], input.pinch[1]);
    bool const leftSignal = handPinchSignal(input.handActive[0], input.pinchActive[0], input.pinch[0]) &&
        worldHands[0].valid;
    // Aim is needed to target, not to sustain a tracked pinch or palm grab.
    bool const leftPressed = leftPinch.update(leftSignal, input.pinch[0]);
    bool const rightPressed = rightPinch.update(handSignal, input.pinch[1]);
    bool const handHeld = handPinchHeld(handSignal, input.pinch[1]);
    bool const graspSignal = handPinchSignal(handRight, input.graspActive, input.grasp) && worldHands[1].valid;
    rightGrasp.update(graspSignal, input.grasp);
    bool const graspHeld = handPinchHeld(graspSignal, input.grasp);
    bool const leftHeld = handPinchHeld(leftSignal, input.pinch[0]);
    graspGate.update(graspSignal, graspHeld, handHeld,
                     app.handUiOwned || app.galleryActive || handMenuOpen || app.stickSession);
    if (app.physicalGrab) {
      rightPinch.reset(); // a pinch made during a physical grab cannot click on release
    }
    if (!worldHands[1].valid || !handRight) { selectionLatch.reset(); }
    if (!leftHeld) { leftUiOwned = false; }
    if (handRight || rightWasHand) {
      input.selectPressed = handRight && rightPressed && !app.physicalGrab;
      input.selectHeld = handHeld;
      input.selectReleased = !handHeld;
    }
    if (rightWasHand != handRight || app.galleryActive) {
      selectionLatch.reset();
      graspGate.reset();
      rightGrasp.reset();
    }
    rightWasHand = handRight;
    bool handCarouselToggled = false;
    palmUp = input.handActive[0] && palmFacingUp(worldHands[0], palmUp);
    if (leftPressed && palmUp && !handHeld && !app.handGrab) {
      selectionLatch.reset();
      graspGate.reset();
      rightGrasp.reset();
      leftUiOwned = true;
      handMenuOpen = !handMenuOpen;
      if (handMenuOpen) { prepareHandPanel(app); }
      else { endEdits(app); }
      app.handUiOwned = true;
      if (!handMenuOpen) { galleryClose(app); }
      handCarouselToggled = true;
      std::printf("[hands] menu %s\n", handMenuOpen ? "open" : "closed");
    }
    // Latch the stationary carousel anchor on its opening frame.
    if (app.galleryActive && !app.ringLatched) {
      galleryTick(app, renderer, 0.0F, headWorld, headForward);
    }
    bool const panelWanted = handRight && handMenuOpen && !app.galleryActive;
    if (panelWanted && (!app.handPanel.visible || app.handPanelCarousel != app.galleryActive)) {
      prepareHandPanel(app);
      app.handPanelCarousel = app.galleryActive;
      app.handPanel.orientation = quatFromAxisAngle({0,1,0}, std::atan2(-headForward.x, -headForward.z));
      app.handPanel.origin = headWorld + headForward * .65F + Vec3{0,.08F,0};

      app.handUiOwned = handHeld;
      std::printf("[hands] %s panel open\n", app.galleryActive ? "carousel" : "tools");
    }
    if (!panelWanted && app.handPanel.visible) {
      endEdits(app);
    }
    app.handPanel.visible = panelWanted;
    app.handHover = handSignal && input.aimValid[1] ? app.handPanel.hit(worldRay) : -1;
    if (app.handHover >= 1 && app.handHover <= 7) {
      app.handHover = -1;
    }
    if (!handHeld) {
      if ((app.handGrab && !app.physicalGrab) || app.handTool >= 0) {
        endEdits(app);
      }
      app.handUiOwned = false;
    }
    if (rightPressed && !app.physicalGrab && app.handHover >= 0 && !handCarouselToggled) {
      int const button = app.handHover;
      prepareHandTool(app, button);
      app.handUiOwned = true;
      std::printf("[hands] tool: %s\n", kHandTools[button]);
      if (button == 0) {
        galleryOpen(app, renderer);
      } else if (button == 8) {
        input.undoPressed = true;
      } else if (button == 9) {
        input.redoPressed = true;
      } else if (button == 10 && !app.selection.empty()) {
        app.handTool = button;
        app.deleteHolding = false;
      } else if (button == 11) {
        toggleBackdrop(app, renderer, &shell);
      }
    } else if (rightPressed && !app.physicalGrab && app.handPanel.visible && !app.galleryActive) {
      app.handUiOwned = true; // tool panel owns misses too; do not click through
    }
    if (app.handGrab) {
      if ((app.physicalGrab ? graspHeld : handHeld) && worldHands[1].valid && !app.handUiOwned && !panelWanted && !app.galleryActive) {
        if (app.physicalGrab) {
          updateHandGrab(app, worldHands[1]); // unfiltered, 1:1 palm pose
        } else if (app.handScaling) {
          if (leftHeld && worldHands[0].valid) {
            updateHandScale(app, worldHands[0], worldHands[1], dt);
          } else {
            endEdits(app);
            app.handUiOwned = true; // right must release before another edit
          }
        } else if (leftHeld && !leftUiOwned && leftPinch.pressed() && worldHands[0].valid) {
          beginHandScale(app, worldHands[0], worldHands[1]);
          if (!app.handScaling) {
            app.grabFiltered = smoothHandPose(app.grabFiltered, worldHands[1], dt);
            updateHandGrab(app, app.grabFiltered);
          }
        } else {
          app.grabFiltered = smoothHandPose(app.grabFiltered, worldHands[1], dt);
          updateHandGrab(app, app.grabFiltered);
        }
      } else {
        bool const wasPhysical = app.physicalGrab;
        endEdits(app);
        if (wasPhysical) { graspGate.reset(); rightGrasp.reset(); }
        // A lost palm is a tracking interruption, not a release. Require a
        // real release before this held pinch can start a new interaction.
        if (handHeld && !worldHands[1].valid) {
          rightPinch.reset();
          app.handUiOwned = true;
        }
      }
    }

    if (handRight && (app.handUiOwned || app.physicalGrab)) {
      input.selectPressed = false;
    }

    // The left controller's MENU carries two controls (the user's binding):
    // TAP = the model carousel (open/close, the same rule as A), HOLD (0.4 s)
    // = backdrop toggle. The tap fires on release so a hold can be told apart
    // from a tap (the hold cancels the tap); thumbstick clicks do NOTHING.
    // The tap used to switch MOVE/ROTATE — that comes back as gestures.
    bool carouselToggled = handCarouselToggled;
    if (input.menuPressed || (input.menuHeld && !app.menuPrevHeld)) {
      // The app-side rising edge backs up the shell's edge: a press swallowed
      // by a skipped frame must still initialise the tap/hold state, or a
      // later hold resumes from a stale charge.
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
      // Same open/close rule as A (galleryOpen ends any live stick gesture).
      if (app.galleryActive) {
        galleryClose(app);
      } else {
        galleryOpen(app, renderer);
      }
      // The stick that was cycling the ring is still deflected: it must
      // re-centre before it can start a scale/height gesture on the model the
      // close just revealed (this frame's stick block runs after this one).
      app.stickArmed = false;
      carouselToggled = true;
    }
    bool cancelledLiveEdit = false;
    if (input.undoPressed || input.redoPressed) { selectionLatch.reset(); graspGate.reset(); rightGrasp.reset(); }
    if (input.undoPressed) {
      if (app.stickSession || app.handGrab || app.handTool >= 0 || app.galleryActive) {
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
      if (app.stickSession || app.handGrab || app.handTool >= 0 || app.galleryActive) {
        cancelLiveEdit(app);
      } else {
        vredit::EditorSnapshot state = snapshot(app);
        if (app.undo.redo(state)) {
          applySnapshot(app, state);
          app.stickArmed = false; // sticks must re-centre before a new gesture
        }
      }
    }

    // The right controller's aim ray drives the same state machine as the
    // mouse. A body press only SELECTS —
    // movement and rotation are the sticks' job (controller aim is too
    // jittery for accurate placement).
    app.camera.position = headWorld;
    // Cache only an actual closest selected surface while released. A current
    // hit on another object always wins over the grace; never use gaze fallback.
    PointerTarget handTarget;
    bool const handCanTarget = handSignal && input.aimValid[1];
    bool const sceneHandTargeting = handCanTarget && !app.handUiOwned &&
        !panelWanted && !app.galleryActive && !app.stickSession;
    if (app.handUiOwned || panelWanted || app.galleryActive || app.stickSession) {
      selectionLatch.clearHit();
    }
    if (sceneHandTargeting && selectionLatch.primed() && !handHeld) {
      handTarget = resolvePointerTarget(app, worldRay);
      bool const selected = handTarget.node != kInvalidNode &&
          std::find(app.selection.begin(), app.selection.end(), handTarget.node) != app.selection.end();
      if (selected) {
        Mat4 inverse{};
        if (invertAffine(app.scene.world(handTarget.node), inverse)) {
          recentHandNode = handTarget.node;
          recentHandLocalPoint = transformPoint(inverse, handTarget.point);
          selectionLatch.rememberHit();
        } else {
          selectionLatch.clearHit();
        }
      } else if (handTarget.node != kInvalidNode && !selected) {
        selectionLatch.clearHit();
      }
    }
    // Physical contact wins an overlapping pinch, but an unprimed first
    // selection never becomes an edit. A held, armed squeeze may enter contact.
    if (graspGate.canAcquire(rightGrasp.pressed(), selectionLatch.primed()) &&
        !app.physicalGrab && !app.handUiOwned && !panelWanted && !handMenuOpen &&
        !app.galleryActive && !app.stickSession && !handCarouselToggled &&
        !input.undoPressed && !input.redoPressed && !app.selection.empty() &&
        selectedHandContact(app, input, rigWorld)) {
      if (app.handGrab) {
        // Upgrade the projected pinch in place: keep its original undo
        // transaction, but rebase the physical motion at the current pose.
        app.grabStart = worldHands[1];
        app.grabFiltered = worldHands[1];
        app.handScaling = false;
        for (auto& [id, world] : app.grabWorlds) { world = app.scene.world(id); }
        for (auto& [id, local] : app.grabLocals) { local = app.scene.local(id); }
        for (auto& [id, pose] : app.grabPoseStarts) { pose = app.scene.poseOffset(id); }
      } else {
        beginHandGrab(app, worldHands[1]);
      }
      app.physicalGrab = app.handGrab;
      if (app.physicalGrab) { input.selectPressed = false; rightPinch.reset(); }
    }
    if (input.selectPressed && !handCarouselToggled) {
      if (app.galleryActive) {
        // A hand without a valid aim must never act on the gaze fallback.
        if (!handRight || handCanTarget) { galleryPointerPress(app, renderer, worldRay); }
        graspGate.reset();
        rightGrasp.reset();
        if (!app.galleryActive) { handMenuOpen = false; }
        if (handRight) { app.handUiOwned = true; }
      } else {
        bool const partSelect = !handRight && input.gripHeld > 0.5F;
        PointerTarget target = handRight && !handCanTarget ? PointerTarget{} :
            resolvePointerTarget(app, worldRay);
        bool const selectedHit = target.node != kInvalidNode &&
            std::find(app.selection.begin(), app.selection.end(), target.node) != app.selection.end();
        bool const recentSelected = handRight && target.node == kInvalidNode &&
            selectionLatch.recentHit() &&
            std::find(app.selection.begin(), app.selection.end(), recentHandNode) != app.selection.end();
        if (handSignal && worldHands[1].valid &&
            selectionLatch.startsGrab(selectedHit, recentSelected)) {
          Vec3 const point = selectedHit ? target.point :
              transformPoint(app.scene.world(recentHandNode), recentHandLocalPoint);
          beginHandGrab(app, worldHands[1], point, {headWorld, normalized(cameraOrientation(rig) * input.headOrient), true});
        } else if (!handRight || handCanTarget) {
          pointerPress(app, target, partSelect);
          graspGate.reset();
          rightGrasp.reset();
          selectionLatch.selected(handSignal && !app.selection.empty());
        }
      }
    }

    // Stick-driven editing (the coloured handles are gone): the LEFT stick
    // slides the selection across the floor relative to the head's yaw
    // (pushing forward walks it away from you) in BOTH modes; the RIGHT stick
    // is mode-driven (move: X uniform scale, Y height; rotate: X spins about
    // the scene's up axis, Y tips). One undo step per gesture.
    //
    // The gesture's frame is latched at start (pivot, the pre-signed viewer
    // axes, fine rate): turning your head or toggling grip mid-gesture must
    // not steer the rotation axes, change the rate mid-step, or walk
    // asymmetric models sideways through a per-frame re-centred pivot.
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
    // The LEFT stick moves (X/Z) in both modes. The RIGHT stick is
    // mode-driven (desktop 1/2 only — VR has no switch until the gestures
    // land): MOVE = X uniform scale, Y height;
    // ROTATE = X spins about the scene's up axis, Y tips the top away ("the
    // model should rotate in the direction of the stick").
    bool const sticksActive = std::fabs(input.rightStick.x) > stickThreshold ||
                              std::fabs(input.rightStick.y) > stickThreshold ||
                              std::fabs(input.leftStick.x) > stickThreshold ||
                              std::fabs(input.leftStick.y) > stickThreshold;
    bool const gestureAlive = sticksActive;
    // While the carousel is open the sticks are ITS: the right stick's X
    // cycles the staged model (a flick = one step; held = one step per
    // 0.25 s). All editing stays blocked (the session gates below).
    galleryTick(app, renderer, dt, headWorld, headForward);
    if (app.galleryActive) {
      app.galleryCycleCooldown -= dt;
      if (std::fabs(input.rightStick.x) > 0.5F && app.galleryCycleCooldown <= 0.0) {
        galleryCycle(app, renderer, input.rightStick.x > 0.0F ? 1 : -1);
        app.galleryCycleCooldown = 0.25;
      }
    }
    // After an undo/redo the sticks must return to neutral before a new
    // gesture may start, or the same frame's drift cancels-and-restarts and
    // swallows the undo.
    if (!sticksActive) {
      app.stickArmed = true;
    }
    if (!panelWanted && sticksActive && !app.stickSession && !app.handGrab && app.handTool < 0 && !app.galleryActive &&
        !app.selection.empty() && app.stickArmed) {
      app.undo.begin(snapshot(app));
      app.stickSession = true;
      app.stickFineRate = input.gripHeld > 0.5F ? 0.25F : 1.0F;
      // VIEWER-RELATIVE axes only ("only ever rotate the model around the
      // axis that stretches to the left and right of the viewer" = tip; the
      // spin is the scene's up axis), latched at gesture start so a live
      // head pose cannot steer a step. The two axes are perpendicular and
      // level with the horizon.
      vredit::viewerRotationAxes(headRightFlat, app.stickYawAxis, app.stickTipAxis);
      vredit::Aabb bounds{};
      if (vredit::movingSetBounds(app.scene, app.visuals, app.selection, bounds)) {
        app.stickPivot = vredit::aabbCentre(bounds);
      } else {
        app.stickPivot = app.scene.worldPosition(app.selection.front());
      }
    }
    if (app.stickSession && !app.galleryActive && !panelWanted) {
      if (gestureAlive && !app.selection.empty()) {
        Vec3 const moveDelta = (headRightFlat * leftX + headForwardFlat * leftY) *
                               (kStickMoveMps * app.stickFineRate * dt);
        // Guard the call, not just the maths: a zero delta would still
        // round-trip every local transform through matrix decompose (float
        // noise, epoch churn, junk in the undo snapshot).
        if (moveDelta.x != 0.0F || moveDelta.y != 0.0F || moveDelta.z != 0.0F) {
          vredit::moveSelectionHorizontal(app.scene, app.selection, moveDelta);
        }
        // RIGHT STICK (mode-driven): MOVE = X uniform scale (right = bigger),
        // Y = height (up = lift). ROTATE = X spins about the scene's up axis
        // (stick right sweeps the face toward your right), Y tips the top
        // away / lifts the face ("the model should rotate in the direction of
        // the stick"). ROTATE MODE ONLY — nothing rotates outside it.
        switch (app.mode) {
        case EditMode::move: {
          if (std::fabs(rightX) > 1e-4F) {
            float const factor =
                std::exp(rightX * kStickScalePerS * app.stickFineRate * dt);
            vredit::scaleSelectionUniform(app.scene, app.visuals, app.selection, factor);
          }
          if (std::fabs(rightY) > 1e-4F) {
            // Explicit height: no floor rule (lifting is deliberate).
            float const deltaY = rightY * kStickHeightMps * app.stickFineRate * dt;
            vredit::moveSelectionVertical(app.scene, app.selection, deltaY);
          }
          break;
        }
        case EditMode::rotate:
        default: {
          // SPIN + TIP (the signs live in the core seams): stick X spins
          // about the scene's vertical — never the model's own up — and a
          // spin never changes height, so no floor rule applies. Stick Y
          // tips the top away / lifts the face (grip = fine).
          if (std::fabs(rightX) > 1e-4F) {
            vredit::stickRotateYaw(app.scene, app.selection, app.stickPivot,
                                   app.stickYawAxis, rightX, kStickTurnRadps,
                                   app.stickFineRate * dt);
          }
          if (vredit::stickRotateTip(app.scene, app.selection, app.stickPivot,
                                     app.stickTipAxis, rightY, kStickTurnRadps,
                                     app.stickFineRate * dt)) {
            // The floor rule is about MODELS ("models stay stuck to the
            // floor"): tipping digs a resting model's corners below it;
            // tipping a PART must not lift it out of its model by flooring
            // through the part's own bounds.
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
    if ((input.deleteHeld || (app.handTool == 10 && handHeld && panelWanted)) &&
        !app.selection.empty() && !app.galleryActive) {
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
    if (input.spawnPressed && !carouselToggled) {
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
    assembleFrame(app, items, depthLines, overlayLines);
    drawHandPanel(app, overlayLines);
    // Two reusable meshes follow current tracked joints and render over models.
    // Do not retain a stale hand when tracking or mesh generation fails.
    for (int hand = 0; hand < 2; ++hand) {
      if (!input.handActive[hand]) { continue; }
      Vec3 joints[kHandJointCount]{};
      for (std::size_t joint = 0; joint < kHandJointCount; ++joint) {
        joints[joint] = transformPoint(rigWorld, input.handJoints[hand][joint]);
      }
      auto const mesh = makeHandMesh(joints, input.handJointValid[hand]);
      if (mesh.indices.empty()) { continue; }
      auto const handle = renderer.updateHandMesh(static_cast<std::size_t>(hand), mesh);
      if (handle == kInvalidMesh) { continue; }
      DrawItem handItem;
      handItem.mesh = handle;
      items.push_back(handItem);
    }
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
    // Render each raw located eye in the same rig as hands and controllers.
    for (int eye = 0; eye < 2; ++eye) {
      Vec3 eyeWorld{};
      Mat4 view{};
      if (!composeStereoEye(rigWorld, input.eyePos[eye], input.eyeOrient[eye], eyeWorld, view)) {
        std::printf("[xr] eye pose is singular\n");
        drew = false;
        break;
      }
      Mat4 viewProj = mul(input.eyeProj[eye], view);
      if (!renderer.drawToView(shell.eyeView(eye), shell.eyeImage(eye), view, viewProj, eyeWorld,
                               items, depthLines, overlayLines)) {
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
    std::printf("[xr] controller sticks (left = move, right = edit): "
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
// move = X uniform scale, Y height; rotate = X spins about the scene's up
// axis, Y tips about the viewer's left-right line. Same
// one-undo-step-per-gesture lifecycle as the sticks: begun on the first
// deflection, committed when the arrows settle.
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
    // VIEWER-RELATIVE axes (same rule as VR): the scene's vertical (spin)
    // and the viewer's left-right line (tip) — perpendicular and level,
    // latched per gesture.
    vredit::viewerRotationAxes(rightFlat, app.stickYawAxis, app.stickTipAxis);
    vredit::Aabb bounds{};
    app.stickPivot = vredit::movingSetBounds(app.scene, app.visuals, app.selection, bounds)
                        ? vredit::aabbCentre(bounds)
                        : app.scene.worldPosition(app.selection.front());
  }
  // Same rules as the VR sticks: X spins about the scene's up axis in rotate
  // mode (a spin never changes height), Y = height in move (tips in rotate)
  // — the model rotates in the direction of the stick.
  if (axisX != 0.0F && app.mode == EditMode::rotate) {
    vredit::stickRotateYaw(app.scene, app.selection, app.stickPivot,
                           app.stickYawAxis, axisX, kStickTurnRadps,
                           app.stickFineRate * dt);
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
      if (vredit::stickRotateTip(app.scene, app.selection, app.stickPivot,
                                 app.stickTipAxis, axisY, kStickTurnRadps,
                                 app.stickFineRate * dt)) {
        if (wholeModelSelection(app)) {
          vredit::dropSelectionToFloor(app.scene, app.visuals, app.selection);
        }
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
      "  arrows edit (mode-driven): X = scale / spin   Y = height / tip (shift: fine)\n"
      "  1 move   2 rotate   N model carousel ([ ] cycle, click places)   Esc quit\n",
      renderer.deviceName());

  auto previous = std::chrono::steady_clock::now();
  while (app.running) {
    app.handContactCache.prune();
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
    tickAnimations(app, dt);
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
    SmokeBlend,
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
    } else if (std::strcmp(arg, "--smoke-blend") == 0) {
      mode = Mode::SmokeBlend;
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
          "                 --smoke-blend | --smoke-perf | --smoke-xr |\n"
          "                 --smoke-window [FRAMES]]\n"
          "keys: WASD/QE fly (RMB look), arrows edit: left/right = X (scale in\n"
          "      move mode, spin in rotate), up/down = Y (height / tip),\n"
          "      shift = fine, 1 move / 2 rotate,\n"
          "      N model carousel ([ ] cycle, click places), Delete remove,\n"
          "      B backdrop, Ctrl-Z/Y undo/redo, shift+click part, Esc quit\n"
          "vr:   trigger select (grip+trigger part), left stick move on the floor,\n"
          "      hands: right pinch = select, hand aim = pointer (no buttons/sticks),\n"
          "      right stick: scale (X) and height (Y), grip = fine,\n"
          "      A or menu tap = model carousel (right stick X cycles,\n"
          "      trigger places), hold X delete, B/Y undo/redo,\n"
          "      menu hold = backdrop\n");
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
  case Mode::SmokeBlend:
    return runSmokeBlend(shaderDir);
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
