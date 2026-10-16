// vr-editor: desktop shell over the scene-editor core — viewport, camera,
// selection, gizmo handles. Typed C++ only; the shell is a thin layer over
// vredit::Scene/GizmoDrag and VkRenderer.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <xcb/xcb.h>

#include "camera.h"
#include "gizmo.h"
#include "gltf.h"
#include "handles.h"
#include "mesh.h"
#include "obj.h"
#include "picking.h"
#include "scene.h"
#include "vk_renderer.h"

using namespace vredit;

namespace {

constexpr float kOrbitSpeed = 0.005F;
constexpr float kGizmoPixels = 120.0F;
constexpr float kPickThresholdPx = 8.0F;
constexpr float kSnapTranslate = 0.25F;
constexpr float kSnapRotate = 0.2617994F; // 15 degrees
constexpr float kSnapScale = 0.25F;

struct NodeVisual {
  NodeId id{kInvalidNode};
  MeshHandle mesh{0};
  Aabb localBounds{{-0.5F, -0.5F, -0.5F}, {0.5F, 0.5F, 0.5F}};
  Vec3 colour{1.0F, 1.0F, 1.0F};
};

struct App {
  Scene scene;
  std::vector<NodeVisual> visuals;
  std::vector<NodeId> selection;
  Camera camera;
  GizmoMode mode{GizmoMode::Translate};
  GizmoSpace space{GizmoSpace::World};
  GizmoDrag drag;
  bool dragging{false};
  bool orbiting{false};
  bool panning{false};
  bool running{true};
  bool snap{false};
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
  NodeId a = app.scene.create();
  Transform ta;
  ta.translation = {1.2F, 0.5F, 0.0F};
  app.scene.setLocal(a, ta);
  addCubeVisual(app, a, {0.85F, 0.25F, 0.20F});

  NodeId b = app.scene.create();
  Transform tb;
  tb.translation = {-1.8F, 0.35F, 1.2F};
  tb.scale = {0.7F, 0.7F, 0.7F};
  app.scene.setLocal(b, tb);
  addCubeVisual(app, b, {0.25F, 0.45F, 0.85F});

  NodeId parent = app.scene.create();
  Transform tp;
  tp.translation = {0.0F, 0.5F, -2.4F};
  tp.rotation = quatFromAxisAngle({0.0F, 1.0F, 0.0F}, 0.7853982F);
  app.scene.setLocal(parent, tp);
  addCubeVisual(app, parent, {0.35F, 0.75F, 0.35F});

  NodeId child = app.scene.create(parent);
  Transform tc;
  tc.translation = {0.0F, 1.4F, 0.0F};
  tc.scale = {0.55F, 0.55F, 0.55F};
  app.scene.setLocal(child, tc);
  addCubeVisual(app, child, {0.85F, 0.75F, 0.25F});
}

// Adds a loaded model's meshes and node hierarchy to the scene and visuals.
bool addModelToScene(App& app, VkRenderer& renderer, Model const& model) {
  std::vector<MeshHandle> meshHandles(model.meshes.size(), 0U);
  for (std::size_t m = 0; m < model.meshes.size(); ++m) {
    meshHandles[m] = renderer.addMesh(model.meshes[m].data);
    if (meshHandles[m] == kInvalidMesh) {
      std::printf("mesh registration failed for mesh %zu\n", m);
      return false;
    }
  }
  std::vector<NodeId> handles(model.nodes.size(), kInvalidNode);
  for (std::size_t n = 0; n < model.nodes.size(); ++n) {
    ModelNode const& node = model.nodes[n];
    NodeId parent = node.parent >= 0 ? handles[static_cast<std::size_t>(node.parent)] : kInvalidNode;
    NodeId id = app.scene.create(parent);
    app.scene.setLocal(id, node.local);
    handles[n] = id;
    if (node.mesh >= 0 && static_cast<std::size_t>(node.mesh) < model.meshes.size()) {
      ModelMesh const& mesh = model.meshes[static_cast<std::size_t>(node.mesh)];
      NodeVisual visual;
      visual.id = id;
      visual.mesh = meshHandles[static_cast<std::size_t>(node.mesh)];
      visual.localBounds = meshBounds(mesh.data);
      visual.colour = mesh.colour;
      app.visuals.push_back(visual);
    }
  }
  return true;
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
  app.camera.target = centre;
  app.camera.distance = std::fmax(2.2F * radius, 1.0F);
  app.camera.pitch = -0.35F;
  app.camera.yaw = 0.6F;
}

Ray appRay(App const& app, Vec2 pixel) {
  Vec2 viewport{static_cast<float>(app.width), static_cast<float>(app.height)};
  return cameraRay(app.camera, viewport.x / viewport.y, pixel, viewport);
}

ProjectFn makeProject(App const& app) {
  Vec2 viewport{static_cast<float>(app.width), static_cast<float>(app.height)};
  Mat4 viewProj = cameraViewProjection(app.camera, viewport.x / viewport.y);
  return [viewProj, viewport](Vec3 const& p) {
    float x = viewProj.m[0] * p.x + viewProj.m[4] * p.y + viewProj.m[8] * p.z + viewProj.m[12];
    float y = viewProj.m[1] * p.x + viewProj.m[5] * p.y + viewProj.m[9] * p.z + viewProj.m[13];
    float w = viewProj.m[3] * p.x + viewProj.m[7] * p.y + viewProj.m[11] * p.z + viewProj.m[15];
    if (w < 1e-6F) {
      // Behind (or on) the camera plane: not projectable. NaN makes the shape
      // unpickable; the rasteriser clips such geometry independently.
      float nan = std::numeric_limits<float>::quiet_NaN();
      return Vec2{nan, nan};
    }
    return Vec2{(x / w + 1.0F) * 0.5F * viewport.x, (y / w + 1.0F) * 0.5F * viewport.y};
  };
}

bool gizmoFrame(App& app, Vec3& pivot, Quat& orientation) {
  if (app.selection.empty()) {
    return false;
  }
  pivot = app.scene.worldPosition(app.selection[0]);
  orientation = Quat{};
  if (app.space == GizmoSpace::Local) {
    Vec3 t;
    Vec3 s;
    decompose(app.scene.world(app.selection[0]), t, orientation, s);
  }
  return true;
}

float gizmoSize(App const& app, Vec3 pivot) {
  float distance = length(cameraPosition(app.camera) - pivot);
  float worldPerPixel =
      2.0F * distance * std::tan(app.camera.fovY * 0.5F) / static_cast<float>(app.height);
  return kGizmoPixels * worldPerPixel;
}

std::optional<GizmoHandle> handleUnderCursor(App& app, Vec2 mouse) {
  Vec3 pivot;
  Quat orientation;
  if (!gizmoFrame(app, pivot, orientation)) {
    return std::nullopt;
  }
  auto shapes =
      buildHandleShapes(app.mode, pivot, orientation, gizmoSize(app, pivot), makeProject(app));
  return pickHandle(shapes, mouse, kPickThresholdPx);
}

std::optional<NodeId> pickNode(App& app, Ray const& ray) {
  std::vector<std::pair<NodeId, Aabb>> candidates;
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
    candidates.push_back({visual.id, box});
  }
  auto hit = pickClosest(ray, candidates);
  if (!hit.has_value()) {
    return std::nullopt;
  }
  return hit->node;
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

void assembleFrame(App& app,
                    std::vector<DrawItem>& items,
                    std::vector<RendererVertex>& depthLines,
                    std::vector<RendererVertex>& overlayLines) {
  appendGrid(depthLines, 8.0F, 1.0F);
  for (NodeVisual const& visual : app.visuals) {
    items.push_back({visual.mesh, app.scene.world(visual.id), visual.colour, true});
  }
  Vec3 pivot;
  Quat orientation;
  if (gizmoFrame(app, pivot, orientation)) {
    auto geometry = buildHandleGeometry(app.mode, pivot, orientation, gizmoSize(app, pivot));
    appendHandleLines(overlayLines, geometry);
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
    if (press->detail == 1U) {
      if (auto handle = handleUnderCursor(app, mouse)) {
        Vec3 pivot;
        Quat orientation;
        if (gizmoFrame(app, pivot, orientation)) {
          GizmoSettings settings;
          settings.mode = app.mode;
          settings.space = app.space;
          settings.snapStep = snapStepFor(app);
          app.dragging = app.drag.begin(app.scene, app.selection, pivot, orientation, *handle,
                                        appRay(app, mouse), cameraForward(app.camera), settings);
        }
      } else {
        auto hit = pickNode(app, appRay(app, mouse));
        app.selection.clear();
        if (hit.has_value()) {
          app.selection.push_back(*hit);
        }
      }
    } else if (press->detail == 3U) {
      if ((press->state & XCB_MOD_MASK_SHIFT) != 0U) {
        app.panning = true;
        app.panButton = 3U;
      } else {
        app.orbiting = true;
      }
    } else if (press->detail == 2U) {
      app.panning = true;
      app.panButton = 2U;
    } else if (press->detail == 4U) {
      cameraDolly(app.camera, 1.0F);
    } else if (press->detail == 5U) {
      cameraDolly(app.camera, -1.0F);
    }
    break;
  }
  case XCB_BUTTON_RELEASE: {
    auto* release = reinterpret_cast<xcb_button_release_event_t*>(event);
    if (release->detail == 1U) {
      app.drag.end();
      app.dragging = false;
    }
    if (release->detail == 3U) {
      app.orbiting = false;
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
    if (app.dragging) {
      app.drag.update(appRay(app, mouse));
    } else if (app.orbiting) {
      cameraOrbit(app.camera, delta.x * kOrbitSpeed, -delta.y * kOrbitSpeed);
    } else if (app.panning) {
      cameraPan(app.camera, delta.x, delta.y, static_cast<float>(app.height));
    }
    break;
  }
  case XCB_KEY_PRESS: {
    auto* press = reinterpret_cast<xcb_key_press_event_t*>(event);
    xcb_keysym_t keysym = keysymFor(window, press->detail);
    // Only keys that change gizmo state cancel an active drag; modifier presses
    // (e.g. Ctrl for snap) must not.
    bool stateChanged = false;
    if (keysym == 0xFF1BU) {
      app.running = false;
      stateChanged = true;
    } else if (keysym == 'g' || keysym == 'G') {
      app.mode = GizmoMode::Translate;
      stateChanged = true;
    } else if (keysym == 'r' || keysym == 'R') {
      app.mode = GizmoMode::Rotate;
      stateChanged = true;
    } else if (keysym == 's' || keysym == 'S') {
      app.mode = GizmoMode::Scale;
      stateChanged = true;
    } else if (keysym == 'l' || keysym == 'L') {
      app.space = (app.space == GizmoSpace::World) ? GizmoSpace::Local : GizmoSpace::World;
      std::printf("gizmo space: %s\n", app.space == GizmoSpace::World ? "world" : "local");
      stateChanged = true;
    }
    if (stateChanged && app.dragging) {
      app.drag.end();
      app.dragging = false;
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
    std::string name(path);
    bool ok = false;
    if (name.size() >= 4U && name.compare(name.size() - 4U, 4U, ".glb") == 0) {
      ok = loadGlbFile(path, model, error);
    } else if (name.size() >= 4U && name.compare(name.size() - 4U, 4U, ".obj") == 0) {
      ok = loadObjFile(path, model, error);
    } else {
      std::printf("unsupported model type: %s (want .glb or .obj)\n", path);
      return false;
    }
    if (!ok) {
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

int runSmokeOffscreen(char const* shaderDir, std::vector<char const*> const& modelPaths) {
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
  }
  writePpm(".build/smoke.ppm", pixels, config.width, config.height);
  std::printf("SMOKE pixels: covered=%d reddish=%d (of %d) -> .build/smoke.ppm\n", covered,
              reddish, config.width * config.height);
  int minCovered = modelPaths.empty() ? 2000 : 100;
  int minReddish = modelPaths.empty() ? 200 : 0;
  if (covered < minCovered || reddish < minReddish) {
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
    cameraOrbit(app.camera, 0.01F, 0.0F);
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

int runInteractive(char const* shaderDir, std::vector<char const*> const& modelPaths) {
  App app;
  if (modelPaths.empty()) {
    buildDemoScene(app);
  }
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
      "  LMB: select / drag gizmo   RMB: orbit   MMB or Shift+RMB: pan   wheel: zoom\n"
      "  G translate   R rotate   S scale   L world/local space   Ctrl snap   Esc quit\n",
      renderer.deviceName());

  while (app.running) {
    xcb_generic_event_t* event = nullptr;
    while ((event = xcb_poll_for_event(window.connection)) != nullptr) {
      handleEvent(app, window, event, renderer);
      std::free(event);
    }
    if (app.width == 0U || app.height == 0U) {
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
  }

  renderer.shutdown();
  closeWindow(window);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  enum class Mode { Interactive, SmokeOffscreen, SmokeWindow };
  Mode mode = Mode::Interactive;
  char const* shaderDir = ".build/shaders";
  int frames = 30;
  std::vector<char const*> modelPaths;

  for (int i = 1; i < argc; ++i) {
    char const* arg = argv[i];
    if (std::strcmp(arg, "--shaders") == 0 && i + 1 < argc) {
      shaderDir = argv[++i];
    } else if (std::strcmp(arg, "--smoke-offscreen") == 0) {
      mode = Mode::SmokeOffscreen;
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
    } else if (std::strcmp(arg, "--help") == 0) {
      std::printf(
          "usage: vr-editor [model.glb|model.obj ...] [--shaders DIR]\n"
          "                [--smoke-offscreen | --smoke-window [FRAMES]]\n");
      return 0;
    } else if (arg[0] == '-') {
      std::printf("unknown option: %s\n", arg);
      return 1;
    } else {
      modelPaths.push_back(arg);
    }
  }

  switch (mode) {
  case Mode::SmokeOffscreen:
    return runSmokeOffscreen(shaderDir, modelPaths);
  case Mode::SmokeWindow:
    return runSmokeWindow(shaderDir, frames, modelPaths);
  default:
    return runInteractive(shaderDir, modelPaths);
  }
}
