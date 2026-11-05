# Genesis — 3D scene editor

Multi-model scene manipulator (move, rotate, scale) for VR, with free-flight
first-person controls and an OpenXR shell. Built to have the DLSS5 neural
renderer bolted on later through the seam contract
(`NEURAL-SEAM.md`, `neural-seam.h`).

## Layout

- `include/vecmath.h` — Vec2/Vec3/Quat/Mat4/Ray/Aabb, TRS, affine inverse, decompose
- `include/scene.h`, `src/scene.cpp` — transform hierarchy, cached world
  matrices, `epoch()` discontinuity counter
- `include/picking.h`, `src/picking.cpp` — ray/AABB, ray/triangle (Möller–
  Trumbore), closest-hit selection
- `include/gizmo.h`, `src/gizmo.cpp` — translate/rotate/scale drags, world/local
  space, axis/plane/free handles, snapping, drift-free cumulative application
- `include/camera.h`, `src/camera.cpp` — orbit camera, screen→ray, pan/dolly
- `include/handles.h`, `src/handles.cpp` — gizmo handle geometry (single source
  for drawing *and* hit testing) + screen-space pick resolution
- `include/mesh.h`, `src/mesh.cpp` — vertex/mesh/model data types
- `include/json.h`, `src/json.cpp` — strict JSON DOM (arena-based) for glTF
- `include/gltf.h`, `src/gltf.cpp` — glTF 2.0 binary (`.glb`) loader
- `include/obj.h`, `src/obj.cpp` — Wavefront OBJ + MTL (`Kd`) loader
- `app/vk_renderer.*` — Vulkan 1.3 viewport renderer (dynamic rendering;
  XCB/swapchain windowed mode, offscreen mode with RGBA8 readback; registered
  indexed meshes, built-in cube is mesh 0)
- `app/geometry.*` — cube, floor grid, handle-line geometry
- `app/editor_app.cpp` — XCB/OpenXR shell input, selection, controller
  workflow, model loading (`.glb`/`.obj` on the command line), smoke modes
- `include/controller_workflow.h` — controller grip-pose quality, the
  carousel's settled/loaded placement gate, the trigger select-then-grab state
  machine and the tools-panel press/Delete-hold capture
- `shaders/` — GLSL (compiled by `glslc` at launch)
- `tests/test_core.cpp` — maths, parser and behaviour tests

## Build and test

```sh
bash run-tests.sh        # core + controller/hand state checks; fixtures via ImageMagick
bash run-editor.sh       # build + interactive editor
bash run-editor.sh model.glb other.obj   # load models (textures via glTF/OBJ materials)
bash run-editor.sh model.glb --xr        # VR session via OpenXR (WiVRn etc.)
bash run-editor.sh --smoke-offscreen     # render + pixel assertions
bash run-editor.sh --smoke-textured      # texture sampling end-to-end
bash run-editor.sh --smoke-blend         # blended surfaces
bash run-editor.sh --smoke-perf          # offscreen rendering timings
bash run-editor.sh --smoke-skinned       # rig: bind pose -> animated deformation
bash run-editor.sh --bench-skin          # CPU skinning cost at 100k/300k/1.2M vertices
bash run-editor.sh --smoke-window 30     # windowed present path
bash run-editor.sh --smoke-xr            # XR frame loop (SKIP without a runtime)
```

`run-tests.sh` compiles with `g++ -std=c++20 -Wall -Wextra -Werror`;
`run-editor.sh` compiles the shell and shaders and runs it. Thin shell launchers
only: **all project logic is typed C/C++ (project rule — no Python)**.

The offscreen renderer regressions are a separate binary because they need a
live Vulkan 1.3 device and the compiled shaders (build them with
`run-editor.sh` once):

```sh
g++ -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude -Iapp tests/test_renderer.cpp \
  src/camera.cpp src/handles.cpp src/triangle_sort.cpp src/mesh.cpp src/image.cpp \
  app/geometry.cpp app/vk_renderer.cpp -lvulkan -lxcb -lz -o .build/test-renderer
./.build/test-renderer
```

## Controls

Free-flight, first-person (desktop debugging): **WASD** move, **Q/E** down/up,
**Shift** boost, **RMB** mouse-look, **MMB** or **Shift+RMB** pan, **wheel** fly
forward/back.

**LMB** selects: a click selects the whole model under the cursor,
**Shift+LMB** or **Ctrl+LMB** selects a single part. There is no gizmo — the
coloured axis handles were removed; desktop editing is the **arrow keys** and
it is **mode-driven** with two modes:
**1** = move (**left/right** uniform scale — per-axis scaling does not exist;
**up/down** height, no floor rule), **2** = rotate (**left/right** spins the
model about the scene's up axis, **up/down** tips the top away / lifts the
face — strictly around the viewer's axes). **Shift** runs
the arrows at quarter speed (fine). Nothing rotates outside rotate mode. The
desktop **2** key still rotates strictly around the VIEWER's axes (the scene's
vertical and the left-right line through your eye, latched per gesture): whole
models re-rest on the floor after a tip, part selections tip in place. Each
arrow gesture is one undo step. **Ctrl-Z** undo (cancels a live edit),
**Ctrl-Y** / **Ctrl-Shift-Z** redo, **N** opens the model carousel (the
ring of models rotates in/out of view around the camera; **[** **]** cycle
the library; clicking the exact surface of the centred preview places the
staged model below the centred preview as one undo step and takes priority
over browsing. Otherwise, clicking anywhere left or right of centre browses
one model per click (previous on the left, next on the right — there is no
central dead zone; a grazing clip is inert). **N**
closes without placing), **Delete** removes the selection (model or part),
**Esc** quits. The selection is marked by a thin warm rim in the shader (a
part selection marks the part) and the current mode's name shows as
sans-serif text floating in front of the camera.

In VR the right controller's **physical grip pose** is the manipulation input;
its aim ray only selects and points at UI. Placement in the carousel needs no
pointing. There is one owner grab at a time, and the controls are:

- **Right trigger, viewer:** the first press selects the exact model (or, with
  **grip+trigger**, the single part) under the tracked aim and is consumed:
  holding it cannot then move the model. Release. The next press-and-hold grabs
  the current selection and moves/rotates it with the controller's physical 3D
  position AND orientation, through the same latched-view projection the hand
  grab used: lateral motion follows the controller in view, and depth changes by
  the same fraction as the grip's distance from the eye. The pivot is the picked
  surface point; release commits one undo step. A press on a different model
  selects it; the grab can start without pointing back at the selection. Nothing snaps on
  grab start because every update applies the cumulative delta to the captured
  start transforms.
- **Grip pose quality:** the shell locates a separate grip action space for each
  controller and records VALID and TRACKED independently. A VALID-but-untracked
  grip can neither start nor update a grab, and there is no head-gaze fallback
  for the physical pose. Losing tracking during a grab commits at the last good
  pose and requires a real trigger release before another grab. Profile changes
  and skipped/paused XR frames do the same.
- **Left Menu:** one press opens/closes the real tools panel (**Models**,
  **Undo**, **Redo**, **Delete**, **Backdrop**), latched in the world in front of
  the viewer so pointing at it is stable. Aim with the right controller and pull
  the trigger: Models/Undo/Redo/Backdrop fire on release over the same button;
  Delete is a 0.6 s hold with the red charge ramp and cancels if released early.
  A miss is captured and swallowed (nothing clicks through). Opening the panel
  commits a live grab; a trigger held across a panel/carousel transition must
  return to release before it can act again.
- **Carousel:** **A** or the panel's **Models** button opens it. **Right stick
  X** browses the ring (one step per flick, held repeats every 0.25 s). The
  **right trigger loads the centred model — no pointing at it** — but only once
  the ring is at rest, the right stick is active and neutral, and that centre slot has finished
  loading and rested for 0.12 s. A trigger press while it is loading or
  unsettled is consumed: it never replays when the load completes, so it cannot
  place a different model than the one visible at the press. The centre
  preview wears the cool rim while a trigger press would place now (readiness,
  not aim). Profiles without an active right stick cannot place from the carousel.
  Placement is at natural size on the floor below the centred preview,
  one undo step, and closes the carousel.
- **A** still opens/closes the carousel directly, undo (**B**)/redo (**Y**) and
  hold-**X** delete remain, and **grip** remains the part-selection modifier.
  Thumbstick clicks do nothing. The VR thumbsticks never move, rotate or scale a
  model: the right stick only browses the carousel and the left stick is unused.
- **Hands:** tracked hand/arm surfaces still render, depth-tested against the
  scene, and the **left palm-up pinch** menu still opens the tools panel. Hand
  pinch/grasp no longer selects, grabs, moves or scales models; the old hand-grab
  maths and tests are retained for future arm interaction. A hand menu follows
  the palm and closes on palm tracking loss; the controller menu stays latched where it opened.
  Models in the hand-only menu requires switching to the right controller.

**Verification status:** strict build; core, hand, XR-view and hand-mesh tests;
state-machine regressions for carousel placement gating (loading, settling,
browsing, held-trigger latch, changed index), trigger select-then-grab,
untracked poses, tools-panel capture/Delete hold and history; offscreen,
textured, blend and perf smokes. The offscreen smoke also renders and checks the
panel, and covers a controller-owner grab with position+orientation, refusal of
an untracked pose, undo and release. The **XR frame smoke was not available**:
it was not run during this handover to avoid disturbing the active XR session.
Runtime bindings and headset usability remain **unverified**.

The **bounding box is not a click target**: pressing empty space next to a model
selects nothing. The selection is shown by a thin warm rim; the mode label is
desktop-only (there is no fixed MOVE label in VR). The carousel is a
**stationary ring** of library models: the centred model is staged at eye height
and the previous/next models sit visible at the sides as cheap geometry-only
previews (textures load when a model is placed; each preview is labelled with
its filename stem). The library folder (`~/Downloads/VR` by default,
`--library DIR`) is rescanned every time the carousel opens — drop a file in and
place it without restarting. **Hold X** to delete: the selection ramps red and
the delete fires at full charge (0.6 s); release before that to cancel. **B**
(right) / **Y** (left) on Touch undo/redo, trackpad click on Vive. Passthrough
is the DEFAULT view: the real world with the models drawn over it (the
**Backdrop** button, or desktop **B**, toggles the grid back). There are no
flight controls in VR by design: passthrough means you walk around the real
world with the models placed in it, so locomotion is physical — you walk, turn
and crouch. The sticks never edit or drive locomotion (their output is still
logged at session end as controller diagnostics). The view uses the headset's
located eye poses, including head roll, while the world rig remains level with
gravity. Desktop keeps its keyboard
flight (WASD, Q/E, Shift boost) and mouse look for desk debugging.

## Animation and rendering status

The first usable glTF node-TRS clip plays automatically and loops on placed
models and command-line imports. STEP, LINEAR and CUBICSPLINE translation,
rotation and scale are supported; playback is separate from undoable edits.
Unsupported clips are logged and skipped. Nodes with singular (zero-scale) rest
transforms cannot currently be animated. There is no clip selector yet.

### Skeletal skinning (rigged characters)

glTF skins are loaded (`skins`, node `skin`, `inverseBindMatrices`,
`JOINTS_0`/`WEIGHTS_0`) and animated with four-influence linear blend skinning.
A clip that targets a skin joint is preferred over an object clip, so a rigged
character plays its walk cycle rather than a turntable. The clip choice is
still automatic (first joint-targeting clip); there is no selector.

Joint matrices are `inverse(meshWorld) * jointWorld * inverseBindMatrix` per
the glTF spec, built from the scene's live world matrices after the node pose
is applied, so animated *and* hand-edited hierarchies both drive the skin. The
node's own transform cancels exactly: moving the whole model re-parents the
draw but does not reskin it. Inverse bind matrices must be MAT4 float, affine
(bottom row `0,0,0,1`) and one per joint; a malformed skin is logged and its
meshes render rigid instead of hiding the model. `JOINTS_1+` (more than four
influences), sparse skin accessors, morph targets and joints outside the
selected scene stay unsupported; those primitives render in the bind pose.
Models with skins but no playable joint clip keep their bind-pose draw, and
`JOINTS_0`/`WEIGHTS_0` are only parsed when the file actually defines skins
(the static library exports carry unused joint attributes).

**CPU skinning is deliberate**: weights are ordinary CPU data, exact picking
and selection bounds need the deformed vertices anyway, both stereo eyes share
one pose, and the upload uses the renderer's fence-safe dynamic path instead of
a per-frame `vkDeviceWaitIdle`. Measured single-threaded on platinum (Radeon
8060S, `--bench-skin`, run-to-run variation about ±20%): 100k vertices
0.9 ms deform + 0.3 ms queue, 300k vertices 3.1 + 0.9 ms, 1.2M vertices
11.0 + 5.1 ms. Small and medium rigs play inside the 90 Hz stereo budget; the
library's largest static characters (1.0–1.7 M vertices) would spend ~16 ms per
frame in the CPU path alone, so multi-million-vertex rigs need GPU skinning or
a threaded deformer next.
Skinned primitives are per-instance dynamic meshes (two copies of a character
deform independently), and a mesh used only by skinned nodes skips the shared
static registration entirely, so a placement costs one copy rather than two.
Each placed copy keeps a device vertex buffer, a persistent host-visible
staging buffer and the CPU pending vertices: about 132 bytes per vertex plus 12
bytes per triangle, i.e. roughly 165 MB for a 1.2 M-vertex (600k-triangle)
character. That memory is never reclaimed, because the renderer has no release
path (see Design notes) — spawning and deleting large rigged characters
repeatedly grows memory for the rest of the session.
Deform results are re-uploaded only while the joint
matrices change, and opaque skinned meshes skip the blend-sort key pass.
Ray picking and selection bounds use the live deformed vertices. Physical hand
contact still builds its triangle BVH once per mesh (a build costs ~1 µs per
triangle, measured: 211 ms at 200k, 2.4 s at 2M), so grasping an *animated*
character tests the surface from when the hand first reached it, not the
current pose; use pinch/ray selection for exact contact. Deleting an animated
model still leaves its registered GPU mesh (resources are never released, see
Design notes).

`--smoke-skinned` renders the synthetic two-joint fixture that
`run-tests.sh` writes to `.build/test-skinned.glb` (bind pose → animated,
moved bounds/picking, changed pixels); it needs an offscreen Vulkan device.
Headset-side appearance still has to be judged by eye.

A comparison with Centre Stage identified the missing PBR materials,
transmission, emissive, sRGB material pipeline and studio lighting. This change
only fixes normal transforms and Vulkan blend-upload synchronisation; it does
not claim Centre Stage appearance parity. The hands are procedural tracked surfaces, not the downloaded rig; its library
GLB has no skin, while the original archive contains a rig. Rig retargeting and
collision need further work. Visible hands/arms and their depth-tested
rendering are retained in the controller workflow, but no arm-asset work was
done in this change. Passthrough does not supply real room geometry for
collision.

## Design notes

- **Renderer resources are registered, never released** (there is no
  removeMesh/removeTexture API): image-content dedupe means re-staging or
  re-placing a model reuses its registrations, but distinct models
  accumulate. The texture registry holds 256 (a device-limit check at init
  fails loudly below that) — after ~17 heavily textured models in one
  session, further textured loads degrade to untextured with a log message.
  A refcounted release path tied to undo history is the eventual fix.
- **Gizmo drags are cumulative from drag start**, not incremental per event:
  every update recomputes the total delta and re-applies it to captured start
  transforms, so rounding cannot accumulate.
- Text uses pre-rasterised DejaVu Sans glyphs (licence in `app/DEJAVU-LICENSE.txt`); unsupported Unicode characters display as `?`.
- **Scene transforms are TRS only.** Nested non-uniform scale under rotation can
  introduce shear that TRS cannot represent; the gizmo maths assumes shear-free
  hierarchies. Hand edits reject samples that cannot be represented as local TRS rather than distorting the hierarchy; revisit if a rig needs arbitrary affine transforms.
- **Discontinuity**: any transform or structural change bumps `Scene::epoch()`;
  the seam maps that to `reset = 1` for the temporal loop. Selection changes do
  not bump it.
- The caller resolves which gizmo handle was grabbed (the renderer knows where
  handles were drawn); this core only does the drag maths and applies it.

## Roadmap

1. ~~Vulkan desktop shell~~ — done: windowed + offscreen renderer, camera,
   selection, gizmo handles and drag wiring (16/10/2026).
2. ~~glTF 2.0 (`.glb`) + OBJ loading in C++~~ — done (17/10/2026): strict JSON
   DOM, GLB container/accessor reader (strided/interleaved, normalised colours),
   OBJ/MTL loader, registered indexed meshes, per-mesh bounds picking.
3. ~~Textures + free-flight camera~~ — done (17/10/2026): hand-written PNG and
   baseline JPEG decoders (typed C++, zlib for inflate), glTF baseColorTexture
   and OBJ map_Kd with alpha modes (opaque/mask/blend) and material alpha,
   first-person free flight (WASD/QE, mouse-look, wheel dolly).
4. ~~WiVRn/OpenXR VR shell: stereo views through the same interface~~ — done
   (17/10/2026): XR_KHR_vulkan_enable2 shell in app/xr_shell.cpp (per-eye
   swapchains, LOCAL space, action set for Quest/Vive/simple controllers),
   rig-based free-flight locomotion composed with the head pose, controller
   aim-ray gizmo interaction (pickHandleByRay in the core), `--xr` /
   `--smoke-xr` (honest SKIP without a runtime). Needs a live OpenXR runtime
   (WiVRn) for first in-headset tuning of bindings and comfort.
5. Bind `neural_seam_evaluate` to the native evaluator when the migration
   stream's full native evaluation lands; conform the seam's convention table to
   the pinned native input conventions at that point.

## Git

Standalone project at `~/Projects/Genesis`, extracted from the DLSS5 VR
repo's `scene-editor` branch (history preserved). The DLSS5 side consumes
the seam contract (`NEURAL-SEAM.md`) when the native evaluation stream is
ready.
