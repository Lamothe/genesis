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
- `app/editor_app.cpp` — XCB window, input, selection, gizmo wiring, model
  loading (`.glb`/`.obj` on the command line), smoke modes
- `shaders/` — GLSL (compiled by `glslc` at launch)
- `tests/test_core.cpp` — maths, parser and behaviour tests

## Build and test

```sh
bash run-tests.sh        # core + hand gesture checks; fixtures via ImageMagick
bash run-editor.sh       # build + interactive editor
bash run-editor.sh model.glb other.obj   # load models (textures via glTF/OBJ materials)
bash run-editor.sh model.glb --xr        # VR session via OpenXR (WiVRn etc.)
bash run-editor.sh --smoke-offscreen     # render + pixel assertions
bash run-editor.sh --smoke-textured      # texture sampling end-to-end
bash run-editor.sh --smoke-window 30     # windowed present path
bash run-editor.sh --smoke-xr            # XR frame loop (SKIP without a runtime)
```

`run-tests.sh` compiles with `g++ -std=c++20 -Wall -Wextra -Werror`;
`run-editor.sh` compiles the shell and shaders and runs it. Thin shell launchers
only: **all project logic is typed C/C++ (project rule — no Python)**.

## Controls

Free-flight, first-person (desktop debugging): **WASD** move, **Q/E** down/up,
**Shift** boost, **RMB** mouse-look, **MMB** or **Shift+RMB** pan, **wheel** fly
forward/back.

**LMB** selects: a click selects the whole model under the cursor,
**Shift+LMB** or **Ctrl+LMB** selects a single part. There is no gizmo — the
coloured axis handles were removed; all editing is the **arrow keys** (the
desktop mirror of the sticks) and it is **mode-driven** with two modes:
**1** = move (**left/right** uniform scale — per-axis scaling does not exist;
**up/down** height, no floor rule), **2** = rotate (**left/right** spins the
model about the scene's up axis, **up/down** tips the top away / lifts the
face — strictly around the viewer's axes). **Shift** runs
the arrows at quarter speed (fine). Nothing rotates outside rotate mode. Each
arrow gesture is one undo step. **Ctrl-Z** undo (cancels a live edit),
**Ctrl-Y** / **Ctrl-Shift-Z** redo, **N** opens the model carousel (the
ring of models rotates in/out of view around the camera; **[** **]** cycle
the library; clicking the exact surface of the centred preview places the
staged model below the centred preview as one undo step. Otherwise,
clicking the left/right ray zones browses one model per click; a central miss
does nothing. **N**
closes without placing), **Delete** removes the selection (model or part),
**Esc** quits. The selection is marked by a thin warm rim in the shader (a
part selection marks the part) and the current mode's name shows as
sans-serif text floating in front of the camera.

In VR the trigger only *selects* — whole model, or a single part with grip+trigger.
Bare hands (Quest through WiVRn) use a separate control path; controller
buttons and sticks are unchanged:

- **Left pinch while the left palm faces up** toggles the stationary hand menu;
  palm-up alone does nothing. Choose **Models** to open the carousel.
- In the carousel, **right pinch** the exact surface of the settled centre
  preview to place it. Otherwise, the left/right ray zones browse one model
  per pinch; a central miss does nothing. Placing a model closes the menu.
  The centre marker identifies the model that can be placed. Desktop and controller use the same hit rules.
- Outside the menu, **right pinch and release** selects the whole model under
  the aim beam. Pinch and hold the selected model again to move it with the
  palm and rotate it with the wrist around the picked surface. Lateral motion is
  view-relative: lateral motion follows the hand in the view, and depth changes
  by the same fraction as the hand's distance from the viewer. The grab view is
  latched; releasing commits one undo step. Hand selection is whole-model only;
  controllers retain grip+trigger part selection.
- While holding a right-hand grab, pinch and hold with the left hand and vary
  the distance between the palms to scale uniformly about the model's centre.
  Releasing either hand ends the edit; release the right hand before another
  edit. Both hands are shown as tracked, procedurally generated 3D surfaces
  depth-tested against models; they are not the downloaded rigged asset.
- After selecting a model, bring the right hand near its surface and close it
  to grab and rotate it directly; opening the hand releases it. This uses the
  runtime grasp gesture and does not provide collision or haptic resistance.
- The hand menu offers **Models**, **Undo**, **Redo**, **Delete** and
  **Backdrop**. Delete requires a 0.6 s hold with red charge feedback.
  A held menu interaction stays captured until release.
- Opening the tools panel pauses an active controller-stick edit; choosing
  UNDO/REDO cancels it without discarding the existing redo branch. Closing
  the panel or choosing another edit commits it.
- Tracking loss or a skipped/paused XR frame ends the current hand edit;
  release the pinch before starting again. Aim-and-pinch select/place remains
  available without joint tracking, but grab/rotation and the palm-up menu
  toggle need tracked palm joints. Direct grasp also needs runtime grasp input. The console reports joint availability
  and tracking, left-pinch menu actions and tool activations.

**Verification status:** build, automated hand/core tests and offscreen hand
editing/panel smoke checks are available; these do **not** verify the runtime
bindings or gesture usability in the headset. Michael's headset confirmation
is still required for every feature above.

Movement and editing are the **sticks** (controller aim is too jittery for
accurate placement): the **left stick** slides the selection across the floor
relative to your head at **3 m/s** at full deflection (previously 1.5 m/s;
pushing forward walks it away from you; it never lifts
off the floor), and the **right stick** scales it (X, right = bigger) and
raises it (Y, up = lift). The **left controller's menu** is two controls:
**tap** = the model carousel, **hold** = the backdrop (thumbstick clicks do
nothing). The old menu-tap MOVE/ROTATE switch is gone — hand grabs provide rotation; the desktop **2** key still
rotates, strictly around the VIEWER's axes (the scene's vertical and the
left-right line through your eye, always perpendicular and level, latched per
gesture: turning your head mid-gesture does not steer it; whole models
re-rest on the floor after a tip, part selections tip in place). **Grip** held
runs the sticks at quarter speed for fine control. Each stick gesture is one
undo step. The bounding box is not a click target: pressing empty space next
to a model selects nothing. The selection is shown by a thin warm rim; the mode label is desktop-only (there is no fixed MOVE label in VR). **A** or a **menu tap**
is the model carousel: a STATIONARY RING of library models around you — the
centred model is staged at eye height in front of you and the previous/next
models sit visible at the sides as cheap geometry-only previews (all
previews stay untextured; textures load when a model is placed). **right
stick X**
rotates
the ring in and out of view INSTANTLY (decoding runs on a worker thread, so
a huge model never blocks the switch; each preview is labelled with its filename stem), and the
trigger places the centred model only when its exact preview surface is hit,
at natural size on the floor below the centred preview — one undo step
(undo removes it entirely), and the carousel closes. Otherwise, left/right
ray zones browse one model per trigger press; a central miss is inert. **A**, a **menu tap** or undo (B) closes without placing. The library folder
(`~/Downloads/VR` by default, `--library DIR`) is rescanned every time it
opens — drop a file in and place it without restarting. **Hold X** to
delete: the selection ramps red and the delete fires at full charge (0.6 s);
release
before that to cancel. **B** (right) / **Y** (left) on Touch undo/redo,
trackpad click on Vive. Passthrough is the DEFAULT view: the real world with
the models drawn over it (hold the menu button, or desktop B, toggles the grid backdrop
back). There are no flight controls in VR by design: passthrough means you
walk around the real world with the
models placed in it, so locomotion is physical — you walk, turn and crouch.
The sticks are for EDITING, not locomotion (their output is still logged at
session end as controller diagnostics). The view is horizon-locked: the head's physical
roll is stripped (the look direction is kept exactly), so the ground stays
level with gravity however you move or turn. Desktop keeps its keyboard
flight (WASD, Q/E, Shift boost) and mouse look for desk debugging.

## Animation and rendering status

The first usable glTF node-TRS clip plays automatically and loops on placed
models and command-line imports. STEP, LINEAR and CUBICSPLINE translation,
rotation and scale are supported; playback is separate from undoable edits.
Unsupported clips are logged and skipped. This is **not skinning**: rigged
characters, morph targets and clips requiring them will not deform. Nodes with
singular (zero-scale) rest transforms cannot currently be animated. There is no
clip selector yet.

A comparison with Centre Stage identified the missing PBR materials,
transmission, emissive, sRGB material pipeline and studio lighting. This change
only fixes normal transforms and Vulkan blend-upload synchronisation; it does
not claim Centre Stage appearance parity. The hands are procedural tracked surfaces, not the downloaded rig; its library
GLB has no skin, while the original archive contains a rig. Rig retargeting and
collision need further work. Passthrough does not supply real
room geometry for collision.

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
