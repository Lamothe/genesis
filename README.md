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
bash run-tests.sh        # core tests (603 checks; image fixtures via ImageMagick)
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
**up/down** height, no floor rule), **2** = rotate (**left/right** spins about
the selection's own up; **up/down** tip, **Shift** = roll). **Shift** runs
the arrows at quarter speed (fine). Nothing rotates outside rotate mode. Each
arrow gesture is one undo step. **Ctrl-Z** undo (cancels a live edit),
**Ctrl-Y** / **Ctrl-Shift-Z** redo, **N** opens the model carousel (one
staged model appears at eye height in front of the camera; **[** **]** cycle
the library, a click places the
staged model at the floor point under the cursor as one undo step; **N**
closes without placing), **Delete** removes the selection (model or part),
**Esc** quits. The selection is marked by a thin warm rim in the shader (a
part selection marks the part) and the current mode's name shows as
line-font text floating in front of the camera.

In VR the trigger only *selects* — whole model, or a single part with grip+trigger.
Movement and rotation are the **sticks** (controller aim is too jittery for
accurate placement): the **right stick** slides the selection across the floor
relative to your head (pushing forward walks it away from you; it never lifts
off the floor) in **both** modes, and the **left stick is mode-driven** — the
**menu button taps** toggle the two modes: **MOVE** (X = uniform scale,
right = bigger; Y = height, up = lift) and **ROTATE** (X = spin about the
model's own up; Y = tip, up tips the top away; grip+Y = roll). Nothing
rotates outside ROTATE. The rotation axes come from the model's own frame and
are mutually perpendicular (a spin follows the model's tipped up axis); the
axes, rate and grip mode are latched per gesture (turning your head or
toggling grip mid-gesture does not steer a gesture in flight). Whole models
re-rest on the floor after a tip; part selections tip in place. **Grip** held
runs both sticks at quarter speed for fine control. Each stick gesture is one
undo step. The bounding box is not a click target: pressing empty space next
to a model selects nothing. The selection is shown by a thin warm rim; the
current mode (and the staged model's name) shows as line-font text floating
in front of your head. **A** is the model carousel: one library model is
staged at eye height in front of you (loaded and textured), **stick X**
cycles the library (only one model is ever loaded at a time), and the trigger
places the staged model at natural size on the floor where your aim beam
meets it — one undo step (undo removes it entirely), and the carousel
closes. **A** or undo (B) closes without placing. The library folder
(`~/Downloads/VR` by default, `--library DIR`) is rescanned every time it
opens — drop a file in and place it without restarting. **Hold X** to
delete: the selection ramps red and the delete fires at full charge (0.6 s);
release
before that to cancel. **B** (right) / **Y** (left) on Touch undo/redo,
trackpad click on Vive. Passthrough is the DEFAULT view: the real world with
the models drawn over it (grip + menu, or desktop B, toggles the grid backdrop
back). There are no flight controls in VR by design: passthrough means you
walk around the real world with the
models placed in it, so locomotion is physical — you walk, turn and crouch.
The sticks are for EDITING, not locomotion (their output is still logged at
session end as controller diagnostics). The view is horizon-locked: the head's physical
roll is stripped (the look direction is kept exactly), so the ground stays
level with gravity however you move or turn. Desktop keeps its keyboard
flight (WASD, Q/E, Shift boost) and mouse look for desk debugging.

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
- **Scene transforms are TRS only.** Nested non-uniform scale under rotation can
  introduce shear that TRS cannot represent; the gizmo maths assumes shear-free
  hierarchies. Fine for editor content; revisit if a rig ever needs it.
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
