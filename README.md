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
bash run-tests.sh        # core tests (548 checks; image fixtures via ImageMagick)
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

**LMB** selects and drags the gizmo: a click selects the whole model under the
cursor, **Shift+LMB** or **Ctrl+LMB** selects a single part. **1**/**G**
translate, **2**/**R** rotate, **3** scale, **L** world/local space, **Ctrl**
snap, **Ctrl-Z** undo (cancels a live edit), **Ctrl-Y** / **Ctrl-Shift-Z**
redo, **N** opens the model gallery (small previews of every model in the
library folder arranged in an arc; the pointed-at one brightens and a click
spawns that model at the mouse aim point as one undo step; **N** closes
without spawning), **Delete** removes the selection (model or part),
**Esc** quits. Scale is UNIFORM only — every scale handle drives one factor on
all three axes; per-axis scaling does not exist. The gizmo sits at the centre
of the selection's bounds; the selection is tinted.

In VR the trigger only *selects* — whole model, or a single part with grip+trigger.
Movement and rotation are the **sticks** (controller aim is too jittery for
accurate placement): the **right stick** slides the selection across the floor
relative to your head (pushing forward walks it away from you; it never lifts
off the floor), the **left stick** turns it around vertical — turntable-style,
never tipping it over. Left stick up/down does *nothing* (it briefly zoomed;
that ruined rotation and was removed). To make the model bigger or smaller:
scale mode (menu button cycles translate → rotate → scale; desktop key **3**)
and the scale handles (uniform), or walk closer. **Grip** held
runs both sticks at quarter speed for fine control. Each stick gesture is one
undo step. Aim at a gizmo handle for precise axis/rotate/uniform-scale work —
a handle wins only where it is nearer than the model surface. The bounding box
is not a click target: pressing empty space next to a model selects nothing.
Handles brighten when hovered; the selection is shown by the warm tint and the
gizmo at the selection's centre. **A** is the model gallery: small previews of
every model in the library folder appear in an arc in front of you (two rows
beyond eight), the aimed preview brightens, and the trigger spawns that model
on the floor where your aim beam meets it — one undo step, and the gallery
closes. **A** or undo (B) closes without spawning. The library folder
(`~/Downloads/VR` by default, `--library DIR`) is rescanned every time it
opens — drop a file in and spawn it without restarting. **Hold X** to delete: the
selection ramps red and the delete fires at full charge (0.6 s); release
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
