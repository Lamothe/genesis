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
- `include/controller_arm_rig.h` — controller arm pose and finger solver
- `include/controller_workflow.h` — controller grip-pose quality, the
  carousel's settled/loaded placement gate, the trigger select-then-grab state
  machine and the tools-panel press/Delete-hold capture
- `shaders/` — GLSL (compiled by `glslc` at launch)
- `tests/test_core.cpp` — maths, parser and behaviour tests

## Build and test

```sh
bash run-tests.sh        # core + controller/hand state checks; fixtures via ImageMagick
bash run-xr-frame-tests.sh # shell frame lifecycle with stubbed XR; Vulkan and OpenXR loader libraries
bash run-editor.sh       # build + interactive editor
bash run-editor.sh model.glb other.obj   # load models (textures via glTF/OBJ materials)
bash run-editor.sh model.glb --xr        # VR session via OpenXR (WiVRn etc.)
bash run-editor.sh --smoke-offscreen     # render + pixel assertions
bash run-editor.sh --smoke-textured      # texture sampling end-to-end
bash run-editor.sh --smoke-blend         # blended surfaces
bash run-editor.sh --smoke-perf          # offscreen rendering timings
bash run-editor.sh --smoke-skinned       # rig: bind pose -> animated deformation
GENESIS_CONTROLLER_ARMS_GLB=/path/to/rigged.glb bash run-editor.sh --smoke-controller-arms # real asset + depth
bash run-editor.sh --bench-skin          # CPU skinning cost at 100k/300k/1.2M vertices
bash run-editor.sh --smoke-window 30     # windowed present path
bash run-editor.sh --smoke-xr            # XR frame loop (SKIP without a runtime)
```

`run-tests.sh` compiles with `g++ -std=c++20 -Wall -Wextra -Werror`;
`run-editor.sh` compiles the shell and shaders and runs it. Thin shell launchers
only: **all project logic is typed C/C++ (project rule — no Python)**.
The separate XR frame tests need Vulkan headers and Vulkan/OpenXR loader libraries,
but no active OpenXR runtime or headset; core tests do not link those loaders.

The offscreen renderer regressions are a separate binary because they need a
live Vulkan 1.3 device and the compiled shaders (build them with
`run-editor.sh` once):

```sh
g++ -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude -Iapp tests/test_renderer.cpp \
  src/camera.cpp src/handles.cpp src/triangle_sort.cpp src/mesh.cpp src/image.cpp \
  app/geometry.cpp app/vk_renderer.cpp -lvulkan -lxcb -lz -o .build/test-renderer
./.build/test-renderer
```

### Hair opacity and animation asset limitations

BLEND materials whose whole name token is `hair` or `scalp` use Centre Stage's
live-view default opacity remap `1-(1-alpha)^3`, intended to reduce scalp
show-through. Name matching conservatively keeps non-ASCII bytes inside a token
(rather than attempting Centre Stage's Unicode tokenisation). Other materials and opaque/masked draws are unchanged. This is only one piece
of rendering parity: Genesis does not implement Centre Stage's PBR studio
lighting, specular-glossiness shading, emissive or transmission.

Genesis reads COLOR_0 VEC4 alpha (VEC3/absent default to one) for BLEND and
MASK. Una/Rose have no vertex colours, so no visible improvement for those
assets is established. Specular-glossiness shading, nonzero texture coordinate
sets and colour texture sRGB decoding remain unsupported. Partial diffuse-only
specular-glossiness support was avoided because it can turn metallic materials
black. These changes do not establish material parity.

An offline inventory of 109 GLBs in platinum ~/Downloads/VR on 06/11/2026
found no skins or animation clips. No compatible approved motion source or
approved production rig has been established; synthetic deformation is not
a quality or retargeting validation.

The platinum library assets `~/Downloads/VR/Characters/Una/Una.glb` (SHA-256
`8aa497ba4b34c74321dc3eb87ed1adb6d4f200944fed7be86989c45b65bbfe96`) and
`~/Downloads/VR/Characters/Rose/Rose.glb` (SHA-256
`dd0ae874f5d5e3b41f431177fc9c518c970915993d7a3061e19afa057696b3fc`)
and their archived original `model.gltf` files contain no skins, clips or
morph targets. Separate experimental files on mercury,
`~/Projects/UniRig/work/Una_Rose_20261104/{Una/Una_rigged,Rose/Rose_rigged}.glb`,
contain skins but no clips; their generated weights and appearance are not
approved. Playback cannot invent missing animation or facial expressions.

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
**Esc** quits. The selection is marked by a neutral angle-based shader rim (not a silhouette border) (a
part selection marks the part and its descendants) and the current mode's name shows as
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
  selects it; the grab can start without pointing back at the selection only when the tracked ray misses other models. An untracked or unavailable aim consumes a new press without selecting or grabbing; release and press again once tracking returns. Losing aim during an active grab does not stop it while the grip and view remain tracked. A pickable surrounding skydome therefore currently blocks an off-target grab. Nothing snaps on
  grab start because every update applies the cumulative delta to the captured
  start transforms.
- **Left trigger:** a left-only press and release clears the selection, regardless of duration without pointing, including when aiming at a skydome. While a right-trigger grab is already active, a fresh left-trigger press joins uniform scaling based on the distance between the two physical controller grip positions. Release the left trigger to resume the right-hand move/rotation without a jump; release the right to commit the whole gesture as one undo step. The scale pivot is the selection bounds centre at join; a 4 cm minimum initial separation avoids unstable ratios. A left press held before the right grab cannot scale or deselect on release. If either pose is lost during scaling, the edit commits at the last tracked pose and requires release before another grab. Menu/carousel transitions suppress held triggers.
- **Thumbsticks:** left stick flies forward/back and strafes left/right along
  the current combined virtual and physical view axes, even when looking
  vertically. Looking down and moving forward descends; strafing can also move
  vertically when the view is tilted or rolled. Right stick smoothly looks left,
  right, up and down around the current head position. Physical head movement
  and roll remain active. Rates are 1.5 m/s and 1.5 rad/s at full deflection,
  with a radial 0.2 deadzone and a 50 ms maximum locomotion timestep. Because
  virtual pitch tilts the whole tracking space, a subsequent physical head turn
  can make the horizon appear tilted; removing that effect dynamically would
  also rotate the scene during physical head movement. Sticks never manipulate a selection. The right stick browses exclusively while the
  carousel is open. Locomotion pauses while using the tools panel or triggers,
  during a grab, and on untracked/stale views or focus/profile interruption;
  after a pause each stick must return to neutral before it can resume. Vive
  trackpads use touch: an active, untouched pad is neutral; an inactive pad or
  touch action is not treated as a release.
- **Grip pose quality:** the shell locates a separate grip action space for each
  controller and records VALID and TRACKED independently. A VALID-but-untracked
  grip can neither start nor update a grab, and there is no head-gaze fallback
  for the physical pose. Losing grip or head/view tracking during a grab commits
  at the last good pose. Selection and new grabs are disabled while the view is untracked; a
  real trigger release after tracking returns is required. Profile changes
  and skipped/paused XR frames do the same.
- **Left Menu:** one press opens/closes the real tools panel (**Models**,
  **Undo**, **Redo**, **Delete**, **Backdrop**, **Lighting**), latched in the world in front of
  the viewer so pointing at it is stable. Aim with the right controller and pull
  the trigger: Models/Undo/Redo/Backdrop fire on release over the same button;
  Delete is a 0.6 s hold with the red charge ramp and cancels if released early.
  A miss is captured and swallowed (nothing clicks through). Opening the panel
  commits a live grab; a trigger held across a panel/carousel transition must
  return to release before it can act again.
- **Lighting:** select Lighting on the tools panel. Light left/right rotates a
  distant light around the vertical axis; Light up/down adjusts its elevation
  in 15-degree steps. These are world-space controls, not a movable local lamp:
  the direction applies equally to the entire scene. Ambient -/+ adjusts the
  uniform fill in 5% steps (0–100%). Reset light restores the default; Back
  returns to the tools panel. Each change is an undo/redo step, including
  reset. Settings last for the current editor session only. Materials use a
  simple directional diffuse plus ambient approximation, not shadows, light
  colour, reflections or measured real-world lighting. The controls do not
  access passthrough imagery. On 06/11/2026 platinum WiVRn Flatpak 26.9
  returned 61 instance extensions via xrEnumerateInstanceExtensionProperties;
  none exposed light estimation (including XR_ANDROID_light_estimation). This
  does not establish capabilities of other runtimes or future WiVRn versions.
  Meta documents passthrough as a compositor layer without app image access:
  https://developers.meta.com/vr/documentation/native/android/mobile-passthrough/
  Khronos describes XR_ANDROID_light_estimation as an unratified extension:
  https://registry.khronos.org/OpenXR/specs/1.1/man/html/XR_ANDROID_light_estimation.html
- **Carousel:** **A** or the panel's **Models** button opens it. **Right stick
  X** browses the ring (one step per flick, held repeats every 0.25 s). The
  View loss or a skipped frame pauses browsing; the right stick must return to
  neutral after tracking resumes before it can browse again. The
  **right trigger loads the centred model — no pointing at it** — but only once
  the ring is at rest, the right stick is active and neutral (or the right
  controller is confirmed as a stick-less simple controller), and that centre
  slot has finished loading and rested for 0.12 s. A trigger press while it is loading or
  unsettled is consumed: it never replays when the load completes, so it cannot
  place a different model than the one visible at the press. The centre
  preview wears the cool rim while a trigger press would place now (readiness,
  not aim). A temporarily inactive or invalid stick on other profiles cannot
  authorise placement. A simple controller can place the centred item, but
  cannot browse using a stick.
  Placement is at natural size on the floor below the centred preview,
  one undo step, and closes the carousel.
- **Menu**, **A**, **B** and **Y** require a tracked view and an active physical release after
  an inactive action, focus/profile interruption or skipped frame before a new
  press can toggle the panel or carousel or invoke undo/redo. Interrupted presses are discarded.
- **A** still opens/closes the carousel directly, undo (**B**)/redo (**Y**) and
  hold-**X** delete remain, and **grip** remains the part-selection modifier.
  Thumbstick clicks do nothing. Vive trackpad clicks are not undo/redo: use
  the tools panel instead, so navigation cannot accidentally alter history.
  Quest face-button shortcuts are unchanged. A view interruption cancels a
  Delete hold; release after tracking returns is required before another hold.
  Neither stick moves, rotates or
  scales a model.
- **Hands:** tracked hand/arm surfaces still render, depth-tested against the
  scene, and the **left palm-up pinch** (using the physical tracking pose) still opens the tools panel, latched
  in the scene rather than following the palm. Hand
  pinch/grasp no longer selects, grabs, moves or scales models; the old hand-grab
  maths and tests are retained for future arm interaction. The hand menu closes
  on palm tracking loss.
  Models in the hand-only menu requires switching to the right controller.

**Verification status:** strict build; core, hand, XR-view and hand-mesh tests;
state-machine regressions for carousel placement gating (including stick-less
placement, loading, settling, browsing and held-trigger latch), trigger
select-then-grab, head and controller tracking loss, Vive/Quest bindings,
tools-panel capture/Delete hold and history; offscreen,
textured, blend and perf smokes. The offscreen smoke also renders and checks the
panel, and covers controller move/rotation, joined scaling and hierarchy, refusal of
an untracked pose, undo and release. The **XR frame smoke was not available**:
it was not run during this handover to avoid disturbing the active XR session.
Runtime bindings and headset usability remain **unverified**.

The **bounding box is not a click target**: pressing empty space next to a model
selects nothing. The selection is shown by a neutral angle-based shader rim, not a silhouette border; the mode label is
desktop-only (there is no fixed MOVE label in VR). The carousel is a
**stationary, upright ring** of library models: the centred model is staged along the opening view direction
and the previous/next models sit visible at the sides as cheap geometry-only
previews (textures load when a model is placed; each preview is labelled with
its filename stem). The library folder (`~/Downloads/VR` by default,
`--library DIR`) is rescanned every time the carousel opens — drop a file in and
place it without restarting. **Hold X** to delete: the selection ramps red and
the delete fires at full charge (0.6 s); release before that to cancel. Changing
the selection or opening or closing the carousel or tools panel cancels the
charge; release X and press it again before deleting. **B**
(right) / **Y** (left) on Touch undo/redo, trackpad click on Vive. Passthrough
is the DEFAULT view: the real world with the models drawn over it (the
**Backdrop** button, or desktop **B**, toggles the grid back). VR flight and look change the virtual tracking-to-world rig, not the physical
room. In passthrough, virtual models will no longer align with physical surroundings
after stick movement or rotation; artificial pitch also tilts the virtual world
against gravity. This can be disorienting and create collision risk: move slowly,
keep a clear physical play area, and rely on the headset boundary, not virtual
models, for real-world safety. There is no real-room collision detection or
rig reset control. The compositor receives the original eye poses, retaining
stereo and physical head roll. Desktop keeps its keyboard
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
A clip targeting a skin joint is preferred, followed by one targeting a joint
ancestor, then an unrelated object clip. The choice is automatic; there is no
selector.

Joint matrices are `inverse(meshWorld) * jointWorld * inverseBindMatrix` per
the glTF spec, built from the scene's live world matrices after the node pose
is applied, so animated *and* hand-edited hierarchies both drive the skin. The
node's own transform cancels exactly: moving the whole model re-parents the
draw but does not reskin it. Inverse bind matrices must be MAT4 float, affine
(bottom row `0,0,0,1`) and one per joint; a malformed skin is logged and its
meshes render rigid instead of hiding the model. `JOINTS_1+` (more than four
influences), sparse skin accessors, morph targets and joints outside the
selected scene stay unsupported; those primitives render in the bind pose.
Placed skins with no playable clip are evaluated at the static pose, and
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
Ray picking and selection bounds use the live deformed vertices. The ray-pick
BVH is rebuilt lazily after a deformation; repeated picks of large changing
meshes may therefore be costly. Physical hand
contact still builds its triangle BVH once per mesh (a build costs ~1 µs per
triangle, measured: 211 ms at 200k, 2.4 s at 2M), so grasping an *animated*
character tests the surface from when the hand first reached it, not the
current pose; use pinch/ray selection for exact contact. Deleting an animated
model still leaves its registered GPU mesh (resources are never released, see
Design notes).

`--smoke-skinned` renders the synthetic two-joint fixture that
`run-tests.sh` writes to `.build/test-skinned.glb` (bind pose → animated,
moved bounds/picking, changed pixels, and immediate joint undo/redo/cancel
picking and rendering without an animation tick); it needs an offscreen Vulkan device.
Headset-side appearance still has to be judged by eye.

A comparison with Centre Stage identified the missing PBR materials,
transmission, emissive, sRGB material pipeline and studio lighting. This change
only fixes normal transforms and Vulkan blend-upload synchronisation; it does
not claim Centre Stage appearance parity. Passthrough does not supply real
room geometry for collision.

### Optional controller arm visuals

Controller arms load automatically from `.build/controller-arm-source.glb` when
present. Set `GENESIS_CONTROLLER_ARMS_GLB` to another path, or to an empty value
to disable them. No asset is bundled. The tested source is Felipe Bueno’s
[Male Hand Rigged - Human Anatomy](https://sketchfab.com/3d-models/male-hand-rigged-human-anatomy-ecbe20f0d3764050b2e8a0487908ee99).
The local source archive is a **rip**, not an official download. The
[Sketchfab model API](https://api.sketchfab.com/v3/models/ecbe20f0d3764050b2e8a0487908ee99)
reported `isDownloadable: false` and an empty licence on 06/11/2026; the
archive contains no model licence. Rights to use or redistribute this asset
remain unverified. Do not redistribute the archive or asset; obtain the
rights holder’s permission before use. No asset is distributed with this project
and the application performs no download.

On platinum, if permission has been obtained, extract the skinned GLB from
the existing archive (this writes or replaces only the specified local build
file):

```sh
mkdir -p .build
tmp="$(mktemp .build/controller-arm-source.XXXXXX)"
if unzip -p "$HOME/Downloads/VR/Props/Male_Hand_Rigged_-_Human_Anatomy/original/source-download.zip" \
  "Male_Hand_Rigged_-_Human_Anatomy.glb" > "$tmp" && [ -s "$tmp" ]; then
  mv -f "$tmp" .build/controller-arm-source.glb
else
  rm -f "$tmp"
  echo "Could not extract controller rig" >&2
fi
```

The installed rig is picked up automatically by a normal XR launch. To override
it with another compatible GLB for which you have rights:

```sh
bash run-editor.sh --xr
GENESIS_CONTROLLER_ARMS_GLB="/path/to/other-rig.glb" bash run-editor.sh --xr
GENESIS_CONTROLLER_ARMS_GLB="" bash run-editor.sh --xr
```

The library copy described above has no skin and will be rejected; the original
source archive contains a skinned GLB. The rig must provide both named deform-bone
chains and separate skinned meshes. Invalid or unavailable files disable the
visuals with a diagnostic. Initialisation checkpoints renderer registrations and
rolls back registered meshes and textures on failure or exception, preserving
existing assets. A failed rollback is retried up to three times unless the
renderer reports device loss; if it cannot be completed, the XR session shuts
down rather than reusing the renderer. Shutdown must establish GPU idle (or
device loss); an unexpected persistent wait failure still aborts rather than
risking GPU use-after-free. A per-frame solver failure hides only that side for the frame, resets its elbow
continuity and retries on later frames; its diagnostic is limited to once per
five seconds per side. Mesh/renderer update failures and exceptions disable only
the affected side with a diagnostic; a fatal renderer error stops the XR session. Arm
visuals are not
scene objects, pick targets or part of undo/history. They are depth-tested with
scene geometry; blended materials follow the renderer’s normal transparency
ordering. The offscreen smoke uses the supplied real asset and checks each
side, mesh inversion and deformation validation including a singular transform,
yaw trajectories across both verticals in both directions, vertical startup
and focused/tracked placement gating, transient solver recovery, frame exception
isolation, material alpha, transactional rollback and reuse, curl, pose motion,
front/back occlusion and each side obscured while
the opposite side remains visible. It requires Vulkan and does
not verify headset appearance.

Both controller sides use their independently tracked physical grip pose and
analogue trigger/squeeze values, without changing control actions. A lost grip,
view or focus hides the affected visuals rather than displaying a stale pose;
an unknown profile hides only that side and clears its elbow continuity. Profile
changes reset the affected elbow; an OpenXR global profile-change event resets
both because it does not identify a side, including an unobserved round trip.
A focus interruption resets both, even if focus has already returned. These
resets do not hide independently tracked controllers or alter their inputs;
actual tracked hands and the hand menu remain unchanged and replace the
controller visual on that side. The shoulder is estimated 20 cm below the head
with yaw-only orientation. Near vertical gaze, yaw freezes below a normalised
horizontal-forward length of 0.10 and resumes above 0.20. Ordinary tracked yaw
follows immediately; after vertical gaze, recovery is limited to 2 radians per
second (with a 50 ms timestep cap) until caught up. This avoids a sudden
180-degree flip, though shoulders can lag during recovery. Starting vertically
uses a default heading until a reliable one is
available. Two-bone IK and finger curls are approximations,
not body or finger tracking. Elbow bend is transported across frames; it is
path-dependent and may drift, and resets after tracking loss. The wrist currently uses the controller grip pose
without a device-specific grip-to-wrist calibration, and reach is clamped.
Expect alignment, elbow and finger shape to need headset/device testing and
calibration. CPU skinning and large embedded textures have a performance and
memory cost. No real-world collision is provided.

## Design notes

- **Committed renderer resources are registered, never individually released** (there is no
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
