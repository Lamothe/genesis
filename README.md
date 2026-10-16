# vr/editor — 3D scene editor core

Multi-model scene manipulator (move, rotate, scale) for the DLSS5 VR testbed,
built to run **in parallel** with the native neural-render migration stream and
have the neural renderer bolted on later through the seam contract
(`vr/NEURAL-SEAM.md`, `vr/neural-seam.h`).

## Layout

- `include/vecmath.h` — Vec3/Quat/Mat4, TRS, affine inverse, decompose
- `include/scene.h`, `src/scene.cpp` — transform hierarchy, cached world
  matrices, `epoch()` discontinuity counter
- `include/picking.h`, `src/picking.cpp` — ray/AABB, ray/triangle (Möller–
  Trumbore), closest-hit selection
- `include/gizmo.h`, `src/gizmo.cpp` — translate/rotate/scale drags, world/local
  space, axis/plane/free handles, snapping, drift-free cumulative application
- `tests/test_core.cpp` — maths and behaviour tests

## Build and test

```sh
bash vr/editor/run-tests.sh
```

Compiles with `g++ -std=c++20 -Wall -Wextra -Werror` and runs the test binary.
Thin shell launcher only: **all project logic is typed C/C++ (project rule — no
Python)**.

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

1. Vulkan desktop shell (single stack; matches the native CUDA evaluator) —
   viewport, camera, handle drawing, mesh picking.
2. glTF 2.0 (`.glb`) + OBJ loading in C++ (no converter scripts).
3. WiVRn/OpenXR VR shell: stereo views through the same interface.
4. Bind `neural_seam_evaluate` to the native evaluator when the migration
   stream's full native evaluation lands; conform the seam's convention table to
   the pinned native input conventions at that point.

## Git

Developed on branch `scene-editor` (worktree `~/Projects/dlss-testbed-editor`)
to stay isolated from the migration session working on `main`. Merge into
`main` when the native evaluation stream is ready to consume the seam.
