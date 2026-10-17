# Neural seam contract — 16/10/2026

Single boundary between any renderer and the neural evaluator. The scene editor
(this project) produces `NeuralSeamFrame` inputs and consumes `NeuralSeamOutput`
results through `neural-seam.h`; the native evaluation stream binds
`neural_seam_evaluate` when the full native chain lands. The editor itself never
touches evaluator internals.

## Per-eye inputs

| Field | Meaning | Convention |
| --- | --- | --- |
| `colour_srgb_f16` | jittered low-res render | display-referred sRGB, RGBA FP16 |
| `depth` | D3D projected depth | near 0.1 m / far 20 m (vr/SCENE.md) |
| `motion` | previous-minus-current | pixels, (dx, dy) |
| `jitter_x/y` | sub-pixel jitter of this render | pixels |
| `reset` | no valid temporal history | 1 = drop history |

Two eyes, two independent evaluator contexts — `neural_seam_evaluate` takes
`contexts[2]` to make this part of the ABI. Do not share a context between eyes
or reuse one across features: the Proton-era add-on kept global mutable state per
feature and would mix eye state (see findings in `vr/INTEGRATION.md`).

## Discontinuity rule

Moving, rotating, scaling, creating, deleting or reparenting an object is a
large-motion event the temporal loop must not smear. The editor bumps
`vredit::Scene::epoch()` on any such change and copies it into
`NeuralSeamFrame::discontinuity_id`. Composition is defined precisely:

1. First frame, or a context with no prior evaluation: evaluate with `reset = 1`.
2. Otherwise, the code invoking `neural_seam_evaluate` compares
   `discontinuity_id` against the last value it evaluated; if it differs, it ORs
   1 into both eyes' `reset` before the call and stores the new value.
3. The editor may set `reset` itself for its own reasons (camera teleport,
   resolution change). If editor-set `reset` and rule 2 disagree, `reset` wins.

Object *selection* does not bump the epoch (highlighting is drawn after the
neural pass or in the composited layer, never inside the neural inputs).

## Ownership

The caller allocates output buffers; the evaluator fills them. The seam carries
pointers for the duration of one `neural_seam_evaluate` call only.

## Status

Provisional: conventions are inherited from the Proton-era scene harness
(`vr/SCENE.md`). When the native evaluation stream pins its expected inputs
(depth range, motion units, colour encoding), conform this document and header to
those values — the structure of the contract should not change, only the
convention table.
