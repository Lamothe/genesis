/* neural-seam.h — contract between a renderer and the neural evaluator.
 *
 * Provisional (16/10/2026): input conventions follow vr/SCENE.md — display-referred
 * sRGB FP16 colour, D3D-projected depth (near 0.1 m / far 20 m), previous-minus-current
 * motion vectors in pixels. Revisit when the native evaluation stream pins its final
 * conventions; the shape of this contract should not have to change.
 *
 * Two independent evaluator contexts (eyes) are modelled from the start because the
 * Proton-era add-on path kept global mutable state per feature (vr/SCENE.md findings);
 * alternating eyes through one context mixes state.
 */
#ifndef NEURAL_SEAM_H
#define NEURAL_SEAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NeuralSeamEye {
  uint32_t render_width;  /* jittered low-res input */
  uint32_t render_height;
  uint32_t output_width;  /* presented resolution */
  uint32_t output_height;

  /* render_width * render_height * 4, RGBA, display-referred sRGB, FP16. */
  const uint16_t* colour_srgb_f16;

  /* render_width * render_height, D3D projected depth, near 0.1 m / far 20 m. */
  const float* depth;

  /* render_width * render_height * 2 (dx, dy), previous-minus-current, pixels. */
  const float* motion;

  /* Sub-pixel jitter applied to this render, in pixels. */
  float jitter_x;
  float jitter_y;

  /* 1 = no valid temporal history for this eye. */
  int32_t reset;
} NeuralSeamEye;

typedef struct NeuralSeamFrame {
  NeuralSeamEye eyes[2]; /* 0 = left, 1 = right. */
  /* Monotonic counter: the frame producer bumps it on any scene change that
   * invalidates temporal history — moving, rotating, scaling, creating,
   * deleting or reparenting an object (vredit::Scene::epoch() in
   * vr/editor/include/scene.h). Discontinuity handling:
   *   - First frame, or a context with no prior evaluation: reset = 1.
   *   - The code invoking neural_seam_evaluate tracks the last evaluated value;
   *     if it differs, it ORs 1 into both eyes' reset before the call.
   *   - If editor-set reset and the discontinuity rule disagree, reset wins. */
  uint64_t discontinuity_id;
} NeuralSeamFrame;

typedef struct NeuralSeamOutputEye {
  /* output_width * output_height * 4, caller-allocated, display-referred sRGB FP16. */
  uint16_t* colour_srgb_f16;
} NeuralSeamOutputEye;

typedef struct NeuralSeamOutput {
  NeuralSeamOutputEye eyes[2];
} NeuralSeamOutput;

typedef struct NeuralSeamContext NeuralSeamContext;

/* Evaluation entry point. Returns 0 on success. One context per eye, and a
 * context must never be shared between eyes or reused across features: the
 * Proton-era path kept global mutable state per feature, so the two-context
 * requirement is part of the ABI, not just of the docs. Binding to the native
 * evaluator is renderer-side work; the editor only produces NeuralSeamFrame
 * inputs and consumes NeuralSeamOutput results. */
int32_t neural_seam_evaluate(NeuralSeamContext* contexts[2],
                             const NeuralSeamFrame* frame,
                             NeuralSeamOutput* output);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_SEAM_H */
