/* neural-seam-native.h — the native evaluation stream's lifecycle surface:
 * the three functions the renderer needs alongside neural-seam.h's
 * neural_seam_evaluate (whose opaque NeuralSeamContext this factory makes).
 * The contract header itself is not modified.
 */
#ifndef NEURAL_SEAM_NATIVE_H
#define NEURAL_SEAM_NATIVE_H

#include "neural-seam.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One context per eye (neural-seam.h's rule): create two, never share one
 * between eyes and never reuse one across features. */
NeuralSeamContext* neural_seam_native_create(void);
void neural_seam_native_destroy(NeuralSeamContext* context);

/* The in-band label every presented output must carry (synthetic weights:
 * the transform is arbitrary). */
const char* neural_seam_native_label(void);

/* The boundary conversion self-test (clamping, ties-to-even anchors):
 * returns 0 when every internal check holds. The conversions are otherwise
 * unobservable through the synthetic transform, so this hook is their
 * independent anchor. */
int neural_seam_native_conversion_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_SEAM_NATIVE_H */
