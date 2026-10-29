/* neural-seam-native.cc — the "native evaluation stream": Genesis's
 * neural-seam.h ABI implemented over two dlss5_nr host-boundary handles
 * (third_party/dlss5-native/native-host.h), one context per eye exactly as
 * the contract requires (a context is never shared between eyes or reused
 * across features; in-call aliasing of both eyes to one context is
 * rejected). Contract rules implemented or enforced here:
 *
 *   rule 1 — a context with no prior evaluation runs with reset = 1
 *            (enforced here even if the caller forgot);
 *   rule 2 — a discontinuity_id change ORs 1 into both eyes' reset
 *            (the contract puts this on the invoker; it is enforced here
 *            too, idempotently, so an epoch change can never smear);
 *   rule 3 — editor-set reset wins: reset composition is a plain OR.
 *
 * Conventions (vr/VR-SEAM-M1-ALIGNMENT.md): colour_srgb_f16 is display-
 * referred sRGB RGBA FP16 across this boundary and RGB8 inside it (alpha
 * dropped on input, 1.0 on output; values clamp to [0, 1], ties to even
 * on the 8-bit rounding); depth, motion and jitter are carried but not
 * consumed by the synthetic evaluator. The evaluator is same-extent by
 * design (the host presents the upscaled result itself), so an output
 * extent that differs from the render extent is refused with the output
 * untouched. Synthetic weights: the transform is arbitrary — any presented
 * output must carry neural_seam_native_label() (dlss5_nr_label()).
 */
#include "neural-seam-native.h"

#include <cstdlib>
#include <cstring>

#include "native-host.h"

struct NeuralSeamContext {
    dlss5_nr_handle* handle = nullptr;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned long long last_discontinuity_id = 0;
    bool evaluated = false;
};

NeuralSeamContext* neural_seam_native_create(void) {
    return new NeuralSeamContext();
}

void neural_seam_native_destroy(NeuralSeamContext* context) {
    if (context == nullptr) {
        return;
    }
    if (context->handle != nullptr) {
        dlss5_nr_destroy(context->handle);
    }
    delete context;
}

const char* neural_seam_native_label(void) {
    return dlss5_nr_label();
}

namespace {

/* float <-> fp16 (round to nearest even). Clamp semantics of the
 * float_to_fp16 direction: magnitudes below the fp16 subnormal floor round
 * to signed zero and overflow rounds to infinity; NaN becomes +Inf. The
 * only caller feeds rgb/255 in [0, 1], so neither is reachable in-band;
 * the behaviour is documented so the helper is never reused blind. */
float fp16_to_float(unsigned short bits) {
    unsigned int sign = (unsigned int)(bits & 0x8000u) << 16;
    unsigned int exp = (unsigned int)(bits & 0x7C00u) >> 10;
    unsigned int frac = (unsigned int)(bits & 0x03FFu);
    unsigned int out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            /* Subnormal: normalise. */
            exp = 127 - 15 + 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x03FFu;
            out = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

unsigned short float_to_fp16(float value) {
    unsigned int bits;
    std::memcpy(&bits, &value, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xFFu) - 127 + 15;
    unsigned int frac = bits & 0x007FFFFFu;
    if (exp <= 0) {
        return (unsigned short)sign; /* clamp to signed zero */
    }
    if (exp >= 31) {
        return (unsigned short)(sign | 0x7C00u); /* clamp to infinity */
    }
    /* Round to nearest even on the 10-bit mantissa. */
    unsigned int mant = frac >> 13;
    unsigned int rest = frac & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (mant & 1u) != 0)) {
        ++mant;
        if (mant == 0x0400u) {
            mant = 0;
            ++exp;
            if (exp >= 31) {
                return (unsigned short)(sign | 0x7C00u);
            }
        }
    }
    return (unsigned short)(sign | ((unsigned int)exp << 10) | mant);
}

/* RGB8 <-> FP16 conversion at the boundary; colours clamp to [0, 1]. */
void f16_rgba_to_rgb8(const unsigned short* rgba, int pixels,
                      unsigned char* rgb) {
    for (int i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = fp16_to_float(rgba[4 * i + c]);
            if (!(v > 0.0f)) {
                v = 0.0f;
            }
            if (v > 1.0f) {
                v = 1.0f;
            }
            rgb[3 * i + c] = (unsigned char)(v * 255.0f + 0.5f);
        }
    }
}

void rgb8_to_f16_rgba(const unsigned char* rgb, int pixels,
                      unsigned short* rgba) {
    for (int i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            rgba[4 * i + c] =
                float_to_fp16((float)rgb[3 * i + c] / 255.0f);
        }
        rgba[4 * i + 3] = 0x3C00u; /* alpha = 1.0 */
    }
}

/* Everything evaluate_eye needs, checked up front so a validation failure
 * can never leave one eye's output written (the boundary's untouched-output
 * convention, preserved end to end). */
int validate_eye(const NeuralSeamEye* eye) {
    if (eye == nullptr || eye->colour_srgb_f16 == nullptr) {
        return 1;
    }
    if (eye->render_width == 0 || eye->render_height == 0) {
        return 1;
    }
    /* Same-extent by design: the host presents the upscaled result itself,
     * so an output extent that differs is a contract error, not a resize. */
    if (eye->output_width != eye->render_width
        || eye->output_height != eye->render_height) {
        return 1;
    }
    return 0;
}

int evaluate_eye(NeuralSeamContext* context, const NeuralSeamEye* eye,
                 unsigned short* out_f16) {
    unsigned int w = eye->render_width;
    unsigned int h = eye->render_height;

    /* A resolution change is an editor-side reset (rule 3): the temporal
     * history cannot span extents. The replacement is created before the
     * old handle is released, so a doomed create cannot drop a usable
     * context's history. */
    if (context->handle == nullptr
        || context->width != w || context->height != h) {
        dlss5_nr_handle* replacement = nullptr;
        if (dlss5_nr_create((int)w, (int)h, 0, &replacement) != DLSS5_NR_OK) {
            return 1;
        }
        if (context->handle != nullptr) {
            dlss5_nr_destroy(context->handle);
        }
        context->handle = replacement;
        context->width = w;
        context->height = h;
        context->evaluated = false;
    }

    int pixels = (int)(w * h);
    unsigned char* in_rgb = (unsigned char*)std::malloc((size_t)(3 * pixels));
    unsigned char* out_rgb = (unsigned char*)std::malloc((size_t)(3 * pixels));
    if (in_rgb == nullptr || out_rgb == nullptr) {
        std::free(in_rgb);
        std::free(out_rgb);
        return 1;
    }
    f16_rgba_to_rgb8(eye->colour_srgb_f16, pixels, in_rgb);

    int reset = eye->reset != 0 ? 1 : 0;
    if (!context->evaluated) {
        reset = 1; /* rule 1 */
    }
    dlss5_nr_status status =
        dlss5_nr_evaluate(context->handle, in_rgb, 3 * (int)w, out_rgb,
                          3 * (int)w, reset);
    std::free(in_rgb);
    if (status != DLSS5_NR_OK) {
        std::free(out_rgb);
        return 1;
    }
    rgb8_to_f16_rgba(out_rgb, pixels, out_f16);
    std::free(out_rgb);
    context->evaluated = true;
    return 0;
}

} /* namespace */

int neural_seam_native_conversion_selftest(void) {
    int failures = 0;
    unsigned short rgba[4];
    unsigned char rgb[3];

    /* Clamping: negative and NaN land on 0 (the !(v > 0) guard), positive
     * infinity and any value above 1 land on 255. */
    rgba[0] = 0xBC00u; /* -1.0 -> 0 */
    rgba[1] = 0x7C00u; /* +inf -> 255 */
    rgba[2] = 0x7E00u; /* NaN -> 0 */
    rgba[3] = 0x3C00u; /* alpha (ignored) */
    f16_rgba_to_rgb8(rgba, 1, rgb);
    failures += (rgb[0] == 0 && rgb[1] == 255 && rgb[2] == 0) ? 0 : 1;

    /* Round-trip anchors (ties to even): 1.0, 0.5 and 254/255. */
    failures += (float_to_fp16(1.0f) == 0x3C00u) ? 0 : 1;
    failures += (float_to_fp16(0.5f) == 0x3800u) ? 0 : 1;
    unsigned char one[3] = {255, 128, 0};
    unsigned short back[4];
    rgb8_to_f16_rgba(one, 1, back);
    failures += (back[0] == 0x3C00u && back[3] == 0x3C00u) ? 0 : 1;
    failures += (fp16_to_float(back[1]) > 0.501f
                 && fp16_to_float(back[1]) < 0.503f)
        ? 0 : 1;
    return failures;
}

int32_t neural_seam_evaluate(NeuralSeamContext* contexts[2],
                             const NeuralSeamFrame* frame,
                             NeuralSeamOutput* output) {
    if (contexts == nullptr || frame == nullptr || output == nullptr) {
        return 1;
    }
    if (contexts[0] == nullptr || contexts[1] == nullptr) {
        return 1;
    }
    /* The ABI forbids sharing a context between eyes (the Proton-era
     * global-state finding); in-call aliasing is detectable and rejected. */
    if (contexts[0] == contexts[1]) {
        return 1;
    }

    /* Validate BOTH eyes before touching either output: a validation
     * failure leaves every output buffer untouched. */
    for (int e = 0; e < 2; ++e) {
        if (output->eyes[e].colour_srgb_f16 == nullptr) {
            return 1;
        }
        if (validate_eye(&frame->eyes[e]) != 0) {
            return 1;
        }
    }

    /* Rule 2, enforced here as well as on the invoker side: an epoch change
     * ORs 1 into both eyes' reset (rule 3: reset wins either way). */
    int epoch_reset = 0;
    for (int e = 0; e < 2; ++e) {
        if (contexts[e]->evaluated
            && contexts[e]->last_discontinuity_id != frame->discontinuity_id) {
            epoch_reset = 1;
        }
    }

    int failed = 0;
    for (int e = 0; e < 2; ++e) {
        NeuralSeamEye eye = frame->eyes[e];
        if (epoch_reset != 0) {
            eye.reset = 1;
        }
        failed |= evaluate_eye(contexts[e], &eye,
                               output->eyes[e].colour_srgb_f16);
    }
    /* The new id is stored only after both eyes land: a partial failure
     * re-fires the epoch reset on retry (redundant resets, never a missed
     * one — deliberately NOT stored per eye, which would risk smearing). */
    if (failed == 0) {
        for (int e = 0; e < 2; ++e) {
            contexts[e]->last_discontinuity_id = frame->discontinuity_id;
        }
    }
    return failed;
}
