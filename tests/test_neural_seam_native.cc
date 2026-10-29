/* test_neural_seam_native.cc — the seam adapter over Genesis's contract ABI
 * (neural-seam.h) fed with synthetic stereo frames: rule 1 (first frame
 * resets), rule 2 (an epoch change resets, enforced even when the caller
 * forgets), rule 3 (editor-set reset wins), per-eye independence, the
 * FP16/RGB8 boundary conventions (alpha = 1.0, clamped colour) and the
 * output-untouched-on-error contract. Full-network frames: each
 * neural_seam_evaluate costs two ~8 s evaluations, so the suite runs for
 * about two minutes. Synthetic weights only.
 */
#include <cstdio>
#include <cstring>
#include <vector>

#include "neural-seam-native.h"

static int g_failures = 0;
static long long g_checks = 0;

static void check(int ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        std::printf("FAILED (%d): %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check((expr) != 0, #expr, __LINE__)

enum {
    kWidth = 64,
    kHeight = 72,
    kPixels = kWidth * kHeight,
    kParallax = 4
};

/* Test-side float -> fp16 (round to nearest even; the helper is only for
 * generating inputs, so a few ulps of bias cannot hide a seam defect). */
static unsigned short to_half(float value) {
    unsigned int bits;
    std::memcpy(&bits, &value, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xFFu) - 127 + 15;
    unsigned int frac = bits & 0x007FFFFFu;
    if (exp <= 0) {
        return (unsigned short)sign;
    }
    if (exp >= 31) {
        return (unsigned short)(sign | 0x7C00u);
    }
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

/* Synthetic per-eye input: deterministic, with real horizontal disparity. */
static void fill_eye(std::vector<unsigned short>& rgba, int frame, int eye) {
    rgba.resize((size_t)(4 * kPixels));
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            int sx = x + (eye == 1 ? kParallax : 0);
            float r = (float)((y * 3 + sx * 5 + 37 * frame) % 256) / 255.0f;
            float g = r * 0.5f + 0.1f;
            float b = (float)((sx % 16) * (frame % 17)) / 16.0f;
            int i = y * kWidth + x;
            rgba[(size_t)(4 * i) + 0] = to_half(r);
            rgba[(size_t)(4 * i) + 1] = to_half(g);
            rgba[(size_t)(4 * i) + 2] = to_half(b);
            rgba[(size_t)(4 * i) + 3] = 0x3C00u;
        }
    }
}

static void make_frame(NeuralSeamFrame& frame, int seed,
                       std::vector<unsigned short>& left,
                       std::vector<unsigned short>& right,
                       unsigned long long discontinuity_id) {
    fill_eye(left, seed, 0);
    fill_eye(right, seed, 1);
    for (int e = 0; e < 2; ++e) {
        NeuralSeamEye& eye = frame.eyes[e];
        std::memset(&eye, 0, sizeof(eye));
        eye.render_width = kWidth;
        eye.render_height = kHeight;
        eye.output_width = kWidth;
        eye.output_height = kHeight;
    }
    frame.eyes[0].colour_srgb_f16 = left.data();
    frame.eyes[1].colour_srgb_f16 = right.data();
    frame.discontinuity_id = discontinuity_id;
}

static void make_output(NeuralSeamOutput& output,
                        std::vector<unsigned short>& out_left,
                        std::vector<unsigned short>& out_right) {
    out_left.assign((size_t)(4 * kPixels), 0u);
    out_right.assign((size_t)(4 * kPixels), 0u);
    output.eyes[0].colour_srgb_f16 = out_left.data();
    output.eyes[1].colour_srgb_f16 = out_right.data();
}

static int outputs_equal(const std::vector<unsigned short>& a,
                         const std::vector<unsigned short>& b) {
    return a.size() == b.size()
        && std::memcmp(a.data(), b.data(), a.size() * sizeof(unsigned short))
            == 0;
}

int main(void) {
    /* The boundary conversions' independent anchor (clamping, RNE). */
    CHECK(neural_seam_native_conversion_selftest() == 0);

    const char* label = neural_seam_native_label();
    CHECK(label != nullptr);
    CHECK(label != nullptr && std::strstr(label, "synthetic weights") != nullptr);
    CHECK(label != nullptr && std::strstr(label, "arbitrary transform") != nullptr);

    std::vector<unsigned short> l, r, ol, or_, rl, rr, cl, cr;

    /* ---- rule 1: a fresh context resets even when the caller says 0 ----
     * (The comparison is as strong as outputs allow: on a fresh handle the
     * evaluator's history is already zero, so this cannot distinguish the
     * adapter's rule-1 enforcement from a clean start. The enforcement
     * claim rests on the code path, not on this check.) */
    NeuralSeamFrame frame = {};
    make_frame(frame, 1, l, r, 100);
    NeuralSeamOutput output = {};
    make_output(output, ol, or_);
    NeuralSeamContext* a0 = neural_seam_native_create();
    NeuralSeamContext* a1 = neural_seam_native_create();
    NeuralSeamContext* contexts_a[2] = {a0, a1};
    CHECK(a0 != nullptr && a1 != nullptr);
    frame.eyes[0].reset = 0;
    frame.eyes[1].reset = 0;
    CHECK(neural_seam_evaluate(contexts_a, &frame, &output) == 0);

    NeuralSeamOutput ref = {};
    make_output(ref, rl, rr);
    NeuralSeamContext* b0 = neural_seam_native_create();
    NeuralSeamContext* b1 = neural_seam_native_create();
    NeuralSeamContext* contexts_b[2] = {b0, b1};
    frame.eyes[0].reset = 1;
    frame.eyes[1].reset = 1;
    CHECK(neural_seam_evaluate(contexts_b, &frame, &ref) == 0);
    CHECK(outputs_equal(ol, rl));
    CHECK(outputs_equal(or_, rr));

    /* History is actually accumulated when no reset is due: a same-id
     * follow-up frame with reset = 0 over frame-1 history must DIFFER from
     * a fresh forced-reset evaluation of that frame (an adapter that resets
     * on every call fails here). */
    {
        make_frame(frame, 2, l, r, 100); /* same id: no epoch reset */
        frame.eyes[0].reset = 0;
        frame.eyes[1].reset = 0;
        NeuralSeamOutput step = {};
        make_output(step, cl, cr);
        CHECK(neural_seam_evaluate(contexts_a, &frame, &step) == 0);

        NeuralSeamOutput fresh2 = {};
        make_output(fresh2, rl, rr); /* the rule-1 refs are spent; reuse */
        NeuralSeamContext* d0 = neural_seam_native_create();
        NeuralSeamContext* d1 = neural_seam_native_create();
        NeuralSeamContext* contexts_d[2] = {d0, d1};
        frame.eyes[0].reset = 1;
        frame.eyes[1].reset = 1;
        CHECK(neural_seam_evaluate(contexts_d, &frame, &fresh2) == 0);
        CHECK(!outputs_equal(cl, rl)); /* left: history vs fresh */
        CHECK(!outputs_equal(cr, rr)); /* right: history vs fresh */
        neural_seam_native_destroy(d0);
        neural_seam_native_destroy(d1);
    }

    /* The two eyes' outputs differ (disparity in, disparity out). ABI
     * independence itself is enforced by the aliasing rejection below. */
    CHECK(!outputs_equal(ol, or_));

    /* ---- rule 2: an epoch change resets even when the caller forgets --- */
    make_frame(frame, 2, l, r, 101); /* discontinuity_id changed vs 100 */
    frame.eyes[0].reset = 0;
    frame.eyes[1].reset = 0;
    NeuralSeamOutput out2 = {};
    make_output(out2, cl, cr);
    CHECK(neural_seam_evaluate(contexts_a, &frame, &out2) == 0);

    NeuralSeamOutput ref2 = {};
    make_output(ref2, rl, rr);
    NeuralSeamContext* c0 = neural_seam_native_create();
    NeuralSeamContext* c1 = neural_seam_native_create();
    NeuralSeamContext* contexts_c[2] = {c0, c1};
    frame.eyes[0].reset = 1;
    frame.eyes[1].reset = 1;
    CHECK(neural_seam_evaluate(contexts_c, &frame, &ref2) == 0);
    CHECK(outputs_equal(cl, rl));
    CHECK(outputs_equal(cr, rr));

    /* ---- rule 3: editor-set reset wins (same id, reset = 1) ----------- */
    make_frame(frame, 2, l, r, 101); /* same frame, same id as the call above */
    frame.eyes[0].reset = 1;
    frame.eyes[1].reset = 1;
    NeuralSeamOutput out3 = {};
    make_output(out3, ol, or_);
    CHECK(neural_seam_evaluate(contexts_a, &frame, &out3) == 0);
    CHECK(outputs_equal(ol, rl));
    CHECK(outputs_equal(or_, rr));

    /* ---- the boundary conventions: alpha = 1.0, clamped colour --------- */
    {
        int alpha_ok = 1;
        int colour_ok = 1;
        for (size_t i = 0; i < ol.size(); ++i) {
            if ((i % 4) == 3 && ol[i] != 0x3C00u) {
                alpha_ok = 0;
            }
            if ((i % 4) != 3 && (ol[i] & 0x7C00u) == 0x7C00u) {
                colour_ok = 0; /* no infinities/NaNs */
            }
        }
        CHECK(alpha_ok != 0);
        CHECK(colour_ok != 0);
    }

    /* ---- the output-untouched-on-error contract ----------------------- */
    {
        std::vector<unsigned short> untouched((size_t)(4 * kPixels), 0xDEADu);
        NeuralSeamOutput bad = {};
        bad.eyes[0].colour_srgb_f16 = untouched.data();
        bad.eyes[1].colour_srgb_f16 = untouched.data();
        CHECK(neural_seam_evaluate(nullptr, &frame, &bad) != 0);
        CHECK(neural_seam_evaluate(contexts_a, nullptr, &bad) != 0);
        NeuralSeamOutput missing = {};
        missing.eyes[0].colour_srgb_f16 = nullptr;
        missing.eyes[1].colour_srgb_f16 = untouched.data();
        CHECK(neural_seam_evaluate(contexts_a, &frame, &missing) != 0);
        /* Context aliasing (both eyes on one context) is ABI-forbidden. */
        NeuralSeamContext* aliased[2] = {a0, a0};
        CHECK(neural_seam_evaluate(aliased, &frame, &bad) != 0);
        /* A mismatched output extent (presented size != render size) is
         * refused: the evaluator is same-extent by design. */
        NeuralSeamFrame wide = frame;
        wide.eyes[0].output_width = kWidth / 2;
        CHECK(neural_seam_evaluate(contexts_a, &wide, &bad) != 0);
        int unchanged = 1;
        for (size_t i = 0; i < untouched.size(); ++i) {
            if (untouched[i] != 0xDEADu) {
                unchanged = 0;
            }
        }
        CHECK(unchanged != 0);
    }

    neural_seam_native_destroy(a0);
    neural_seam_native_destroy(a1);
    neural_seam_native_destroy(b0);
    neural_seam_native_destroy(b1);
    neural_seam_native_destroy(c0);
    neural_seam_native_destroy(c1);
    neural_seam_native_destroy(nullptr);

    if (g_failures == 0) {
        std::printf("neural-seam-native: %lld checks passed\n", g_checks);
        return 0;
    }
    std::printf("neural-seam-native: %d check(s) FAILED\n", g_failures);
    return 1;
}
