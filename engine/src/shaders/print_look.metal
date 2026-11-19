// print_look.metal -- the print look's kernels (`pipeline/print_look.cpp`): Print, Color Head,
// Expand and the positive Film Grain on the finished linear print. Reference: grain-lab/look.py.
#include "spk_common.h"

constant float3 kLookY = float3(0.2126f, 0.7152f, 0.0722f);
constant float kLookGamma = 2.2f;

static inline float3 look_enc(float3 x) { return sign(x) * pow(abs(x), 1.0f / kLookGamma); }
static inline float3 look_dec(float3 x) { return sign(x) * pow(abs(x), kLookGamma); }

// Shadow / midtone / highlight weights of an encoded luminance.
static inline float3 look_zones(float l) {
    l = clamp(l, 0.0f, 1.0f);
    float ws = (1.0f - l) * (1.0f - l), wh = l * l;
    return float3(ws, max(1.0f - ws - wh, 0.0f), wh);
}

// p: {exposure, contrast, density, saturation, yb, mg, cr, tone_s, tone_m, tone_h, preserve, impact,
//     black, white, -, -}
kernel void spk_look_tone(device const float* rgb [[buffer(0)]],
                          device const float* p [[buffer(1)]],
                          device const uint* count [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    float3 c = max(float3(rgb[3u * i], rgb[3u * i + 1u], rgb[3u * i + 2u]), 0.0f);
    // Print
    c *= exp2(p[0]);
    float y = max(dot(c, kLookY), 1e-6f);
    if (p[1] != 1.0f) {
        float y2 = 0.18f * pow(y / 0.18f, p[1]);
        c *= y2 / y; y = y2;
    }
    if (p[2] != 0.0f) {
        float3 d = -log10(max(c, 1e-6f));
        float dm = (d.x + d.y + d.z) / 3.0f;
        c = pow(float3(10.0f), -(dm + (d - dm) * (1.0f + p[2] / 100.0f)));
        c *= y / max(dot(c, kLookY), 1e-6f);
    }
    if (p[3] != 100.0f) c = y + (c - y) * (p[3] / 100.0f);
    // Color Head
    if ((p[4] != 0.0f || p[5] != 0.0f || p[6] != 0.0f) && p[11] > 0.0f) {
        float3 z = look_zones(look_enc(float3(max(dot(c, kLookY), 0.0f))).x);
        float kz = z.x * (1.0f + p[7] / 100.0f) + z.y * (1.0f + p[8] / 100.0f) + z.z * (1.0f + p[9] / 100.0f);
        float3 dens = float3(p[6], p[5], -p[4]) * 0.01f * (p[11] / 100.0f);   // +red, +green, +yellow
        float3 c2 = c * pow(float3(10.0f), dens * kz);
        if (p[10] > 0.0f) {
            float r = pow(max(dot(c, kLookY), 1e-6f) / max(dot(c2, kLookY), 1e-6f), p[10] / 100.0f);
            c2 *= r;
        }
        c = c2;
    }
    // Expand
    if (p[12] != 0.0f || p[13] != 100.0f) {
        float3 e = look_enc(c);
        e = (e - p[12] / 100.0f) / max((p[13] - p[12]) / 100.0f, 0.05f);
        c = look_dec(e);
    }
    out[3u * i] = c.x;
    out[3u * i + 1u] = c.y;
    out[3u * i + 2u] = c.z;
}

// Three standard normal draws a pixel. meta: {count, stream, seed}
kernel void spk_look_noise(device const uint* meta [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= meta[0]) return;
    Rng rng(i, meta[1], meta[2]);
    out[3u * i] = rng.normal();
    out[3u * i + 1u] = rng.normal();
    out[3u * i + 2u] = rng.normal();
}

// mono: (fine, coarse, -); colf / colc: colour fine / coarse RGB, all blurred.
// p: {amount, shadows, midtones, highlights, color, norm_fine, norm_coarse, toe, -, -}
kernel void spk_look_grain(device const float* rgb [[buffer(0)]],
                           device const float* mono [[buffer(1)]],
                           device const float* colf [[buffer(2)]],
                           device const float* colc [[buffer(3)]],
                           device const float* p [[buffer(4)]],
                           device const uint* count [[buffer(5)]],
                           device float* out [[buffer(6)]],
                           uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    float3 c = float3(rgb[3u * i], rgb[3u * i + 1u], rgb[3u * i + 2u]);
    float nf = p[5], nc = p[6], toe = p[7];
    float m = 0.8f * mono[3u * i] * nf + 0.6f * mono[3u * i + 1u] * nc;
    float3 col = 0.8f * float3(colf[3u * i], colf[3u * i + 1u], colf[3u * i + 2u]) * nf
               + 0.6f * float3(colc[3u * i], colc[3u * i + 1u], colc[3u * i + 2u]) * nc;
    float cm = p[4] / 100.0f;
    float3 g = sqrt(1.0f - cm) * m + sqrt(cm) * col;
    float3 z = look_zones(dot(look_enc(max(c, 0.0f)), kLookY));
    float strength = (p[1] * z.x + p[2] * z.y + p[3] * z.z) / 50.0f;
    float k = (p[0] / 100.0f) * 0.6f * 0.6931471805599453f * strength;
    c = (c + toe) * exp(k * g - 0.5f * k * k) - toe;
    out[3u * i] = c.x;
    out[3u * i + 1u] = c.y;
    out[3u * i + 2u] = c.z;
}
