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

// sRGB transfer, linear <-> encoded, per channel (sign-symmetric). meta: {count, 1 = encode / 0 = decode}
static inline float look_srgb_enc(float x) {
    float a = abs(x);
    float e = a <= 0.0031308f ? 12.92f * a : 1.055f * pow(a, 1.0f / 2.4f) - 0.055f;
    return x < 0.0f ? -e : e;
}
static inline float look_srgb_dec(float x) {
    float a = abs(x);
    float l = a <= 0.04045f ? a / 12.92f : pow((a + 0.055f) / 1.055f, 2.4f);
    return x < 0.0f ? -l : l;
}
kernel void spk_look_srgb(device const float* in [[buffer(0)]],
                          device const uint* meta [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= meta[0]) return;
    for (uint c = 0u; c < 3u; ++c)
        out[3u * i + c] = meta[1] != 0u ? look_srgb_enc(in[3u * i + c]) : look_srgb_dec(in[3u * i + c]);
}

// A square, zero-phase 2-D filter on all three channels, mirrored at the edges. meta: {w, h, radius, count}
kernel void spk_look_fir(device const float* in [[buffer(0)]],
                         device const float* k [[buffer(1)]],
                         device const uint* meta [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= meta[3]) return;
    int w = int(meta[0]), h = int(meta[1]), r = int(meta[2]), n = 2 * r + 1;
    int x = int(i % uint(w)), y = int(i / uint(w));
    float3 acc = float3(0.0f);
    for (int dy = -r; dy <= r; ++dy) {
        int yy = mirror_index(y + dy, h);
        for (int dx = -r; dx <= r; ++dx) {
            int xx = mirror_index(x + dx, w);
            uint j = uint(yy * w + xx);
            acc += k[(dy + r) * n + (dx + r)] * float3(in[3u * j], in[3u * j + 1u], in[3u * j + 2u]);
        }
    }
    out[3u * i] = acc.x;
    out[3u * i + 1u] = acc.y;
    out[3u * i + 2u] = acc.z;
}

// The measured grain on the sRGB-encoded positive; writes linear.
// f0: (mono, colour R, colour G), f1: (colour B, -, -), both through the grain filter.
// t: std[L], shift[L] (encoded units), then {sqrt(1-w), sqrt(w), gR, gG, gB, level0, level step, L}
kernel void spk_look_grain2(device const float* enc [[buffer(0)]],
                            device const float* f0 [[buffer(1)]],
                            device const float* f1 [[buffer(2)]],
                            device const float* t [[buffer(3)]],
                            device const uint* count [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    uint L = uint(t[2u * 21u + 7u]);
    device const float* x = t + 2u * L;
    float3 e = float3(enc[3u * i], enc[3u * i + 1u], enc[3u * i + 2u]);
    float l = clamp(dot(e, kLookY), 0.0f, 1.0f);
    float pos = clamp((l - x[5]) / x[6], 0.0f, float(L - 1u));
    uint i0 = min(uint(pos), L - 2u);
    float fr = pos - float(i0);
    float sd = mix(t[i0], t[i0 + 1u], fr), sh = mix(t[L + i0], t[L + i0 + 1u], fr);
    float mono = f0[3u * i];
    float3 col = float3(f0[3u * i + 1u], f0[3u * i + 2u], f1[3u * i]);
    float3 g = float3(x[2], x[3], x[4]) * (x[0] * mono + x[1] * col);
    e += sh + sd * g;
    out[3u * i] = look_srgb_dec(e.x);
    out[3u * i + 1u] = look_srgb_dec(e.y);
    out[3u * i + 2u] = look_srgb_dec(e.z);
}
