// scene_tone.metal -- Tone's kernels (`pipeline/scene_tone.cpp`): Capture
// One-style Highlight, Shadow, White and Black on the scene before the film.
// Reference: c1-hdr-lab/c1tone.py; every step below is one line there.
#include <metal_stdlib>
using namespace metal;

// (L, L^2, 0), L = log2(Y / mid) + c, clamped to [-16, 8] stops.
// p: {yr, yg, yb, mid, c}
kernel void spk_tone_prep(device const float* rgb [[buffer(0)]],
                          device const float* p [[buffer(1)]],
                          device const uint* count [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    float y = p[0] * max(rgb[3u * i], 0.0f) + p[1] * max(rgb[3u * i + 1u], 0.0f) + p[2] * max(rgb[3u * i + 2u], 0.0f);
    float L = clamp(log2(max(y, 1e-6f) / p[3]) + p[4], -16.0f, 8.0f);
    out[3u * i] = L;
    out[3u * i + 1u] = L * L;
    out[3u * i + 2u] = 0.0f;
}

// The self-guided filter's coefficients from the means of L and L^2:
// a = var / (var + eps), b = mean - a * mean.  p: {eps}
kernel void spk_tone_coef(device const float* m [[buffer(0)]],
                          device const float* p [[buffer(1)]],
                          device const uint* count [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    float mL = m[3u * i], mLL = m[3u * i + 1u];
    float var = max(mLL - mL * mL, 0.0f);
    float a = var / (var + p[0]);
    out[3u * i] = a;
    out[3u * i + 1u] = mL - a * mL;
    out[3u * i + 2u] = 0.0f;
}

// Nearly the larger of two same-signed pulls (p = 6); opposite signs add.
static float tone_combine(float a, float b) {
    if (a * b <= 0.0f) return a + b;
    float m = pow(pow(abs(a), 6.0f) + pow(abs(b), 6.0f), 1.0f / 6.0f);
    return a < 0.0f ? -m : m;
}

// out = rgb * 2^dEV.  guide: (L, L^2, 0); ab: means of (a, b, 0).
// p: {highlight, white, shadow, black (stops at full weight), local_hs, local_wb, dev_min, dev_max}
kernel void spk_tone_apply(device const float* rgb [[buffer(0)]],
                           device const float* guide [[buffer(1)]],
                           device const float* ab [[buffer(2)]],
                           device const float* p [[buffer(3)]],
                           device const uint* count [[buffer(4)]],
                           device float* out [[buffer(5)]],
                           uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint i = thread_position_in_grid.x;
    if (i >= count[0]) return;
    float L = guide[3u * i];
    float base = ab[3u * i] * L + ab[3u * i + 1u];
    float bh = L + p[4] * (base - L);
    float bw = L + p[5] * (base - L);

    float w_h = 0.3f * smoothstep(-1.5f, 2.0f, bh) + 0.7f * smoothstep(1.2f, 3.2f, bh);
    float w_w = 0.1f * smoothstep(-1.0f, 2.0f, bw) + 0.9f * smoothstep(1.6f, 2.9f, bw);
    float z = 1.5f * (0.2f - bh);                                   // softplus(z) / 1.5 / 3.6, at most 1.5
    float w_s = min((max(z, 0.0f) + log(1.0f + exp(-abs(z)))) / 1.5f / 3.6f, 1.5f);
    float w_k = (1.0f - smoothstep(-4.5f, -0.5f, bw)) + 0.5f * max(-4.5f - bw, 0.0f);

    float dev = tone_combine(p[0] * w_h, p[1] * w_w) + p[2] * w_s + p[3] * w_k;
    float g = exp2(clamp(dev, p[6], p[7]));
    out[3u * i] = rgb[3u * i] * g;
    out[3u * i + 1u] = rgb[3u * i + 1u] * g;
    out[3u * i + 2u] = rgb[3u * i + 2u] * g;
}
