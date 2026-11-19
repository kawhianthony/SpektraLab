// print_look.cpp -- the print look: Expand, Print, Color Head and a positive Film Grain on the
// finished print (linear, in the output colour space), before EDR and the output encoding.
//
// Controls after Dehancer's panels; the maths is our own (darkroom and film-grain behaviour).
// Reference: grain-lab/look.py, step for step:
//
//   Print       exposure (stops), tonal contrast (log slope of luminance about mid-grey), colour
//               density (chroma in log density, luminance kept), saturation (100 = unchanged)
//   Color Head  yellow-blue / magenta-green / cyan-red filtration (a point = 0.01 density), scaled
//               per tonal zone by its Tone (-100..100: none..double), luminance restored by
//               Preserve Exposure, faded by Impact
//   Expand      black / white point on the encoded positive (per cent of the range)
//   Grain       the measured model (print_look_grain_data.hpp, grain-lab): on the sRGB-encoded
//               positive, softening (a narrow share inside a wide blur, by Resolution, Size and
//               Amount), then white noise through the measured grain filter (resampled to this
//               render's pixel size, so preview and export match), mono / colour by Color, at a
//               strength and tone shift read from tables by tone and the three zone sliders.
//   Order       Print, Color Head, Grain, then Expand (it wins back what the grain takes from the
//               black and white points).
//
// A whole-plane stage (`kPrintStages`): every default is a bypass and the stage hands its input on
// untouched. Metal only; the CMake targets hand the image on.
#include <algorithm>
#include <cmath>
#include <vector>

#include "pipeline.hpp"
#include "print_look_grain_data.hpp"
#include "timer.hpp"

namespace spk {

namespace {

namespace gm = grain_model;

double interp_colour(double c, int col) {
    for (int i = 0; i + 1 < gm::kColours; ++i) {
        const double c0 = gm::kColour[4 * i], c1 = gm::kColour[4 * (i + 1)];
        if (c <= c1 || i + 2 == gm::kColours) {
            const double t = std::clamp((c - c0) / std::max(c1 - c0, 1e-9), 0.0, 1.0);
            return gm::kColour[4 * i + col] * (1 - t) + gm::kColour[4 * (i + 1) + col] * t;
        }
    }
    return gm::kColour[col];
}

// The measured filters blended in log size, widened by kKernelScale: a continuous kernel in reference
// pixels, read bilinearly.
struct RefKernel {
    std::vector<double> k;
    int r = gm::kKernelRadius, n = 2 * gm::kKernelRadius + 1;
    explicit RefKernel(double size) {
        const double s = std::clamp(size, 8.0, 32.0), t = std::log2(s / 8.0);
        k.resize(size_t(n * n));
        for (int i = 0; i < n * n; ++i)
            k[size_t(i)] = t <= 1.0 ? (1 - t) * gm::kKernel8[i] + t * gm::kKernel16[i]
                                    : (2 - t) * gm::kKernel16[i] + (t - 1) * gm::kKernel32[i];
    }
    // at (x, y) reference px from the centre, the kernel widened by kKernelScale
    double at(double x, double y) const {
        const double fx = x / gm::kKernelScale + r, fy = y / gm::kKernelScale + r;
        const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
        const double ax = fx - x0, ay = fy - y0;
        auto v = [&](int xx, int yy) { return (xx < 0 || yy < 0 || xx >= n || yy >= n) ? 0.0 : k[size_t(yy * n + xx)]; };
        return (1 - ay) * ((1 - ax) * v(x0, y0) + ax * v(x0 + 1, y0)) + ay * ((1 - ax) * v(x0, y0 + 1) + ax * v(x0 + 1, y0 + 1));
    }
};

// The grain filter at this render's pixel size: q = pixel / reference pixel. Each tap is q times the
// kernel's mean over the tap's footprint (q reference px wide), so the field is the reference grain as
// these pixels see it: finer grain than a pixel averages down (the preview), larger pixels never.
// Normalised so that a q <= 1 render (export at or above the reference) has unit variance.
std::vector<float> tier_kernel(double size, double q, int& radius) {
    RefKernel ref(size);
    const double reach = gm::kKernelRadius * gm::kKernelScale + 0.5 * q;
    radius = std::max(0, int(std::ceil(reach / q)));
    const int n = 2 * radius + 1, sub = 4;
    std::vector<double> t(size_t(n * n));
    double ss = 0;
    for (int j = -radius; j <= radius; ++j)
        for (int i = -radius; i <= radius; ++i) {
            double acc = 0;
            for (int b = 0; b < sub; ++b)
                for (int a = 0; a < sub; ++a)
                    acc += ref.at(q * (i + (a + 0.5) / sub - 0.5), q * (j + (b + 0.5) / sub - 0.5));
            const double v = q * acc / (sub * sub);
            t[size_t((j + radius) * n + (i + radius))] = v;
            ss += v * v;
        }
    // The same sampling at q = 1 is the reference field's variance: normalise it to 1. A render with
    // larger pixels (q > 1, the preview) keeps its natural share (< 1); one with smaller pixels (an
    // export above the reference) is held at 1, as the reference sees its grain.
    double ss_ref = 0;
    for (int j = -gm::kKernelRadius - 1; j <= gm::kKernelRadius + 1; ++j)
        for (int i = -gm::kKernelRadius - 1; i <= gm::kKernelRadius + 1; ++i) {
            double acc = 0;
            for (int b = 0; b < sub; ++b)
                for (int a = 0; a < sub; ++a) acc += ref.at(i + (a + 0.5) / sub - 0.5, j + (b + 0.5) / sub - 0.5);
            ss_ref += (acc / (sub * sub)) * (acc / (sub * sub));
        }
    const double scale = q <= 1.0 ? 1.0 / std::sqrt(std::max(ss, 1e-12)) : 1.0 / std::sqrt(std::max(ss_ref, 1e-12));
    std::vector<float> out(t.size());
    for (size_t i = 0; i < t.size(); ++i) out[i] = float(t[i] * scale);
    return out;
}

}  // namespace

bool Pipeline::print_look_wanted() const {
    const PrintLookParams& k = params_.print_look;
    const bool any = k.print_exposure != 0.0 || k.tonal_contrast != 1.0 || k.color_density != 0.0 ||
                     k.saturation != 100.0 ||
                     ((k.head_yellow_blue != 0.0 || k.head_magenta_green != 0.0 || k.head_cyan_red != 0.0) &&
                      k.head_impact > 0.0) ||
                     k.black_point != 0.0 || k.white_point != 100.0 || k.grain_amount > 0.0;
    // The print LUT bake stands for the paper alone; the DI is its own encoded output.
    return any && !params_.debug.lut_mode && !di_active_;
}

#ifndef __APPLE__
bool Pipeline::print_look(const Chain& in, Chain& out, std::string&) { out = in; return true; }
#else
bool Pipeline::print_look(const Chain& in, Chain& out, std::string& error) {
    out = in;
    if (!print_look_wanted()) return true;
    Timer t(this, "print.look");
    const PrintLookParams& k = params_.print_look;
    Image cur = in.cur, next;
    const uint32_t n[1] = {uint32_t(cur.pixels())};

    auto tone_pass = [&](bool print_and_head, bool expand) -> bool {
        const float p[16] = {print_and_head ? float(k.print_exposure) : 0.0f,
                             print_and_head ? float(k.tonal_contrast) : 1.0f,
                             print_and_head ? float(k.color_density) : 0.0f,
                             print_and_head ? float(k.saturation) : 100.0f,
                             print_and_head ? float(k.head_yellow_blue) : 0.0f,
                             print_and_head ? float(k.head_magenta_green) : 0.0f,
                             print_and_head ? float(k.head_cyan_red) : 0.0f,
                             float(k.head_shadows), float(k.head_midtones), float(k.head_highlights),
                             float(k.head_preserve), print_and_head ? float(k.head_impact) : 0.0f,
                             expand ? float(k.black_point) : 0.0f, expand ? float(k.white_point) : 100.0f,
                             0.0f, 0.0f};
        gpu::BufferRef p_buf = gpu_->upload(p, sizeof p, error);
        if (!p_buf || !alloc_like(cur, next, error)) return false;
        if (!gpu_->dispatch("spk_look_tone",
                            {gpu::Arg::buf(cur.buf), gpu::Arg::buf(p_buf), gpu::Arg::inline_bytes(n, 1),
                             gpu::Arg::buf(next.buf)},
                            cur.pixels(), error)) return false;
        cur = next;
        return true;
    };

    // 1. Print and Color Head.
    if (!tone_pass(true, false)) return false;

    // 2. Film Grain (the measured model), on the sRGB-encoded positive: softening, then grain.
    if (k.grain_amount > 0.0) {
        const double pixel_um = pixel_size_um_ > 0.0 ? pixel_size_um_ : gm::kRefPixelUm;
        const double q = pixel_um / gm::kRefPixelUm;
        const double a = k.grain_amount / 40.0, sz = std::clamp(k.grain_size, 1.0, 48.0);
        // encode
        Image enc;
        {
            const uint32_t meta[2] = {uint32_t(cur.pixels()), 1u};
            if (!alloc_like(cur, enc, error)) return false;
            if (!gpu_->dispatch("spk_look_srgb", {gpu::Arg::buf(cur.buf), gpu::Arg::inline_bytes(meta, 2),
                                                  gpu::Arg::buf(enc.buf)}, cur.pixels(), error)) return false;
        }
        // softening: a narrow share inside a wide blur
        const double sig = std::pow(a, gm::kBlurAmountExp) * std::pow(sz / 16.0, gm::kBlurSizeExp) *
                           std::hypot(gm::kBlurS0, gm::kBlurK * (100.0 - k.grain_resolution));
        const double share = std::min(1.0, gm::kSharpShare * std::pow(sig / gm::kSharpRef, -gm::kSharpExp));
        {
            const double sn = gm::kNarrowSigma / q, sw = gm::kWideScale * sig / q;
            const double s_n[3] = {sn, sn, sn}, s_w[3] = {sw, sw, sw};
            Image narrow, wide, mixed;
            if (!blur_.gaussian(enc, s_n, narrow, error)) return false;
            if (!blur_.gaussian(enc, s_w, wide, error)) return false;
            const double wa[3] = {share, share, share}, wb[3] = {1 - share, 1 - share, 1 - share};
            if (!blur_.lincomb(narrow, wide, wa, wb, mixed, error)) return false;
            enc = mixed;
        }
        // noise: (mono, colour R, colour G), (colour B, -, -), each filtered by the tier's grain kernel
        int radius = 0;
        const std::vector<float> kern = tier_kernel(sz, q, radius);
        gpu::BufferRef k_buf = gpu_->upload(kern.data(), kern.size() * sizeof(float), error);
        if (!k_buf) return false;
        Image field[2];
        for (uint32_t j = 0; j < 2; ++j) {
            Image white;
            if (!alloc_like(cur, white, error) || !alloc_like(cur, field[j], error)) return false;
            const uint32_t meta[3] = {uint32_t(cur.pixels()), 211u + 13u * j, 0x6A1Du};
            if (!gpu_->dispatch("spk_look_noise", {gpu::Arg::inline_bytes(meta, 3), gpu::Arg::buf(white.buf)},
                                cur.pixels(), error)) return false;
            const uint32_t fm[4] = {cur.w, cur.h, uint32_t(radius), uint32_t(cur.pixels())};
            if (!gpu_->dispatch("spk_look_fir", {gpu::Arg::buf(white.buf), gpu::Arg::buf(k_buf),
                                                 gpu::Arg::inline_bytes(fm, 4), gpu::Arg::buf(field[j].buf)},
                                cur.pixels(), error)) return false;
        }
        // tables for this render: std and shift at the 21 levels
        const double amt = std::pow(a, gm::kAmountExp), szs = std::pow(sz / 16.0, gm::kSizeStdExp);
        const double zs = k.grain_shadows / 100.0, zm = k.grain_midtones / 100.0, zh = k.grain_highlights / 100.0;
        std::vector<float> tab(2 * gm::kLevels + 8);
        for (int i = 0; i < gm::kLevels; ++i) {
            tab[size_t(i)] = float((gm::kStdBase[i] + zs * gm::kStdShadows[i] + zm * gm::kStdMidtones[i] +
                                    zh * gm::kStdHighlights[i]) * amt * szs / 255.0);
            tab[size_t(gm::kLevels + i)] = float((gm::kShiftBase[i] + zs * gm::kShiftShadows[i] +
                                                  zm * gm::kShiftMidtones[i] + zh * gm::kShiftHighlights[i]) * amt / 255.0);
        }
        const double w = std::clamp(interp_colour(k.grain_color, 1), 0.0, 1.0), red = interp_colour(k.grain_color, 2);
        const double lg = interp_colour(k.grain_color, 3);
        const double y[3] = {0.2126, 0.7152, 0.0722}, g[3] = {red, 1.0, 1.0};
        const double ysum = y[0] * g[0] + y[1] * g[1] + y[2] * g[2];
        const double lum_std = std::sqrt(ysum * ysum * (1 - w) + w * (y[0] * y[0] * g[0] * g[0] + y[1] * y[1] + y[2] * y[2]));
        float* extra = tab.data() + 2 * gm::kLevels;
        extra[0] = float(std::sqrt(1 - w)); extra[1] = float(std::sqrt(w));
        extra[2] = float(red * lg / lum_std); extra[3] = float(lg / lum_std); extra[4] = float(lg / lum_std);
        extra[5] = gm::kLevel[0]; extra[6] = gm::kLevel[1] - gm::kLevel[0]; extra[7] = float(gm::kLevels);
        gpu::BufferRef t_buf = gpu_->upload(tab.data(), tab.size() * sizeof(float), error);
        if (!t_buf || !alloc_like(cur, next, error)) return false;
        if (!gpu_->dispatch("spk_look_grain2",
                            {gpu::Arg::buf(enc.buf), gpu::Arg::buf(field[0].buf), gpu::Arg::buf(field[1].buf),
                             gpu::Arg::buf(t_buf), gpu::Arg::inline_bytes(n, 1), gpu::Arg::buf(next.buf)},
                            cur.pixels(), error)) return false;
        cur = next;
    }

    // 3. Expand, after the grain: the contrast the grain takes from the black and white points, back.
    if (k.black_point != 0.0 || k.white_point != 100.0) {
        if (!tone_pass(false, true)) return false;
    }
    out.cur = cur;
    return true;
}
#endif

}  // namespace spk
