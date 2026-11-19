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
//   Grain       softening by Resolution (100 = none); then grain as density: Gaussian-blurred
//               white noise sized in micrometres on the film (so preview and export match),
//               clustered (fine + 1.6x coarse), mono / colour by Color, a mean-preserving ratio in
//               stops about a small toe, strength Amount x the tonal zone's own
//
// A whole-plane stage (`kPrintStages`): every default is a bypass and the stage hands its input on
// untouched. Metal only; the CMake targets hand the image on.
#include <algorithm>
#include <cmath>

#include "pipeline.hpp"
#include "timer.hpp"

namespace spk {

namespace {

constexpr double kRefPixelUm = 36000.0 / 4240.0;   // grain is tuned on a 35 mm frame at 4240 px
constexpr double kGrainUmPerStep = 0.6;            // grain sigma per Size step, micrometres
constexpr double kClusterScale = 1.6;
constexpr double kSoftenPxAtZero = 2.0;            // Resolution 0's softening, px at the reference

// What one pixel of this size sees of a grain field of sigma `s` (um), against a pixel of the
// reference size, times the normalisation of a blur of unit white noise at `st` pixels.
double grain_norm(double s_um, double pixel_um) {
    const double st = s_um / pixel_um;
    return std::sqrt(1.0 + 4.0 * M_PI * st * st) *
           std::sqrt(s_um * s_um + kRefPixelUm * kRefPixelUm / 12.0) /
           std::sqrt(s_um * s_um + pixel_um * pixel_um / 12.0);
}

}  // namespace

bool Pipeline::print_look_wanted() const {
    const PrintLookParams& k = params_.print_look;
    const bool any = k.print_exposure != 0.0 || k.tonal_contrast != 1.0 || k.color_density != 0.0 ||
                     k.saturation != 100.0 ||
                     ((k.head_yellow_blue != 0.0 || k.head_magenta_green != 0.0 || k.head_cyan_red != 0.0) &&
                      k.head_impact > 0.0) ||
                     k.black_point != 0.0 || k.white_point != 100.0 || k.grain_amount > 0.0 ||
                     k.grain_resolution < 100.0;
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

    // 1. Print, Color Head, Expand: one pointwise pass.
    {
        const float p[16] = {float(k.print_exposure), float(k.tonal_contrast), float(k.color_density),
                             float(k.saturation), float(k.head_yellow_blue), float(k.head_magenta_green),
                             float(k.head_cyan_red), float(k.head_shadows), float(k.head_midtones),
                             float(k.head_highlights), float(k.head_preserve), float(k.head_impact),
                             float(k.black_point), float(k.white_point), 0.0f, 0.0f};
        gpu::BufferRef p_buf = gpu_->upload(p, sizeof p, error);
        if (!p_buf || !alloc_like(cur, next, error)) return false;
        if (!gpu_->dispatch("spk_look_tone",
                            {gpu::Arg::buf(cur.buf), gpu::Arg::buf(p_buf), gpu::Arg::inline_bytes(n, 1),
                             gpu::Arg::buf(next.buf)},
                            cur.pixels(), error)) return false;
        cur = next;
    }

    const double pixel_um = pixel_size_um_ > 0.0 ? pixel_size_um_ : kRefPixelUm;

    // 2. Resolution: the print's own softness, sized on the film like the grain.
    if (k.grain_resolution < 100.0) {
        const double s = (100.0 - k.grain_resolution) / 100.0 * kSoftenPxAtZero * kRefPixelUm / pixel_um;
        const double s3[3] = {s, s, s};
        if (!blur_.gaussian(cur, s3, next, error)) return false;
        cur = next;
    }

    // 3. Grain.
    if (k.grain_amount > 0.0) {
        const double sf = k.grain_size * kGrainUmPerStep, sc = sf * kClusterScale;
        const double pf = sf / pixel_um, pc = sc / pixel_um;
        // Three planes of white noise: (mono fine, mono coarse, -), colour fine RGB, colour coarse RGB.
        Image noise[3], blurred[3];
        const double sig[3][3] = {{pf, pc, 0.0}, {pf, pf, pf}, {pc, pc, pc}};
        for (uint32_t j = 0; j < 3; ++j) {
            if (!alloc_like(cur, noise[j], error)) return false;
            const uint32_t meta[3] = {uint32_t(cur.pixels()), 101u + 7u * j, 0x6A1Du};
            if (!gpu_->dispatch("spk_look_noise",
                                {gpu::Arg::inline_bytes(meta, 3), gpu::Arg::buf(noise[j].buf)},
                                cur.pixels(), error)) return false;
            if (!blur_.gaussian(noise[j], sig[j], blurred[j], error)) return false;
        }
        const double nf = grain_norm(sf, pixel_um), nc = grain_norm(sc, pixel_um);
        const float p[10] = {float(k.grain_amount), float(k.grain_shadows), float(k.grain_midtones),
                             float(k.grain_highlights), float(k.grain_color), float(nf), float(nc),
                             0.012f, 0.0f, 0.0f};
        gpu::BufferRef p_buf = gpu_->upload(p, sizeof p, error);
        if (!p_buf || !alloc_like(cur, next, error)) return false;
        if (!gpu_->dispatch("spk_look_grain",
                            {gpu::Arg::buf(cur.buf), gpu::Arg::buf(blurred[0].buf), gpu::Arg::buf(blurred[1].buf),
                             gpu::Arg::buf(blurred[2].buf), gpu::Arg::buf(p_buf), gpu::Arg::inline_bytes(n, 1),
                             gpu::Arg::buf(next.buf)},
                            cur.pixels(), error)) return false;
        cur = next;
    }
    out.cur = cur;
    return true;
}
#endif

}  // namespace spk
