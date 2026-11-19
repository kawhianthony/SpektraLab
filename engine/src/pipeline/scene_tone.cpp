// scene_tone.cpp -- Tone: Capture One-style High Dynamic Range (Highlight,
// Shadow, White, Black) on the scene before the film.
//
// Kept out of `pipeline.cpp` as `scene_latitude.cpp` is: the film chain gained
// one guarded call at the end of `film_prefix`, and with all four sliders at 0
// nothing in this file runs -- the render is the same dispatches as before.
//
// What it does (reference: c1-hdr-lab/c1tone.py, fitted to Capture One 16.8):
//
//   L     = log2(Y / mid) + c          stops about the metered mid-grey, moved by
//                                      Film Exposure (the Decode curve's axis)
//   base  = guided filter of L on L    edge-aware local brightness: Gaussian
//                                      windows, sigma = 1 % of the frame's long
//                                      edge (so every tier sees the same picture),
//                                      eps = 0.25 stops^2
//   Bh/Bw = L moved 80 % / 50 % of the way to base   (Highlight-Shadow / White-Black)
//   dEV   = Highlight (+) White combined nearly as a max, + Shadow + Black,
//           each a smooth zone weight times the slider's strength, clamped
//   out   = rgb * 2^dEV                one gain for R, G and B: hue and
//                                      saturation are untouched
//
// Metal only. The CMake targets (Windows, Linux) never dispatch it: the calls
// are under `__APPLE__`.
#ifdef __APPLE__
#include <algorithm>
#include <cmath>

#include "latitude_fit.hpp"
#include "pipeline.hpp"
#include "timer.hpp"

namespace spk {

namespace {

constexpr double kSigmaFrac = 0.01;      // blur sigma, fraction of the long edge
constexpr double kEps = 0.25;            // guided filter's edge threshold, stops^2
constexpr double kLocalHS = 0.8;         // how local Highlight / Shadow are
constexpr double kLocalWB = 0.5;         // how local White / Black are
constexpr double kSliderPower = 0.7;     // -50 is ~60 % of -100, as Capture One
// Strength in stops at |slider| = 100: (negative side, positive side).
constexpr double kHighlight[2] = {1.35, 0.6};
constexpr double kWhite[2] = {0.88, 0.45};
constexpr double kShadow[2] = {1.2, 2.2};
constexpr double kBlack[2] = {1.0, 2.3};
constexpr double kDevMin = -3.0, kDevMax = 4.0;

double amount(double s, const double k[2]) {
    if (s == 0.0) return 0.0;
    const double m = std::pow(std::min(std::abs(s), 100.0) / 100.0, kSliderPower);
    return s < 0.0 ? -m * k[0] : m * k[1];
}

}  // namespace

bool Pipeline::scene_tone_wanted() const {
    const CameraParams& c = params_.camera;
    const bool any = c.tone_highlight != 0.0 || c.tone_shadow != 0.0 || c.tone_white != 0.0 || c.tone_black != 0.0;
    // The print LUT bake renders a neutral chain, as for the Decode curve.
    return any && !params_.debug.lut_mode;
}

bool Pipeline::node_scene_tone(const Image& in, Image& out, std::string& error) {
    Timer t(this, "preprocess.scene_tone");
    const CameraParams& cam = params_.camera;
    const uint32_t n[1] = {uint32_t(in.pixels())};

    // 1. (L, L^2, 0)
    const float prep[5] = {float(rgb_to_xyz_ae_.m[1][0]), float(rgb_to_xyz_ae_.m[1][1]),
                           float(rgb_to_xyz_ae_.m[1][2]), float(slm::kMidgray),
                           float(cam.exposure_compensation_ev)};
    gpu::BufferRef prep_buf = gpu_->upload(prep, sizeof prep, error);
    Image guide, means, coef, coef_means;
    if (!prep_buf || !alloc_like(in, guide, error)) return false;
    if (!gpu_->dispatch("spk_tone_prep",
                        {gpu::Arg::buf(in.buf), gpu::Arg::buf(prep_buf), gpu::Arg::inline_bytes(n, 1),
                         gpu::Arg::buf(guide.buf)},
                        in.pixels(), error)) return false;

    // 2. Gaussian means of L and L^2; the sigma is a share of the frame, so the
    // live, preview and full tiers make the same mask.
    const double sigma = std::max(1.0, kSigmaFrac * double(std::max(in.w, in.h)));
    const double s2[3] = {sigma, sigma, 0.0};
    if (!blur_.gaussian(guide, s2, means, error)) return false;

    // 3. The guided filter's (a, b), then their means.
    const float coef_p[1] = {float(kEps)};
    gpu::BufferRef coef_buf = gpu_->upload(coef_p, sizeof coef_p, error);
    if (!coef_buf || !alloc_like(in, coef, error)) return false;
    if (!gpu_->dispatch("spk_tone_coef",
                        {gpu::Arg::buf(means.buf), gpu::Arg::buf(coef_buf), gpu::Arg::inline_bytes(n, 1),
                         gpu::Arg::buf(coef.buf)},
                        in.pixels(), error)) return false;
    if (!blur_.gaussian(coef, s2, coef_means, error)) return false;

    // 4. The gain.
    const float apply_p[8] = {float(amount(cam.tone_highlight, kHighlight)), float(amount(cam.tone_white, kWhite)),
                              float(amount(cam.tone_shadow, kShadow)), float(amount(cam.tone_black, kBlack)),
                              float(kLocalHS), float(kLocalWB), float(kDevMin), float(kDevMax)};
    gpu::BufferRef apply_buf = gpu_->upload(apply_p, sizeof apply_p, error);
    if (!apply_buf || !alloc_like(in, out, error)) return false;
    return gpu_->dispatch("spk_tone_apply",
                          {gpu::Arg::buf(in.buf), gpu::Arg::buf(guide.buf), gpu::Arg::buf(coef_means.buf),
                           gpu::Arg::buf(apply_buf), gpu::Arg::inline_bytes(n, 1), gpu::Arg::buf(out.buf)},
                          in.pixels(), error);
}

}  // namespace spk
#endif  // __APPLE__
