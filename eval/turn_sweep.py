"""Phase 3 exit-criterion eval: head turns up to ~30 degrees without a
detectable edit.

Setup per identity: randomized donor expression (smile of varying strength,
slight jaw/brow variation), randomized head pitch and off-center framing,
auto-exposure jitter between shots — the kind of variance real bursts have.
Metrics: PSNR and SSIM of the swap against the ideal render (same person,
base pose, donor expression) inside the ground-truth face region, mean and
WORST CASE across identities, plus the pipeline's own gate rejections.

Usage:  PYTHONPATH=.. python3 eval/turn_sweep.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.pipeline import DonorInput, LevelBConfig, LevelBSwap

OUT = Path(__file__).parent / "out"
SIZE = 320


def make_persons(rng: np.random.Generator):
    """Identities spanning tone/structure; expression scripts per person."""
    persons = []
    tone = [-0.45, -0.1, 0.2, 0.45, 0.6, -0.6]
    hue = [0.3, -0.4, 0.5, 0.0, -0.55, 0.4]
    for i in range(6):
        cid = np.concatenate([
            rng.uniform(-1, 1, 10) * np.array([0.5, 0.5, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.5])])
        cid[0] = tone[i]
        cid[1] = hue[i]
        cex_d = np.zeros(6)
        cex_d[1] = rng.uniform(0.4, 0.7)              # smile strength
        cex_d[0] = rng.uniform(0.0, 0.25)             # slight jaw
        cex_b = np.zeros(6)
        cex_b[2] = rng.uniform(0.5, 0.9)              # brows (mid-blink feel)
        cex_b[3] = rng.uniform(0.0, 0.5)              # eye closing
        persons.append((f"p{i}", cid, cex_b, cex_d, int(rng.integers(0, 10_000))))
    return persons


def psnr(a, b, mask):
    d = (a - b)[mask]
    return 10 * np.log10(1.0 / max(float(np.mean(d ** 2)), 1e-12))


def ssim(a, b, mask, win=11):
    """Mean SSIM over the masked region (luminance, uniform window)."""
    x, y = a.mean(-1).astype(np.float64), b.mean(-1).astype(np.float64)
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    mu_x = uniform_filter(x, win)
    mu_y = uniform_filter(y, win)
    var_x = uniform_filter(x * x, win) - mu_x * mu_x
    var_y = uniform_filter(y * y, win) - mu_y * mu_y
    cov = uniform_filter(x * y, win) - mu_x * mu_y
    smap = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / \
        ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
    return float(smap[mask].mean())


def run_case(model, cid, cex_b, cex_d, seed, base_yaw, donor_yaws):
    rng = np.random.default_rng(seed)
    cap = SyntheticCapture(model, size=SIZE, seed=seed)
    capture = cap.make_capture(
        cid, base_yaw=base_yaw, donor_yaws=donor_yaws,
        base_c_ex=cex_b, donor_c_ex=cex_d,
        exposure_jitter=0.12,
        base_pitch=float(rng.uniform(-6, 6)),
        donor_pitch=float(rng.uniform(-4, 4)),
        center_offset=(float(rng.uniform(-40, 40)), float(rng.uniform(-30, 30))))
    swap = LevelBSwap(model, LevelBConfig(mode="pinhole"))
    t0 = time.time()
    res = swap.run(capture.base.frame,
                   FaceObservation("p0", capture.base.obs_landmarks),
                   [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
                    for d in capture.donors])
    dt = time.time() - t0
    reg = capture.face_region_gt
    p = psnr(res.image, capture.ideal, reg)
    s = ssim(res.image, capture.ideal, reg)
    return res, capture, p, s, dt, res.method == "rejected"


def turn_sweep(model, persons, angles=range(0, 45, 5)):
    print(f"{'yaw':>4} {'PSNR mean':>10} {'worst':>7} {'SSIM mean':>10} "
          f"{'face cov':>9} {'rej':>4} {'s/case':>7}")
    rows = {}
    for yaw in angles:
        ps, ss, covs, rej, dts, panels = [], [], [], 0, [], []
        for name, cid, cex_b, cex_d, seed in persons:
            res, capture, p, s, dt, rejected = run_case(
                model, cid, cex_b, cex_d, seed, float(yaw), [5.0])
            ps.append(p)
            ss.append(s)
            covs.append(res.coverage)
            rej += rejected
            dts.append(dt)
            panels.append(np.concatenate([capture.base.frame.rgb, res.image,
                                          capture.ideal], axis=1))
        rows[yaw] = (float(np.mean(ps)), float(np.min(ps)))
        print(f"{yaw:>4} {np.mean(ps):>10.2f} {np.min(ps):>7.2f} "
              f"{np.mean(ss):>10.3f} {np.mean(covs):>9.2f} {rej:>4d} "
              f"{np.mean(dts):>7.1f}")
        if yaw % 15 == 0:
            Image.fromarray((np.clip(np.concatenate(panels, axis=1), 0, 1) * 255)
                            .astype(np.uint8)).save(OUT / f"sweep_yaw{yaw:02d}.png")
    return rows


def fill_ablation(model, persons, base_yaw=35.0):
    """Donor-count ablation: what extra burst frames buy at a large turn
    (coverage is face-relative: fraction of the face area swapped)."""
    print(f"\nfill ablation at base yaw {base_yaw:.0f} "
          f"({len(persons)} identities, face-relative coverage):")
    for donors in ([5.0], [-20.0], [-20.0, 35.0], [5.0, -20.0, 35.0]):
        ps, covs, fills, worst = [], [], [], 1e9
        for name, cid, cex_b, cex_d, seed in persons:
            cap = SyntheticCapture(model, size=SIZE, seed=seed)
            res, _, p, _, _, rej = run_case(model, cid, cex_b, cex_d, seed,
                                            base_yaw, donors)
            if not rej:
                ps.append(p)
                covs.append(res.coverage)
                fills.append(res.fill_fraction)
                worst = min(worst, p)
        print(f"  donors {str(donors):<20} PSNR {np.mean(ps):6.2f}±{np.std(ps):<4.2f} "
              f"(worst {worst:5.2f})  face coverage {np.mean(covs):.2f}  "
              f"fill {np.mean(fills):.3f}")


def main():
    OUT.mkdir(exist_ok=True)
    model = SyntheticHeadModel()
    rng = np.random.default_rng(123)
    persons = make_persons(rng)
    print(f"Phase 3 eval — Level B vs. base head yaw; {len(persons)} identities, "
          f"randomized expression/pitch/off-center framing, "
          f"auto-exposure jitter ±12%, donor at 5°\n")
    rows = turn_sweep(model, persons)
    fill_ablation(model, persons)

    mean30, worst30 = rows[30]
    print(f"\nExit criterion (plan): handles head turns up to 30° without a "
          f"detectable edit.")
    print(f"  PSNR vs. ideal at 30°: mean {mean30:.2f} dB, worst identity "
          f"{worst30:.2f} dB "
          f"({'PASS' if mean30 >= 21 and worst30 >= 19 else 'INVESTIGATE'} "
          f"vs. mean >= 21 / worst >= 19 dB prototype bar)")
    print("  PSNR/SSIM stand in for the planned blind 'was this edited?' test;")
    print("  gate rejections above should stay 0 for genuine swaps.")


if __name__ == "__main__":
    main()
