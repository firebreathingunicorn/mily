import numpy as np

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.pipeline import DonorInput, LevelBConfig, LevelBSwap


def _swap(mode="pinhole", seed=11, size=288):
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=size, seed=seed)
    return m, cap


CID = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])


def _run(cap, base_yaw, donor_yaws, base_cex, donor_cex, mode="pinhole", model=None):
    model = model
    capture = cap.make_capture(CID, base_yaw=base_yaw, donor_yaws=donor_yaws,
                               base_c_ex=base_cex, donor_c_ex=donor_cex)
    swap = LevelBSwap(model, LevelBConfig(mode=mode))
    res = swap.run(capture.base.frame,
                   FaceObservation("p0", capture.base.obs_landmarks),
                   [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
                    for d in capture.donors])
    reg = capture.face_region_gt
    diff = (res.image - capture.ideal)[reg]
    psnr = 10 * np.log10(1.0 / max(float(np.mean(diff ** 2)), 1e-12))
    return res, psnr, capture


def test_end_to_end_smile_swap_20deg():
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8            # brows raised in base
    cex_d = np.zeros(6); cex_d[1] = 0.6            # smiling in donor
    res, psnr, _ = _run(cap, 20.0, [5.0], cex_b, cex_d, model=m)
    assert res.coverage > 0.4
    assert psnr > 22.0, f"PSNR {psnr:.2f} dB"
    assert res.checks["artifact"] and res.checks["identity"]
    assert res.method == "B"


def test_turn_30deg_exit_criterion():
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    res, psnr, _ = _run(cap, 30.0, [5.0], cex_b, cex_d, model=m)
    assert psnr > 21.0, f"PSNR at 30 deg: {psnr:.2f} dB"


def test_two_donors_fill_improves_coverage():
    """Base at +35 reveals the canonical +x side; a primary donor turned the
    other way (-20) cannot see it, a second donor at +35 can -> fill."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    res1, _, _ = _run(cap, 35.0, [-20.0], cex_b, cex_d, model=m)
    res2, _, _ = _run(cap, 35.0, [-20.0, 35.0], cex_b, cex_d, model=m)
    assert res2.coverage > res1.coverage + 0.02
    assert res2.fill_fraction > 0.01
    assert res2.method == "B+fill"


def test_weak_mode_pipeline_runs():
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    res, psnr, _ = _run(cap, 15.0, [5.0], cex_b, cex_d, mode="weak", model=m)
    assert psnr > 20.0


def test_donor_ordering_frontal_first():
    """Donor order in the input must not matter: the most frontal donor
    becomes primary even when passed last."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    res, psnr, capture = _run(cap, 25.0, [30.0, 5.0], cex_b, cex_d, model=m)
    assert res.method in ("B", "B+fill")
    assert psnr > 20.0
    # after sorting, index 0 is the frontal donor and should dominate
    swapped = res.face_region & (res.weight > 0.5)
    frac_primary = float((res.source_map[swapped] == 0).mean())
    assert frac_primary > 0.7, f"primary donor covers only {frac_primary:.2f}"


def test_off_center_framing():
    """The person is not centered in frame; fit translation must absorb it."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=20.0, donor_yaws=[5.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d,
                               center_offset=(38.0, -26.0))
    swap = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    res = swap.run(capture.base.frame,
                   FaceObservation("p0", capture.base.obs_landmarks),
                   [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
                    for d in capture.donors])
    reg = capture.face_region_gt
    psnr = 10 * np.log10(1.0 / max(float(np.mean((res.image - capture.ideal)[reg] ** 2)), 1e-12))
    assert res.method in ("B", "B+fill")
    assert psnr > 20.0, f"off-center PSNR {psnr:.2f} dB"


def test_bad_donor_is_dropped_not_fatal():
    """A failed detection (NaN landmarks) must be skipped: the run succeeds
    on the remaining donors instead of crashing or shipping garbage."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=20.0, donor_yaws=[5.0, 12.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d)
    broken_obs = FaceObservation("p0", capture.donors[0].obs_landmarks.copy())
    broken_obs.landmarks[:, 0] = np.nan
    swap = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    res = swap.run(capture.base.frame,
                   FaceObservation("p0", capture.base.obs_landmarks),
                   [DonorInput(capture.donors[0].frame, broken_obs),
                    DonorInput(capture.donors[1].frame,
                               FaceObservation("p0", capture.donors[1].obs_landmarks))])
    assert res.method in ("B", "B+fill")
    assert res.coverage > 0.3
    assert res.checks["identity"]


def test_base_occluder_is_repaired_by_swap():
    """A hand over the base's cheek is base-side damage: donor pixels paste
    straight over it, so the result approaches the ideal despite it."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=20.0, donor_yaws=[5.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d)
    base_occ = cap.shoot(CID, cex_b, 20.0,
                         occluder=(cap.intrinsics.cx + 40, cap.intrinsics.cy + 45,
                                   30.0, 24.0, 1.8), person_id="base_occ")
    swap = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    res = swap.run(base_occ.frame,
                   FaceObservation("p0", capture.base.obs_landmarks),
                   [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
                    for d in capture.donors])
    reg = capture.face_region_gt
    psnr = 10 * np.log10(1.0 / max(float(np.mean((res.image - capture.ideal)[reg] ** 2)), 1e-12))
    assert res.method in ("B", "B+fill")
    assert psnr > 18.0, f"occluded-base PSNR {psnr:.2f} dB"


def test_sector_coverage_map_sane():
    """The smart-shutter hook: 3x3 sector map, values in [0,1], area-weighted
    mean equals overall coverage."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=20.0, donor_yaws=[5.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d)
    swap = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    res = swap.run(capture.base.frame,
                   FaceObservation("p0", capture.base.obs_landmarks),
                   [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
                    for d in capture.donors])
    s = res.coverage_sectors
    assert s is not None and s.shape == (3, 3)
    assert (s >= 0).all() and (s <= 1).all()
    # back-weighted mean of sectors equals the scalar coverage
    reg = res.face_region
    ys, xs = np.nonzero(reg)
    iy = np.clip((ys - ys.min()) * 3 // (ys.max() - ys.min() + 1), 0, 2)
    ix = np.clip((xs - xs.min()) * 3 // (xs.max() - xs.min() + 1), 0, 2)
    area = np.zeros((3, 3))
    np.add.at(area, (iy, ix), 1.0)
    mean = float((s * area).sum() / area.sum())
    assert abs(mean - res.coverage) < 0.02


def test_occluded_donor_falls_back_to_other_frames():
    """An occluder (hand) in the primary donor's frame must be excluded by
    the depth/visibility test and filled from a clean second donor."""
    m, cap = _swap()
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=25.0, donor_yaws=[5.0, 5.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d)
    # a hand clearly in front of the cheek (Z=1.8 vs face ~2.1-2.8)
    d0 = capture.donors[0].frame
    occ = (d0.intrinsics.cx + 55, d0.intrinsics.cy + 45, 32.0, 25.0, 1.8)
    shot_occ = cap.shoot(CID, cex_d, 5.0, occluder=occ, person_id="occ")
    from besttake.level_b.pipeline import DonorInput as DI
    from besttake.common.types import FaceObservation as FO

    def run(donors):
        swap = LevelBSwap(m, LevelBConfig(mode="pinhole"))
        r = swap.run(capture.base.frame,
                     FaceObservation("p0", capture.base.obs_landmarks), donors)
        reg = capture.face_region_gt
        p = 10 * np.log10(1.0 / max(float(np.mean((r.image - capture.ideal)[reg] ** 2)), 1e-12))
        return r, p

    r_occ, p_occ = run([DI(shot_occ.frame, FO("p0", shot_occ.obs_landmarks))])
    r_fix, p_fix = run([DI(shot_occ.frame, FO("p0", shot_occ.obs_landmarks)),
                        DI(capture.donors[1].frame,
                           FO("p0", capture.donors[1].obs_landmarks))])
    # the depth guard excludes the blob from the occluded donor; the clean
    # donor restores swapped coverage there. (PSNR barely moves: the excluded
    # area falls back to base pixels, which match the ideal at the cheek.)
    assert r_fix.coverage > r_occ.coverage + 0.02
    assert r_fix.fill_fraction > 0.02
    assert r_fix.method == "B+fill"
    assert p_fix > p_occ - 0.5
