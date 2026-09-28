"""Level A (Phase 1) tests: same closed-loop synthetic captures Level B uses."""
import numpy as np

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_a import (ClassicalArtifactChecker, GeometricIdentityChecker,
                              LevelASwap, ADonor, descriptor, estimate_sigma,
                              fit_similarity, key_points, refine_burst_yaw,
                              relative_distance, score_frame)
from besttake.level_a.identity import relative_distance as rel_dist

CID = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])


def _model(seed=11, size=288):
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=size, seed=seed)
    return m, cap


def _psnr(a, b, mask):
    diff = (np.asarray(a, float) - np.asarray(b, float))[mask]
    mse = max(float(np.mean(diff ** 2)), 1e-12)
    return 10 * np.log10(1.0 / mse)


# ------------------------------------------------------------- fundamentals

def test_similarity_fit_recovers_translation():
    src = np.array([[10.0, 20], [40, 20], [25, 60], [30, 35]])
    dst = src + np.array([12.0, -7.0])
    m = fit_similarity(src, dst)
    a, b = m[0, 0], m[1, 0]
    assert abs(np.hypot(a, b) - 1.0) < 1e-3
    assert abs(m[0, 2] - 12) < 1e-3 and abs(m[1, 2] + 7) < 1e-3


def test_similarity_fit_recovers_scale_rotation():
    t = np.pi / 2
    src = np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1], [1, 1]])
    dst = 2 * np.stack([src[:, 0] * np.cos(t) - src[:, 1] * np.sin(t),
                        src[:, 0] * np.sin(t) + src[:, 1] * np.cos(t)], axis=1)
    m = fit_similarity(src, dst)
    assert abs(np.hypot(m[0, 0], m[1, 0]) - 2.0) < 1e-3
    hom = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    out = hom @ m.T
    assert np.allclose(out, dst, atol=1e-3)


def test_noise_sigma_matches_swift_estimator():
    rng = np.random.default_rng(99)
    img = np.full((200, 200, 3), 0.5, np.float32)
    noise = rng.normal(0, 0.02, (200, 200, 1)).astype(np.float32)
    img = img + noise  # luma noise: same sample on all channels
    est = estimate_sigma(img)
    assert abs(est - 0.02) < 0.2 * 0.02, f"sigma {est}"


def test_identity_descriptor_separates_proportions():
    m, cap = _model()
    cap_a = cap.make_capture(CID, base_yaw=0.0, donor_yaws=[0.0])
    # Different identity: scaled identity coefficients.
    cid_b = CID * -0.8
    cap_b = cap.make_capture(cid_b, base_yaw=0.0, donor_yaws=[0.0])
    d = relative_distance(descriptor(cap_a.base.obs_landmarks),
                          descriptor(cap_b.base.obs_landmarks))
    d_same = relative_distance(descriptor(cap_a.base.obs_landmarks),
                               descriptor(cap_a.donors[0].obs_landmarks))
    assert d > 0.08, f"distinct identities too close: {d}"
    assert d_same < 0.05, f"same person apart: {d_same}"


# ------------------------------------------------------------ end-to-end A

def _one_person(base_yaw, donor_yaw, base_ex, donor_ex, seed=11, exposure_jitter=0.0):
    """Level A's home scenario: a burst with locked exposure (real bursts do
    not re-meter mid-burst), same or near-same pose, different expression."""
    m, cap = _model(seed=seed)
    capture = cap.make_capture(CID, base_yaw=base_yaw, donor_yaws=[donor_yaw],
                               base_c_ex=base_ex, donor_c_ex=donor_ex,
                               exposure_jitter=exposure_jitter)
    swap = LevelASwap()
    base_obs = FaceObservation("p0", capture.base.obs_landmarks)
    donors = [ADonor(frame=d.frame, obs=FaceObservation("p0", d.obs_landmarks))
              for d in capture.donors]
    res = swap.run(capture.base.frame, base_obs, donors)
    return capture, res


def _psnr_blur(a, b, mask, sigma=1.2):
    """PSNR with grain suppressed (Gaussian pre-filter): how humans judge
    structure; raw PSNR over grainy frames is dominated by matched noise."""
    from scipy import ndimage
    ab = np.stack([ndimage.gaussian_filter(np.asarray(a, float)[..., c], sigma) for c in range(3)], -1)
    bb = np.stack([ndimage.gaussian_filter(np.asarray(b, float)[..., c], sigma) for c in range(3)], -1)
    return _psnr(ab, bb, mask)


def test_level_a_swap_small_yaw_beats_base():
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6  # smiling donor
    capture, res = _one_person(8.0, 8.0, cex_b, cex_d)
    assert res is not None and res.method == "A"
    assert res.checks["artifact"] and res.checks["identity"]
    assert res.coverage > 0.2
    reg = capture.face_region_gt
    # Phase 1 exit criterion (proxy): the composite beats the base frame
    # against the ideal render of the donor's expression, measured with grain
    # suppressed (raw PSNR over grainy frames is dominated by matched noise).
    p_base = _psnr_blur(capture.base.frame.rgb, capture.ideal, reg)
    p_res = _psnr_blur(res.image, capture.ideal, reg)
    assert p_res > p_base, f"no improvement: {p_base:.2f} -> {p_res:.2f} dB"


def test_level_a_color_matches_exposure_offset():
    """A donor that re-metered brighter must land at the base's exposure."""
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture, res = _one_person(8.0, 8.0, cex_b, cex_d)
    assert res is not None

    # Brighten the donor by ~8% and re-run through the swap.
    bright = np.clip(capture.donors[0].frame.rgb.astype(np.float32) * 1.08, 0, 1)
    m, cap = _model()
    cap2 = cap.make_capture(CID, base_yaw=8.0, donor_yaws=[8.0],
                            base_c_ex=cex_b, donor_c_ex=cex_d, exposure_jitter=0.0)
    cap2.donors[0].frame.rgb = bright
    swap = LevelASwap()
    res2 = swap.run(cap2.base.frame, FaceObservation("p0", cap2.base.obs_landmarks),
                    [ADonor(frame=cap2.donors[0].frame,
                            obs=FaceObservation("p0", cap2.donors[0].obs_landmarks))])
    assert res2 is not None and res2.method == "A"
    solid = res2.weight > 0.5
    base_mean = np.asarray(cap2.base.frame.rgb)[solid].mean(axis=0)
    out_mean = np.asarray(res2.image)[solid].mean(axis=0)
    shift = float(np.linalg.norm(out_mean - base_mean))
    assert shift < 0.02, f"exposure offset not corrected: {shift:.3f}"


def test_level_a_rejects_noise_mismatch():
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture, res = _one_person(5.0, 0.0, cex_b, cex_d)
    assert res is not None
    # Deliberately degrade the composite: heavy grain in the pasted area.
    rng = np.random.default_rng(3)
    bad = res.image.copy()
    bad[res.face_region] = np.clip(
        bad[res.face_region] + rng.normal(0, 0.1, bad[res.face_region].shape), 0, 1)
    checker = ClassicalArtifactChecker()
    solid = res.weight > 0.5
    assert not checker.check(bad, res.face_region, weight=res.weight,
                             references=[capture.donors[0].frame.rgb]), \
        "noise-mismatched composite must be rejected"


def test_level_a_refuses_large_yaw():
    """Beyond the pose envelope Level A must decline (orchestrator → B)."""
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture, res = _one_person(30.0, 5.0, cex_b, cex_d)
    assert res is None, "30° donor should be outside Level A's envelope"


def test_scoring_prefers_smile():
    m, cap = _model()
    neutral = np.zeros(6)
    smile = np.zeros(6); smile[1] = 0.6
    cap_n = cap.make_capture(CID, base_yaw=0.0, donor_yaws=[0.0], base_c_ex=neutral)
    cap_s = cap.make_capture(CID, base_yaw=0.0, donor_yaws=[0.0], base_c_ex=smile)
    s_n = score_frame(cap_n.base.frame, FaceObservation("p0", cap_n.base.obs_landmarks))
    s_s = score_frame(cap_s.base.frame, FaceObservation("p0", cap_s.base.obs_landmarks))
    assert s_s > s_n, f"smile {s_s} should score above neutral {s_n}"


# ------------------------------------------------- accuracy: pose/noise/gate

def test_pose_refinement_uses_foreshortening():
    """A head turned 20° away foreshortens inter-ocular distance; the burst
    maximum anchors the frontal frame and acos recovers the turn magnitude.
    The sign follows the raw nose-offset proxy (this model's convention)."""
    m, cap = _model()
    cap0 = cap.make_capture(CID, base_yaw=0.0, donor_yaws=[20.0])
    base_obs = FaceObservation("p0", cap0.base.obs_landmarks)
    donor_obs = FaceObservation("p0", cap0.donors[0].obs_landmarks)
    raw_sign = np.sign(key_points(donor_obs.landmarks).yaw_proxy)
    refine_burst_yaw([base_obs, donor_obs])
    kb = key_points(base_obs.landmarks, base_obs.yaw_override)
    kd = key_points(donor_obs.landmarks, donor_obs.yaw_override)
    assert abs(kb.yaw_proxy) < 0.06, f"frontal reference should be ~0: {kb.yaw_proxy}"
    assert abs(abs(kd.yaw_proxy) - np.deg2rad(20)) < 0.1, \
        f"20° turn should be recovered from foreshortening: {kd.yaw_proxy}"
    assert np.sign(kd.yaw_proxy) == (raw_sign if raw_sign != 0 else 1), \
        "sign must stay consistent with the nose-offset proxy"


def test_noise_sigma_robust_to_texture_edges():
    """A high-contrast stripe across a noisy field must not inflate sigma."""
    rng = np.random.default_rng(42)
    img = np.full((300, 300, 3), 0.5, np.float32)
    img[144:156, :, 0] = 0.9
    img[144:156, :, 1] = 0.1
    img[144:156, :, 2] = 0.2
    noise = rng.normal(0, 0.02, (300, 300, 1)).astype(np.float32)
    img = img + noise
    est = estimate_sigma(img)
    assert abs(est - 0.02) < 0.25 * 0.02, f"stripe inflated sigma: {est}"


def test_identity_limit_adapts_to_pose_delta():
    checker = GeometricIdentityChecker()
    m, cap = _model()
    cap0 = cap.make_capture(CID, base_yaw=0.0, donor_yaws=[0.0])
    base_obs = FaceObservation("p0", cap0.base.obs_landmarks)
    same_pose = FaceObservation("p0", cap0.donors[0].obs_landmarks)
    # Raw proxies have small noise at "same pose" — the limit sits at ~base.
    assert abs(checker.limit(base_obs, same_pose) - 0.08) < 0.02
    # Exact pose difference via the override: limit is exact.
    turned = FaceObservation("p0", cap0.donors[0].obs_landmarks, yaw_override=0.4)
    frontal = FaceObservation("p0", cap0.base.obs_landmarks, yaw_override=0.0)
    assert abs(checker.limit(frontal, turned) - min(0.14, 0.08 + 0.15 * 0.4)) < 1e-6


# Silence unused-import lint: rel_dist re-exported alias used by some callers.
_ = rel_dist
