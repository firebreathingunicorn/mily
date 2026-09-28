import numpy as np

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.geometry import inv_rodrigues
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.fitter import FaceFitter


def _case():
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=288, seed=11)
    rng = np.random.default_rng(5)
    cid = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])
    return m, cap, cid


def test_landmark_fit_recovers_pose():
    m, cap, cid = _case()
    fitter = FaceFitter(m, mode="pinhole")
    cex = np.zeros(6); cex[1] = 0.5
    shot = cap.shoot(cid, cex, 20.0)
    fit = fitter.fit(shot.obs_landmarks, intrinsics=cap.intrinsics,
                     c_identity=cid, fix_identity=True)
    w = inv_rodrigues(fit.R)
    assert abs(np.degrees(w[1]) - 20.0) < 1.0
    assert fit.landmark_rmse < 1.2
    # jaw/smile/brow/pucker/lip are landmark-observable; eye-close is almost
    # purely textural (tiny geometry) and only ridge-bounded
    obs = [0, 1, 2, 4, 5]
    assert np.abs(fit.c_expression[obs] - cex[obs]).max() < 0.35
    assert abs(fit.c_expression[3]) < 1.0


def test_depth_refine_anchors_metric_pose():
    m, cap, cid = _case()
    fitter = FaceFitter(m, mode="pinhole")
    shot = cap.shoot(cid, np.zeros(6), 25.0)
    fit_ld = fitter.fit(shot.obs_landmarks, intrinsics=cap.intrinsics,
                        c_identity=cid, fix_identity=True)
    fit_dep = fitter.fit(shot.obs_landmarks, depth=shot.frame.depth,
                         intrinsics=cap.intrinsics, c_identity=cid, fix_identity=True)
    assert abs(fit_dep.T[2] - 3.0) < 0.02
    assert abs(fit_dep.T[2] - 3.0) <= abs(fit_ld.T[2] - 3.0) + 1e-3
    w = inv_rodrigues(fit_dep.R)
    assert abs(np.degrees(w[1]) - 25.0) < 1.0


def test_weak_mode_fit():
    """Weak perspective alone over-rotates on turned heads (foreshortening
    bias, ~10° at 15° true yaw). With assumed-FOV intrinsics the fallback
    refines focal from foreshortening and recovers the pose."""
    m, cap, cid = _case()
    from besttake.common.types import Intrinsics
    fitter = FaceFitter(m, mode="weak")
    shot = cap.shoot(cid, np.zeros(6), -15.0)
    assumed = Intrinsics(fx=0.75 * 288, fy=0.75 * 288, cx=143.5, cy=143.5)
    fit = fitter.fit(shot.obs_landmarks, intrinsics=assumed,
                     c_identity=cid, fix_identity=True)
    w = inv_rodrigues(fit.R)
    assert abs(np.degrees(w[1]) - (-15.0)) < 1.5
    assert fit.landmark_rmse < 4.0
    assert fit.mode == "pinhole"
    # the estimated focal lands near the true one from a wrong prior
    assert 300.0 < fit.intrinsics.fx < 460.0


def test_burst_identity_beats_single_frame():
    """Identity fitted per donor and median-combined is closer to the person
    than any single frame's fit (plan: identity from the person's burst)."""
    m, cap, cid = _case()
    fitter = FaceFitter(m, mode="pinhole")
    cex = np.zeros(6); cex[1] = 0.5
    per_frame = []
    for yaw in (-12.0, 4.0, 18.0):
        shot = cap.shoot(cid, cex, yaw)
        fit = fitter.fit(shot.obs_landmarks, depth=shot.frame.depth,
                         intrinsics=cap.intrinsics)
        per_frame.append(fit.c_identity)
    combined = np.median(np.stack(per_frame), axis=0)
    err_single = np.abs(np.asarray(per_frame) - cid).max()
    err_combined = np.abs(combined - cid).max()
    assert err_combined < err_single, \
        f"burst identity {err_combined:.2f} not better than single {err_single:.2f}"


def test_yaw_hint_matches_full_multistart():
    """The tracker-proxy hint must not cost accuracy: hinted fit converges to
    the same solution as the full multi-start (and skips 4 of 5 stage-A
    solves, with the rmse fallback covering bad hints)."""
    m, cap, cid = _case()
    fitter = FaceFitter(m, mode="pinhole")
    cex = np.zeros(6); cex[1] = 0.5
    for yaw_true in (-25.0, 10.0, 33.0):
        shot = cap.shoot(cid, cex, yaw_true)
        fit_hint = fitter.fit(shot.obs_landmarks, depth=shot.frame.depth,
                              intrinsics=cap.intrinsics, c_identity=cid,
                              fix_identity=True, yaw_hint_deg=yaw_true + 4.0)
        fit_full = fitter.fit(shot.obs_landmarks, depth=shot.frame.depth,
                              intrinsics=cap.intrinsics, c_identity=cid,
                              fix_identity=True)
        assert abs(np.degrees(inv_rodrigues(fit_hint.R)[1]) - yaw_true) < 1.0, \
            f"hinted yaw off at {yaw_true}"
        assert np.allclose(fit_hint.R, fit_full.R, atol=1e-3), "hint != full"


def test_hint_fallback_recovers_from_bad_hint():
    """A wildly wrong hint must be caught by the rmse fallback, not trusted."""
    m, cap, cid = _case()
    fitter = FaceFitter(m, mode="pinhole")
    shot = cap.shoot(cid, np.zeros(6), 28.0)
    fit = fitter.fit(shot.obs_landmarks, intrinsics=cap.intrinsics,
                     c_identity=cid, fix_identity=True, yaw_hint_deg=-38.0)
    assert abs(np.degrees(inv_rodrigues(fit.R)[1]) - 28.0) < 1.0


def test_huber_survives_landmark_outliers():
    """A few bad detections (detector glitches) must not bend the pose."""
    m, cap, cid = _case()
    fitter = FaceFitter(m, mode="pinhole")
    cex = np.zeros(6); cex[1] = 0.5
    shot = cap.shoot(cid, cex, 18.0)
    obs = shot.obs_landmarks.copy()
    rng = np.random.default_rng(9)
    bad = rng.choice(len(obs), 6, replace=False)
    obs[bad] += rng.uniform(-18, 18, (6, 2))
    fit = fitter.fit(obs, intrinsics=cap.intrinsics,
                     c_identity=cid, fix_identity=True)
    w = inv_rodrigues(fit.R)
    assert abs(np.degrees(w[1]) - 18.0) < 1.5, f"yaw pulled by outliers"
    assert fit.landmark_rmse < 6.0  # robust rmse stays bounded
