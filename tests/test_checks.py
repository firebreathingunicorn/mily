"""Calibration-locked tests for the prototype verification gates."""
import numpy as np
from scipy.ndimage import gaussian_filter

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.adapters.simple_checks import (ArtifactCheckerProto,
                                             PaletteIdentityChecker)
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.pipeline import DonorInput, LevelBConfig, LevelBSwap

CID = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])


class _Pass:
    def check(self, *a, **k):
        return True


def _composite(yaw=25.0, seed=11):
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=288, seed=seed)
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=yaw, donor_yaws=[5.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d)
    res = LevelBSwap(m, LevelBConfig(mode="pinhole"),
                     artifact_checker=_Pass(), identity_checker=_Pass()).run(
        capture.base.frame,
        FaceObservation("p0", capture.base.obs_landmarks),
        [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
         for d in capture.donors])
    return res, capture


def test_artifact_gate_accepts_good_composite():
    res, capture = _composite()
    ac = ArtifactCheckerProto()
    assert ac.check(res.image, res.face_region, weight=res.weight,
                    references=[capture.donors[0].frame.rgb])


def test_artifact_gate_rejects_plastic_skin():
    res, capture = _composite()
    sm = gaussian_filter(res.image, (1.6, 1.6, 0))
    bad = res.image.copy()
    sel = res.weight > 0.5
    bad[sel] = sm[sel]
    ac = ArtifactCheckerProto()
    assert not ac.check(bad, res.face_region, weight=res.weight,
                        references=[capture.donors[0].frame.rgb])


def test_artifact_gate_rejects_noise_mismatch():
    res, capture = _composite()
    rng = np.random.default_rng(0)
    sel = res.weight > 0.5
    noisy = np.clip(res.image + rng.normal(0, 0.03, res.image.shape) * sel[..., None],
                    0, 1)
    ac = ArtifactCheckerProto()
    assert not ac.check(noisy, res.face_region, weight=res.weight,
                        references=[capture.donors[0].frame.rgb])


def test_identity_gate_same_vs_wrong_person():
    res, capture = _composite()
    ic = PaletteIdentityChecker()
    assert ic.check(res.image, res.face_region, [capture.donors[0].frame.rgb])
    m = SyntheticHeadModel(nu=64, nv=64)
    cap2 = SyntheticCapture(m, size=288, seed=2)
    other_cid = CID.copy()
    other_cid[0] = -0.6   # much lighter skin, different structure
    other_cid[1] = 0.5
    other = cap2.shoot(other_cid, np.zeros(6), 0.0)
    assert not ic.check(res.image, res.face_region, [other.frame.rgb])


def test_identity_gate_survives_auto_exposure():
    """Real bursts have auto-exposure between shots. Chromaticity is
    gain-invariant, so ±15% exposure references must not false-reject, while
    a different person under the same jitter still fails."""
    res, capture = _composite()
    ic = PaletteIdentityChecker()
    rng = np.random.default_rng(4)
    refs = [np.clip(capture.donors[0].frame.rgb * rng.uniform(0.85, 1.15), 0, 1)
            for _ in range(3)]
    assert ic.check(res.image, res.face_region, refs)
    m = SyntheticHeadModel(nu=64, nv=64)
    cap2 = SyntheticCapture(m, size=288, seed=2)
    other_cid = CID.copy()
    other_cid[0] = -0.6
    other_cid[1] = 0.5
    other = cap2.shoot(other_cid, np.zeros(6), 0.0)
    assert not ic.check(res.image, res.face_region,
                        [np.clip(other.frame.rgb * 1.1, 0, 1)])


def test_pipeline_rejects_corrupted_result():
    """End-to-end: a checker that (wrongly) flags everything forces the
    pipeline to leave the person unchanged rather than ship a bad composite."""
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=288, seed=11)
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=25.0, donor_yaws=[5.0],
                               base_c_ex=cex_b, donor_c_ex=cex_d)

    class _Reject:
        def check(self, *a, **k):
            return False

    res = LevelBSwap(m, LevelBConfig(mode="pinhole"),
                     artifact_checker=_Reject()).run(
        capture.base.frame,
        FaceObservation("p0", capture.base.obs_landmarks),
        [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))
         for d in capture.donors])
    assert res.method == "rejected"
    assert not res.checks["artifact"]
    assert np.array_equal(res.image, capture.base.frame.rgb)
    assert res.weight.sum() == 0
