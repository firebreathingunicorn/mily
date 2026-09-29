"""Orchestrator tests: least-invasive-first level choice on real Level B
synthetic captures."""
import numpy as np

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_a import LevelASwap
from besttake.orchestrator import BestTakeOrchestrator, BDonor

CID = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])


def _capture(base_yaw, donor_yaws, seed=11):
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=288, seed=seed)
    cex_b = np.zeros(6); cex_b[2] = 0.8
    cex_d = np.zeros(6); cex_d[1] = 0.6
    capture = cap.make_capture(CID, base_yaw=base_yaw, donor_yaws=donor_yaws,
                               base_c_ex=cex_b, donor_c_ex=cex_d)
    return m, capture


def _donors(capture):
    return [BDonor(frame=d.frame, obs=FaceObservation("p0", d.obs_landmarks))
            for d in capture.donors]


def test_small_yaw_uses_level_a():
    m, capture = _capture(8.0, [2.0])
    orch = BestTakeOrchestrator()  # no Level B installed
    out = orch.swap(capture.base.frame,
                    FaceObservation("p0", capture.base.obs_landmarks),
                    _donors(capture))
    assert out.level == "A" and out.result is not None
    assert out.result.method == "A"
    assert "A:accepted" in out.history


def test_large_yaw_skips_a_without_level_b():
    m, capture = _capture(30.0, [5.0])
    orch = BestTakeOrchestrator()  # Level B unavailable
    out = orch.swap(capture.base.frame,
                    FaceObservation("p0", capture.base.obs_landmarks),
                    _donors(capture))
    assert out.level == "none" and out.result is None
    assert "A:skipped(pose)" in out.history
    assert "B:unavailable" in out.history


def test_a_rejected_escalates_to_level_b():
    """When Level A's artifact check rejects a same-pose swap (simulated here
    with a broken checker), the orchestrator must escalate to Level B."""
    m, capture = _capture(8.0, [8.0])  # small delta: A would normally win

    class _AlwaysFailChecker:
        def check(self, *a, **k):
            return False

    from besttake.level_b.pipeline import LevelBConfig, LevelBSwap
    level_b = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    broken_a = LevelASwap(artifact_checker=_AlwaysFailChecker())
    orch = BestTakeOrchestrator(level_a=broken_a, level_b=level_b)
    out = orch.swap(capture.base.frame,
                    FaceObservation("p0", capture.base.obs_landmarks),
                    _donors(capture))
    assert out.level == "B" and out.result is not None
    assert out.result.method.startswith("B")
    assert "A:rejected" in out.history and "B:accepted" in out.history


def test_large_yaw_escalates_to_level_b():
    m, capture = _capture(25.0, [5.0])
    from besttake.level_b.pipeline import LevelBConfig, LevelBSwap
    level_b = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    orch = BestTakeOrchestrator(level_b=level_b)
    out = orch.swap(capture.base.frame,
                    FaceObservation("p0", capture.base.obs_landmarks),
                    _donors(capture))
    assert out.level == "B" and out.result is not None
    assert out.result.method.startswith("B")
    assert "A:skipped(pose)" in out.history and "B:accepted" in out.history


def test_b_rejected_falls_through_to_none():
    """Level B returning the unchanged base (method == "rejected", e.g. the
    identity gate failed) must NOT be reported as B:accepted — the
    orchestrator falls through to "leave the person unchanged"."""
    m, capture = _capture(25.0, [5.0])

    class _AlwaysFailIdentity:
        def check(self, *a, **k):
            return False

    from besttake.level_b.pipeline import LevelBConfig, LevelBSwap
    level_b = LevelBSwap(m, LevelBConfig(mode="pinhole"),
                         identity_checker=_AlwaysFailIdentity())
    orch = BestTakeOrchestrator(level_b=level_b)
    out = orch.swap(capture.base.frame,
                    FaceObservation("p0", capture.base.obs_landmarks),
                    _donors(capture))
    assert out.level == "none" and out.result is None
    assert "B:rejected" in out.history
    assert "B:accepted" not in out.history


def test_b_unusable_fit_does_not_crash():
    """NaN landmarks mean Level B cannot fit anything: LevelBSwap.run raises
    ValueError. The orchestrator must catch it and degrade to "leave the
    person unchanged" instead of crashing."""
    m, capture = _capture(25.0, [5.0])
    from besttake.level_b.pipeline import LevelBConfig, LevelBSwap
    level_b = LevelBSwap(m, LevelBConfig(mode="pinhole"))
    orch = BestTakeOrchestrator(level_b=level_b)

    base_obs = FaceObservation("p0", capture.base.obs_landmarks.copy() * np.nan)
    donors = []
    for d in capture.donors:
        lm = d.obs_landmarks.copy()
        lm[:, 0] = np.nan
        donors.append(BDonor(frame=d.frame,
                             obs=FaceObservation("p0", lm)))
    out = orch.swap(capture.base.frame, base_obs, donors)
    assert out.level == "none" and out.result is None
    assert any(h.startswith("B:failed(") for h in out.history)


def test_level_a_rejected_is_never_accepted():
    """A Level A result that carries method="none" (identity gate failed)
    must not pass the orchestrator's acceptance gate even if a broken
    checker stuffed artifact=True into its checks."""
    from besttake.common.types import SwapResult
    from besttake.orchestrator import _accepted
    res = SwapResult(
        image=np.zeros((4, 4, 3), np.float32),
        weight=np.zeros((4, 4), np.float32),
        source_map=np.full((4, 4), -1, np.int32),
        face_region=np.zeros((4, 4), bool),
        method="none", checks={"artifact": True, "identity": False},
    )
    assert not _accepted(res)
