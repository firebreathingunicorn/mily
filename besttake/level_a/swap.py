"""LevelASwap — Level A entry point returning the shared SwapResult.

Same shape as Level B's ``LevelBSwap.run``: base frame + observation, donor
candidates, injected Phase 1 providers (matting, noise, checkers). Returns a
``SwapResult`` with ``method == "A"``, or ``None`` when no donor is safely
transplantable (caller escalates to Level B — least invasive first).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..common.types import FaceObservation, Frame, SwapResult
from .checks import ClassicalArtifactChecker, GeometricIdentityChecker
from .landmarks import key_points
from .matting import EllipseFeatherMatting
from .pose import refine_burst_yaw
from .scoring import risk, score_frame
from .transplant import direct_transplant


@dataclass
class ADonor:
    frame: Frame
    obs: FaceObservation


@dataclass
class LevelAConfig:
    """Pose envelope of direct transplantation: beyond this, escalate to B.

    The yaw proxy's scale depends on the face model (stylized heads have
    close-set eyes, inflating proxy-per-degree), so the envelope is expressed
    in proxy units calibrated on the research head: ±0.55 ≈ a 10-12° donor
    delta. Risk only ranks donor choice; the envelope gates.
    """
    max_yaw_delta: float = 0.55
    max_roll_delta: float = 0.35
    max_risk: float = 1.5
    min_gain: float = 0.03


class LevelASwap:
    def __init__(self,
                 matting=None,
                 noise=None,
                 artifact_checker=None,
                 identity_checker=None,
                 config: LevelAConfig | None = None,
                 seed: int = 7):
        self.matting = matting if matting is not None else EllipseFeatherMatting()
        self.noise = noise
        self.artifact_checker = artifact_checker if artifact_checker is not None else ClassicalArtifactChecker()
        self.identity_checker = identity_checker if identity_checker is not None else GeometricIdentityChecker()
        self.config = config if config is not None else LevelAConfig()
        self.rng = np.random.default_rng(seed)

    def run(self, base: Frame, base_obs: FaceObservation,
            donors: list[ADonor], base_score: float | None = None) -> SwapResult | None:
        if not donors:
            return None
        cfg = self.config

        # Burst-relative pose refinement (foreshortening-based yaw) — makes
        # the pose gates and the adaptive identity threshold far more accurate.
        refine_burst_yaw([base_obs] + [d.obs for d in donors])

        # Donor choice: best expression gain discounted by swap risk, gated by
        # the Level A pose envelope (least invasive first).
        best: tuple[float, ADonor] | None = None
        for d in donors:
            r = risk(base_obs, d.obs)
            if r > cfg.max_risk:
                continue
            kp_d = key_points(d.obs.landmarks, d.obs.yaw_override)
            kp_b = key_points(base_obs.landmarks, base_obs.yaw_override)
            if abs(kp_d.yaw_proxy - kp_b.yaw_proxy) > cfg.max_yaw_delta:
                continue
            if abs(kp_d.roll - kp_b.roll) > cfg.max_roll_delta:
                continue
            gain = score_frame(d.frame, d.obs) - (base_score if base_score is not None else 0.0)
            value = gain - 0.6 * r
            if best is None or value > best[0]:
                best = (value, d)
        if best is None:
            return None
        donor = best[1]

        # Identity gate before touching pixels.
        id_ok, id_dist = self.identity_checker.check_pair(base_obs, donor.obs)
        if not id_ok:
            return SwapResult(
                image=base.rgb, weight=np.zeros(base.shape, np.float32),
                source_map=-np.ones(base.shape, np.int32),
                face_region=np.zeros(base.shape, bool),
                method="none", checks={"identity": False, "identity_distance": id_dist},
            )

        alpha = self.matting.alpha(donor.frame, donor.obs)
        sigma_base = self.noise.sigma(base.frame) if self.noise is not None else None
        draft = direct_transplant(
            base, base_obs, donor.frame, donor.obs, alpha,
            noise_sigma_base=sigma_base, rng=self.rng,
        )
        if draft is None:
            return None

        art_ok = self.artifact_checker.check(
            draft.image, draft.region, weight=draft.weight,
            references=[donor.frame.rgb],
        )

        weight = draft.weight
        swapped = weight > 0.5
        # Fraction of the transplant region actually pasted (same semantics
        # as Level B's coverage: relative to the face area, not the frame).
        coverage = float(swapped.sum()) / max(float(draft.region.sum()), 1.0)
        return SwapResult(
            image=draft.image,
            weight=weight,
            source_map=np.where(swapped, 0, -1).astype(np.int32),
            face_region=draft.region,
            method="A",
            coverage=coverage,
            fill_fraction=0.0,
            checks={
                "artifact": bool(art_ok),
                "identity": True,
                "identity_distance": id_dist,
            },
        )
