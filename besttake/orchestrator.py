"""Best-take orchestrator: least-invasive-first synthesis policy.

This is the product policy layer (plan principle 2: "choose the least invasive
method that works"). It imports both workstreams:

- Level A (``besttake.level_a``, Phase 1): direct transplant, gated by the
  pose envelope — used whenever the donor's head pose is close to the base's.
- Level B (``besttake.level_b``, Phase 3): 3D-aware re-projection — used for
  larger head turns, or when Level A's checks reject the result.

Either workstream may be omitted (e.g. Level B not installed); the
orchestrator degrades to whatever is available, and finally to "leave the
person unchanged" — a real pixel beats a generated one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from besttake.common.types import FaceObservation, Frame, SwapResult
from besttake.level_a import ADonor, LevelASwap


@dataclass
class OrchestratorConfig:
    """Yaw-proxy delta below which Level A is preferred. Proxy units match
    LevelAConfig.max_yaw_delta (see level_a.swap for the calibration)."""
    level_a_yaw_split: float = 0.55


@dataclass
class BDonor:
    """Donor in Level B's input shape, built lazily to avoid a hard import."""
    frame: Frame
    obs: FaceObservation


@dataclass
class Outcome:
    result: SwapResult | None
    level: str            # "A", "B", or "none"
    reason: str = ""
    history: list = field(default_factory=list)


class BestTakeOrchestrator:
    """swap(base, base_obs, donors) -> Outcome with the best safe result."""

    def __init__(self, level_a: LevelASwap | None = None,
                 level_b=None,  # optional LevelBSwap
                 config: OrchestratorConfig | None = None):
        self.level_a = level_a if level_a is not None else LevelASwap()
        self.level_b = level_b
        self.config = config if config is not None else OrchestratorConfig()

    def swap(self, base: Frame, base_obs: FaceObservation,
             donors: list[BDonor], base_score: float | None = None) -> Outcome:
        history: list[str] = []
        a_donors = [ADonor(frame=d.frame, obs=d.obs) for d in donors]

        # Prefer Level A when a donor exists inside the direct-transplant
        # envelope (small pose delta) — cheapest, and every pixel is real.
        from besttake.level_a.landmarks import key_points
        kb = key_points(base_obs.landmarks, base_obs.yaw_override)
        near = [d for d in donors
                if abs(key_points(d.obs.landmarks, d.obs.yaw_override).yaw_proxy - kb.yaw_proxy)
                <= self.config.level_a_yaw_split]
        if near:
            res_a = self.level_a.run(base, base_obs, a_donors, base_score=base_score)
            if res_a is not None and res_a.checks.get("artifact", False):
                history.append("A:accepted")
                return Outcome(res_a, "A", history=history)
            history.append("A:" + ("rejected" if res_a is not None else "no-viable-donor"))
        else:
            history.append("A:skipped(pose)")

        # Escalate to Level B for larger turns (or when A failed checks).
        if self.level_b is not None:
            from besttake.level_b.pipeline import DonorInput
            b_donors = [DonorInput(d.frame, d.obs) for d in donors]
            res_b = self.level_b.run(base, base_obs, b_donors)
            if res_b is not None and res_b.checks.get("artifact", False):
                history.append("B:accepted")
                return Outcome(res_b, "B", history=history)
            history.append("B:" + ("rejected" if res_b is not None else "failed"))
        else:
            history.append("B:unavailable")

        # Fall back to leaving the person unchanged.
        return Outcome(None, "none", "no method passed verification", history)
