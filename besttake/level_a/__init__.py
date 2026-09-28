"""Level A (Phase 1) — direct pixel transplant and its supporting Phase 1
signals, implementing the contracts in ``besttake.common.contracts``.

Least-invasive-first: everything shown is a real pixel of this person from
this capture session. This module mirrors the production Swift core
(``Sources/BestTakeCore``) so the Python research loop exercises the same
algorithm and thresholds:

- ``noise``      4-neighbor Laplacian grain estimator (same constant)
- ``identity``   geometric identity descriptor + relative distance gate
- ``similarity`` landmark-fitted similarity transform (Umeyama, closed form)
- ``matting``    soft feathered ellipse matting (MattingProvider)
- ``transplant`` warp donor -> base, feather, noise match, sharpness match
- ``checks``     classical artifact checker (seam noise / color / sharpness)
- ``swap``       LevelASwap returning the shared ``SwapResult`` (method "A")
"""
from .landmarks import key_points, KeyPoints
from .similarity import fit_similarity, invert_similarity, apply_similarity
from .noise import estimate_sigma
from .identity import descriptor, relative_distance
from .matting import EllipseFeatherMatting
from .scoring import score_frame, risk, plan_burst
from .checks import ClassicalArtifactChecker, GeometricIdentityChecker
from .pose import refine_burst_yaw, MIN_RATIO as pose_min_ratio
from .transplant import direct_transplant, TransplantDraft
from .swap import ADonor, LevelAConfig, LevelASwap

__all__ = [
    "KeyPoints", "key_points",
    "fit_similarity", "invert_similarity", "apply_similarity",
    "estimate_sigma",
    "descriptor", "relative_distance",
    "EllipseFeatherMatting",
    "score_frame", "risk", "plan_burst",
    "ClassicalArtifactChecker", "GeometricIdentityChecker", "refine_burst_yaw",
    "direct_transplant", "TransplantDraft",
    "ADonor", "LevelAConfig", "LevelASwap",
]
