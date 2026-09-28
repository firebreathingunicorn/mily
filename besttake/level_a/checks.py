"""Classical (model-free) verification for Level A.

The learned composite detector replaces ``ClassicalArtifactChecker`` via the
same ``ArtifactChecker`` contract; the geometric identity gate is the Level A
stand-in for face embeddings (see ``identity.py``).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..common.types import FaceObservation
from .identity import descriptor, relative_distance
from .landmarks import key_points
from .noise import estimate_sigma


def _lap_var(g: np.ndarray) -> float:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _lap_rms(rgb: np.ndarray, mask: np.ndarray) -> float:
    """RMS of the 4-neighbor Laplacian of the luma inside a boolean mask —
    high-frequency energy; sensitive to added grain, insensitive to smooth
    warp error and constant color offsets."""
    if mask.sum() < 16:
        return 0.0
    g = _luma(rgb)
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    mk = np.asarray(mask, bool)[1:-1, 1:-1]
    return float(np.sqrt((lap[mk] ** 2).mean()))


def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, float)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


class ClassicalArtifactChecker:
    """Composite detector for Level A.

    Primary test (needs a reference): the pasted pixels must match the donor
    frame at the same coordinates — measured as the Laplacian energy of the
    composite-minus-donor residual (added grain is white noise and shows up
    strongly; sub-pixel warp error is smooth and does not), plus a mean-color
    term that catches constant tints (which a Laplacian cannot see). Without a
    reference, falls back to seam-grain and sharpness bounds.
    """

    def __init__(self, seam_noise_ratio: float = 2.2, max_color_shift: float = 0.14,
                 sharp_lo: float = 0.3, sharp_hi: float = 3.0,
                 residual_floor: float = 0.08, residual_k: float = 4.0):
        self.seam_noise_ratio = seam_noise_ratio
        self.max_color_shift = max_color_shift
        self.sharp_lo, self.sharp_hi = sharp_lo, sharp_hi
        self.residual_floor = residual_floor
        self.residual_k = residual_k

    def check(self, image: np.ndarray, region: np.ndarray,
              weight: np.ndarray | None = None,
              references: list[np.ndarray] | None = None) -> bool:
        image = np.asarray(image, float)
        mask = np.asarray(region, bool)
        solid = mask if weight is None else (np.asarray(weight, float) > 0.5) & mask
        if solid.sum() < 32:
            return False

        if references:
            donor = np.asarray(references[0], float)
            if donor.shape == image.shape:
                residual = np.clip(image, 0.0, 1.0) - donor
                art = _lap_rms(residual, solid)
                limit = max(self.residual_floor, self.residual_k * _lap_rms(donor, solid))
                if art > limit:
                    return False
                color_shift = float(np.linalg.norm(image[solid].mean(axis=0) - donor[solid].mean(axis=0)))
                return color_shift <= self.max_color_shift

        # Fallback: ring-based seam checks (noisier; content can confound).
        it = max(2, int(round(np.sqrt(solid.sum()) / 8)))
        ring = ndimage.binary_dilation(mask, iterations=it) & ~solid

        sigma_in = estimate_sigma(image, solid)
        sigma_out = estimate_sigma(image, ring)
        noise_ratio = (sigma_in / sigma_out) if sigma_out > 1e-6 else 1.0

        inside = image[solid].mean(axis=0)
        d_in = image[ring].mean(axis=0) if ring.any() else inside
        color_shift = float(np.linalg.norm(inside - d_in))

        sharp_in = _lap_var(_luma(image) * solid)
        sharp_ring = _lap_var(_luma(image) * ring) if ring.any() else sharp_in
        sharp_ratio = sharp_in / max(sharp_ring, 1e-8)

        return bool(noise_ratio <= self.seam_noise_ratio
                    and color_shift <= self.max_color_shift
                    and self.sharp_lo <= sharp_ratio <= self.sharp_hi)


class GeometricIdentityChecker:
    """Identity gate from landmark proportions (base observation vs donor).

    Rejects wrong-person swaps at Level A. The threshold adapts to the pose
    difference between the frames (2D proportions drift as the head turns):
    tau = min(cap, base + slope * |delta yaw|). Mirrors Swift IdentityChecker.
    """

    def __init__(self, base_max_distance: float = 0.08, pose_slope: float = 0.15,
                 max_distance_cap: float = 0.14):
        self.base_max_distance = base_max_distance
        self.pose_slope = pose_slope
        self.max_distance_cap = max_distance_cap

    def limit(self, base_obs: FaceObservation, donor_obs: FaceObservation) -> float:
        kb = key_points(base_obs.landmarks, base_obs.yaw_override)
        kd = key_points(donor_obs.landmarks, donor_obs.yaw_override)
        return min(self.max_distance_cap,
                   self.base_max_distance + self.pose_slope * abs(kb.yaw_proxy - kd.yaw_proxy))

    def check_pair(self, base_obs: FaceObservation, donor_obs: FaceObservation) -> tuple[bool, float]:
        d = relative_distance(descriptor(base_obs.landmarks), descriptor(donor_obs.landmarks))
        return d <= self.limit(base_obs, donor_obs), d
