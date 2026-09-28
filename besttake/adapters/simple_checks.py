"""Prototype verification gates (plan §Verification).

Honest-but-simple stand-ins for the Phase 1 trained models: they make the
pipeline's reject path real today and pin the wiring that the production
checkers (composite detector, face-embedding identity threshold) drop into
unchanged. Calibration numbers live in tests/test_checks.py so threshold
changes stay traceable.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace


def _sigma_hf(image: np.ndarray, mask: np.ndarray) -> float:
    """Robust high-frequency amplitude inside a mask (MAD of Laplacian)."""
    gray = image.astype(np.float64).mean(-1)
    lap = laplace(gray)
    vals = np.abs(lap[mask] - np.median(lap[mask]))
    return float(np.median(vals) / 0.6745 / np.sqrt(20.0))


class ArtifactCheckerProto:
    """Texture-energy artifact gate (prototype for the trained detector).

    Compares the composite's high-frequency energy (robust MAD-of-Laplacian
    sigma, full frame) against the donor frames it was built from. Correct
    finishing preserves the source texture level; the classic failures move
    it far off — plastic-skin smoothing collapses it, noise mismatches and
    recompression blow it up.

    Calibration (288 px synthetic captures, donor yaw ~5 deg, base yaw
    5-35 deg): correct composites 0.64-0.67 of donor energy (hair pixels
    aren't swapped), smoothed-paste corruption 0.13-0.19, added-noise
    corruption 1.8-2.1. Pass band [0.35, 1.4].
    """

    def __init__(self, lo: float = 0.35, hi: float = 1.4):
        self.lo, self.hi = lo, hi

    def check(self, image: np.ndarray, region: np.ndarray,
              weight: np.ndarray | None = None,
              references: list[np.ndarray] | None = None) -> bool:
        if not references:
            return True
        s_img = _sigma_hf(image, np.ones(image.shape[:2], bool))
        s_ref = float(np.median([_sigma_hf(r, np.ones(r.shape[:2], bool))
                                 for r in references]))
        if s_ref < 1e-6:
            return True
        return self.lo <= s_img / s_ref <= self.hi


# Backwards-compatible alias for the earlier experimental name.
SeamArtifactChecker = ArtifactCheckerProto


class PaletteIdentityChecker:
    """Catches gross wrong-person/wrong-appearance swaps by color statistics.

    Exposure-robust: compares *chromaticity* (rgb / luma) of the swapped
    region against each reference's own face palette — auto-exposure between
    shots is a multiplicative gain that cancels in chromaticity, so the gate
    does not false-reject bursts, while a genuinely different skin/hair tone
    still fails. A loose luminance-ratio guard catches gross brightness
    identity errors. Pose-independent: a reference may be a (frame, region)
    pair, and the region is where that frame's face actually is. This is NOT
    an embedding check and cannot catch intra-palette identity drift; it
    exists to exercise the reject path until the Phase 1/4 identity model
    lands.

    Calibration: same person under auto-exposure jitter differs by <0.005 in
    chromaticity (gain-invariant); a different skin/hue tone differs by ~0.05.
    """

    def __init__(self, chroma_threshold: float = 0.03,
                 luma_band: tuple[float, float] = (0.45, 2.2)):
        self.chroma_threshold = chroma_threshold
        self.luma_band = luma_band

    @staticmethod
    def _stats(entry, fallback_region):
        frame, region = (entry if isinstance(entry, tuple) else (entry, fallback_region))
        if frame.shape[:2] != region.shape or region.sum() < 100:
            return None
        px = frame[region].reshape(-1, 3).astype(np.float64) + 1e-6
        chroma = px / px.sum(1, keepdims=True)
        return chroma.mean(0), float(px.mean())

    def check(self, image: np.ndarray, region: np.ndarray,
              references: list) -> bool:
        if not references or region.sum() < 100:
            return True
        stats = [self._stats(r, region) for r in references]
        stats = [s for s in stats if s is not None]
        if not stats:
            return True
        px = image[region].reshape(-1, 3).astype(np.float64) + 1e-6
        chroma_img = (px / px.sum(1, keepdims=True)).mean(0)
        luma_img = float(px.mean())
        d_chroma = min(float(np.linalg.norm(c - chroma_img)) for c, _ in stats)
        r_luma = luma_img / max(np.median([l for _, l in stats]), 1e-6)
        return d_chroma < self.chroma_threshold and \
            self.luma_band[0] <= r_luma <= self.luma_band[1]
