"""Grain (noise sigma) estimation — 4-neighbor Laplacian estimator.

Same estimator and constants as the MilyCore Swift core: under additive
Gaussian noise the 4-neighbor Laplacian has variance 20*sigma^2. Uses the
MEDIAN of |L| — for the half-normal distribution of |L|, median =
0.6745*std — which makes the estimate robust to sparse texture edges that
would inflate a mean-based one.
"""
from __future__ import annotations

import numpy as np


def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, float)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def estimate_sigma(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Estimate sensor grain sigma on a frame (or inside a boolean mask)."""
    g = _luma(rgb)
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    if mask is not None:
        m = np.asarray(mask, bool)[1:-1, 1:-1]
        if m.sum() < 16:
            return 0.0
        vals = np.abs(lap[m])
    else:
        vals = np.abs(lap)
    if vals.size == 0:
        return 0.0
    return float(np.median(vals) / (0.6745 * np.sqrt(20.0)))
