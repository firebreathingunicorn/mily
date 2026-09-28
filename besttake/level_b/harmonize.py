"""Finishing: color harmonization, shading transfer and grain matching (plan §5).

The shading transfer is the prototype stand-in for learned relighting
(plan §risks #2): donor pixels carry the donor pose's shading; a masked,
low-frequency luminance ratio against the base frame corrects the dominant
illumination difference without touching albedo detail.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def estimate_noise_sigma(rgb: np.ndarray) -> float:
    """Sensor grain level from the median absolute deviation of the Laplacian
    (robust to edges/shading): Var[Laplacian] = 20 sigma^2 for white noise.
    Channels are estimated separately (averaging them first would shrink the
    noise by sqrt(3))."""
    sigmas = []
    for c in range(rgb.shape[-1]):
        lap = laplace(rgb[..., c].astype(np.float64))
        mad = np.median(np.abs(lap - np.median(lap)))
        sigmas.append(mad / 0.6745 / np.sqrt(20.0))
    return float(np.median(sigmas))


def color_transfer(pasted: np.ndarray, base: np.ndarray, weight: np.ndarray,
                   strength: float = 0.8
                   ) -> tuple[np.ndarray, dict]:
    """Affine per-channel match of the pasted face to the base image, computed
    on the confident overlap region. Returns corrected paste + params."""
    sel = weight > 0.6
    params = {"gain": np.ones(3), "bias": np.zeros(3), "n": int(sel.sum())}
    if sel.sum() < 50:
        return pasted, params
    a = pasted[sel].astype(np.float64)
    b = base[sel].astype(np.float64)
    gain = b.std(0) / np.maximum(a.std(0), 1e-6)
    gain = np.clip(gain, 0.5, 2.0)
    bias = b.mean(0) - a.mean(0) * gain
    params = {"gain": gain, "bias": bias, "n": int(sel.sum())}
    out = pasted.astype(np.float64) * gain + bias
    out = pasted + strength * (out - pasted)
    return np.clip(out, 0.0, 1.0).astype(np.float32), params


def shading_transfer(pasted: np.ndarray, base: np.ndarray, weight: np.ndarray,
                     sigma_px: float = 24.0, strength: float = 0.8,
                     gain_clip: tuple[float, float] = (0.55, 1.8),
                     fill_sigma: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Low-frequency luminance-ratio relight of the pasted face.

    On confident overlap the base and the paste show (nearly) the same
    geometry, so blur(base)/blur(paste) estimates the illumination ratio;
    the ratio field is extended across the whole pasted support by normalized
    convolution and applied as a per-pixel scalar gain. High-frequency
    (albedo/detail) structure is untouched.
    """
    sel = (weight > 0.5)
    support = weight > 0.02
    if sel.sum() < 200:
        return pasted, np.ones_like(weight)
    lb = base.astype(np.float64).mean(-1)
    lp = pasted.astype(np.float64).mean(-1)
    w = sel.astype(np.float64)
    den = gaussian_filter(w, sigma_px)
    ok = den > 0.05
    ratio = np.ones_like(lb)
    if ok.any():
        ratio[ok] = (gaussian_filter(lb * w, sigma_px)[ok] / den[ok]) / \
                    (gaussian_filter(lp * w, sigma_px)[ok] / den[ok] + 1e-3)
    # extend the ratio across the pasted region
    m = (ok & support).astype(np.float64)
    num = gaussian_filter(np.where(m > 0, ratio, 0.0), fill_sigma)
    div = gaussian_filter(m, fill_sigma)
    gain = np.ones_like(ratio)
    good = div > 1e-4
    gain[good] = num[good] / div[good]
    gain = np.clip(gain, *gain_clip) ** strength
    gain[~support] = 1.0
    out = np.clip(pasted.astype(np.float64) * gain[..., None], 0.0, 1.0)
    return out.astype(np.float32), gain.astype(np.float32)


def grain_match(out: np.ndarray, weight: np.ndarray, sigma_base: float,
                sigma_donor: float, rng: np.random.Generator) -> np.ndarray:
    """Add base-matched grain inside pasted regions (the most common giveaway
    once seams are fixed). Only the deficit vs. the donor's own grain is added."""
    deficit = np.sqrt(max(sigma_base ** 2 - sigma_donor ** 2, 0.0))
    if deficit < 1e-5:
        return out
    noise = rng.normal(0.0, deficit, out.shape[:2])[..., None]
    return np.clip(out + (noise * weight[..., None]).astype(out.dtype), 0.0, 1.0)
