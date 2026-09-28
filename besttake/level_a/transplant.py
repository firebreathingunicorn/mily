"""Level A synthesis: direct donor->base face transplant.

Every pasted pixel is a real pixel of the donor frame (least invasive that
works). Steps mirror the Swift ``DirectTransplantSynthesizer`` + ``Finisher``:

1. similarity fit donor landmarks -> base landmarks
2. warp donor RGB + donor matting alpha into the base face region
3. multiply by a feathered face ellipse (soft edges, never a hard cutout)
4. sharpness match against the base's own face region
5. grain matching: add sqrt(sigma_base^2 - sigma_donor^2) of noise
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from ..common.types import FaceObservation, Frame
from .landmarks import face_ellipse, key_points
from .matting import soft_ellipse
from .noise import estimate_sigma
from .similarity import fit_similarity, invert_similarity


@dataclass
class TransplantDraft:
    image: np.ndarray          # composited full frame
    weight: np.ndarray         # (H, W) blend weight (0 outside region)
    region: np.ndarray         # (H, W) bool, the transplant region
    content: np.ndarray        # region-sized warped donor pixels
    donor_sigma: float
    transform: np.ndarray


def _bilinear_warp(src: np.ndarray, m_inv: np.ndarray, region: tuple[slice, slice]) -> np.ndarray:
    """Sample `src` at inverse-mapped coordinates over the region grid.

    Works for (H, W) and (H, W, C) sources; out-of-bounds samples are 0.
    """
    ys, xs = region
    h, w = ys.stop - ys.start, xs.stop - xs.start
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx += xs.start
    yy += ys.start
    sx = m_inv[0, 0] * xx + m_inv[0, 1] * yy + m_inv[0, 2]
    sy = m_inv[1, 0] * xx + m_inv[1, 1] * yy + m_inv[1, 2]
    coords = np.stack([sy, sx])
    if src.ndim == 2:
        return ndimage.map_coordinates(src, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
    channels = [
        ndimage.map_coordinates(src[..., c], coords, order=1, mode="constant", cval=0.0)
        for c in range(src.shape[2])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def _color_match(content: np.ndarray, base_face: np.ndarray,
                 weight: np.ndarray, min_shift: float = 0.006) -> np.ndarray:
    """Match content's per-channel mean/sigma to the base face's, over the
    alpha-solid support. Skipped when the shift is within noise."""
    solid = weight > 0.5
    if int(solid.sum()) < 64:
        return content
    d = content[solid].astype(np.float64)
    b = base_face[solid].astype(np.float64)
    mu_d, mu_b = d.mean(axis=0), b.mean(axis=0)
    sd, sb = d.std(axis=0), b.std(axis=0)
    gains = np.where(sd > 1e-4, np.clip(sb / np.maximum(sd, 1e-6), 0.6, 1.6), 1.0)
    if (np.abs(mu_b - mu_d) < min_shift).all() and (np.abs(gains - 1.0) < 0.05).all():
        return content
    matched = (content - mu_d) * gains + mu_b
    return (matched * weight[..., None] + content * (1.0 - weight[..., None])).astype(np.float32)


def _face_region(shape, kp, inflation: float = 1.55) -> tuple[slice, slice]:
    h, w = shape
    pts = np.stack([kp.eye_a, kp.eye_b, kp.nose_tip, kp.mouth_l, kp.mouth_r, kp.chin])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * inflation / 2, (y1 - y0) * inflation / 2
    xs = slice(int(max(0, cx - hw)), int(min(w, cx + hw)))
    ys = slice(int(max(0, cy - hh)), int(min(h, cy + hh)))
    return ys, xs


def _lap_var(g: np.ndarray) -> float:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, float)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def direct_transplant(
    base: Frame,
    base_obs: FaceObservation,
    donor: Frame,
    donor_obs: FaceObservation,
    donor_alpha: np.ndarray,
    feather_fraction: float = 0.30,
    region_inflation: float = 1.55,
    sharpness_ratio_threshold: float = 1.8,
    noise_sigma_base: float | None = None,
    rng: np.random.Generator | None = None,
) -> TransplantDraft | None:
    """Warp the donor's face onto the base frame with matched grain.

    ``donor_alpha``: soft matting alpha for the donor in the donor frame
    (from a MattingProvider). ``noise_sigma_base`` overrides the measured
    base grain (e.g. when the caller already knows it).
    """
    m = fit_similarity(donor_obs.landmarks, base_obs.landmarks)
    if m is None:
        return None
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    if not (0.2 < scale < 5.0):
        return None
    m_inv = invert_similarity(m)

    kb = key_points(base_obs.landmarks)
    kd = key_points(donor_obs.landmarks)
    region = _face_region(base.shape, kb, region_inflation)
    ys, xs = region
    if ys.stop - ys.start < 8 or xs.stop - xs.start < 8:
        return None

    content = _bilinear_warp(donor.rgb, m_inv, region)
    warped_alpha = _bilinear_warp(donor_alpha[..., None], m_inv, region)[..., 0]

    # Feathered face ellipse in region-local coordinates, sized from the
    # landmark extent (stylized head models are not io-proportioned).
    feather_c, feather_rx, feather_ry = face_ellipse(kb)
    cx = feather_c[0] - xs.start
    cy = feather_c[1] - ys.start
    feather = soft_ellipse(content.shape[:2], (cx, cy), feather_rx, feather_ry,
                           max(4.0, min(feather_rx, feather_ry) * feather_fraction))
    weight_region = np.clip(warped_alpha * feather, 0.0, 1.0)

    # Sharpness match against the base's own face region (same content type;
    # a face is always sharper than the surrounding background).
    g_content = _luma(content)
    g_base_face = _luma(base.rgb[region])
    sharp_c, sharp_b = _lap_var(g_content), _lap_var(g_base_face)
    if sharp_b > 1e-8 and sharp_c > sharp_b * sharpness_ratio_threshold:
        sigma_blur = float(np.clip(0.6 * np.log2(sharp_c / sharp_b), 0.5, 1.2))
        content = np.stack([
            ndimage.gaussian_filter(content[..., c], sigma_blur) for c in range(3)
        ], axis=-1)

    # Color harmonization: per-channel mean/sigma transfer of the donor
    # content toward the base's own face pixels. Bursts can re-meter between
    # frames; a pasted face must sit at the base frame's exposure.
    content = _color_match(content, base.rgb[region], weight_region)

    # Grain matching: add only the noise deficit vs the base frame.
    io = kb.inter_ocular
    ring = ndimage.binary_dilation(np.ones(content.shape[:2], bool), iterations=max(4, int(io / 2)))
    base_ring = np.zeros(base.shape, bool)
    base_ring[region] = ring
    sb = noise_sigma_base if noise_sigma_base is not None else estimate_sigma(base.rgb, base_ring)
    donor_face = np.zeros(donor.shape, bool)
    kpd_pts = np.stack([kd.eye_a, kd.eye_b, kd.nose_tip, kd.mouth_l, kd.mouth_r, kd.chin])
    x0, y0 = kpd_pts.min(axis=0).astype(int)
    x1, y1 = np.ceil(kpd_pts.max(axis=0)).astype(int)
    donor_face[max(0, y0):y1, max(0, x0):x1] = True
    sd = estimate_sigma(donor.rgb, donor_face)
    extra = float(np.sqrt(max(0.0, sb * sb - sd * sd)))
    if extra > 5e-4:
        gen = rng if rng is not None else np.random.default_rng()
        content = content + gen.normal(0.0, extra, content.shape).astype(np.float32) * weight_region[..., None]

    weight_region = weight_region.astype(np.float32)
    image = base.rgb.astype(np.float32).copy()
    image[region] = (
        content * weight_region[..., None] + image[region] * (1.0 - weight_region[..., None])
    )

    full_weight = np.zeros(base.shape, np.float32)
    full_weight[region] = weight_region
    full_region = np.zeros(base.shape, bool)
    full_region[region] = True

    return TransplantDraft(
        image=np.clip(image, 0.0, 1.0),
        weight=full_weight,
        region=full_region,
        content=content,
        donor_sigma=sd,
        transform=m,
    )
