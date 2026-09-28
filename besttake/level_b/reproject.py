"""Level B synthesis: 3D-aware re-projection of a donor face onto the base
frame's head pose, with occlusion-aware visibility and gap filling from
other frames of the same person.

Correspondence model: base and donor fits share identity coefficients (same
person), so a base surface point's anatomical location in the donor frame is
its canonical position minus the base expression displacement plus the donor
expression displacement, projected with the donor fit. Visibility requires
(a) the surface normal to face the donor camera and (b) the predicted depth
to match the donor's z-buffer (nothing occluding it in that frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from ..common.rendering import Render, interp_attr
from ..common.types import Fit

_EPS = 1e-6


@dataclass
class WarpedDonor:
    rgb: np.ndarray            # (H, W, 3) sampled donor appearance
    weight: np.ndarray         # (H, W) confidence in [0, 1]
    valid: np.ndarray          # (H, W) bool
    donor_index: int


def fit_scale_depth_tol(fit: Fit, tol_canonical: float) -> float:
    """Convert a tolerance in canonical units to the fit's depth units."""
    return tol_canonical * (fit.s if fit.mode == "weak" else 1.0)


def reproject(base_render: Render, base_disp: np.ndarray, donor_disp: np.ndarray,
              donor_fit: Fit, donor_rgb: np.ndarray, donor_zbuf: np.ndarray,
              base_fit: Fit, donor_index: int = 0,
              size: tuple[int, int] | None = None,
              z_tol: float = 0.03, facing_min: float = 0.04,
              border_feather: float = 3.0,
              donor_depth: np.ndarray | None = None) -> WarpedDonor:
    """Warp donor appearance onto the base surface.

    base_disp / donor_disp: per-vertex expression displacements (N, 3) of the
    base/donor shapes relative to the shared identity shape.
    donor_depth: the donor's captured depth map (LiDAR). Where it is much
    closer than the predicted surface, something real is in front of the
    face in that frame (hand, drink) and the sample is invalid — the model's
    own z-buffer cannot know about foreground objects.
    """
    H, W = base_render.mask.shape
    if size is None:
        size = (H, W)
    dh, dw = donor_rgb.shape[:2]

    # anatomical correspondence: same surface point, donor's expression
    ddisp = interp_attr(base_render, (donor_disp - base_disp).astype(np.float32))
    P = base_render.canonical + ddisp

    pix = donor_fit.project(P)
    z = donor_fit.depth(P)

    # donor-facing confidence: rotate base camera normals into donor camera
    # space (canonical = base_fit.R^T @ cam, then donor_fit.R)
    R_rel = (donor_fit.R @ base_fit.R.T).astype(np.float64)
    n_donor = (R_rel @ base_render.cam_normal.reshape(-1, 3).T).T \
        .reshape(base_render.cam_normal.shape)
    facing_donor = np.clip(-n_donor[..., 2] / (np.linalg.norm(n_donor, axis=-1) + _EPS),
                           0.0, 1.0)

    x, y = pix[..., 0], pix[..., 1]
    inside = (x >= 0) & (x <= dw - 1) & (y >= 0) & (y <= dh - 1)
    xi = np.clip(np.floor(x).astype(int), 0, dw - 1)
    yi = np.clip(np.floor(y).astype(int), 0, dh - 1)
    zbuf_at = donor_zbuf[yi, xi]
    tol = fit_scale_depth_tol(donor_fit, z_tol)
    z_ok = inside & (z <= zbuf_at + tol) & (z >= zbuf_at - 3.0 * tol)
    if donor_depth is not None:
        z_cap = donor_depth[yi, xi]
        z_ok &= ~np.isfinite(z_cap) | (z_cap >= z - 2.0 * tol)

    facing_base = base_render.facing
    # plateau-then-ramp: full confidence while a surface point is reasonably
    # front-facing; fade only in the last sliver before the silhouette
    def ramp(x, lo, hi):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    valid = inside & z_ok & (facing_donor > facing_min) & (facing_base > facing_min)

    # bicubic sample of donor color (sub-pixel warp keeps more detail than
    # bilinear; clip guards the spline's overshoot at edges)
    coords = np.stack([np.clip(y, 0, dh - 1), np.clip(x, 0, dw - 1)])
    rgb = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        rgb[..., c] = np.clip(map_coordinates(donor_rgb[..., c].astype(np.float32),
                                              coords, order=3, mode="nearest"),
                              0.0, 1.0)

    # confidence: how front-facing the point is in both views, feathered at
    # the donor frame border so samples never touch the edge hard
    feather = np.ones((H, W), np.float32)
    if border_feather > 0:
        dist_in = np.minimum(np.minimum(x, dw - 1 - x), np.minimum(y, dh - 1 - y))
        feather = np.clip(dist_in / border_feather, 0.0, 1.0)
    weight = (ramp(facing_base, facing_min, 3.0 * facing_min + 0.06) *
              ramp(facing_donor, facing_min, 3.0 * facing_min + 0.14) ** 0.75) * feather
    weight[~valid] = 0.0
    return WarpedDonor(rgb=rgb.astype(np.float32), weight=weight.astype(np.float32),
                       valid=valid, donor_index=donor_index)


def blend_sources(warped: list[WarpedDonor], primary_index: int,
                  feather_sigma: float = 1.5
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Combine donor warpings: per-pixel confidence-weighted average with a
    Gaussian cross-fade between sources. Returns (rgb, weight, source_map,
    fill_fraction); pixels only non-primary donors cover count as filled."""
    if not warped:
        raise ValueError("no donors")
    H, W = warped[0].weight.shape
    share = np.zeros((len(warped), H, W), np.float32)
    rgb_acc = np.zeros((H, W, 3), np.float32)
    for i, w in enumerate(warped):
        share[i] = gaussian_filter(w.weight * w.valid, feather_sigma)
        rgb_acc += share[i][..., None] * w.rgb
    total = share.sum(0)
    src = share.argmax(0)
    has = total > 1e-4
    out_rgb = np.zeros((H, W, 3), np.float32)
    out_rgb[has] = rgb_acc[has] / total[has][:, None]
    out_w = np.clip(total, 0.0, 1.0)

    swapped = has & (out_w > 0.02)
    fill_fraction = float(np.mean((src != primary_index) & swapped)) if swapped.any() else 0.0
    source_map = np.where(has, src, -1).astype(np.int32)
    return out_rgb, out_w, source_map, fill_fraction


def fallback_feather_alpha(region: np.ndarray, feather_px: float = 5.0) -> np.ndarray:
    """Matting fallback (until Phase 1 matting plugs in): soft alpha ramping
    from the region boundary inward — never a hard cutout."""
    inside = distance_transform_edt(region)
    alpha = np.clip(inside / max(feather_px, 1e-6), 0.0, 1.0)
    return alpha.astype(np.float32)
