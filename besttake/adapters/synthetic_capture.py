"""Synthetic capture: renders multi-frame group bursts with full ground truth.

Stands in for the consented capture dataset + rig ground truth (plan §Data):
known identity/expression/pose per frame, metric depth, projected landmark
observations with realistic pixel noise, and the ideal output image a correct
Level B swap should reproduce (base pose, donor expression).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.geometry import rodrigues
from ..common.rendering import render, shade
from ..common.types import Fit, Frame, Intrinsics
from ..face_model.canonical import SyntheticHeadModel

LIGHT = np.array([-0.4, -0.5, -0.75])
AMBIENT, KD = 0.35, 0.75
CAM_DIST = 3.0


@dataclass
class Shot:
    frame: Frame
    obs_landmarks: np.ndarray     # observed (noisy) 2D landmarks
    gt_fit: Fit                   # ground-truth pose fit
    gt_c_expression: np.ndarray


@dataclass
class Capture:
    base: Shot
    donors: list[Shot]
    person_id: str
    c_identity: np.ndarray
    ideal: np.ndarray             # ground truth of a perfect swap (base pose, donor expression)
    face_region_gt: np.ndarray    # evaluated face area in the ideal render


def make_person(rng: np.random.Generator) -> np.ndarray:
    """Random but plausible identity coefficients (|c| <= 0.5)."""
    return (rng.uniform(-1, 1, 10) * np.array([0.5, 0.5, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.5]))


def _pose(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    return rodrigues(np.array([np.radians(pitch_deg), np.radians(yaw_deg), 0.0]))


class SyntheticCapture:
    """Renders frames of one synthetic person with a pinhole camera."""

    def __init__(self, model: SyntheticHeadModel | None = None,
                 size: int = 320, seed: int = 0):
        self.model = model or SyntheticHeadModel()
        self.size = size
        self.f = 380.0
        self.intrinsics = Intrinsics(fx=self.f, fy=self.f,
                                     cx=size / 2 - 0.5, cy=size / 2 - 0.5)
        self.rng = np.random.default_rng(seed)

    def _fit(self, yaw, pitch, c_id, c_ex, center_offset=(0.0, 0.0)) -> Fit:
        T = np.array([center_offset[0] * CAM_DIST / self.f,
                      center_offset[1] * CAM_DIST / self.f, CAM_DIST])
        return Fit(R=_pose(yaw, pitch), T=T,
                   mode="pinhole", intrinsics=self.intrinsics,
                   c_identity=c_id, c_expression=c_ex)

    def shoot(self, c_id: np.ndarray, c_ex: np.ndarray, yaw: float,
              pitch: float = 0.0, exposure: float = 1.0,
              noise_sigma: float = 0.008, lm_noise: float = 0.4,
              with_depth: bool = True, person_id: str = "p0",
              center_offset: tuple[float, float] = (0.0, 0.0),
              occluder: tuple[float, float, float, float, float] | None = None) -> Shot:
        """occluder: (cx, cy, rx, ry, z) in pixels / camera units — a matte
        blob in front of the face (hand, drink). Inserted into BOTH rgb and
        depth so downstream occlusion tests behave as with a real sensor.
        Place it away from landmarks; the tracker stub observes through it.
        center_offset: framing offset of the head in pixels (off-center
        person; the fit's translation must absorb it).
        """
        m = self.model
        fit = self._fit(yaw, pitch, c_id, c_ex, center_offset)
        verts = m.verts(c_id, c_ex)
        r = render(fit, verts, m.faces, m.normals_can(c_id, c_ex),
                   m.albedo(c_id, c_ex), np.stack([m.uu, m.vv], 1),
                   (self.size, self.size))
        img = shade(r, LIGHT, AMBIENT, KD) * exposure
        img += self.rng.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1).astype(np.float32)

        # background fill
        bg = np.full_like(img, 0.02)
        bg += self.rng.normal(0, noise_sigma, bg.shape).astype(np.float32)
        img = np.where(r.mask[..., None], img, np.clip(bg, 0, 1))

        depth = np.where(r.mask, r.zbuf, np.nan).astype(np.float32) \
            if with_depth else None
        if occluder is not None:
            cx, cy, rx, ry, zo = occluder
            yy, xx = np.mgrid[0:self.size, 0:self.size]
            inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            img = np.where(inside[..., None], np.array([0.09, 0.07, 0.06], np.float32),
                           img)
            if depth is not None:
                depth = np.where(inside, np.minimum(np.nan_to_num(depth, nan=zo), zo),
                                 depth).astype(np.float32)
        lm_gt = fit.project(verts[m.landmark_idx])
        lm_obs = lm_gt + self.rng.normal(0, lm_noise, lm_gt.shape)

        frame = Frame(rgb=img, depth=depth, intrinsics=self.intrinsics,
                      name=f"{person_id}_y{yaw:g}")
        return Shot(frame=frame, obs_landmarks=lm_obs, gt_fit=fit,
                    gt_c_expression=c_ex)

    def make_capture(self, c_id: np.ndarray, base_yaw: float,
                     donor_yaws: list[float], base_c_ex: np.ndarray | None = None,
                     donor_c_ex: np.ndarray | None = None,
                     exposure_jitter: float = 0.05,
                     base_pitch: float = 0.0, donor_pitch: float = 0.0,
                     center_offset: tuple[float, float] = (0.0, 0.0),
                     **kw) -> Capture:
        """Base frame at (base_yaw, base_pitch); donors at donor_yaws. All
        frames share center_offset (off-center framing). The ideal output is
        the base pose rendered with the primary donor's expression."""
        base_c_ex = base_c_ex if base_c_ex is not None else np.zeros(self.model.num_expr)
        donor_c_ex = donor_c_ex if donor_c_ex is not None \
            else np.zeros(self.model.num_expr)
        base = self.shoot(c_id, base_c_ex, base_yaw, pitch=base_pitch,
                          center_offset=center_offset, person_id="base", **kw)
        donors = [self.shoot(c_id, donor_c_ex, y, pitch=donor_pitch,
                             center_offset=center_offset, person_id=f"d{i}",
                             exposure=1.0 + self.rng.uniform(-exposure_jitter, exposure_jitter),
                             **kw)
                  for i, y in enumerate(donor_yaws)]
        ideal_ex = donors[0].gt_c_expression
        ideal_fit = self._fit(base_yaw, base_pitch, c_id, ideal_ex, center_offset)
        m = self.model
        verts = m.verts(c_id, ideal_ex)
        r = render(ideal_fit, verts, m.faces, m.normals_can(c_id, ideal_ex),
                   m.albedo(c_id, ideal_ex), np.stack([m.uu, m.vv], 1),
                   (self.size, self.size))
        ideal = np.clip(shade(r, LIGHT, AMBIENT, KD), 0, 1).astype(np.float32)
        face_region = (~m.hair_mask(r.uv.reshape(-1, 2)).reshape(self.size, -1)) & \
            (r.uv[..., 1] > 0.28) & (r.uv[..., 1] < 0.82) & r.mask & (r.facing > 0.05)
        return Capture(base=base, donors=donors, person_id="p0",
                       c_identity=c_id, ideal=ideal, face_region_gt=face_region)
