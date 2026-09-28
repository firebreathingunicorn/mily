"""Level B pipeline: one person, one base frame, N donor frames.

Inputs come from capture + scene understanding (Phase 1 contracts); Level B
fits the 3D face model, re-projects donor appearance onto the base pose,
fills what the primary donor cannot see from other frames, harmonizes, and
composites. Everything shown is real donor pixels unless a region is
unfillable — nothing here is generated (that is Level C's job).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common.contracts import (ArtifactChecker, IdentityChecker,
                                MattingProvider, NoiseEstimator)
from ..common.rendering import render
from ..common.types import FaceObservation, Frame, Fit, Intrinsics, SwapResult
from ..face_model.canonical import FaceModel
from .fitter import FaceFitter
from .harmonize import (color_transfer, estimate_noise_sigma, grain_match,
                        shading_transfer)
from .reproject import blend_sources, fallback_feather_alpha, reproject


@dataclass
class LevelBConfig:
    mode: str = "pinhole"         # 'pinhole' (uses depth) | 'weak' (fallback)
    assumed_fov_factor: float = 0.75  # f = factor * width when intrinsics unknown
    z_tol: float = 0.03           # occlusion test tolerance, canonical units
    facing_min: float = 0.04
    color_strength: float = 0.8
    shading_strength: float = 0.8
    feather_px: float = 5.0
    depth_weight: float = 1.0
    coef_ridge: float = 10.0
    use_yaw_hint: bool = True      # order/frontality proxy seeds the fit start
    max_donor_rmse: float = 10.0   # drop donors whose fit missed badly
    identity_from: str = "primary_donor"  # person library later


@dataclass
class DonorInput:
    frame: Frame
    obs: FaceObservation


class _PassChecker:
    def check(self, *a, **k) -> bool:
        return True


class LevelBSwap:
    """Phase 3 synthesis stage. Optional Phase 1 providers plug in here.

    By default the verification gates are the prototype checkers in
    adapters.simple_checks (seam detector + palette identity); a failed gate
    leaves the person unchanged — never a detectable composite.
    """

    def __init__(self, model: FaceModel, config: LevelBConfig | None = None,
                 matting: MattingProvider | None = None,
                 noise: NoiseEstimator | None = None,
                 artifact_checker: ArtifactChecker | None = None,
                 identity_checker: IdentityChecker | None = None,
                 seed: int = 0):
        if artifact_checker is None:
            from ..adapters.simple_checks import ArtifactCheckerProto
            artifact_checker = ArtifactCheckerProto()
        if identity_checker is None:
            from ..adapters.simple_checks import PaletteIdentityChecker
            identity_checker = PaletteIdentityChecker()
        self.model = model
        self.cfg = config or LevelBConfig()
        self.matting = matting
        self.noise = noise
        self.artifact_checker = artifact_checker
        self.identity_checker = identity_checker
        self.fitter = FaceFitter(model, mode=self.cfg.mode,
                                 depth_weight=self.cfg.depth_weight,
                                 coef_ridge=self.cfg.coef_ridge)
        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------------- fitting
    def _yaw_proxy_deg(self, obs: FaceObservation) -> float:
        """Signed, fit-free yaw estimate from landmarks (nose-tip offset from
        the inner-canthi midpoint). Coarse — used to order donors and to hint
        the fitter's yaw start; the fitter's rmse-fallback covers misses."""
        lm = obs.landmarks
        nt = self.model.nose_tip
        i1, i2 = self.model.inner_canthi
        proxy = float(lm[nt, 0] - 0.5 * (lm[i1, 0] + lm[i2, 0]))
        return -2.15 * proxy

    def _frontality(self, obs: FaceObservation) -> float:
        """Horizontal nose-tip offset from the inner canthi midpoint — a
        fit-free yaw proxy used only to order donors."""
        return abs(self._yaw_proxy_deg(obs))

    def _fit_frame(self, frame: Frame, obs: FaceObservation,
                   c_identity: np.ndarray | None, fix_identity: bool) -> Fit:
        if obs.fit is not None and obs.fit.c_identity is not None:
            return obs.fit
        H, W = frame.shape
        intrinsics = frame.intrinsics
        if intrinsics is None:
            # on-device this always comes from capture; fallback assumes a
            # typical phone main-camera field of view. Used by pinhole mode
            # and by the weak mode's assumed-FOV refinement stage.
            intrinsics = Intrinsics(fx=self.cfg.assumed_fov_factor * W,
                                    fy=self.cfg.assumed_fov_factor * W,
                                    cx=(W - 1) / 2, cy=(H - 1) / 2)
        hint = self._yaw_proxy_deg(obs) if self.cfg.use_yaw_hint else None
        return self.fitter.fit(obs.landmarks, depth=frame.depth,
                               intrinsics=intrinsics, c_identity=c_identity,
                               fix_identity=fix_identity, yaw_hint_deg=hint)

    @staticmethod
    def _fit_usable(fit: Fit | None) -> bool:
        """Guardrail against failed detections: a fit with NaN landmarks or a
        degenerate pose must never reach rendering or the identity median."""
        if fit is None or not np.isfinite(fit.landmark_rmse):
            return False
        if not (np.all(np.isfinite(fit.R)) and np.isfinite(fit.c_identity).all()
                and np.isfinite(fit.c_expression).all()):
            return False
        if fit.mode == "weak":
            return np.isfinite(fit.s) and fit.s > 0 and np.all(np.isfinite(fit.t))
        return fit.T is not None and np.all(np.isfinite(fit.T))

    def _expression_disp(self, c_ex: np.ndarray) -> np.ndarray:
        return np.tensordot(np.asarray(c_ex, float),
                            self.model.D_expression, (0, 0))

    def _face_region(self, r) -> np.ndarray:
        """Evaluated face area of a render: real face zone, hair excluded."""
        return (~self.model.hair_mask(r.uv.reshape(-1, 2)).reshape(r.mask.shape)) & \
            (r.uv[..., 1] > 0.28) & (r.uv[..., 1] < 0.82) & r.mask & \
            (r.facing > 0.05)

    @staticmethod
    def _sector_coverage(face_region: np.ndarray, weight: np.ndarray
                         ) -> np.ndarray:
        """3x3 coverage of the face bounding box — the capture-guidance hook
        (plan §Smart shutter): a sector with face area but ~0 swap coverage
        means the burst has no frame that sees it yet."""
        ys, xs = np.nonzero(face_region)
        out = np.zeros((3, 3), np.float32)
        if len(ys) == 0:
            return out
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        iy = np.clip((ys - y0) * 3 // max(y1 - y0, 1), 0, 2)
        ix = np.clip((xs - x0) * 3 // max(x1 - x0, 1), 0, 2)
        swapped = (weight[ys, xs] > 0.1).astype(np.float64)
        area = np.zeros(9); done = np.zeros(9)
        np.add.at(area, iy * 3 + ix, 1.0)
        np.add.at(done, iy * 3 + ix, swapped)
        ok = area > 0
        out.ravel()[ok] = (done[ok] / area[ok]).astype(np.float32)
        return out

    # ------------------------------------------------------------------- run
    def run(self, base: Frame, base_obs: FaceObservation,
            donors: list[DonorInput], reference_images: list[np.ndarray] | None = None
            ) -> SwapResult:
        assert donors, "Level B needs at least one donor frame"
        H, W = base.shape

        # most-frontal donor becomes primary (identity + main appearance)
        donors = sorted(donors, key=lambda d: self._frontality(d.obs))

        # burst-wide identity (plan §4C): fit identity independently on every
        # donor and take the coefficient-wise median — one frame's noise no
        # longer defines the person. Expressions are then refit against the
        # shared identity. Failed detections (NaN landmarks, degenerate fits)
        # are dropped here instead of poisoning the median or the render.
        id_fits = []
        usable = []
        for d in donors:
            try:
                f = self._fit_frame(d.frame, d.obs, None, fix_identity=False)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if self._fit_usable(f) and f.landmark_rmse <= self.cfg.max_donor_rmse:
                id_fits.append(f)
                usable.append(d)
        if not usable:
            raise ValueError("no donor frame produced a usable fit "
                             "(check landmark source / detection quality)")
        donors = usable
        c_id = np.median(np.stack([f.c_identity for f in id_fits]), axis=0)
        fits_d = [self._fit_frame(d.frame, d.obs, c_id, fix_identity=True)
                  for d in donors]
        fit_b = self._fit_frame(base, base_obs, c_id, fix_identity=True)
        if not self._fit_usable(fit_b):
            raise ValueError("base frame fit is not usable "
                             "(check landmark source / detection quality)")

        # base surface with its own expression
        c_ex_b = fit_b.c_expression
        verts_b = self.model.verts(c_id, c_ex_b)
        base_render = render(fit_b, verts_b, self.model.faces,
                             self.model.normals_can(c_id, c_ex_b),
                             self.model.albedo(c_id, c_ex_b),
                             np.stack([self.model.uu, self.model.vv], 1), (H, W))
        disp_b = self._expression_disp(c_ex_b)

        warped = []
        donor_regions = []
        for i, (donor, fit_d) in enumerate(zip(donors, fits_d)):
            c_ex_d = fit_d.c_expression
            verts_d = self.model.verts(c_id, c_ex_d)
            d_render = render(fit_d, verts_d, self.model.faces,
                              self.model.normals_can(c_id, c_ex_d),
                              self.model.albedo(c_id, c_ex_d),
                              np.stack([self.model.uu, self.model.vv], 1),
                              donor.frame.shape)
            donor_regions.append(self._face_region(d_render))
            warped.append(reproject(
                base_render, disp_b, self._expression_disp(c_ex_d), fit_d,
                donor.frame.rgb, d_render.zbuf, base_fit=fit_b,
                donor_index=i, z_tol=self.cfg.z_tol,
                facing_min=self.cfg.facing_min,
                donor_depth=donor.frame.depth))

        pasted, weight, source_map, fill_frac = blend_sources(warped, primary_index=0)

        # face region for checks/metrics: real face area, hair excluded
        face_region = self._face_region(base_render)

        # finishing: match color, shading and grain to the base frame
        pasted, color_params = color_transfer(pasted, base.rgb, weight,
                                              strength=self.cfg.color_strength)
        pasted, gain = shading_transfer(pasted, base.rgb, weight,
                                        strength=self.cfg.shading_strength)
        sigma_base = self.noise.sigma(base) if self.noise else estimate_noise_sigma(base.rgb)
        sigma_donor = self.noise.sigma(donors[0].frame) if self.noise else \
            estimate_noise_sigma(donors[0].frame.rgb)
        w_final = weight * (self.matting.alpha(base, base_obs) if self.matting
                            else fallback_feather_alpha(weight > 0.02,
                                                        self.cfg.feather_px))
        out = base.rgb * (1 - w_final[..., None]) + pasted * w_final[..., None]
        out = grain_match(out, w_final, sigma_base, sigma_donor, self.rng)
        out = np.clip(out, 0, 1).astype(np.float32)

        # verification gates (plan §Verification): a failed gate leaves the
        # person unchanged — never show a detectable composite
        if reference_images is None:
            reference_images = [(d.frame.rgb, reg) for d, reg in zip(donors, donor_regions)]
        checks = {
            "artifact": bool(self.artifact_checker.check(out, face_region,
                                                         weight=w_final,
                                                         references=[r[0] if isinstance(r, tuple) else r
                                                                     for r in reference_images])),
            "identity": bool(self.identity_checker.check(
                out, face_region, reference_images)),
        }
        if not all(checks.values()):
            return SwapResult(
                image=base.rgb.copy(), weight=np.zeros((H, W), np.float32),
                source_map=np.full((H, W), -1, np.int32),
                face_region=face_region, method="rejected",
                coverage=0.0, fill_fraction=0.0, checks=checks)

        swapped = face_region & (w_final > 0.1)
        method = "B" if fill_frac < 0.01 else "B+fill"
        return SwapResult(
            image=out, weight=w_final,
            source_map=source_map, face_region=face_region, method=method,
            # face-relative: fraction of the person's face area that carries
            # the swapped moment (frame-relative was the old, misleading form)
            coverage=float(np.mean(swapped[face_region])) if face_region.any() else 0.0,
            fill_fraction=fill_frac,
            coverage_sectors=self._sector_coverage(face_region, w_final),
            checks=checks,
        )
