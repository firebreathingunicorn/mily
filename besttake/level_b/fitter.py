"""3D face fitting: pose + identity/expression from 2D landmarks (+ depth).

Fits the parametric head model to one frame's landmarks with Levenberg-
Marquardt over pose (rotation, scale/translation — or full pinhole
translation) and linear-in-mesh expression/identity coefficients. Yaw
multi-start handles the weak-perspective left/right ambiguity; optional
metric depth anchors scale and out-of-plane pose ("3D face fitting with
depth" — the Phase 3 front-end).
"""
from __future__ import annotations

import numpy as np

from ..common.types import Fit, Intrinsics
from ..common.geometry import rodrigues
from ..face_model.canonical import FaceModel


class FaceFitter:
    def __init__(self, model: FaceModel, mode: str = "weak",
                 yaw_starts=(-40.0, -20.0, 0.0, 20.0, 40.0),
                 depth_weight: float = 1.0, coef_ridge: float = 1e-3,
                 z_inlier: float = 0.06, hint_rmse_fallback: float = 2.5):
        self.model = model
        self.mode = mode
        self._proj = mode   # active projection during staged fitting
        self.yaw_starts = tuple(yaw_starts)
        self.hint_rmse_fallback = hint_rmse_fallback
        self.depth_weight = depth_weight
        self.coef_ridge = coef_ridge
        self.z_inlier = z_inlier
        self.z_nominal = 3.0      # fixed camera distance for 'weakrefine'
        self._fx = None           # active focal (free parameter in weakrefine)
        lmk = model.landmark_idx
        self.L0 = model.neutral[lmk]
        self.Did_l = model.D_identity[:, lmk, :]
        self.Dex_l = model.D_expression[:, lmk, :]
        # depth sample points: front-facing vertices of the neutral shape
        n = model.normals_can()
        front = np.nonzero(n[:, 2] < -0.15)[0]
        self.depth_samples = front[:: max(1, len(front) // 250)]
        self.depth_normals = n[self.depth_samples]

    # ---------------------------------------------------------------- shapes
    def landmarks3d(self, c_id: np.ndarray, c_ex: np.ndarray) -> np.ndarray:
        """(M, 3) canonical landmark positions for coefficients."""
        L = self.L0
        if c_id is not None:
            L = L + np.tensordot(c_id, self.Did_l, (0, 0))
        if c_ex is not None:
            L = L + np.tensordot(c_ex, self.Dex_l, (0, 0))
        return L

    def fit(self, landmarks: np.ndarray, depth=None, intrinsics=None,
            c_identity: np.ndarray | None = None, fix_identity: bool = False,
            yaw_hint_deg: float | None = None) -> Fit:
        """Fit one face. `landmarks` (M, 2) in model landmark order.

        Stages: multi-start landmark LM (yaw starts handle the ambiguity) →
        metric depth anchors scale and out-of-plane pose (occlusion-gated).
        mode 'weak' with intrinsics runs one more stage: the weak-perspective
        solution is converted to pinhole with the given (or assumed-FOV)
        intrinsics and refined — this removes the systematic ~10° yaw
        over-rotation of pure weak perspective on turned heads. The returned
        fit is then pinhole; 'weak' without intrinsics stays legacy.
        c_identity: initial/known identity; fix_identity freezes it.
        yaw_hint_deg: coarse prior (e.g. from the tracker). The fit then runs
        from the nearest yaw start only — ~5× fewer stage-A solves — and
        falls back to the full multi-start if the hinted fit misses the
        rmse bar.
        """
        obs = np.asarray(landmarks, float)
        self._proj = self.mode
        self._fx = None
        starts = self.yaw_starts
        if yaw_hint_deg is not None:
            starts = (min(self.yaw_starts, key=lambda s0: abs(s0 - yaw_hint_deg)),)
        best_x, best_cost, best_free = self._multi_start(
            obs, intrinsics, starts, c_identity, fix_identity)
        if best_x is None:
            raise ValueError("landmark fit failed: non-finite observations or "
                             "degenerate configuration")
        if yaw_hint_deg is not None and len(starts) < len(self.yaw_starts):
            if self._lm_rmse(best_x, obs, intrinsics, best_free) > self.hint_rmse_fallback:
                best_x, best_cost, best_free = self._multi_start(
                    obs, intrinsics, self.yaw_starts, c_identity, fix_identity)
        if self._proj == "weak" and intrinsics is not None:
            w, s, t, T, c_id, c_ex = self._unpack(best_x, best_free)
            self._proj = "weakrefine"
            self._fx = intrinsics.fx  # starting point; refined from foreshortening
            parts = [w, np.array([np.log(intrinsics.fx)]), t]
            if best_free:
                parts.append(c_id)
            parts.append(c_ex)
            best_x = np.concatenate([np.atleast_1d(np.asarray(p, float)) for p in parts])
            best_x, best_cost = self._run_lm(best_x, obs, None, intrinsics,
                                             best_free, iters=60)
        if depth is not None and self.depth_weight > 0:
            for _ in range(3):
                gate = self._depth_gate(best_x, depth, intrinsics, best_free)
                best_x, best_cost = self._run_lm(best_x, obs, depth, intrinsics,
                                                 best_free, iters=10, depth_gate=gate)
        w, s, t, T, c_id, c_ex = self._unpack(best_x, best_free)
        res = self._residual(best_x, obs, None, intrinsics, best_free, 1.0,
                             with_penalty=False)
        rmse = float(np.sqrt(np.mean(np.sum(res.reshape(-1, 2) ** 2, 1))))
        out_mode, out_intr = self._proj, intrinsics
        if self._proj == "weakrefine":
            # the estimated focal is part of the calibration now: downstream
            # render/re-projection must use it to stay consistent
            out_mode = "pinhole"
            out_intr = Intrinsics(fx=self._fx, fy=self._fx,
                                  cx=intrinsics.cx, cy=intrinsics.cy)
        return Fit(R=rodrigues(w), s=s, t=None if t is None else np.asarray(t, float),
                   T=None if T is None else np.asarray(T, float), mode=out_mode,
                   intrinsics=out_intr if out_mode == "pinhole" else None,
                   c_identity=np.asarray(c_id, float), c_expression=np.asarray(c_ex, float),
                   landmark_rmse=rmse)

    def _multi_start(self, obs, intrinsics, starts, c_identity, fix_identity):
        best_x, best_cost, best_free = None, np.inf, None
        for yaw0 in starts:
            x0, free_id = self._init_x(obs, intrinsics, yaw0, c_identity, fix_identity)
            x, cost = self._run_lm(x0, obs, None, intrinsics, free_id, iters=40)
            if cost < best_cost:
                best_x, best_cost, best_free = x, cost, free_id
        return best_x, best_cost, best_free

    def _lm_rmse(self, x, obs, intrinsics, free_id):
        res = self._residual(x, obs, None, intrinsics, free_id, 1.0,
                             with_penalty=False)
        return float(np.sqrt(np.mean(np.sum(res.reshape(-1, 2) ** 2, 1))))

    def _init_x(self, obs, intrinsics, yaw0, c_identity, fix_identity):
        free_id = not fix_identity
        self._c_id_fixed = np.zeros(self.model.num_id) if c_identity is None \
            else np.asarray(c_identity, float)
        n_id = self.model.num_id if free_id else 0
        Lc = self.landmarks3d(self._c_id_fixed, np.zeros(self.model.num_expr))
        t0 = obs.mean(0) - Lc[:, :2].mean(0)
        spread = np.linalg.norm(obs - obs.mean(0)).mean() / max(
            np.linalg.norm(Lc[:, :2] - Lc[:, :2].mean(0)).mean(), 1e-9)
        w0 = np.array([0.0, np.radians(yaw0), 0.0])
        if self._proj == "weak":
            x = np.concatenate([w0, [np.log(spread)], t0,
                                np.zeros(n_id + self.model.num_expr)])
        else:
            assert intrinsics is not None, "pinhole mode needs intrinsics"
            Z0 = 3.0
            T0 = np.array([(obs.mean(0)[0] - intrinsics.cx) * Z0 / intrinsics.fx,
                           (obs.mean(0)[1] - intrinsics.cy) * Z0 / intrinsics.fy,
                           Z0])
            x = np.concatenate([w0, T0, np.zeros(n_id + self.model.num_expr)])
        return x, free_id

    @staticmethod
    def _huber_weights(r: np.ndarray, n_lm: int, n_depth: int,
                       k_lm: float = 2.0, k_depth: float = 4.0) -> np.ndarray:
        """IRLS weights: landmark gross outliers (bad detections) and depth
        gate-edge samples get down-weighted; regularization rows weight 1."""
        w = np.ones_like(r)
        a = np.abs(r[:n_lm])
        w[:n_lm] = np.where(a <= k_lm, 1.0, k_lm / np.maximum(a, 1e-9))
        if n_depth:
            a = np.abs(r[n_lm:n_lm + n_depth])
            w[n_lm:n_lm + n_depth] = np.where(a <= k_depth, 1.0,
                                              k_depth / np.maximum(a, 1e-9))
        return w

    def _run_lm(self, x, obs, depth, intrinsics, free_id, iters, depth_gate=None):
        px_scale = 1.0  # reserved for weak-mode depth weighting
        n_lm = 2 * obs.shape[0]
        n_depth = len(self.depth_samples) if depth is not None else 0
        r = self._residual(x, obs, depth, intrinsics, free_id, px_scale,
                           depth_gate=depth_gate)
        rw = r * 1.0
        cost = float(rw @ rw)
        lam = 1e-3
        for _ in range(iters):
            w = self._huber_weights(r, n_lm, n_depth)
            sw = np.sqrt(w)
            J = self._numeric_jac(x, obs, depth, intrinsics, free_id, px_scale,
                                  depth_gate=depth_gate) * sw[:, None]
            rw = r * sw
            JTJ = J.T @ J
            JTr = J.T @ rw
            cost = float(rw @ rw)
            step_ok = False
            for _ in range(6):
                dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-9), -JTr)
                x_new = x + dx
                r_new = self._residual(x_new, obs, depth, intrinsics, free_id,
                                       px_scale, depth_gate=depth_gate)
                rw_new = r_new * np.sqrt(self._huber_weights(r_new, n_lm, n_depth))
                cost_new = float(rw_new @ rw_new)
                if np.isfinite(cost_new) and cost_new < cost:
                    x, r, cost = x_new, r_new, cost_new
                    lam = max(lam * 0.5, 1e-9)
                    step_ok = True
                    break
                lam *= 5.0
            if not step_ok or cost < 1e-10:
                break
        return x, cost

    # -------------------------------------------------------------- LM core
    def _unpack(self, x, free_id):
        i = 0
        w = x[i:i + 3]; i += 3
        if self._proj == "weak":
            s = float(np.exp(x[i])); i += 1
            t = x[i:i + 2]; i += 2
            T = None
        elif self._proj == "weakrefine":
            # assumed-FOV fallback: focal is free, camera distance is fixed —
            # perspective foreshortening makes the focal observable, and with
            # Z fixed the (f, Z) gauge degeneracy disappears
            self._fx = float(np.exp(x[i])); i += 1
            t = x[i:i + 2]; i += 2
            s = 1.0
            T = np.array([t[0], t[1], self.z_nominal])
        else:
            s, t = 1.0, None
            T = x[i:i + 3]; i += 3
        n_id = self.model.num_id if free_id else 0
        c_id = x[i:i + n_id] if free_id else self._c_id_fixed
        i += n_id
        c_ex = x[i:i + self.model.num_expr]
        return w, s, t, T, c_id, c_ex

    def _fx_eff(self, intrinsics):
        return self._fx if self._fx is not None else intrinsics.fx

    def _depth_gate(self, x, depth, intrinsics, free_id):
        """Inlier mask for depth samples: must face the camera in the current
        pose and agree with the observed depth within tolerance (kills
        occluded samples that would otherwise drag the pose). Recomputed
        between LM rounds, frozen within a round for Jacobian stability."""
        w, s, t, T, c_id, c_ex = self._unpack(x, free_id)
        R = rodrigues(w)
        n = (R @ self.depth_normals.T).T
        facing = -n[:, 2]
        z_pred, pix = self._depth_pred(x, intrinsics, free_id)
        z_obs = self._sample_depth(depth, pix)
        return np.isfinite(z_obs) & (facing > 0.10) & \
            (np.abs(z_pred - z_obs) < self.z_inlier)

    def _depth_pred(self, x, intrinsics, free_id):
        """Depths and pixels of depth-sample points under current params."""
        w, s, t, T, c_id, c_ex = self._unpack(x, free_id)
        R = rodrigues(w)
        S = self.model.verts(c_id, c_ex)[self.depth_samples]
        Sc = (R @ S.T).T
        if self._proj == "weak":
            return s * Sc[:, 2], s * Sc[:, :2] + t
        Sc = Sc + T
        fx = self._fx_eff(intrinsics)
        pix = np.stack([fx * Sc[:, 0] / Sc[:, 2] + intrinsics.cx,
                        fx * Sc[:, 1] / Sc[:, 2] + intrinsics.cy], 1)
        return Sc[:, 2], pix

    def _residual(self, x, obs, depth, intrinsics, free_id, px_scale,
                  with_penalty: bool = True, depth_gate=None):
        w, s, t, T, c_id, c_ex = self._unpack(x, free_id)
        R = rodrigues(w)
        L = self.landmarks3d(c_id, c_ex)
        Xc = (R @ L.T).T
        if self._proj == "weak":
            pred = s * Xc[:, :2] + t
            res = (pred - obs).ravel()
        else:
            Xc = Xc + T
            fx = self._fx_eff(intrinsics)
            pred = np.stack([fx * Xc[:, 0] / Xc[:, 2] + intrinsics.cx,
                             fx * Xc[:, 1] / Xc[:, 2] + intrinsics.cy], 1)
            res = (pred - obs).ravel()
        if depth is not None and self.depth_weight > 0:
            S = self.model.verts(c_id, c_ex)[self.depth_samples]
            Sc = (R @ S.T).T
            if self._proj == "weak":
                z_pred = s * Sc[:, 2]
                pix = s * Sc[:, :2] + t
                wgt = px_scale
            else:
                Sc = Sc + T
                z_pred = Sc[:, 2]
                pix = np.stack([intrinsics.fx * Sc[:, 0] / Sc[:, 2] + intrinsics.cx,
                                intrinsics.fy * Sc[:, 1] / Sc[:, 2] + intrinsics.cy], 1)
                # pixel-equivalent weighting: 1 unit of Z error ~ fx/Z px of parallax
                wgt = intrinsics.fx / max(float(np.mean(z_pred)), 1e-6)
            z_obs = self._sample_depth(depth, pix)
            # fixed-length residual: gated or off-silhouette samples
            # contribute zero, so the numeric Jacobian keeps a stable shape
            bad = ~np.isfinite(z_obs)
            if depth_gate is not None:
                bad = bad | ~depth_gate
            res_d = np.where(bad, 0.0,
                             self.depth_weight * wgt * (z_pred - np.where(bad, 0.0, z_obs)))
            res = np.concatenate([res, res_d])
        # Tikhonov on free coefficients: bounds null-direction drift
        if with_penalty:
            pen = np.sqrt(self.coef_ridge) * np.concatenate([
                (c_id if free_id else np.zeros(0)), c_ex])
            res = np.concatenate([res, pen])
        return res

    @staticmethod
    def _sample_depth(depth, pix):
        H, W = depth.shape
        x = pix[:, 0]
        y = pix[:, 1]
        x0 = np.clip(np.floor(x).astype(int), 0, W - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
        fx = np.clip(x - x0, 0, 1)
        fy = np.clip(y - y0, 0, 1)
        d = (depth[y0, x0] * (1 - fx) * (1 - fy) + depth[y0, x0 + 1] * fx * (1 - fy) +
             depth[y0 + 1, x0] * (1 - fx) * fy + depth[y0 + 1, x0 + 1] * fx * fy)
        return d

    def _numeric_jac(self, x, obs, depth, intrinsics, free_id, px_scale,
                     depth_gate=None):
        r0 = self._residual(x, obs, depth, intrinsics, free_id, px_scale,
                            depth_gate=depth_gate)
        J = np.zeros((len(r0), len(x)))
        for i in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[i]))
            xp = x.copy(); xp[i] += h
            J[:, i] = (self._residual(xp, obs, depth, intrinsics, free_id,
                                      px_scale, depth_gate=depth_gate) - r0) / h
        return J
