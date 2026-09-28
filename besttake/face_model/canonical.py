"""SyntheticHeadModel: a parametric 3D face model (3DMM stand-in).

Phase 3 needs a deformable, textured, landmark-able head with linear
identity and expression modes to develop and evaluate fitting, re-projection
and gap filling offline. This is a procedurally authored stand-in for a
FLAME-class model; `FaceModel` below is the interface a licensed production
model would implement. Everything downstream (fitter, re-projection, fill)
depends only on the interface.

Canonical space: +x right, +y down, +z INTO the head (front of the face is
the -z side; the nose tip has the smallest z). Surface coordinates:
u = azimuth / 2pi (u=0.75 is the face center), v = latitude / pi (v=0 top).
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..common.geometry import vertex_normals

FRONT_U = 0.75


def _smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _gauss(du, dv, su, sv):
    return np.exp(-0.5 * ((du / su) ** 2 + (dv / sv) ** 2))


class FaceModel:
    """Interface used by the fitter / re-projection / pipeline."""

    num_id: int
    num_expr: int

    def verts(self, c_identity=None, c_expr=None) -> np.ndarray: ...
    def albedo(self, c_identity=None, c_expr=None) -> np.ndarray: ...
    def normals_can(self, c_identity=None, c_expr=None) -> np.ndarray: ...


class SyntheticHeadModel(FaceModel):
    """Procedural head: neutral mesh + linear identity and expression modes."""

    def __init__(self, nu: int = 96, nv: int = 96, seed: int = 7):
        self.nu, self.nv = nu, nv
        u = (np.arange(nu) + 0.5) / nu
        v = (np.arange(nv) + 0.5) / nv
        self.u, self.v = u, v
        uu, vv = np.meshgrid(u, v)               # (nv, nu)
        self.uu, self.vv = uu.ravel(), vv.ravel()
        self.num_id = 10
        self.num_expr = 6

        self._build_neutral()
        self._build_faces()
        self._build_identity_modes()
        self._build_expression_modes()
        self._landmarks()
        self._hair_lut()

    # ------------------------------------------------------------------ mesh
    def _build_neutral(self):
        uu, vv = self.uu, self.vv
        az = uu * 2 * np.pi
        lat = vv * np.pi
        dx, dy, dz = np.sin(lat) * np.cos(az), -np.cos(lat), np.sin(lat) * np.sin(az)
        front = np.clip(-dz, 0.0, 1.0)           # 1 at face center direction
        side = np.abs(dx)

        # superellipsoid base: boxier than a sphere reads as a head
        n = 2.6
        base = (np.abs(dx / 0.80) ** n + np.abs(dy / 1.02) ** n + np.abs(dz / 0.90) ** n) ** (-1.0 / n)
        x, y, z = dx * base, dy * base, dz * base

        # flatten the face plane, elongate the skull back
        z = z * (1.0 - 0.10 * front) * (1.0 + 0.06 * _smoothstep(0.3, 1.0, dz))
        # jaw taper below the cheekbones
        x = x * (1.0 - 0.38 * _smoothstep(0.52, 0.98, vv))
        y = y * (1.0 + 0.04 * _smoothstep(0.5, 0.0, vv))  # slightly taller cranium

        du = np.abs(((uu - FRONT_U + 0.5) % 1.0) - 0.5)  # signed distance in u from face center
        sgn = np.sign(((uu - FRONT_U + 0.5) % 1.0) - 0.5)

        def bump(du_c, dv_c, su, sv, amp):
            return amp * _gauss(du - du_c, vv - dv_c, su, sv) * (0.35 + 0.65 * front ** 0.5)

        # features displace along -z (toward the camera)
        z = z - bump(0.0, 0.505, 0.050, 0.045, 0.16)                       # nose
        z = z - bump(0.0, 0.470, 0.018, 0.030, 0.05) * 0.5                 # bridge
        z = z - bump(0.0, 0.355, 0.075, 0.012, 0.028)                      # brow ridge
        z = z + bump(0.0541, 0.42, 0.020, 0.018, 0.030)                    # eye sockets
        z = z + bump(-0.0541, 0.42, 0.020, 0.018, 0.030)
        z = z - bump(0.042, 0.480, 0.030, 0.025, 0.020)                    # cheekbones
        z = z - bump(-0.042, 0.480, 0.030, 0.025, 0.020)
        z = z - bump(0.0, 0.627, 0.042, 0.007, 0.020)                      # upper lip
        z = z - bump(0.0, 0.657, 0.038, 0.008, 0.024)                      # lower lip
        z = z + bump(0.0, 0.642, 0.040, 0.004, 0.012)                      # mouth groove
        z = z - bump(0.0, 0.800, 0.045, 0.030, 0.045)                      # chin
        # ears: protrude along +-x near az = 0 / pi
        w_side = _smoothstep(0.70, 0.95, side) * _gauss(vv, 0.47, 0.075, 0.055)
        x = x + np.sign(dx) * 0.13 * w_side

        self.neutral = np.stack([x, y, z], axis=1).astype(np.float64)

    def _build_faces(self):
        nu, nv = self.nu, self.nv
        idx = np.arange(nv * nu).reshape(nv, nu)
        idx_next = np.roll(idx, -1, axis=1)      # wrap in u (azimuth)
        quads = np.stack([idx[:-1].ravel(), idx_next[:-1].ravel(),
                          idx_next[1:].ravel(), idx[1:].ravel()], axis=1)
        a, b, c, d = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
        self.faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        # canonical winding -> outward normals (verified by nose-normal test)
        self._check_orientation()

    def _check_orientation(self):
        n = vertex_normals(self.neutral, self.faces)
        nose = int(np.argmin(self.neutral[:, 2]))
        if n[nose, 2] > 0:
            self.faces = self.faces[:, ::-1]

    # ----------------------------------------------------------------- modes
    def _fields(self):
        """Shared weight fields over vertices, used by both mode families."""
        uu, vv = self.uu, self.vv
        du = ((uu - FRONT_U + 0.5) % 1.0) - 0.5      # signed, in [-0.5, 0.5)
        dz = np.sin(vv * np.pi) * np.sin(uu * 2 * np.pi)
        front = np.clip(-dz, 0.0, 1.0)
        return uu, vv, du, front

    def _build_identity_modes(self):
        uu, vv, du, front = self._fields()
        D = np.zeros((self.num_id, len(uu), 3))

        D[0] = self.neutral * 0.10                                   # overall size
        D[1][:, 0] = self.neutral[:, 0] * 0.09                       # face width
        D[2] = np.stack([np.zeros_like(uu), np.zeros_like(uu),
                         self.neutral[:, 2] * 0.10 * (0.5 + 0.5 * front)], 1)  # face depth
        D[3][:, 1] = self.neutral[:, 1] * 0.06                       # head height
        jaw = _smoothstep(0.50, 0.98, vv)
        D[4][:, 0] = self.neutral[:, 0] * 0.12 * jaw                 # jaw width
        D[5] = np.stack([np.zeros_like(uu), np.zeros_like(uu),
                         -0.05 * _gauss(du, vv - 0.80, 0.05, 0.03) * front], 1)  # chin
        D[6] = np.stack([np.zeros_like(uu), np.zeros_like(uu),
                         -0.09 * _gauss(du, vv - 0.505, 0.045, 0.040) * front], 1)  # nose size
        eye_w = _gauss(du, vv - 0.42, 0.055, 0.025)
        D[7][:, 0] = 0.05 * np.sign(du + 1e-9) * eye_w               # eye spacing
        cheek = _gauss(np.abs(du), vv - 0.47, 0.030, 0.030)
        D[8] = np.stack([np.zeros_like(uu), np.zeros_like(uu),
                         -0.035 * cheek * front], 1)                 # cheek fullness
        back = _smoothstep(0.4, 1.0, -front)
        D[9] = np.stack([np.zeros_like(uu), np.zeros_like(uu),
                         self.neutral[:, 2] * 0.10 * back], 1)       # skull back
        self.D_identity = D

    def _build_expression_modes(self):
        uu, vv, du, front = self._fields()
        E = np.zeros((self.num_expr, len(uu), 3))
        sgn = np.sign(du + 1e-12)

        jaw_w = _smoothstep(0.55, 0.78, vv) * (0.4 + 0.6 * front ** 0.3)
        E[0] = np.stack([np.zeros_like(uu), 0.17 * jaw_w, -0.02 * jaw_w], 1)  # jaw open
        corner = _gauss(np.abs(np.abs(du) - 0.045), vv - 0.640, 0.020, 0.016)
        cheek = _gauss(np.abs(du) - 0.075, vv - 0.52, 0.020, 0.025)
        E[1] = np.stack([sgn * 0.055 * corner, -0.030 * corner - 0.014 * cheek,
                         np.zeros_like(uu)], 1)                       # smile
        brow = _gauss(du, vv - 0.355, 0.075, 0.022) + \
            0.6 * _gauss(du, vv - 0.30, 0.080, 0.030)
        E[2] = np.stack([np.zeros_like(uu), -0.035 * brow * front,
                         np.zeros_like(uu)], 1)                       # brow raise
        eye_w = _gauss(np.abs(du), vv - 0.42, 0.024, 0.020)
        E[3] = np.stack([np.zeros_like(uu), np.zeros_like(uu),
                         0.012 * eye_w * front], 1)                   # eye close (also in albedo)
        lips = _gauss(du, vv - 0.642, 0.045, 0.018)
        E[4] = np.stack([-sgn * 0.030 * lips, np.zeros_like(uu),
                         -0.030 * lips * front], 1)                   # pucker
        lower = _gauss(du, vv - 0.657, 0.040, 0.012)
        E[5] = np.stack([sgn * 0.015 * lower, 0.050 * lower,
                         np.zeros_like(uu)], 1)                       # lower-lip drop
        self.D_expression = E

    # ------------------------------------------------------------- landmarks
    def _landmarks(self):
        """Landmark surface (u, v) targets, snapped to grid vertices."""
        pts: list[tuple[float, float]] = []

        def add(du_list, v_list):
            for d, vv_ in zip(du_list, v_list):
                pts.append(((FRONT_U + d) % 1.0, vv_))

        # eye contour, 6 per eye: outer canthus, upper-outer, upper, inner,
        # lower-inner, lower-outer
        for su in (+1, -1):
            add([su * 0.028, su * 0.017, 0.0, -su * 0.021,
                 -su * 0.012, -su * 0.020],
                [0.420, 0.408, 0.405, 0.422, 0.435, 0.433])
        # brows, 5 per side
        for su in (+1, -1):
            add([su * d for d in (0.048, 0.026, 0.0, -0.026, -0.048)],
                [0.360, 0.352, 0.349, 0.352, 0.362])
        # nose: bridge 4, tip, base center, alar 2
        add([0.0] * 4, [0.400, 0.432, 0.464, 0.494])
        add([0.0], [0.516])
        add([0.0], [0.548])
        add([0.023, -0.023], [0.548, 0.548])
        # mouth outer: corners, upper 5, lower 5
        add([0.046, -0.046], [0.639, 0.639])
        add([-0.032, -0.016, 0.0, 0.016, 0.032], [0.627] * 5)
        add([-0.030, -0.015, 0.0, 0.015, 0.030], [0.657] * 5)
        # mouth inner: corners, upper 3, lower 3
        add([0.037, -0.037], [0.641, 0.641])
        add([-0.018, 0.0, 0.018], [0.633] * 3)
        add([-0.018, 0.0, 0.018], [0.649] * 3)
        # jaw contour, 11 points left -> chin -> right
        t = np.linspace(-0.205, 0.205, 11)
        add(list(t), list(0.60 + 0.17 * (1 - (t / 0.205) ** 2) ** 0.55))
        # cheeks
        add([0.095, -0.095], [0.510, 0.510])

        uv = np.array(pts)
        j = np.round(uv[:, 0] * self.nu).astype(int) % self.nu
        i = np.clip(np.round(uv[:, 1] * self.nv).astype(int), 0, self.nv - 1)
        self.landmark_idx = i * self.nu + j
        self.landmark_uv = uv
        # nose tip = the landmark closest to (FRONT_U, 0.516)
        self.nose_tip = int(np.argmin(np.abs(uv[:, 0] - FRONT_U) +
                                      np.abs(uv[:, 1] - 0.516)))
        # inner eye canthi, used for the frontality proxy
        self.inner_canthi = (3, 9)

    # ------------------------------------------------------------------ API
    def verts(self, c_identity=None, c_expr=None) -> np.ndarray:
        V = self.neutral.copy()
        if c_identity is not None:
            V = V + np.tensordot(np.asarray(c_identity, float), self.D_identity, (0, 0))
        if c_expr is not None:
            V = V + np.tensordot(np.asarray(c_expr, float), self.D_expression, (0, 0))
        return V

    def normals_can(self, c_identity=None, c_expr=None) -> np.ndarray:
        return vertex_normals(self.verts(c_identity, c_expr), self.faces)

    def albedo(self, c_identity=None, c_expr=None) -> np.ndarray:
        uu, vv, du, front = self._fields()
        cid = np.zeros(self.num_id) if c_identity is None else np.asarray(c_identity, float)
        cex = np.zeros(self.num_expr) if c_expr is None else np.asarray(c_expr, float)
        lum = np.full_like(uu, 0.62 + 0.18 * cid[0] + 0.10 * cid[2])
        hue = float(np.clip(0.30 + 0.25 * cid[1], 0.0, 0.6))
        skin = np.stack([lum * (1.0 + 0.10 * np.cos(2 * np.pi * hue)),
                         lum * (1.0 - 0.04 * np.cos(2 * np.pi * hue)),
                         lum * (1.0 - 0.10 * np.cos(2 * np.pi * hue))], -1)
        rgb = skin.copy()

        # lips / brows / shadowing
        lips = _gauss(du, vv - 0.642, 0.043, 0.017)
        lip_col = np.stack([lum * 0.72, lum * 0.42, lum * 0.40], 1)
        rgb = rgb * (1 - lips[..., None]) + lip_col * lips[..., None]
        brow = np.clip(np.maximum(_gauss(np.abs(du) - 0.048, vv - 0.353, 0.028, 0.008),
                                  _gauss(np.abs(du) - 0.048, vv - 0.360, 0.018, 0.008)), 0, 1)
        rgb = rgb * (1 - brow[..., None]) + np.array([0.13, 0.10, 0.09]) * brow[..., None]

        # eyes: sclera, iris, pupil; closed lids when |c3| large
        closed = float(np.clip(cex[3], -1, 1))
        for su in (+1, -1):
            eu = du - su * 0.0541
            open_v = 0.012 * (1.0 - 0.8 * max(closed, 0.0))
            sclera = _gauss(eu, vv - 0.42, 0.017, max(open_v, 0.002))
            iris = _gauss(eu, vv - 0.42, 0.0078, 0.0078 * min(1.0, open_v / 0.012))
            pupil = _gauss(eu, vv - 0.42, 0.0032, 0.0032)
            white = np.array([0.92, 0.91, 0.89])
            iris_c = np.array([0.28, 0.19, 0.11])
            rgb = rgb * (1 - sclera[..., None]) + white * sclera[..., None]
            rgb = rgb * (1 - iris[..., None]) + iris_c * iris[..., None]
            rgb = rgb * (1 - pupil[..., None]) + np.array([0.03, 0.03, 0.03]) * pupil[..., None]
            if closed > 0.35:  # lid line replaces the open eye
                lid = _gauss(eu, vv - 0.425, 0.016, 0.003)
                rgb = rgb * (1 - lid[..., None]) + skin * lid[..., None]
                lash = _gauss(eu, vv - 0.430, 0.015, 0.0015)
                rgb = rgb * (1 - lash[..., None]) + np.array([0.08, 0.06, 0.05]) * lash[..., None]

        # hair with streak noise (LUT computed once)
        rgb = rgb * (1 - self._hair[..., None]) + self._hair_col * self._hair[..., None]
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def _hair_lut(self):
        uu, vv, du, front = self._fields()
        # hairline: low on the sides/forehead temples, higher mid-forehead,
        # and covers the whole back of the head down to the nape
        side = np.abs(du)
        hairline_front = 0.205 + 0.085 * _smoothstep(0.045, 0.13, side) \
            - 0.015 * _gauss(du, 0.0, 0.015, 1.0)   # widow's peak dip
        hairline_back = 0.52
        is_back = _smoothstep(0.10, 0.20, np.abs(du))
        v_hair = hairline_front * (1 - is_back) + hairline_back * is_back
        hair = (vv < v_hair).astype(np.float64)
        # sideburns
        hair = np.maximum(hair, (_smoothstep(0.115, 0.155, side) * (vv < 0.56)).astype(np.float64))
        # streaks from a hash of grid position
        h = np.array([int(hashlib.md5(f"{i}".encode()).hexdigest()[:8], 16)
                      for i in range(len(uu))]) / 0xFFFFFFFF
        streak = 0.75 + 0.5 * h
        self._hair = hair
        base = np.array([0.10, 0.072, 0.052])
        self._hair_col = np.clip(base[None, :] * streak[:, None], 0, 1)

    def hair_mask(self, uv: np.ndarray) -> np.ndarray:
        """Hair indicator for arbitrary surface (u, v) points (used to build
        the evaluated face region)."""
        uu, vv = uv[:, 0], uv[:, 1]
        du = np.abs(((uu - FRONT_U + 0.5) % 1.0) - 0.5)
        is_back = _smoothstep(0.10, 0.20, du)
        v_hair = (0.205 + 0.085 * _smoothstep(0.045, 0.13, du)) * (1 - is_back) + 0.52 * is_back
        side_hair = (_smoothstep(0.115, 0.155, du) * (vv < 0.56)) > 0
        return (vv < v_hair) | side_hair
