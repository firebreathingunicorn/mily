"""Z-buffer software rasterizer.

Renders a mesh under a face `Fit` into per-pixel buffers needed by Level B:
depth, triangle id + barycentrics (for anatomical correspondence between
frames), camera-space position/normal, canonical-space position, albedo and
model (u, v). Pure numpy; sized for prototype renders (~320 px, ~18k tris).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.types import Fit


@dataclass
class Render:
    mask: np.ndarray          # (H, W) bool — surface coverage
    zbuf: np.ndarray          # (H, W) float32, fit depth units
    face_id: np.ndarray       # (H, W) int32, -1 outside
    bary: np.ndarray          # (H, W, 2) barycentric (b0, b1)
    cam_pos: np.ndarray       # (H, W, 3)
    cam_normal: np.ndarray    # (H, W, 3), outward, unit
    canonical: np.ndarray     # (H, W, 3) canonical-space surface point
    albedo: np.ndarray        # (H, W, 3) linear RGB before shading
    uv: np.ndarray            # (H, W, 2) model surface coordinates
    faces: np.ndarray = None  # (F, 3) mesh connectivity (reference)

    @property
    def facing(self) -> np.ndarray:
        """How much each pixel faces the camera in [0, 1] (-n_z of camera space)."""
        return np.clip(-self.cam_normal[..., 2], 0.0, 1.0)


def interp_attr(render: Render, attr: np.ndarray) -> np.ndarray:
    """Interpolate per-vertex attributes (N, ...) at covered pixels -> (H, W, ...)."""
    m = render.mask
    fi = render.face_id[m]
    f0, f1, f2 = render.faces[fi, 0], render.faces[fi, 1], render.faces[fi, 2]
    bb0, bb1 = render.bary[m, 0], render.bary[m, 1]
    bb2 = 1.0 - bb0 - bb1
    vals = (attr[f0] * bb0[:, None] + attr[f1] * bb1[:, None] + attr[f2] * bb2[:, None])
    out = np.zeros((m.size,) + attr.shape[1:], np.result_type(attr, np.float32))
    out[m.ravel()] = vals
    return out.reshape(render.mask.shape + attr.shape[1:])


def render(fit: Fit, verts_can: np.ndarray, faces: np.ndarray,
           vnorm_can: np.ndarray, albedo: np.ndarray, uv: np.ndarray,
           size: tuple[int, int]) -> Render:
    """Rasterize the posed model. All per-vertex arrays are canonical-space.

    vnorm_can: smooth canonical vertex normals ( shading uses camera-space,
    computed by rotating vnorm_can by the fit).
    """
    H, W = size
    verts_cam = fit.to_camera(verts_can)
    px = fit.project(verts_can)
    nrm_cam = (fit.R @ vnorm_can.T).T

    zbuf = np.full((H, W), np.inf, np.float32)
    fid = np.full((H, W), -1, np.int32)
    b0 = np.zeros((H, W), np.float32)
    b1 = np.zeros((H, W), np.float32)

    v0, v1, v2 = px[faces[:, 0]], px[faces[:, 1]], px[faces[:, 2]]
    z0, z1, z2 = fit.depth(verts_can)[faces[:, 0]], fit.depth(verts_can)[faces[:, 1]], fit.depth(verts_can)[faces[:, 2]]
    # screen-space sign of triangle orientation is consistent across the mesh;
    # cull nothing, but reject back-facing degenerate slivers via area test
    d = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    keep = np.abs(d) > 1e-9
    for t in np.nonzero(keep)[0]:
        x0, y0 = v0[t]
        x1, y1 = v1[t]
        x2, y2 = v2[t]
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))) + 1, W)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))) + 1, H)
        if xmin >= xmax or ymin >= ymax:
            continue
        xs = np.arange(xmin, xmax, dtype=np.float64) + 0.5
        ys = np.arange(ymin, ymax, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        det = d[t]
        w1 = ((gx - x0) * (y2 - y0) - (gy - y0) * (x2 - x0)) / det
        w2 = ((gy - y0) * (x1 - x0) - (gx - x0) * (y1 - y0)) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        z = w0 * z0[t] + w1 * z1[t] + w2 * z2[t]
        sub_z = zbuf[ymin:ymax, xmin:xmax]
        sub_f = fid[ymin:ymax, xmin:xmax]
        upd = inside & (z < sub_z)
        sub_z[upd] = z[upd]
        sub_f[upd] = t
        sub_b0 = b0[ymin:ymax, xmin:xmax]
        sub_b1 = b1[ymin:ymax, xmin:xmax]
        sub_b0[upd] = w0[upd]
        sub_b1[upd] = w1[upd]

    m = fid >= 0
    out = {
        "mask": m,
        "zbuf": zbuf,
        "face_id": fid,
        "bary": np.stack([b0, b1], axis=-1),
    }

    fi = fid[m]
    f0, f1_, f2 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
    bb0, bb1 = b0[m], b1[m]
    bb2 = 1.0 - bb0 - bb1
    gather = np.stack([bb0, bb1, bb2], axis=1)  # (P, 3)

    def interp(attr: np.ndarray) -> np.ndarray:
        vals = attr[f0] * gather[:, 0:1] + attr[f1_] * gather[:, 1:2] + attr[f2] * gather[:, 2:3]
        full = np.zeros((H * W,) + attr.shape[1:], attr.dtype)
        full[m.ravel()] = vals
        return full.reshape((H, W) + attr.shape[1:])

    out["cam_pos"] = interp(verts_cam.astype(np.float32))
    out["cam_normal"] = interp(nrm_cam.astype(np.float32))
    out["canonical"] = interp(verts_can.astype(np.float32))
    out["albedo"] = interp(albedo.astype(np.float32))
    out["uv"] = interp(uv.astype(np.float32))
    out["faces"] = faces
    return Render(**out)


def shade(render: Render, light_dir: np.ndarray, ambient: float = 0.35,
          kd: float = 0.75) -> np.ndarray:
    """Lambert shading of a render; light_dir points from surface toward light."""
    l = light_dir / np.linalg.norm(light_dir)
    lam = np.clip(render.cam_normal @ l, 0.0, None)[..., None]
    return np.clip(render.albedo * (ambient + kd * lam), 0.0, 1.0)
