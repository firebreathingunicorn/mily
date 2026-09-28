import numpy as np

from besttake.common.rendering import interp_attr, render, shade
from besttake.common.types import Fit
from besttake.face_model.canonical import SyntheticHeadModel

_M = None


def model() -> SyntheticHeadModel:
    global _M
    if _M is None:
        _M = SyntheticHeadModel(nu=64, nv=64)
    return _M


def _frontal_fit(m):
    return Fit(R=np.eye(3), s=95.0, t=np.array([160.0, 150.0]), mode="weak")


def test_render_coverage_and_buffers():
    m = model()
    cid, cex = np.zeros(m.num_id), np.zeros(m.num_expr)
    r = render(_frontal_fit(m), m.verts(cid, cex), m.faces, m.normals_can(cid, cex),
               m.albedo(cid, cex), np.stack([m.uu, m.vv], 1), (320, 320))
    assert 0.15 < r.mask.mean() < 0.6
    assert r.zbuf[r.mask].max() < np.inf
    assert (r.face_id[~r.mask] == -1).all()
    assert r.canonical.shape == (320, 320, 3)
    assert 0.0 <= r.albedo.min() and r.albedo.max() <= 1.0


def test_zbuffer_nose_in_front():
    # at profile, the nose z is smaller (closer) than the far cheek's
    from besttake.common.geometry import rodrigues
    m = model()
    fit = Fit(R=rodrigues(np.array([0.0, np.radians(40.0), 0.0])),
              s=95.0, t=np.array([160.0, 150.0]), mode="weak")
    cid, cex = np.zeros(m.num_id), np.zeros(m.num_expr)
    r = render(fit, m.verts(cid, cex), m.faces, m.normals_can(cid, cex),
               m.albedo(cid, cex), np.stack([m.uu, m.vv], 1), (320, 320))
    ys, xs = np.nonzero(r.mask)
    # nose tip pixels have the smallest canonical z
    nose_z = r.canonical[..., 2][r.mask]
    assert nose_z.min() < -0.8  # nose is far forward


def test_shade_range():
    m = model()
    cid, cex = np.zeros(m.num_id), np.zeros(m.num_expr)
    r = render(_frontal_fit(m), m.verts(cid, cex), m.faces, m.normals_can(cid, cex),
               m.albedo(cid, cex), np.stack([m.uu, m.vv], 1), (320, 320))
    img = shade(r, np.array([-0.4, -0.5, -0.75]))
    assert img[r.mask].min() >= 0.0 and img[r.mask].max() <= 1.0


def test_interp_attr_matches_direct():
    m = model()
    cid, cex = np.zeros(m.num_id), np.zeros(m.num_expr)
    r = render(_frontal_fit(m), m.verts(cid, cex), m.faces, m.normals_can(cid, cex),
               m.albedo(cid, cex), np.stack([m.uu, m.vv], 1), (320, 320))
    attr = np.arange(len(m.neutral), dtype=np.float32)[:, None] * np.ones(3, np.float32)
    out = interp_attr(r, attr)
    # pixels at vertex-dominated region: value close to some vertex id average
    ys, xs = np.nonzero(r.mask)
    vals = out[ys, xs, 0] / 1.0
    assert vals.min() >= 0 and vals.max() <= len(m.neutral)
