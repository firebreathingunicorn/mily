import numpy as np

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.rendering import render
from besttake.common.types import FaceObservation, Fit
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.reproject import blend_sources, reproject


def _case(yaw_base=25.0, yaw_donor=8.0):
    m = SyntheticHeadModel(nu=64, nv=64)
    cap = SyntheticCapture(m, size=288, seed=11)
    cid = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])
    cex = np.zeros(6); cex[1] = 0.5
    base = cap.shoot(cid, cex, yaw_base)
    donor = cap.shoot(cid, cex, yaw_donor)
    return m, cap, cid, cex, base, donor


def _render(m, cap, cid, cex, shot):
    fit = shot.gt_fit
    return render(fit, m.verts(cid, cex), m.faces, m.normals_can(cid, cex),
                  m.albedo(cid, cex), np.stack([m.uu, m.vv], 1), shot.frame.shape)


def test_identity_pose_reprojection_recovers_image():
    """Same pose, same expression: re-projection must return the frame itself."""
    m, cap, cid, cex, base, donor = _case(yaw_base=15.0, yaw_donor=15.0)
    br = _render(m, cap, cid, cex, base)
    dr = _render(m, cap, cid, cex, donor)
    zero = np.zeros_like(m.neutral)
    w = reproject(br, zero, zero, base.gt_fit, donor.frame.rgb, dr.zbuf,
                  base_fit=base.gt_fit, z_tol=0.02)
    sel = w.valid & br.mask & (br.facing > 0.2)
    assert sel.sum() > 5000
    err = np.abs(w.rgb[sel] - donor.frame.rgb[sel]).mean()
    assert err < 0.02, f"identity reprojection error {err:.4f}"


def test_visibility_marks_occluded_side_invalid():
    """A 50-degree-turned donor cannot see the far ear region of the base."""
    m, cap, cid, cex, base, donor = _case(yaw_base=0.0, yaw_donor=50.0)
    br = _render(m, cap, cid, cex, base)
    dr = _render(m, cap, cid, cex, donor)
    zero = np.zeros_like(m.neutral)
    w = reproject(br, zero, zero, donor.gt_fit, donor.frame.rgb, dr.zbuf,
                  base_fit=base.gt_fit, z_tol=0.02)
    # far-side surface points (canonical x opposite the turn) must be invalid
    far = br.canonical[..., 0] < -0.5
    sel = far & br.mask & (br.facing > 0.3)
    frac_invalid = 1.0 - w.valid[sel].mean()
    assert frac_invalid > 0.8, f"only {frac_invalid:.2f} of far side invalid"


def test_blend_sources_prefers_confident():
    m = SyntheticHeadModel(nu=48, nv=48)
    shape = (32, 32)
    w1_rgb = np.zeros(shape + (3,), np.float32)
    w2_rgb = np.ones(shape + (3,), np.float32)
    wt1 = np.zeros(shape, np.float32); wt1[:, :16] = 1.0
    wt2 = np.zeros(shape, np.float32); wt2[:, 16:] = 1.0
    from besttake.level_b.reproject import WarpedDonor
    a = WarpedDonor(w1_rgb, wt1, wt1 > 0, 0)
    b = WarpedDonor(w2_rgb, wt2, wt2 > 0, 1)
    rgb, weight, src, fill = blend_sources([a, b], primary_index=0)
    assert src[8, 8] == 0 and src[8, 24] == 1
    assert rgb[8, 8].max() < 0.2 and rgb[8, 24].min() > 0.8
    assert fill > 0.3  # right half only donor 2 covers
