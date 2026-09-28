import numpy as np

from besttake.face_model.canonical import FRONT_U, SyntheticHeadModel


def model():
    return SyntheticHeadModel(nu=64, nv=64)


def test_shapes():
    m = model()
    assert m.neutral.shape == (64 * 64, 3)
    assert m.D_identity.shape == (m.num_id, 64 * 64, 3)
    assert m.D_expression.shape == (m.num_expr, 64 * 64, 3)
    assert len(m.landmark_idx) == m.landmark_uv.shape[0]
    V = m.verts(np.ones(m.num_id), np.ones(m.num_expr))
    assert V.shape == m.neutral.shape and np.isfinite(V).all()


def test_expression_moves_mouth_most():
    m = model()
    lmk = m.landmark_idx
    du = np.abs(((m.landmark_uv[:, 0] - FRONT_U + 0.5) % 1.0) - 0.5)
    mouth = (m.landmark_uv[:, 1] > 0.60) & (m.landmark_uv[:, 1] < 0.68) & (du < 0.06)
    brow = (m.landmark_uv[:, 1] > 0.33) & (m.landmark_uv[:, 1] < 0.38)
    # smile should displace mouth landmarks more than brow landmarks
    d = np.tensordot(np.array([0, 1.0, 0, 0, 0, 0]), m.D_expression, (0, 0))
    mag = np.linalg.norm(d, axis=1)
    assert mag[lmk][mouth].mean() > 4.0 * max(mag[lmk][brow].mean(), 1e-6)
    # brow raise displaces brow landmarks
    d2 = np.tensordot(np.array([0, 0, 1.0, 0, 0, 0]), m.D_expression, (0, 0))
    mag2 = np.linalg.norm(d2, axis=1)
    assert mag2[lmk][brow].mean() > 2.0 * max(mag2[lmk][mouth].mean(), 1e-6)


def test_identity_width_mode_widens():
    m = model()
    w0 = m.neutral[:, 0].max() - m.neutral[:, 0].min()
    V = m.verts(np.array([0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]))
    w1 = V[:, 0].max() - V[:, 0].min()
    assert w1 > w0 * 1.03


def test_landmarks_on_front():
    m = model()
    du = np.abs(((m.landmark_uv[:, 0] - FRONT_U + 0.5) % 1.0) - 0.5)
    assert du.max() < 0.25          # all landmarks near the face
    assert m.landmark_uv[:, 1].min() > 0.30 and m.landmark_uv[:, 1].max() < 0.85


def test_hair_mask_regions():
    m = model()
    uv = np.array([[FRONT_U, 0.45],   # nose bridge: skin
                   [FRONT_U, 0.10],   # crown: hair
                   [FRONT_U + 0.30 % 1.0, 0.30]])  # back of head: hair
    hm = m.hair_mask(uv)
    assert not hm[0] and hm[1] and hm[2]
