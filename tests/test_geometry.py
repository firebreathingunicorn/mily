import numpy as np

from besttake.common.geometry import inv_rodrigues, kabsch_2d, rodrigues, vertex_normals


def test_rodrigues_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.normal(0, 2.5, 3)
        R = rodrigues(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        w2 = inv_rodrigues(R)
        assert np.allclose(rodrigues(w2), R, atol=1e-6)


def test_rodrigues_near_pi():
    R = rodrigues(np.array([0.0, np.radians(179.0), 0.0]))
    w = inv_rodrigues(R)
    assert np.allclose(rodrigues(w), R, atol=1e-4)


def test_vertex_normals_outward():
    # icosphere-ish: use a coarse UV grid from the head model instead
    from besttake.face_model.canonical import SyntheticHeadModel
    m = SyntheticHeadModel(nu=32, nv=32)
    n = vertex_normals(m.neutral, m.faces)
    # nose tip normal should point out of the face (-z)
    nose = int(np.argmin(m.neutral[:, 2]))
    assert n[nose, 2] < -0.9
    # crown normal should point up (-y)
    crown = int(np.argmin(m.neutral[:, 1]))
    assert n[crown, 1] < -0.9


def test_kabsch_2d():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = 2.0 * P + np.array([5.0, -3.0])
    s, t_shift, pc = kabsch_2d(P, Q)
    assert np.isclose(s, 2.0)
    assert np.allclose(Q.mean(0), s * P.mean(0) + t_shift)
