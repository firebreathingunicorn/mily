"""Least-squares 2D similarity fit, closed form (Umeyama without reflection).

dst = M(src) with  x' = a*x - b*y + tx ;  y' = b*x + a*y + ty.
M is a (2, 3) matrix; apply with ``M @ [x, y, 1]``.
"""
from __future__ import annotations

import numpy as np


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    ms = src.mean(axis=0)
    md = dst.mean(axis=0)
    cs = src - ms
    cd = dst - md
    den = float((cs * cs).sum())
    if den < 1e-8:
        return None
    a = float((cs[:, 0] * cd[:, 0] + cs[:, 1] * cd[:, 1]).sum()) / den
    b = float((cs[:, 0] * cd[:, 1] - cs[:, 1] * cd[:, 0]).sum()) / den
    tx = md[0] - (a * ms[0] - b * ms[1])
    ty = md[1] - (b * ms[0] + a * ms[1])
    return np.array([[a, -b, tx], [b, a, ty]])


def invert_similarity(m: np.ndarray) -> np.ndarray:
    a, nb, tx = m[0, 0], m[0, 1], m[0, 2]
    b, _, ty = m[1, 0], m[1, 1], m[1, 2]
    d = a * a + b * b
    if d < 1e-12:
        return np.eye(2, 3)
    ia, ib = a / d, -b / d
    return np.array([
        [ia, ib, -(ia * tx + ib * ty)],
        [-ib, ia, -(-ib * tx + ia * ty)],
    ])


def apply_similarity(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return hom @ m.T
