"""Geometric identity descriptor and the relative-distance identity gate.

Normalized inter-landmark proportions are scale/translation invariant and,
at Level A, guard against wrong-person swaps (tracking misassignment).
Relative Euclidean distance is the metric: cosine is too forgiving for
all-positive distance vectors.
"""
from __future__ import annotations

import numpy as np

from .landmarks import key_points, KeyPoints


def descriptor(landmarks: np.ndarray | KeyPoints) -> np.ndarray:
    kp = landmarks if isinstance(landmarks, KeyPoints) else key_points(landmarks)
    s = max(kp.inter_ocular, 1e-6)

    def d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b)) / s

    mm = kp.mouth_mid
    em = kp.eye_mid
    return np.array([
        d(kp.nose_tip, kp.eye_a), d(kp.nose_tip, kp.eye_b),
        d(kp.mouth_l, kp.eye_a), d(kp.mouth_r, kp.eye_b),
        d(kp.mouth_l, kp.mouth_r),
        d(kp.chin, kp.nose_tip),
        d(kp.chin, kp.mouth_l), d(kp.chin, kp.mouth_r),
        d(kp.chin, kp.eye_a), d(kp.chin, kp.eye_b),
        d(mm, em),
    ], dtype=float)


def relative_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    scale = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    if scale < 1e-9:
        return 1.0
    return float(np.linalg.norm(a - b) / scale)
