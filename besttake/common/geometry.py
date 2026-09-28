"""Small rigid-geometry helpers."""
from __future__ import annotations

import numpy as np


def rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) -> rotation matrix (3, 3)."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def inv_rodrigues(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> rotation vector (3,)."""
    R = np.asarray(R, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-5:  # near-pi: use the diagonal form
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(axis))
        signs = np.sign(R[i] + 1e-18)
        axis = axis * signs
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return k * (theta / (2.0 * np.sin(theta)))


def yaw_pitch_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose R (canonical->camera, y down) into yaw/pitch/roll in degrees."""
    w = inv_rodrigues(R)
    return tuple(np.degrees(w))  # type: ignore[return-value]


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals of a triangle mesh (outward, unit length)."""
    v = vertices[faces]                        # (F, 3, 3)
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return n / norm


def kabsch_2d(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Best-fit similarity (s, t) mapping 2D points P onto Q (in-plane only).

    Returns the residual-optimal in-plane rotation-free alignment; used only
    to initialize the pose fitter.
    """
    pc, qc = P.mean(0), Q.mean(0)
    scale = np.linalg.norm(Q - qc) / max(np.linalg.norm(P - pc), 1e-9)
    return float(scale), qc - scale * pc, pc
