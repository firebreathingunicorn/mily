"""Canonical landmark layout of the 63-point SyntheticHeadModel observations
and the derived key points Level A needs (eyes, nose, mouth corners, chin).

Index map (see ``face_model.canonical.SyntheticHeadModel._landmarks``):
  0-5   eye A contour   6-11  eye B contour
  12-21 brows          22-25  nose bridge     26 nose tip   27 nose base
  28-29 alar           30-31  mouth outer corners
  32-36 mouth upper    37-41  mouth lower
  42-49 mouth inner    50-60  jaw contour     61-62 cheeks

Eye "centers" are the canthi midpoints (outer = 0 / 6, inner = 3 / 9), not
contour centroids: the contour samples cluster at the inner corner, which
would collapse the inter-ocular distance.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EYE_A = slice(0, 6)
EYE_B = slice(6, 12)
OUTER_A, INNER_A = 0, 3
OUTER_B, INNER_B = 6, 9
NOSE_TIP = 26
MOUTH_CORNERS = (30, 31)
MOUTH_UPPER = slice(32, 37)
MOUTH_LOWER = slice(37, 42)
MOUTH_INNER = slice(42, 50)
CHIN = 55
FACE_PTS = list(range(0, 12)) + [NOSE_TIP] + list(MOUTH_CORNERS) + [CHIN]


@dataclass
class KeyPoints:
    eye_a: np.ndarray
    eye_b: np.ndarray
    nose_tip: np.ndarray
    mouth_l: np.ndarray
    mouth_r: np.ndarray
    chin: np.ndarray
    inter_ocular: float
    yaw_proxy: float
    roll: float
    """Face extent from the landmark set (stylized models are not
    io-proportioned, so the matting ellipse sizes from the extent)."""
    face_width: float
    face_height: float

    @property
    def eye_mid(self) -> np.ndarray:
        return 0.5 * (self.eye_a + self.eye_b)

    @property
    def mouth_mid(self) -> np.ndarray:
        return 0.5 * (self.mouth_l + self.mouth_r)


def key_points(landmarks: np.ndarray, yaw_override: float | None = None) -> KeyPoints:
    lm = np.asarray(landmarks, float)
    eye_a = 0.5 * (lm[OUTER_A] + lm[INNER_A])
    eye_b = 0.5 * (lm[OUTER_B] + lm[INNER_B])
    io = max(float(np.linalg.norm(eye_a - eye_b)), 1e-6)
    eye_mid = 0.5 * (eye_a + eye_b)
    nose = lm[NOSE_TIP]
    roll = float(np.arctan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))
    # Refined burst-relative yaw supersedes the raw nose-offset proxy.
    yaw = float(yaw_override) if yaw_override is not None else float((nose[0] - eye_mid[0]) / io)
    face = lm[FACE_PTS]
    width = float(face[:, 0].max() - face[:, 0].min())
    height = float(max(lm[CHIN][1] - eye_mid[1], io))
    return KeyPoints(
        eye_a=eye_a, eye_b=eye_b, nose_tip=nose,
        mouth_l=lm[MOUTH_CORNERS[0]], mouth_r=lm[MOUTH_CORNERS[1]],
        chin=lm[CHIN], inter_ocular=io, yaw_proxy=yaw, roll=roll,
        face_width=width, face_height=height,
    )


def face_ellipse(kp: KeyPoints) -> tuple[tuple[float, float], float, float]:
    """(center, rx, ry) of the soft-matting ellipse covering the face.

    The ellipse extends ~0.25·face-height above the eye line so brows and
    forehead expressions are inside the transplant area.
    """
    center = (kp.eye_mid[0], kp.eye_mid[1] + 0.50 * kp.face_height)
    rx = 0.55 * max(kp.face_width, kp.inter_ocular * 2)
    ry = 0.75 * kp.face_height
    return center, rx, ry
