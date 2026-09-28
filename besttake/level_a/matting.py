"""Soft matting for Level A: feathered elliptical alpha around the face.

Hair-safe soft edges (plan principle: never a hard cutout). In production the
Phase 1 matting model replaces this; the interface is the MattingProvider
contract so the swap is one object.
"""
from __future__ import annotations

import numpy as np

from ..common.types import FaceObservation, Frame
from .landmarks import face_ellipse, key_points


def soft_ellipse(shape: tuple[int, int], center: tuple[float, float],
                 rx: float, ry: float, feather: float) -> np.ndarray:
    """1 inside the core ellipse (r - feather), 0 outside (r + feather),
    smoothstep between."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - center[0]
    dy = yy - center[1]
    f = max(1.0, feather)
    inx, iny = max(1.0, rx - f), max(1.0, ry - f)
    outx, outy = rx + f, ry + f
    n_in = np.hypot(dx / inx, dy / iny)
    n_out = np.hypot(dx / outx, dy / outy)
    alpha = np.zeros((h, w), dtype=np.float32)
    core = n_in <= 1.0
    alpha[core] = 1.0
    band = (~core) & (n_out < 1.0)
    # u = 1 at the core rim, 0 at the outer rim.
    u = (n_out[band] - 1.0) / (n_out[band] - n_in[band])
    alpha[band] = (u * u * (3.0 - 2.0 * u)).astype(np.float32)
    return alpha


class EllipseFeatherMatting:
    """MattingProvider: soft alpha around the observed face ellipse."""

    feather_fraction: float = 0.25

    def alpha(self, frame: Frame, obs: FaceObservation) -> np.ndarray:
        kp = key_points(obs.landmarks)
        (cx, cy), rx, ry = face_ellipse(kp)
        feather = max(4.0, min(rx, ry) * self.feather_fraction)
        return soft_ellipse(frame.shape, (cx, cy), rx, ry, feather)
