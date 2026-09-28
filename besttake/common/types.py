"""Shared data types for the Best Take pipeline.

Coordinate conventions (used everywhere):
- Canonical head space: origin at head center, +x right (image right when
  facing the camera), +y down, +z pointing INTO the head (away from the
  camera when frontal). The nose is the point with the smallest z.
- Camera space: +x right, +y down, +z away from the camera (pinhole:
  pixel = (fx * X/Z + cx, fy * Y/Z + cy), Z > 0).
- A head fit maps canonical -> camera; a frontal head has R = I.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Frame:
    """One captured image with optional metric depth (LiDAR / dual camera)."""
    rgb: np.ndarray                      # (H, W, 3) float32 in [0, 1]
    depth: np.ndarray | None = None      # (H, W) float32, camera-space Z
    intrinsics: Intrinsics | None = None
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]


@dataclass
class Fit:
    """Rigid fit of the face model to one frame.

    mode 'weak':    pixel = s * (R @ X)[:2] + t        (depth = s * (R @ X)[2])
    mode 'pinhole': Xc = R @ X + T; pixel = f * Xc[:2]/Xc[2] + c  (depth = Xc[2])
    """
    R: np.ndarray                        # (3, 3)
    s: float = 1.0                       # weak-perspective scale
    t: np.ndarray | None = None          # (2,) weak-perspective translation
    T: np.ndarray | None = None          # (3,) pinhole translation
    mode: str = "weak"
    intrinsics: "Intrinsics | None" = None
    c_identity: np.ndarray | None = None  # identity coefficients
    c_expression: np.ndarray | None = None
    landmark_rmse: float = float("nan")

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        """Canonical points (..., 3) -> camera-space points (..., 3)."""
        X = np.asarray(X, float)
        lead = X.shape[:-1]
        Xc = (self.R @ X.reshape(-1, 3).T).T
        if self.mode == "pinhole":
            Xc = Xc + self.T
        else:
            Xc = Xc * self.s + np.concatenate([self.t, [0.0]])
        return Xc.reshape(lead + (3,))

    def depth(self, X: np.ndarray) -> np.ndarray:
        """Camera-space depth of canonical points (..., 3)."""
        return self.to_camera(X)[..., 2]

    def project(self, X: np.ndarray) -> np.ndarray:
        """Canonical points (..., 3) -> pixel coordinates (..., 2)."""
        X = np.asarray(X, float)
        lead = X.shape[:-1]
        Xc = (self.R @ X.reshape(-1, 3).T).T
        if self.mode == "pinhole":
            assert self.intrinsics is not None, "pinhole fit needs intrinsics"
            Xc = Xc + self.T
            K = self.intrinsics
            pix = np.stack([K.fx * Xc[:, 0] / Xc[:, 2] + K.cx,
                            K.fy * Xc[:, 1] / Xc[:, 2] + K.cy], axis=-1)
        else:
            pix = self.s * Xc[:, :2] + self.t
        return pix.reshape(lead + (2,))


@dataclass
class FaceObservation:
    """Per-frame, per-person signals from scene understanding (Phase 1/2).

    In production this is produced by the tracker + 3D fitting front-end.
    Here the fitter consumes the landmarks (and optionally depth via Frame).
    """
    person_id: str
    landmarks: np.ndarray                # (M, 2) pixel coordinates
    landmark_ids: np.ndarray | None = None
    fit: Fit | None = None               # filled by the fitter
    frame_index: int = -1
    # Burst-relative yaw (radians, signed) from level_a's PoseRefiner; when
    # set it supersedes the raw nose-offset proxy. Additive, default None —
    # safe for every existing consumer.
    yaw_override: float | None = None


@dataclass
class SwapResult:
    """Output of a Level B (or A) swap for one person in one base frame."""
    image: np.ndarray                    # composited full frame
    weight: np.ndarray                   # (H, W) blend weight of the swap
    source_map: np.ndarray               # (H, W) int, donor index per pixel, -1 = base
    face_region: np.ndarray              # (H, W) bool, evaluated face area
    method: str = "B"                    # which synthesis level produced it
    coverage: float = 0.0                # fraction of face region swapped
    fill_fraction: float = 0.0           # fraction swapped pixels filled by non-primary donors
    coverage_sectors: np.ndarray | None = None  # (3,3) swap coverage of the face bbox — smart-shutter guidance
    checks: dict = field(default_factory=dict)
