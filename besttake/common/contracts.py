"""Contracts between pipeline phases.

Phase 1 (Level A) owns capture, tracking, matting, noise estimation and the
checkers. Phase 3 (this code, Level B) *consumes* them through these
protocols, with simple internal fallbacks so the pipeline runs standalone.
The Phase 1 owner implements these against production models; nothing in
level_b needs to change when that lands.

Integration rule agreed for parallel work: `besttake/level_a/` is owned by
the Phase 1 workstream; Phase 3 never imports from it, only from here.
"""
from __future__ import annotations

from typing import Protocol
import numpy as np

from .types import Frame, FaceObservation, Fit


class LandmarkSource(Protocol):
    """Dense landmarks per person per frame (Phase 1 tracker)."""

    def detect(self, frame: Frame) -> dict[str, FaceObservation]:
        """person_id -> observation with 2D landmarks in pixel coordinates."""
        ...


class FaceFitterSource(Protocol):
    """3D face fitting front-end (may be replaced by an ARKit/Core ML fit)."""

    def fit(self, frame: Frame, obs: FaceObservation) -> Fit:
        ...


class MattingProvider(Protocol):
    """Soft alpha for a person in a frame (hair-safe edges, never a hard cutout)."""

    def alpha(self, frame: Frame, obs: FaceObservation) -> np.ndarray:
        """(H, W) float in [0, 1]."""
        ...


class NoiseEstimator(Protocol):
    """Per-frame sensor grain level, for grain matching (Phase 1 finishing)."""

    def sigma(self, frame: Frame) -> float:
        ...


class ArtifactChecker(Protocol):
    """Trained composite detector. Swap is rejected if it flags artifacts.

    weight: the blend-weight map, for seam-sensitive checkers.
    references: donor/reference frames of the same person, for checkers that
    compare texture statistics against the source material. Both optional.
    """

    def check(self, image: np.ndarray, region: np.ndarray,
              weight: np.ndarray | None = None,
              references: list[np.ndarray] | None = None) -> bool:
        ...


class IdentityChecker(Protocol):
    """Embedding similarity of the result vs. that person's real frames.

    references: frames of the claimed person; entries may be plain arrays or
    (frame, region-mask) pairs, the mask marking that frame's own face area —
    use it when provided, so the comparison is pose-independent.
    """

    def check(self, image: np.ndarray, region: np.ndarray,
              references: list) -> bool:
        ...


class RelightModel(Protocol):
    """Lighting harmonization for re-projected faces (learned relight later)."""

    def relight(self, face_rgb: np.ndarray, target_shading: np.ndarray) -> np.ndarray:
        ...
