"""Expression scoring, swap risk and burst planning for Level A.

Hand-tuned stand-ins for the learned expression-preference model and learned
risk model (plan §Selection). Same weights and saturations as the Swift core,
so research results transfer. Consumed through plain functions so a trained
model can replace them without touching the swap code.
"""
from __future__ import annotations

import numpy as np

from ..common.types import FaceObservation
from .landmarks import key_points


def _clamp01(v):
    return np.clip(v, 0.0, 1.0)


# Weights: eyes, smile, mid-word penalty, frontality, sharpness.
W_EYES, W_SMILE, W_MIDWORD, W_FRONT, W_SHARP = 0.20, 0.35, 0.15, 0.15, 0.15


def score_frame(frame, obs: FaceObservation) -> float:
    """Score one person's moment in one frame, in [0, 1]."""
    kp = key_points(obs.landmarks, obs.yaw_override)
    io = kp.inter_ocular
    lm = np.asarray(obs.landmarks, float)

    eye_a = lm[0:6]
    eye_b = lm[6:12]
    ap_a = (eye_a[:, 1].max() - eye_a[:, 1].min()) / io
    ap_b = (eye_b[:, 1].max() - eye_b[:, 1].min()) / io
    eyes = _clamp01(0.5 * (ap_a + ap_b) / 0.10)

    corners_y = 0.5 * (kp.mouth_l[1] + kp.mouth_r[1])
    mouth_mid_y = 0.5 * (lm[32:37][:, 1].mean() + lm[37:42][:, 1].mean())
    width = max(float(np.linalg.norm(kp.mouth_l - kp.mouth_r)), 1.0)
    smile = _clamp01((mouth_mid_y - corners_y) / width + 0.5)

    inner = lm[42:50]
    aperture = (inner[:, 1].max() - inner[:, 1].min()) / io
    midword = 1.0 - _clamp01(aperture / 0.18)

    frontality = 1.0 - _clamp01(max(abs(kp.yaw_proxy) / 0.3, abs(kp.roll) / 0.6))

    # Local vs global sharpness (log-ratio, centered at 0.5).
    h, w = frame.shape
    cx, cy = kp.eye_mid
    r = int(1.2 * io)
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
    g = _luma_crop(frame.rgb, x0, y0, x1, y1)
    face_sharp = _lap_var(g)
    global_sharp = _lap_var(_luma_crop(frame.rgb, 0, 0, w, h))
    sharp = _clamp01(0.5 + np.log2(max(face_sharp, 1e-8) / max(global_sharp, 1e-8)) / 2)

    return float(W_EYES * eyes + W_SMILE * smile + W_MIDWORD * midword
                 + W_FRONT * frontality + W_SHARP * sharp)


def _luma_crop(rgb, x0, y0, x1, y1):
    rgb = np.asarray(rgb, float)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return g[y0:y1, x0:x1]


def _lap_var(g: np.ndarray) -> float:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def risk(base_obs: FaceObservation, donor_obs: FaceObservation) -> float:
    """Predicted swap risk in [0, 1.5]: pose, roll, scale deltas."""
    max_pose, max_roll, max_scale = 0.3, 0.52, 0.3
    kb, kd = key_points(base_obs.landmarks), key_points(donor_obs.landmarks)
    pose = min(1.0, abs(kb.yaw_proxy - kd.yaw_proxy) / max_pose)
    roll = min(1.0, abs(kb.roll - kd.roll) / max_roll)
    scale = min(1.0, abs(np.log(max(kb.inter_ocular, 1) / max(kd.inter_ocular, 1))) / max_scale)
    return float(min(1.5, pose + roll + scale))


def plan_burst(frames: list, observations: list[FaceObservation],
               risk_weight: float = 0.6, gain_epsilon: float = 0.05) -> tuple[int, int]:
    """Pick (base_frame, donor_frame) for one person over a burst.

    frames: list of Frame; observations: same length, the person's per-frame
    observation (None entries = not visible). Group coherence and multi-person
    planning live in the production planner; the research loop needs the
    single-person choice.
    """
    scores = [score_frame(f, o) if o is not None else -1.0
              for f, o in zip(frames, observations)]
    base = int(np.argmax(scores))
    best, best_v = base, scores[base]
    for i, o in enumerate(observations):
        if o is None:
            continue
        v = scores[i] - risk_weight * risk(observations[base], o)
        if v > best_v:
            best, best_v = i, v
    donor = best if best_v - scores[base] >= gain_epsilon else base
    return base, donor
