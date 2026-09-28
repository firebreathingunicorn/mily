"""Burst-relative pose refinement (mirror of Swift ``PoseRefiner``).

The raw yaw proxy — (nose.x − eyeMid.x) / inter-ocular — is noisy. Within a
burst there is a much stronger signal: a head turning away foreshortens the
apparent inter-ocular distance as cos(yaw), so the person's burst-wide
maximum of inter-ocular distance is their frontal reference and each frame's
|yaw| ≈ acos(io / ioMax). The sign still comes from the nose-offset proxy.

``refine_burst_yaw`` sets ``FaceObservation.yaw_override`` in place; key_points
then reports the refined pose wherever the observation flows (scoring, risk,
identity adaptation).
"""
from __future__ import annotations

import numpy as np

from .landmarks import key_points

MIN_RATIO = 0.55


def refine_burst_yaw(observations: list) -> None:
    """observations: per-frame FaceObservation (or None) for ONE person."""
    valid = [o for o in observations if o is not None]
    if not valid:
        return
    io_max = max(key_points(o.landmarks).inter_ocular for o in valid)
    if io_max < 1e-6:
        return
    for o in observations:
        if o is None:
            continue
        kp = key_points(o.landmarks)
        ratio = min(1.0, kp.inter_ocular / io_max)
        if ratio >= MIN_RATIO:
            magnitude = float(np.arccos(ratio))
        else:
            magnitude = float(min(abs(kp.yaw_proxy), np.pi / 3))
        sign = -1.0 if kp.yaw_proxy < 0 else (1.0 if kp.yaw_proxy > 0 else 0.0)
        o.yaw_override = sign * magnitude
