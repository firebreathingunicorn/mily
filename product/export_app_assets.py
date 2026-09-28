"""Export burst + Level B results as assets for the Flutter picker app.

Renders one person's capture (6 candidate frames with a growing smile on a
turned head), runs Level B per candidate, and writes JPEGs + manifest.json
into the Flutter app's assets. The app is a pure viewer — the heavy pipeline
stays in Python until the Core ML port exists (plan §Data).

Usage:  PYTHONPATH=.. python3 product/export_app_assets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.pipeline import DonorInput, LevelBConfig, LevelBSwap

OUT = Path(__file__).resolve().parents[1] / "app" / "assets" / "burst"
N = 6


def main() -> None:
    model = SyntheticHeadModel(nu=80, nv=80)
    rng = np.random.default_rng(9)
    cid = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])
    cex_b = np.zeros(6); cex_b[2] = 0.8; cex_b[3] = 0.4   # brows raised, half-blink
    cap = SyntheticCapture(model, size=384, seed=5)

    donor_yaws = list(np.linspace(-14, 22, N))
    base = cap.shoot(cid, cex_b, 24.0, person_id="base")
    donors = []
    for i, y in enumerate(donor_yaws):
        e = np.zeros(6)
        e[1] = 0.15 + 0.6 * i / max(N - 1, 1)             # smile grows across the burst
        donors.append(cap.shoot(cid, e, y, person_id=f"d{i}",
                                exposure=1.0 + rng.uniform(-0.08, 0.08)))
    swap = LevelBSwap(model, LevelBConfig(mode="pinhole"))
    base_obs = FaceObservation("p0", base.obs_landmarks)

    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, d in enumerate(donors):
        res = swap.run(base.frame, base_obs,
                       [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))])
        level = "rejected" if res.method == "rejected" else res.method
        Image.fromarray((np.clip(d.frame.rgb, 0, 1) * 255).astype(np.uint8)).save(
            OUT / f"frame{i}_original.jpg", quality=92)
        img = base.rgb if res.method == "rejected" else res.image
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
            OUT / f"frame{i}_result.jpg", quality=92)
        frames.append({"index": i, "level": level,
                       "coverage": round(float(res.coverage), 3)})
        print(f"frame {i}: level {level:8s} coverage {res.coverage:.2f}")

    manifest = {
        "person": "Maya",
        "base": {"original": _jpeg_uri(base.frame.rgb)},
        "frames": [{**f,
                    "original": f"assets/burst/frame{f['index']}_original.jpg",
                    "result": f"assets/burst/frame{f['index']}_result.jpg"}
                   for f in frames],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported {len(frames)} frames -> {OUT}")


def _jpeg_uri(img: np.ndarray) -> str:
    return ""  # base photo shown from a dedicated asset in a later step


if __name__ == "__main__":
    main()
