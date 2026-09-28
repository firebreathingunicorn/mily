#!/usr/bin/env python3
"""Smoothly switch between any two mascot actions: renders a crossfade
transition and writes it as GIF (and optionally APNG).

Example:
  python3 tools/mix.py --from waving --to eating --fade 8 --out mix.gif

The crossfade blends premultiplied alpha (per the guide's resize rule), so
no halo bleeds in mid-fade. The tail of A overlaps the head of B, so the
loop point of either action can sit mid-transition.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TICKS_MS = 80


def load_frames(action):
    d = Path(__file__).resolve().parent.parent / "frames" / action
    return [Image.open(p) for p in sorted(d.glob("*.png"))]


def blend(a, b, alpha):
    """Straight-alpha lerp via premultiplied colors (no halo)."""
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    fa = aa[..., 3:4] / 255 * (1 - alpha)
    fb = bb[..., 3:4] / 255 * alpha
    denom = fa + fb
    rgb = np.where(denom > 1e-4, (aa[..., :3] * fa + bb[..., :3] * fb)
                   / np.maximum(denom, 1e-4), 0)
    out = np.dstack([rgb, (fa + fb)[..., 0] * 255])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def crossfade_into(frames, prev_tail, nxt, fade):
    """Append a fade from prev_tail into nxt's head, then nxt's body."""
    for i in range(1, fade + 1):
        frames.append(blend(prev_tail, nxt[i % len(nxt)], i / (fade + 1)))
    frames += nxt[1:]


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--to", dest="dst")
    ap.add_argument("--chain", help="comma-separated actions, e.g. "
                                    "baking,eating,sleeping — renders one reel")
    ap.add_argument("--fade", type=int, default=8, help="crossfade frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apng", action="store_true", help="also write an .png APNG")
    args = ap.parse_args()

    if args.chain:
        names = [n.strip() for n in args.chain.split(",") if n.strip()]
    else:
        if not (args.src and args.dst):
            print("need --chain or both --from and --to", file=sys.stderr)
            sys.exit(1)
        names = [args.src, args.dst]

    reels = [load_frames(n) for n in names]
    if any(not r for r in reels):
        print("unknown action(s)", file=sys.stderr)
        sys.exit(1)

    frames = list(reels[0])
    for nxt in reels[1:]:
        crossfade_into(frames, frames[-1], nxt, args.fade)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=TICKS_MS, loop=0, disposal=2)
    else:
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=TICKS_MS, loop=0)
    if args.apng:
        apng = out.with_suffix(".png")
        frames[0].save(apng, save_all=True, append_images=frames[1:],
                       duration=TICKS_MS, loop=0)
        print(f"{apng} ({len(frames)} frames)")
    print(f"{out} ({len(frames)} frames, {' -> '.join(names)}, fade {args.fade})")


if __name__ == "__main__":
    main()
