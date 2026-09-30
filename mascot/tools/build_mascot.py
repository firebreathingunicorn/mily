#!/usr/bin/env python3
"""Render the marbled cat mascot: master still + per-action RGBA frame sets.

Usage: python3 build_mascot.py [--out DIR]
Outputs (relative to mascot/):
  master/master_still.png            flat off-white background (guide Step 1)
  master/master_still_transparent.png
  frames/<action>/<action>_NNNN.png  transparent RGBA, 576px, 42ms/frame ticks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

import mascot_actions as actions
from mascot_sprite import (BG_FLAT, CANVAS, draw_cat, draw_cat_curled,
                           draw_cat_running, draw_cat_tired)

SCALE = 6          # 96 -> 576 full size
TICKS_PER_POSE = 2 # hold each pose 2 ticks -> ~12fps stepped at 42ms
TICKS = {"running": 1, "chasing": 1, "treadmill": 1}  # workout actions run ~24fps


def render_pose(action, item):
    if isinstance(item, Image.Image):
        return item
    if action == "peeking":
        pose, ox = item
        cat = draw_cat(pose)
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(cat, (ox, 0), cat)
        return canvas
    if action == "running":
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        if item[0] == "sprint":
            _, phase, ox, *rest = item
            speed = rest[0] if rest else 1.0
            cat = draw_cat_running(phase, speed)
        elif item[0] == "tired":
            _, state, ox = item
            cat = draw_cat_tired(state, headband=False).transpose(
                Image.FLIP_LEFT_RIGHT)
        else:
            ox, cat = 0, Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        canvas.paste(cat, (int(ox), 0), cat)
        return canvas
    if action == "sleeping":
        return draw_cat_curled(item)
    return draw_cat(item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = Path(args.out)

    gens = {
        "idle": actions.frames_idle,
        "waving": actions.frames_wave,
        "floating": actions.frames_float,
        "sleeping": actions.frames_sleep,
        "glitching": actions.frames_glitch,
        "peeking": actions.frames_peek,
        "running": actions.frames_running,
        "zapped": actions.frames_zapped,
        "chasing": actions.frames_chasing,
        "reading": actions.frames_reading,
        "eating": actions.frames_eating,
        "grooming": actions.frames_grooming,
        "boing": actions.frames_boing,
        "tv": actions.frames_tv,
        "treadmill": actions.frames_treadmill,
        "flexing": actions.frames_flexing,
        "stretching": actions.frames_stretching,
        "phone": actions.frames_phone,
        "graduating": actions.frames_graduating,
        "basketball": actions.frames_basketball,
        "snowman": actions.frames_snowman,
        "rain": actions.frames_rain,
        "piano": actions.frames_piano,
        "baking": actions.frames_baking,
        "weightlifting": actions.frames_weightlifting,
        "suit": actions.frames_suit,
        "camera": actions.frames_camera,
        "halloween": actions.frames_halloween,
        "baseball": actions.frames_baseball,
        "volleyball": actions.frames_volleyball,
        "firefighter": actions.frames_firefighter,
        "discovery": actions.frames_discovery,
        "plane": actions.frames_plane,
        "noodles": actions.frames_noodles,
        "gaming": actions.frames_gaming,
        "campfire": actions.frames_campfire,
        "dance": actions.frames_dance,
    }

    master = root / "master"
    master.mkdir(parents=True, exist_ok=True)

    still = draw_cat({})
    big = still.resize((CANVAS * SCALE, CANVAS * SCALE), Image.NEAREST)

    flat = Image.new("RGBA", big.size, BG_FLAT)
    flat.paste(big, (0, 0), big)
    flat.convert("RGB").save(master / "master_still.png")

    transparent = Image.new("RGBA", big.size, (0, 0, 0, 0))
    transparent.paste(big, (0, 0), big)
    transparent.save(master / "master_still_transparent.png")

    for action, gen in gens.items():
        out_dir = root / "frames" / action
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()
        items = gen()
        n = 0
        for item in items:
            frame = render_pose(action, item)
            big_frame = frame.resize((CANVAS * SCALE, CANVAS * SCALE), Image.NEAREST)
            for _ in range(TICKS.get(action, TICKS_PER_POSE)):
                big_frame.save(out_dir / f"{action}_{n:04d}.png")
                n += 1
        print(f"{action}: {n} frames")

    print(f"master + frames written under {root}")


if __name__ == "__main__":
    main()
