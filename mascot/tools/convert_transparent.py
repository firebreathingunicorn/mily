#!/usr/bin/env python3
"""Transparent-asset pipeline (guide Step 3) for the marbled cat mascot.

Subcommands:
  encode  — frames are already transparent RGBA PNGs (our procedural build).
            Audits them, then writes the 5-output pack per action:
            ACTION.gif, ACTION_160.gif, ACTION.png (APNG), ACTION_160.png,
            ACTION.webm (VP9 alpha).
  convert — guide's Higgsfield path: explode hf_*.mp4 clips, chroma-key each
            frame (green screen or flat off-white), dilate/erode the mask 2px
            to eat compression fringe, audit, auto-tighten and reprocess on
            failure, then the same 5-output pack.

Never uses ffmpeg colorkey (fringe) or AI removers (halos on pixel art).
"""

import argparse
import colorsys
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MS_PER_FRAME = 80
SMALL = 160
KEY_COLORS = [
    np.array([242, 243, 238], dtype=np.float32),  # guide's off-white
    np.array([255, 255, 255], dtype=np.float32),  # stray white
    np.array([4, 250, 3], dtype=np.float32),      # green screen
]
AUDIT_TOL = 14


def rgb_to_hsv(arr):
    """arr float32 (H,W,3) in 0..1 -> sat, val arrays."""
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return sat, mx


def key_frame_flat(img, sat_keep=0.25, val_keep=0.35):
    """Flat light background: keep saturated or dark pixels, erode 2px,
    then drop grayish survivors."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    sat, val = rgb_to_hsv(arr)
    keep = (sat > sat_keep) | (val < val_keep)
    keep = ndimage.binary_erosion(keep, iterations=2)
    grayish = (sat < sat_keep) & (val > 0.30)
    keep &= ~grayish
    out = np.asarray(img.convert("RGBA")).copy()
    out[..., 3] = np.where(keep, out[..., 3], 0)
    return Image.fromarray(out)


def key_frame_green(img, sat_min=0.15):
    """Green screen: kill green-dominant pixels or hue 60-180 with any sat,
    then dilate the removal mask 2px to eat the compression fringe."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    sat, val = rgb_to_hsv(arr)
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in arr.reshape(-1, 3)])
    hue = hsv[:, 0].reshape(arr.shape[:2]) * 360
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_dom = (g > r * 1.12) & (g > b * 1.12)
    hue_band = (hue >= 60) & (hue <= 180) & (sat > sat_min)
    remove = green_dom | hue_band
    remove = ndimage.binary_dilation(remove, iterations=2)
    out = np.asarray(img.convert("RGBA")).copy()
    out[..., 3] = np.where(remove, 0, out[..., 3])
    return Image.fromarray(out)


def audit(frames, tol=AUDIT_TOL):
    """Zero suspect (background-colored) opaque pixels allowed."""
    bad = []
    for path, img in frames:
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
        opaque = arr[..., 3] >= 128
        if not opaque.any():
            continue
        rgb = arr[..., :3]
        suspect = np.zeros_like(opaque)
        for kc in KEY_COLORS:
            dist = np.sqrt(((rgb - kc) ** 2).sum(axis=2))
            suspect |= (dist < tol) & opaque
        n = int(suspect.sum())
        if n:
            bad.append((path.name, n))
    return bad


def premultiplied_resize(img, size):
    """Premultiply alpha before scaling (guide's resize trap), then unpremultiply."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    a = arr[..., 3:4] / 255
    prem = arr.copy()
    prem[..., :3] = arr[..., :3] * a
    im = Image.fromarray(prem.astype(np.uint8))
    im = im.resize((size, size), Image.LANCZOS)
    out = np.asarray(im, dtype=np.float32)
    alpha = out[..., 3:4] / 255
    un = out.copy()
    un[..., :3] = np.where(alpha > 1e-4, out[..., :3] / np.maximum(alpha, 1e-4), 0)
    return Image.fromarray(np.clip(un, 0, 255).astype(np.uint8))


def encode_pack(frames, out_dir, name):
    """frames: list of (path, RGBA Image) at full size. Writes 5 outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = [im for _, im in frames]
    w, h = full[0].size

    # APNG full + 160 (8-bit alpha, smooth edges — the good one).
    full[0].save(out_dir / f"{name}.png", save_all=True,
                 append_images=full[1:], duration=MS_PER_FRAME, loop=0)
    small = [premultiplied_resize(im, SMALL) for im in full]
    small[0].save(out_dir / f"{name}_{SMALL}.png", save_all=True,
                  append_images=small[1:], duration=MS_PER_FRAME, loop=0)

    # GIF full + 160 via palettegen (1-bit alpha — last resort format).
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, im in enumerate(full):
            im.save(tdp / f"f_{i:04d}.png")
        gif_filter = ("split[a][b];[a]palettegen=reserve_transparent=on[p];"
                      "[b][p]paletteuse=alpha_threshold=128")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"1000/{MS_PER_FRAME}", "-i", str(tdp / "f_%04d.png"),
            "-vf", gif_filter, str(out_dir / f"{name}.gif"),
        ], check=True)
        tds = tdp / "small"
        tds.mkdir()
        for i, im in enumerate(small):
            im.save(tds / f"f_{i:04d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"1000/{MS_PER_FRAME}", "-i", str(tds / "f_%04d.png"),
            "-vf", gif_filter, str(out_dir / f"{name}_{SMALL}.gif"),
        ], check=True)

    # WebM with real alpha for video editors.
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, im in enumerate(full):
            im.save(tdp / f"f_{i:04d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"1000/{MS_PER_FRAME}", "-i", str(tdp / "f_%04d.png"),
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-lossless", "1",
            str(out_dir / f"{name}.webm"),
        ], check=True)

    print(f"  {name}: gif {_kb(out_dir / f'{name}.gif')}, "
          f"apng {_kb(out_dir / f'{name}.png')}, webm {_kb(out_dir / f'{name}.webm')} + 160px sets")


def _kb(p):
    return f"{p.stat().st_size // 1024}KB"


def load_frames_dir(d, flip=False):
    frames = [(p, Image.open(p)) for p in sorted(d.glob("*.png"))]
    if flip:
        frames = [(p, im.transpose(Image.FLIP_LEFT_RIGHT)) for p, im in frames]
    return frames


def cmd_encode(args):
    src = Path(args.frames)
    frames = load_frames_dir(src, flip=args.flip)
    bad = audit(frames)
    if bad:
        for fname, n in bad[:5]:
            print(f"  AUDIT FAIL {fname}: {n} suspect pixels", file=sys.stderr)
        sys.exit(1)
    encode_pack(frames, Path(args.out), args.name)


def cmd_convert(args):
    inputs = sorted(Path(args.input).glob("hf_*.mp4"))
    if not inputs:
        print(f"no hf_*.mp4 clips in {args.input}", file=sys.stderr)
        sys.exit(1)
    out_root = Path(args.out)
    for clip in inputs:
        name = clip.stem.removeprefix("hf_") or "mascot"
        print(f"converting {clip.name} -> {name}")
        with tempfile.TemporaryDirectory() as td:
            fdir = Path(td) / "raw"
            fdir.mkdir()
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
                str(fdir / "f_%05d.png"),
            ], check=True)
            frames = load_frames_dir(fdir)

            mode = args.mode
            if mode == "auto":
                sample = np.asarray(frames[0][1].convert("RGB"), dtype=np.float32)
                mean = sample.reshape(-1, 3).mean(axis=0)
                mode = "green" if (mean[1] > mean[0] * 1.2 and mean[1] > mean[2] * 1.2) else "flat"
            print(f"  mode={mode}, keying {len(frames)} frames")

            keyed = None
            for attempt in range(3):
                keyed = []
                for path, img in frames:
                    if mode == "green":
                        keyed.append((path, key_frame_green(img, sat_min=0.15 - 0.05 * attempt)))
                    else:
                        keyed.append((path, key_frame_flat(
                            img, sat_keep=0.25 + 0.05 * attempt,
                            val_keep=0.35 - 0.05 * attempt)))
                bad = audit(keyed)
                if not bad:
                    break
                print(f"  audit attempt {attempt + 1} failed on {len(bad)} frames; tightening")
            else:
                for fname, n in bad[:5]:
                    print(f"  AUDIT FAIL {fname}: {n}", file=sys.stderr)
                sys.exit(1)
            encode_pack(keyed, out_root / name, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="encode already-transparent frames")
    e.add_argument("--frames", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--flip", action="store_true",
                   help="mirror every frame (directional one-shots ship both ways)")
    e.set_defaults(fn=cmd_encode)

    c = sub.add_parser("convert", help="convert hf_*.mp4 clips (guide method)")
    c.add_argument("--input", default=str(Path.home() / "Downloads"))
    c.add_argument("--out", required=True)
    c.add_argument("--mode", choices=["auto", "green", "flat"], default="auto")
    c.set_defaults(fn=cmd_convert)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
