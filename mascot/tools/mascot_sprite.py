"""Pixel-art marbled cat mascot: drawing engine + action frame generators.

Logical canvas is 96x96 pixels, drawn with hard integer-pixel primitives and
upscaled nearest-neighbor at export time. All poses are deterministic so the
character stays identical across actions.
"""

import numpy as np
from PIL import Image, ImageDraw

CANVAS = 96

# Palette: golden coat + black marble (ginger marbled tabby), cream accents.
BODY = (232, 180, 88, 255)        # warm golden-yellow coat
BODY_LIGHT = (247, 208, 124, 255) # top-left highlight sheen
BODY_DARK = (198, 144, 60, 255)   # underside / seam shading
GROUND_SHADOW = (20, 18, 16, 56)  # soft contact shadow under the cat
MARBLE = (35, 30, 26, 255)        # black swirl blotches + tail rings
CREAM = (245, 238, 218, 255)      # muzzle, chest, paws, whiskers
EYE = (24, 22, 20, 255)           # black square eyes
NOSE = (205, 110, 96, 255)        # tiny 2px rose nose
WHISKER = (247, 240, 218, 255)    # near-white whiskers; kept outside the
                                  # audit's background-match tolerance
BG_FLAT = (242, 243, 238, 255)    # guide's off-white master-still background


def new_canvas():
    return Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))


def _bezier(p0, p1, p2, p3, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        pts.append((x, y))
    return pts


def draw_tail(d, sway=0.0):
    """Long fluffy tail: rises from behind the body's right, charcoal ringed.
    Non-ring segments get a 1px top sheen so the fur catches the light."""
    p0 = (66, 74)
    p1 = (74 + sway * 0.4, 70)
    p2 = (78 + sway, 52)
    p3 = (70 + sway * 1.3, 40)
    for i, (x, y) in enumerate(_bezier(p0, p1, p2, p3, n=28)):
        t = i / 28
        r = 3.4 if t < 0.25 else (3.0 if t < 0.8 else 2.6)
        ring = (0.26 <= t <= 0.42) or (0.58 <= t <= 0.74) or t > 0.9
        color = MARBLE if ring else BODY
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        if not ring:
            d.rectangle([int(x - 1), int(y - r), int(x), int(y - r)],
                        fill=BODY_LIGHT)


def draw_blob(d, cx, cy, specs, color):
    """Cluster of overlapping circles forming one irregular blotch."""
    for dx, dy, r in specs:
        d.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r], fill=color)


HEAD_MARBLE = [(-6, -4, 2.2), (-3, -6, 2.0), (-7, -1, 1.8), (-2, -3, 1.7)]
BODY_MARBLE = [(1, 1, 2.6), (4, -1, 2.0), (-1, 5, 2.0), (-4, 3, 1.7), (0, 8, 1.8)]


def draw_ear(d, cx, tip_dx=0):
    """One triangular ear with rounded marbled tip, base at head top y=23."""
    d.polygon([(cx - 5, 23), (cx + tip_dx, 13), (cx + 5, 23)], fill=BODY)
    d.ellipse([cx + tip_dx - 2, 13, cx + tip_dx + 2, 17], fill=MARBLE)
    d.polygon([(cx - 2, 22), (cx + round(tip_dx * 0.6), 17), (cx + 2, 22)], fill=BODY_DARK)


def _eye(d, x0, y0, w=4, h=4, gx=0, gy=0):
    """Friendly rounded eye: solid with corners cut, wide 2px glint that
    shifts with the gaze (gx/gy clamped inside the eye)."""
    d.rectangle([x0 + 1, y0, x0 + w - 2, y0 + h - 1], fill=EYE)
    d.rectangle([x0, y0 + 1, x0 + w - 1, y0 + h - 2], fill=EYE)
    cxp = x0 + 1 + max(-1, min(1, gx))
    cyp = y0 + 1 + max(0, min(1, gy))
    d.rectangle([cxp, cyp, cxp + 1, cyp], fill=WHISKER)


HEART = (222, 84, 104, 255)  # bright love-pink, pops on the gold coat


def _heart_eye(d, x, y):
    """4x4 pixel heart for the love-struck eyes."""
    d.rectangle([x, y, x, y], fill=HEART)
    d.rectangle([x + 3, y, x + 3, y], fill=HEART)
    d.rectangle([x, y + 1, x + 3, y + 1], fill=HEART)
    d.rectangle([x + 1, y + 2, x + 2, y + 2], fill=HEART)
    d.rectangle([x + 2, y + 3, x + 2, y + 3], fill=HEART)


def draw_cat(pose):
    """Sitting cat facing the camera. pose keys (all optional):
    head_dy, body_dy, sway (tail tip px), ear_r_dx, blink (bool),
    paw_raise (None or wag int), paw_side ('l'|'r'), mouth (bool), squash (0/1)
    """
    p = {
        "head_dy": 0, "body_dy": 0, "sway": 0.0, "ear_r_dx": 0,
        "blink": False, "paw_raise": None, "paw_side": "r",
        "mouth": False, "squash": 0, "shock": False,
        "gaze": "center", "happy": False, "whisker_dy": 0,
        "paw_dx": 0, "tail_up": False, "both_paws": False,
        "heart_eyes": False, "headband": False, "shadow": True,
    }
    p.update(pose)

    im = new_canvas()
    d = ImageDraw.Draw(im)
    hy, by = p["head_dy"], p["body_dy"]
    sq = p["squash"]

    # Soft contact shadow: grounds the cat unless it's airborne.
    if p["shadow"]:
        d.ellipse([27, 77, 69, 83], fill=GROUND_SHADOW)

    if p["tail_up"]:
        # excited vertical tail, rings near the tip
        for i, (x, y) in enumerate(_bezier((60, 58), (66, 48), (67, 34), (62, 18), n=18)):
            t = i / 18
            r = 2.8 if t < 0.7 else 2.3
            ring = (0.42 <= t <= 0.58) or (0.72 <= t <= 0.84) or t > 0.93
            d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
    else:
        draw_tail(d, p["sway"])

    # Body: sitting blob, rounded base, subtle haunches (kept slim).
    top_w, base_w = 22, 32
    body_top, body_bot = 45 + by + sq, 78 + by
    bw_top = top_w + sq * 2
    d.rounded_rectangle([48 - bw_top / 2, body_top, 48 + bw_top / 2, body_bot], radius=9, fill=BODY)
    d.rounded_rectangle([48 - base_w / 2, 62 + by, 48 + base_w / 2, body_bot], radius=10, fill=BODY)
    d.ellipse([30, 62 + by, 42, 78 + by], fill=BODY)
    d.ellipse([54, 62 + by, 66, 78 + by], fill=BODY)
    # Top-left light: sheen along the crown of the body and haunch tops.
    d.rectangle([42, body_top + 1, 54, body_top + 2], fill=BODY_LIGHT)
    d.rectangle([28, 62 + by, 38, 63 + by], fill=BODY_LIGHT)
    d.rectangle([58, 62 + by, 68, 63 + by], fill=BODY_LIGHT)
    d.rectangle([33, body_bot - 2, 62, body_bot - 1], fill=BODY_DARK)
    d.rectangle([41, 52 + by, 42, 74 + by], fill=BODY_DARK)
    d.rectangle([55, 52 + by, 56, 74 + by], fill=BODY_DARK)

    # Chest patch.
    d.ellipse([43, 52 + by, 53, 72 + by], fill=CREAM)

    # Front paws (or one/both raised, waving). paw_dx = dangling leg drift.
    dx = p["paw_dx"]
    if p["paw_raise"] is None and not p["both_paws"]:
        d.rounded_rectangle([37 + dx, 72 + by, 44 + dx, 77 + by], radius=3, fill=CREAM)
        d.rounded_rectangle([52 - dx, 72 + by, 59 - dx, 77 + by], radius=3, fill=CREAM)

    # Head.
    d.rounded_rectangle([35, 21 + hy, 61, 47 + hy], radius=9, fill=BODY)
    d.ellipse([35, 38 + hy, 61, 50 + hy], fill=BODY)  # cheeks/jaw
    # Light from the top-left: sheen on the crown, fur jags on the cheeks.
    d.rectangle([38, 22 + hy, 58, 23 + hy], fill=BODY_LIGHT)
    for fx, fy in [(34, 43), (33, 46), (62, 43), (63, 46)]:
        d.rectangle([fx, fy + hy, fx, fy + hy], fill=BODY)
    draw_ear(d, 41)
    draw_ear(d, 55, tip_dx=p["ear_r_dx"])

    # Raised arm(s) draw after the head so they read as attached.
    if p["paw_raise"] is not None or p["both_paws"]:
        wag = p["paw_raise"] or 0
        if p["paw_side"] == "r" or p["both_paws"]:
            d.line([(58, 52 + by), (65, 38 + hy)], fill=BODY, width=5)
            d.rounded_rectangle([61 + wag, 26 + hy + wag, 68 + wag, 33 + hy + wag], radius=3, fill=CREAM)
        if p["paw_side"] == "l" or p["both_paws"]:
            d.line([(38, 52 + by), (32, 38 + hy)], fill=BODY, width=5)
            d.rounded_rectangle([27 + wag, 26 + hy + wag, 34 + wag, 33 + hy + wag], radius=3, fill=CREAM)

    # Marble blotches (anchored so they ride with the part they live on).
    draw_blob(d, 40, 27 + hy, HEAD_MARBLE, MARBLE)
    draw_blob(d, 58, 62 + by, BODY_MARBLE, MARBLE)
    d.ellipse([56, 46 + by, 61, 52 + by], fill=MARBLE)  # shoulder patch

    # Sport headband: red band under the ear bases with tie tails on the right.
    if p["headband"]:
        d.rectangle([36, 24 + hy, 60, 27 + hy], fill=BOOK_RED)
        d.rectangle([60, 24 + hy, 66, 26 + hy], fill=BOOK_RED)
        d.rectangle([63, 26 + hy, 68, 29 + hy], fill=BOOK_RED)

    # Face: heart eyes, shock squints, happy ^ eyes, blinks, or soft rounded
    # eyes whose glint moves so the cat can look around.
    if p["heart_eyes"]:
        _heart_eye(d, 40, 29 + hy)
        _heart_eye(d, 52, 29 + hy)
    elif p["shock"]:
        _eye(d, 39, 28, w=6, h=6)
        _eye(d, 50, 28, w=6, h=6)
        d.rectangle([41, 31, 42, 32], fill=CREAM)  # wide startled glints
        d.rectangle([52, 31, 53, 32], fill=CREAM)
    elif p["happy"]:
        for ex in (40, 51):
            d.rectangle([ex, 32 + hy, ex, 32 + hy], fill=EYE)
            d.rectangle([ex + 1, 31 + hy, ex + 1, 31 + hy], fill=EYE)
            d.rectangle([ex + 2, 32 + hy, ex + 2, 32 + hy], fill=EYE)
    elif p["blink"]:
        d.rectangle([40, 33 + hy, 43, 34 + hy], fill=EYE)
        d.rectangle([51, 33 + hy, 54, 34 + hy], fill=EYE)
    else:
        g = p["gaze"]
        gx = {"left": -1, "right": 1}.get(g, 0)
        gy = {"up": -1, "down": 1}.get(g, 0)
        _eye(d, 40, 29 + hy, gx=gx, gy=max(0, gy))
        _eye(d, 52, 29 + hy, gx=gx, gy=max(0, gy))
    d.rectangle([46, 39 + hy, 47, 40 + hy], fill=NOSE)
    d.rectangle([43, 41 + hy, 50, 42 + hy], fill=CREAM)
    if p["mouth"]:
        d.rectangle([46, 43 + hy, 47, 44 + hy], fill=EYE)
    # Whiskers: three per side; whisker_dy droops or lifts the outer ends.
    wd = p["whisker_dy"]
    for y0, x_out in ((36, 27), (39, 25), (42, 27)):
        mid = (x_out + 34) // 2
        if wd:
            d.rectangle([x_out, y0 + hy, mid, y0 + hy], fill=WHISKER)
            d.rectangle([mid, y0 + hy + wd, 34, y0 + hy + wd], fill=WHISKER)
            d.rectangle([61, y0 + hy, 61 + (mid - x_out), y0 + hy], fill=WHISKER)
            d.rectangle([61 + (mid - x_out), y0 + hy + wd, 61 + (34 - x_out), y0 + hy + wd],
                        fill=WHISKER)
        else:
            d.rectangle([x_out, y0 + hy, 34, y0 + hy], fill=WHISKER)
            d.rectangle([61, y0 + hy, 61 + (34 - x_out), y0 + hy], fill=WHISKER)
    return im


def draw_cat_running(phase, speed=1.0, tongue=0, sweat=False, headband=False):
    """Side-view gallop, facing right. SIX phases with big silhouette
    changes so the legs visibly cycle instead of the cat just sliding:
    0 reach (front paws land) · 1 stretch (full split, low) · 2 push (rear
    kicks back) · 3 gather (legs sweep under) · 4 tuck (airborne, folded,
    high) · 5 fall (legs drop, nose dips). speed scales the speed lines.
    tongue: 0 none, 1 hanging, 2 long (exhausted). sweat: drop at the ear."""
    p = phase % 6
    bounce = {0: 1, 1: 3, 2: 2, 3: 0, 4: -4, 5: -2}[p]
    line_len = {0: 1.0, 1: 1.2, 2: 1.1, 3: 0.9, 4: 0.7, 5: 0.8}[p]

    im = new_canvas()
    d = ImageDraw.Draw(im)

    # Speed lines trailing behind the tail, pulsing with the stride.
    if speed > 0:
        base_y = 48 + bounce
        ln = speed * line_len
        for i, (l, dy) in enumerate([(13, -6), (19, 0), (11, 6)]):
            x1 = 28 - int(l * ln)
            y = base_y + dy
            color = MARBLE if i == 1 else BODY_DARK
            d.rectangle([x1, y, 26, y], fill=color)

    # Dust kicked up at ground contact.
    if p in (0, 2):
        for cx, cy, r in [(24, 66, 1.8), (18, 70, 1.4), (29, 69, 1.2)]:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BODY_DARK)

    by = bounce

    # Tail streaming behind, lifting on the tuck.
    wav = {0: 2, 1: 0, 2: -2, 3: 1, 4: 6, 5: 3}[p]
    for i, (x, y) in enumerate(_bezier((34, 52 + by), (24, 50 + by),
                                       (14, 44 + wav * 0.4), (10, 32 + wav), n=18)):
        t = i / 18
        r = 2.8 if t < 0.7 else 2.3
        ring = (0.3 <= t <= 0.48) or (0.62 <= t <= 0.8) or t > 0.9
        d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)

    # Body: stretched in the split, compact and raised in the tuck.
    x0, x1 = {0: (30, 68), 1: (26, 72), 2: (28, 70),
              3: (32, 66), 4: (36, 62), 5: (32, 66)}[p]
    d.rounded_rectangle([x0, 42 + by, x1, 60 + by], radius=9, fill=BODY)
    d.rectangle([x0 + 4, 57 + by, x1 - 4, 58 + by], fill=BODY_DARK)
    d.rectangle([x0 + 5, 43 + by, x1 - 8, 44 + by], fill=BODY_LIGHT)
    draw_blob(d, 44, 50 + by, [(0, 0, 2.4), (3, -2, 1.9), (-2, 3, 1.8)], MARBLE)
    d.ellipse([60, 44 + by, 66, 52 + by], fill=MARBLE)

    # Legs: wide swings so the cycle reads at speed.
    def leg(x_from, y_from, x_to, y_to):
        d.line([(x_from, y_from + by), (x_to, y_to + by)], fill=BODY, width=5)
        d.rounded_rectangle([x_to - 2, y_to + by - 2, x_to + 4, y_to + by + 2],
                            radius=2, fill=CREAM)

    legs = {
        0: [(62, 54, 72, 67), (58, 54, 68, 68), (42, 54, 34, 67), (45, 54, 38, 68)],
        1: [(64, 54, 76, 64), (60, 54, 73, 67), (40, 54, 24, 65), (43, 54, 28, 67)],
        2: [(62, 54, 68, 62), (59, 54, 64, 63), (42, 54, 26, 64), (45, 54, 30, 67)],
        3: [(60, 54, 62, 65), (57, 54, 58, 66), (44, 54, 42, 66), (47, 54, 45, 67)],
        4: [(58, 54, 59, 62), (56, 54, 55, 62), (45, 54, 43, 62), (48, 54, 46, 62)],
        5: [(60, 54, 66, 66), (57, 54, 62, 67), (43, 54, 38, 66), (46, 54, 42, 67)],
    }
    for x_from, y_from, x_to, y_to in legs[p]:
        leg(x_from, y_from, x_to, y_to)

    # Head nods down as the front paws land, lifts in the tuck.
    hy = {0: 1, 1: 1, 2: 0, 3: -1, 4: -2, 5: -1}[p]
    d.rounded_rectangle([58, 30 + hy, 80, 48 + hy], radius=8, fill=BODY)
    d.ellipse([70, 40 + hy, 82, 50 + hy], fill=BODY)  # muzzle/jaw mass

    # Ears swept back with the speed.
    for cx, dy in [(66, 0), (73, -1)]:
        d.polygon([(cx + 4, 33 + hy + dy), (cx - 2, 24 + hy + dy), (cx + 6, 35 + hy + dy)],
                  fill=BODY)
        d.ellipse([cx - 3, 25 + hy + dy, cx + 1, 29 + hy + dy], fill=MARBLE)

    # Face: one soft eye (side view), cream muzzle, nose at the tip.
    _eye(d, 66, 36 + hy, w=4, h=4, gx=1)
    d.rectangle([74, 43 + hy, 80, 46 + hy], fill=CREAM)
    d.rectangle([79, 41 + hy, 80, 42 + hy], fill=NOSE)
    if p in (3, 4):
        d.rectangle([75, 46 + hy, 76, 47 + hy], fill=EYE)  # open mouth, wind in face

    # Whiskers swept back from the muzzle.
    for dy in (-2, 1):
        d.rectangle([82, 43 + hy + dy, 88, 43 + hy + dy], fill=WHISKER)
    # Sport headband with tails trailing behind (treadmill only).
    if headband:
        d.rectangle([58, 33 + hy, 80, 36 + hy], fill=BOOK_RED)
        d.rectangle([50, 33 + hy, 57, 35 + hy], fill=BOOK_RED)
        d.rectangle([46, 35 + hy, 52, 37 + hy], fill=BOOK_RED)
    # Tongue hangs from the mouth, riding with the head bob; longer when
    # exhausted, tip drooping.
    if tongue:
        tl = 5 if tongue == 2 else 4
        d.rounded_rectangle([76, 45 + hy, 78, 45 + hy + tl], radius=1, fill=NOSE)
        tip_dy = 1 if tongue == 2 else 0
        d.rectangle([77, 46 + hy + tl, 78, 46 + hy + tl + tip_dy], fill=NOSE)
    if sweat:
        d.rectangle([83, 28 + hy, 84, 31 + hy], fill=SWEAT)
        d.rectangle([83, 27 + hy, 83, 27 + hy], fill=SWEAT)
    return im


def draw_cat_curled(pose):
    """Curled-up sleeping cat. pose keys: breath (0/1), zz = [(x, y, size)]."""
    p = {"breath": 0, "zz": []}
    p.update(pose)
    b = p["breath"]

    im = new_canvas()
    d = ImageDraw.Draw(im)

    # Curled body + haunch.
    d.rounded_rectangle([28, 50 + b, 68, 78], radius=14, fill=BODY)
    d.ellipse([48, 48 + b, 72, 74], fill=BODY)
    d.rectangle([34, 51 + b, 62, 52 + b], fill=BODY_LIGHT)
    d.rectangle([32, 75, 66, 76], fill=BODY_DARK)

    # Head resting on top of the coil, left side.
    d.rounded_rectangle([30, 38 + b, 56, 60 + b], radius=8, fill=BODY)
    d.ellipse([30, 50 + b, 56, 62 + b], fill=BODY)
    d.rectangle([36, 59 + b, 50, 60 + b], fill=BODY_DARK)  # seam against coil
    draw_ear_curled(d, 37, b)
    draw_ear_curled(d, 49, b)

    draw_blob(d, 36, 45 + b, HEAD_MARBLE, MARBLE)
    draw_blob(d, 58, 58, [(4, 0, 2.6), (7, 3, 2.2), (0, 4, 2.0)], MARBLE)

    # Closed eyes: two dark slits, tiny nose.
    d.rectangle([36, 50 + b, 40, 51 + b], fill=EYE)
    d.rectangle([46, 50 + b, 50, 51 + b], fill=EYE)
    d.rectangle([42, 54 + b, 43, 55 + b], fill=NOSE)

    # Whiskers drooping over the coil.
    d.rectangle([24, 52 + b, 30, 52 + b], fill=WHISKER)
    d.rectangle([25, 55 + b, 30, 55 + b], fill=WHISKER)
    d.rectangle([56, 52 + b, 63, 52 + b], fill=WHISKER)
    d.rectangle([56, 55 + b, 62, 55 + b], fill=WHISKER)

    # Tail wrapped around the front of the coil: dark underlay gives the
    # wrap definition against the body, rings ride on top.
    wrap = _bezier((64, 68), (66, 76), (50, 79), (32, 72), n=20)
    for x, y in wrap:
        d.ellipse([x - 3.4, y - 3.4, x + 3.4, y + 3.4], fill=BODY_DARK)
    for i, (x, y) in enumerate(wrap):
        t = i / 20
        r = 2.6 if t < 0.85 else 2.2
        ring = (0.25 <= t <= 0.40) or (0.55 <= t <= 0.70)
        d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)

    for zx, zy, zsize in p["zz"]:
        draw_z(d, zx, zy, zsize)
    return im


def draw_ear_curled(d, cx, b):
    d.polygon([(cx - 4, 41 + b), (cx, 32 + b), (cx + 4, 41 + b)], fill=BODY)
    d.ellipse([cx - 2, 32 + b, cx + 2, 36 + b], fill=MARBLE)


def draw_z(d, x, y, size):
    """Stepped pixel Z glyph: top bar, diagonal steps, bottom bar.
    Mid-tone so it reads on both light and dark backgrounds."""
    for rect in [(x, y, x + size - 1, y),
                 (x, y + size - 1, x + size - 1, y + size - 1)]:
        d.rectangle(list(rect), fill=BODY_DARK)
    for i in range(1, size - 1):
        d.rectangle([x + size - 1 - i, y + i, x + size - 1 - i, y + i], fill=BODY_DARK)


def draw_bolt(d, strength=1.0):
    """Pixel lightning bolt striking down to the cat's raised paw."""
    pts = [(75, 3), (68, 14), (72, 14), (62, 29)]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=MARBLE, width=6)
        if strength >= 1.0:
            d.line([pts[i], pts[i + 1]], fill=CREAM, width=3)  # white-hot core
        elif strength >= 0.7:
            d.line([pts[i], pts[i + 1]], fill=CREAM, width=2)
        else:
            d.line([pts[i], pts[i + 1]], fill=BODY_DARK, width=2)
    if strength >= 1.0:
        # spark branches
        d.line([(70, 16), (65, 11)], fill=CREAM, width=1)
        d.line([(64, 24), (69, 21)], fill=CREAM, width=1)


def draw_sparks(d):
    """Stray zap sparks around the head."""
    for x, y in [(58, 16), (63, 20), (30, 18), (34, 14), (60, 40), (28, 34)]:
        d.rectangle([x, y, x + 1, y + 1], fill=CREAM)


def draw_fur_spikes(d, hy=0, by=0):
    """Fur standing on end while zapped."""
    for x0, x1, tip_y in [(38, 42, 15), (46, 50, 13), (54, 58, 16),
                          (33, 38, 28), (58, 63, 28),
                          (31, 36, 50), (61, 66, 50)]:
        d.polygon([(x0, (tip_y + 8) + hy), ((x0 + x1) // 2, tip_y + hy), (x1, (tip_y + 8) + hy)],
                  fill=BODY)


def draw_smoke_wisp(d):
    """Little dizzy smoke curl after the shock."""
    for x, y in [(66, 20), (68, 16), (66, 12), (69, 8), (67, 4)]:
        d.rectangle([x, y, x + 1, y + 1], fill=BODY_DARK)


def draw_cat_zapped(state):
    """The zap story: reach -> tease -> zap_a/b/c (shocked) -> dazed."""
    poses = {
        "reach": {"paw_raise": 0, "ear_r_dx": 1, "head_dy": -1, "gaze": "up"},
        "tease": {"paw_raise": 1, "ear_r_dx": 1, "head_dy": -1, "mouth": True,
                  "gaze": "up"},
        "zap_a": {"paw_raise": 1, "shock": True, "head_dy": -1},
        "zap_b": {"paw_raise": 1, "shock": True, "head_dy": 0, "squash": 1},
        "zap_c": {"paw_raise": 1, "shock": True, "head_dy": -1, "mouth": True},
        "dazed": {"blink": True, "head_dy": 1, "ear_r_dx": -2},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    if state == "reach":
        draw_bolt(d, strength=0.4)
    elif state == "tease":
        draw_bolt(d, strength=0.7)
        draw_fur_spikes(d)
    elif state.startswith("zap"):
        draw_bolt(d)
        draw_sparks(d)
        draw_fur_spikes(d)
    else:  # dazed
        draw_smoke_wisp(d)
    return im


def draw_cat_groom(state):
    """Paw-lick groom: raises a paw, pink tongue darts, wipes the cheek,
    perks up, settles. States: raise/lick1/lick2/wipe/perk/settle."""
    poses = {
        "raise": {"paw_side": "l", "paw_raise": 0, "gaze": "down"},
        "lick1": {"paw_side": "l", "paw_raise": -1, "head_dy": 1, "happy": True,
                  "ear_r_dx": 1},
        "lick2": {"paw_side": "l", "paw_raise": 1, "head_dy": 1, "happy": True},
        "wipe": {"paw_side": "l", "paw_raise": 2, "head_dy": 1, "happy": True,
                 "whisker_dy": 1},
        "perk": {"ear_r_dx": 1, "gaze": "right", "sway": 2.0},
        "settle": {"blink": True, "sway": -2.0},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    if state.startswith("lick"):
        # pink tongue darting out under the muzzle
        d.rectangle([44, 43, 45, 44], fill=NOSE)
        d.rectangle([44, 45, 45, 45], fill=NOSE)
    return im


def draw_cat_boing(state):
    """Excited bounce: crouch -> spring up (both paws and tail flying) ->
    hang time -> land with a squash -> shake it off. States:
    crouch/up/hang/land/shake_a/shake_b/settle."""
    poses = {
        "crouch": {"squash": 2, "gaze": "up", "tail_up": True, "ear_r_dx": 1,
                   "headband": True},
        "up": {"both_paws": True, "tail_up": True, "happy": True, "gaze": "up",
               "head_dy": -1, "ear_r_dx": 1, "headband": True, "shadow": False},
        "hang": {"both_paws": True, "tail_up": True, "happy": True, "gaze": "up",
                 "head_dy": -2, "ear_r_dx": 1, "headband": True, "shadow": False},
        "land": {"squash": 2, "blink": True, "sway": 2.5, "headband": True},
        "shake_a": {"whisker_dy": -1, "ear_r_dx": -2, "sway": 2.5,
                    "gaze": "right", "headband": True},
        "shake_b": {"whisker_dy": 1, "ear_r_dx": 2, "sway": -2.5,
                    "gaze": "left", "headband": True},
        "settle": {"gaze": "center", "happy": True, "sway": 1.0,
                   "headband": True},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    if state == "crouch":
        for cx in (30, 65):
            d.ellipse([cx, 76, cx + 4, 79], fill=BODY_DARK)  # gathering dust
    if state == "hang":
        d.rectangle([30, 26, 31, 27], fill=CREAM)   # confetti flecks of joy
        d.rectangle([63, 22, 64, 23], fill=CREAM)
        d.rectangle([24, 34, 25, 35], fill=CREAM)
    return im


def draw_mouse(d, x, y=64):
    """Toy mouse scurrying right: gray body, ear, bead eye, curly tail."""
    fur = (150, 140, 128, 255)
    d.ellipse([x, y, x + 12, y + 7], fill=fur)
    d.ellipse([x + 8, y - 2, x + 11, y + 1], fill=fur)
    d.rectangle([x + 9, y - 1, x + 9, y], fill=BODY_DARK)   # inner ear
    d.rectangle([x + 9, y + 2, x + 9, y + 2], fill=EYE)
    d.rectangle([x + 12, y + 3, x + 12, y + 3], fill=NOSE)
    for px, py in _bezier((x, y + 3), (x - 4, y + 6), (x - 7, y + 2), (x - 9, y - 2), n=8):
        d.rectangle([round(px), round(py), round(px), round(py)], fill=fur)


BOOK_RED = (172, 66, 56, 255)
PAGE = (247, 240, 218, 255)
PAGE_USED = (224, 213, 189, 255)


def draw_cat_reading(**pose):
    """Sitting cat with a small red hardcover held close to its face — the
    red pops against the gold fur where the old white book blended into the
    cream chest. pose extras: look_up, blink, turn (0/1/2 page-flip)."""
    base = {"head_dy": 1, "sway": -1.0, "gaze": "down"}
    base.update(pose)
    look_up = base.pop("look_up", False)
    turn = base.pop("turn", 0)
    if look_up:
        base["head_dy"] = 0
        base["mouth"] = True
        base["gaze"] = "center"
    im = draw_cat(base)
    d = ImageDraw.Draw(im)

    # Red hardcover, open: cover rim around both pages, dark spine seam.
    d.rounded_rectangle([37, 47, 59, 63], radius=2, fill=BOOK_RED)
    d.rectangle([39, 49, 47, 61], fill=PAGE if turn != 2 else PAGE_USED)
    d.rectangle([49, 49, 57, 61], fill=PAGE if turn == 0 else PAGE_USED)
    d.rectangle([47, 49, 49, 61], fill=MARBLE)  # spine seam
    # text lines
    for y in (51, 54, 57, 60):
        d.rectangle([41, y, 45, y], fill=BODY_DARK)
        d.rectangle([51, y, 55, y], fill=BODY_DARK)
    if turn:
        # page mid-flip, sweeping from the right page over the spine
        tip = [(44, 44), (38, 47)][turn - 1]
        d.polygon([(56, 49), (49, 49), tip], fill=PAGE)
        d.rectangle([49, 49, 57, 61], fill=PAGE_USED)
        for y in (51, 54, 57):
            d.rectangle([51, y, 55, y], fill=BODY_DARK)
    # paws wrapped over the cover's outer edges
    d.rounded_rectangle([33, 53, 40, 60], radius=2, fill=CREAM)
    d.rounded_rectangle([56, 53, 63, 60], radius=2, fill=CREAM)
    return im


BUN = (232, 188, 118, 255)
PATTY = (176, 116, 50, 255)
LETTUCE = (140, 175, 80, 255)


def _sandwich(d, x, y):
    """Chicken sandwich: bun, lettuce, fried chicken patty, bun, sesame.
    Dark outline so it pops against the cat's fur."""
    d.rounded_rectangle([x - 1, y - 1, x + 15, y + 11], radius=3, fill=MARBLE)
    d.rounded_rectangle([x, y + 7, x + 14, y + 10], radius=2, fill=BUN)
    d.rectangle([x, y + 6, x + 14, y + 6], fill=LETTUCE)
    d.rectangle([x + 1, y + 3, x + 13, y + 5], fill=PATTY)
    d.rounded_rectangle([x, y, x + 14, y + 3], radius=2, fill=BUN)
    for sx in (x + 3, x + 7, x + 11):
        d.rectangle([sx, y + 1, sx + 1, y + 1], fill=(252, 244, 222, 255))


def _bite(im, rects):
    """Erase alpha in the given rects — cartoon bite notches."""
    import numpy as np
    arr = np.array(im)
    for x0, y0, x1, y1 in rects:
        arr[y0:y1, x0:x1, 3] = 0
    return Image.fromarray(arr)


def _heart(d, x=44, y=4):
    for x0, y0, x1, y1 in [(x, y, x + 1, y), (x + 4, y, x + 5, y),
                           (x, y + 1, x + 5, y + 1), (x + 1, y + 2, x + 4, y + 2),
                           (x + 2, y + 3, x + 3, y + 3), (x + 2, y + 4, x + 3, y + 4)]:
        d.rectangle([x0, y0, x1, y1], fill=NOSE)


SANDWICH_DOWN = (41, 50)   # held at the chest between bites
SANDWICH_UP = (41, 36)     # raised to the mouth for a chomp


def draw_cat_eating(stage, phase="hold"):
    """Chicken sandwich held in both paws, chomped AT THE MOUTH so the bites
    connect. stage: 'look' | 0..3 bites taken | 'gone'. phase: 'hold'
    (sandwich at chest, eyes on it) | 'chomp' (raised, mouth wide open) |
    'chew' (lowered, working on the bite)."""
    if stage == "gone":
        im = draw_cat({"both_paws": True, "heart_eyes": True, "sway": 2.0})
        d = ImageDraw.Draw(im)
        d.rectangle([43, 74, 43, 74], fill=BUN)
        d.rectangle([48, 75, 48, 75], fill=BUN)
        d.rectangle([52, 73, 52, 73], fill=PATTY)
        _heart(d)
        return im

    sx, sy = SANDWICH_UP if phase == "chomp" else SANDWICH_DOWN
    chomping = phase == "chomp"
    pose = {"gaze": "down", "head_dy": 0 if chomping else 1,
            "happy": chomping, "mouth": phase == "chew"}
    im = draw_cat(pose)
    d = ImageDraw.Draw(im)
    if chomping:
        d.rounded_rectangle([45, 41, 48, 45], radius=1, fill=EYE)  # mouth wide

    _sandwich(d, sx, sy)
    notches = {
        1: [(sx + 10, sy - 1, sx + 16, sy + 5)],
        2: [(sx + 6, sy - 1, sx + 16, sy + 7)],
        3: [(sx + 3, sy - 1, sx + 16, sy + 12)],  # final bite clears it all
    }
    if stage in notches:
        im = _bite(im, notches[stage])
        d = ImageDraw.Draw(im)

    # Both paws gripping the sandwich's sides.
    py = sy + 7
    d.rounded_rectangle([sx - 4, py, sx + 1, py + 5], radius=2, fill=CREAM)
    d.rounded_rectangle([sx + 13, py, sx + 18, py + 5], radius=2, fill=CREAM)
    return im


SWEAT = (150, 202, 230, 255)  # one soft-blue accent, only for the sweat drop


def draw_cat_tired(state, headband=True):
    """Worn-out side view, facing right (the caller mirrors it for leftward
    walks): slouched body, folded droopy ears, closed eyes, tongue lolling,
    sweat drop sliding down. state: 'walk0'/'walk1' shuffle steps,
    'pant_a'/'pant_b' huffing stop with head bob."""
    pant = state.startswith("pant")
    by = 1 if state == "pant_b" else 0

    im = new_canvas()
    d = ImageDraw.Draw(im)

    # Tail dragging on the ground behind.
    for i, (x, y) in enumerate(_bezier((34, 54 + by), (24, 57 + by),
                                       (16, 59), (12, 63), n=16)):
        t = i / 16
        r = 2.6 if t < 0.8 else 2.2
        ring = (0.3 <= t <= 0.45) or (0.6 <= t <= 0.75)
        d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
        if not ring:
            d.rectangle([int(x - 1), int(y - r), int(x), int(y - r)], fill=BODY_LIGHT)

    # Slouched body, belly almost on the floor.
    d.rounded_rectangle([32, 45 + by, 66, 61 + by], radius=8, fill=BODY)
    d.rectangle([37, 46 + by, 60, 47 + by], fill=BODY_LIGHT)
    d.rectangle([36, 58 + by, 62, 59 + by], fill=BODY_DARK)
    draw_blob(d, 44, 52 + by, [(0, 0, 2.2), (3, -2, 1.7), (-2, 2, 1.6)], MARBLE)
    d.ellipse([58, 46 + by, 64, 53 + by], fill=MARBLE)

    # Tired shuffle steps.
    def leg(x_from, y_from, x_to, y_to):
        d.line([(x_from, y_from + by), (x_to, y_to + by)], fill=BODY, width=4)
        d.rounded_rectangle([x_to - 2, y_to + by - 2, x_to + 3, y_to + by + 2],
                            radius=2, fill=CREAM)

    if state == "walk0":
        leg(60, 56, 68, 68); leg(56, 56, 62, 69)
        leg(42, 56, 34, 68); leg(45, 56, 38, 69)
    else:
        leg(60, 56, 64, 69); leg(57, 56, 60, 70)
        leg(43, 56, 37, 69); leg(46, 56, 41, 70)

    # Head drooped forward; pants lift it a little on the huff.
    hy = 2 + (-1 if (pant and state == "pant_b") else 0)
    d.rounded_rectangle([56, 32 + hy, 78, 50 + hy], radius=8, fill=BODY)
    d.ellipse([68, 42 + hy, 80, 52 + hy], fill=BODY)
    # Ears folded down with exhaustion.
    for cx, dy in [(64, 0), (71, -1)]:
        d.polygon([(cx + 4, 35 + hy + dy), (cx - 1, 41 + hy + dy), (cx + 5, 39 + hy + dy)],
                  fill=BODY)
        d.ellipse([cx - 2, 39 + hy + dy, cx + 2, 43 + hy + dy], fill=MARBLE)
    # Closed eyes, done with everything.
    d.rectangle([66, 38 + hy, 70, 39 + hy], fill=EYE)
    # Muzzle with the tongue lolling out; it stretches on the huff.
    d.rectangle([73, 44 + hy, 79, 47 + hy], fill=CREAM)
    d.rectangle([78, 42 + hy, 79, 43 + hy], fill=NOSE)
    tongue = 5 if state == "pant_b" else 4
    d.rounded_rectangle([74, 47 + hy, 77, 47 + hy + tongue], radius=1, fill=NOSE)
    if pant and state == "pant_b":
        d.rectangle([84, 45, 85, 45], fill=BODY_DARK)  # huff puff
    # Drooped whiskers + the sweat drop sliding down.
    d.rectangle([80, 44 + hy, 86, 44 + hy], fill=WHISKER)
    d.rectangle([80, 47 + hy, 85, 47 + hy], fill=WHISKER)
    # Still wearing the headband from the workout.
    if headband:
        d.rectangle([57, 35 + hy, 79, 38 + hy], fill=BOOK_RED)
        d.rectangle([50, 35 + hy, 56, 37 + hy], fill=BOOK_RED)
    sy = 31 if state == "pant_b" else 27
    d.rectangle([81, sy, 82, sy], fill=SWEAT)
    d.rectangle([81, sy + 1, 82, sy + 3], fill=SWEAT)
    return im


SCREEN = (40, 44, 52, 255)  # tv screen dark base


def _mini_cat(d, x, y, step):
    """Tiny gold cat walking right across the TV screen — the cat's own
    show, naturally. step alternates the leg pose."""
    d.rectangle([x, y + 6, 90, y + 7], fill=(70, 76, 90, 255))  # stage floor
    d.rectangle([x, y, x + 6, y + 3], fill=BODY)                # body
    d.rectangle([x + 5, y - 2, x + 8, y + 1], fill=BODY)        # head
    d.rectangle([x + 7, y - 3, x + 7, y - 2], fill=MARBLE)      # ear
    d.rectangle([x + 6, y - 1, x + 6, y - 1], fill=SWEAT)       # eye glint
    d.rectangle([x - 2, y + 1, x - 1, y + 2], fill=MARBLE)      # tail
    if step:
        d.rectangle([x + 1, y + 4, x + 2, y + 5], fill=BODY)
        d.rectangle([x + 5, y + 4, x + 6, y + 5], fill=BODY)
    else:
        d.rectangle([x + 2, y + 4, x + 3, y + 5], fill=BODY)
        d.rectangle([x + 4, y + 4, x + 5, y + 5], fill=BODY)


def _tv(d, content, t=0):
    """Little TV on the right: dark frame + real channels, animated by the
    frame index t. content: 'catshow' (a tiny gold cat crossing a stage) |
    'fish' (aquarium: swimming goldfish + bubbles) | 'rocket' (night launch
    flying diagonally) | 'news' (broadcast with drifting weather sun) |
    'bars' | 'static' (channel zap)."""
    d.rounded_rectangle([62, 22, 94, 52], radius=3, fill=MARBLE)
    d.rectangle([65, 25, 91, 49], fill=SCREEN)
    d.rectangle([67, 53, 70, 57], fill=MARBLE)
    d.rectangle([86, 53, 89, 57], fill=MARBLE)

    if content == "bars":
        for i, c in enumerate([BUN, PATTY, LETTUCE, NOSE, CREAM, SWEAT]):
            d.rectangle([66 + i * 4, 26, 66 + i * 4 + 3, 48], fill=c)
    elif content == "static":
        import random
        rng = random.Random(7 + (t % 3))
        shades = [CREAM, BODY_DARK, (96, 102, 118, 255), MARBLE]
        for _ in range(46):
            x = rng.randint(66, 89)
            y = rng.randint(26, 47)
            w = rng.randint(1, 2)
            d.rectangle([x, y, min(x + w, 90), min(y + 1, 48)],
                        fill=rng.choice(shades))
    elif content == "catshow":
        d.rectangle([65, 26, 91, 49], fill=(34, 38, 48, 255))
        walk = (t * 3) % 34
        _mini_cat(d, 66 + walk, 38, step=(t // 2) % 2)
    elif content == "fish":
        d.rectangle([65, 26, 91, 49], fill=(58, 108, 158, 255))
        fx = 70 + (4, 6, 8, 6)[t % 4]
        fy = 36 + (0, 1, 0, -1)[t % 4]
        tail = 1 if t % 2 else -1
        d.polygon([(fx + 3, fy + 1), (fx + 6, fy + 1 - tail),
                   (fx + 6, fy + 3 - tail)], fill=YOLK)          # tail
        d.ellipse([fx, fy, fx + 4, fy + 4], fill=YOLK)           # goldfish
        d.rectangle([fx + 1, fy + 1, fx + 1, fy + 1], fill=MARBLE)
        for i in range(2):
            by = 46 - ((t * 2 + i * 6) % 16)
            d.rectangle([68 + i * 14, by, 68 + i * 14, by + 1], fill=CREAM)
        d.rectangle([65, 47, 91, 49], fill=(90, 132, 178, 255))  # gravel
    elif content == "rocket":
        d.rectangle([65, 26, 91, 49], fill=(16, 18, 30, 255))
        for sx, sy in [(70, 30), (80, 28), (86, 34), (74, 40)]:
            d.rectangle([sx, sy, sx, sy], fill=CREAM)            # stars
        rx = 86 - ((t * 3) % 26)
        ry = 46 - ((t * 3) % 26) // 2
        d.rectangle([rx, ry, rx + 2, ry + 4], fill=CREAM)        # rocket body
        d.polygon([(rx, ry), (rx + 2, ry - 2), (rx + 2, ry)], fill=NOSE)
        d.rectangle([rx + 3, ry + 4, rx + 4, ry + 6], fill=YOLK)  # flame
    else:  # news
        d.rectangle([66, 26, 90, 42], fill=(70, 80, 96, 255))
        for y in (30, 34):
            d.rectangle([69, y, 87, y], fill=CREAM)
            d.rectangle([69, y + 2, 80, y + 2], fill=CREAM)
        sun_x = 82 + (0, 1, 2, 1)[t % 4]
        d.ellipse([sun_x, 28, sun_x + 4, 32], fill=YOLK)         # weather sun
        d.rectangle([66, 43, 90, 48], fill=BOOK_RED)             # banner
        d.rectangle([69, 45, 76, 45], fill=PAGE)


def draw_treadmill(d, belt, level):
    """Side-view treadmill, cat stands on the belt. `belt` is the cumulative
    scroll offset (marks move under the cat); `level` 0..3 lights the console
    speed bars. Deck spans x14-78 at y66-74."""
    # Feet.
    d.rectangle([24, 74, 28, 80], fill=MARBLE)
    d.rectangle([64, 74, 68, 80], fill=MARBLE)
    # Deck + belt band with scrolling tread marks.
    d.rounded_rectangle([13, 65, 79, 75], radius=4, fill=MARBLE)
    d.rectangle([16, 67, 77, 73], fill=(88, 82, 74, 255))
    for k in range(8):
        x = 17 + ((k * 8 - belt) % 58)
        d.rectangle([x, 69, x + 2, 70], fill=MARBLE)
    # Rollers with a rotating spoke so they visibly spin.
    angle = (belt // 4) % 4
    for cx in (20, 73):
        d.ellipse([cx - 4, 67, cx + 4, 75], fill=MARBLE)
        d.rectangle([cx, 70, cx + 1, 71], fill=CREAM)
        sx, sy = [(cx - 1, cx, cx + 1, cx)[angle], (69, 70, 72, 73)[angle]]
        d.rectangle([sx, sy, sx, sy], fill=CREAM)
    # Console post + panel with speed bars (drawn before the cat = behind it).
    d.rectangle([76, 40, 79, 66], fill=MARBLE)
    d.rounded_rectangle([66, 26, 88, 40], radius=2, fill=MARBLE)
    d.rectangle([68, 28, 86, 33], fill=SCREEN)
    for i in range(3):
        c = NOSE if i < level else (70, 74, 84, 255)
        d.rectangle([70 + i * 6, 35, 73 + i * 6, 37], fill=c)
    d.rectangle([82, 29, 84, 30], fill=CREAM)  # blinking button


def draw_cat_tv(state, content="bars", t=0):
    """Movie night: the cat sits beside the TV, eyes glued to whatever is
    actually playing. state: 'watch' | 'lean' (interested) | 'laugh' |
    'scare' (jump-scare). t animates the channel content. The scare frame
    gets glitched by the caller if wanted."""
    poses = {
        "watch": {"gaze": "right", "sway": -3.0},
        "lean": {"gaze": "right", "head_dy": -1, "mouth": True, "sway": -3.0},
        "laugh": {"gaze": "right", "head_dy": -1, "mouth": True, "happy": True,
                  "sway": -3.0, "ear_r_dx": 1},
        "scare": {"gaze": "right", "shock": True, "ear_r_dx": 2, "squash": 1,
                  "sway": -3.0, "tail_up": True},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    _tv(d, content, t)
    return im


def draw_cat_flex(state, headband=True):
    """Gym proof: double-biceps pose. 'ready' stands easy, 'flex' throws the
    pose with bicep bumps on both arms, 'max' crushes it — squint, sparkles,
    everything shook."""
    poses = {
        "ready": {"gaze": "center", "sway": 1.5, "headband": headband},
        "flex": {"both_paws": True, "head_dy": -1, "mouth": True, "sway": -2.0,
                 "headband": headband},
        "max": {"both_paws": True, "head_dy": -1, "happy": True, "squash": 1,
                "whisker_dy": -1, "sway": 2.5, "headband": headband},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    if state != "ready":
        # bicep bumps bulging off both raised arms
        d.ellipse([57, 44, 65, 51], fill=BODY, outline=BODY_DARK)
        d.ellipse([31, 44, 39, 51], fill=BODY, outline=BODY_DARK)
    if state == "max":
        for x, y in [(30, 16), (62, 14), (26, 42), (68, 40)]:
            d.rectangle([x, y, x + 1, y + 3], fill=CREAM)
            d.rectangle([x - 1, y + 1, x + 2, y + 2], fill=CREAM)
    return im


def draw_cat_stretch(phase):
    """Yoga flow on a mat, side view facing right. 'arch': chest low
    between forward-stretched paws, rear high, tail straight up — the deep
    morning stretch. 'dip': front tall, back swayed down, face to the sky.
    'reach': sitting tall, one paw arcing overhead to the far side.
    'settle': easy sit between flows. Everything breathes via the caller's
    1px bob."""
    im = new_canvas()
    d = ImageDraw.Draw(im)

    # Yoga mat: blue with a cream stripe and a rolled end at the right.
    d.rounded_rectangle([10, 70, 84, 79], radius=4, fill=SWEAT)
    d.rectangle([14, 71, 80, 72], fill=(150, 190, 226, 255))
    d.ellipse([78, 66, 88, 80], fill=(120, 165, 208, 255))
    d.ellipse([81, 69, 85, 77], fill=SWEAT)

    def leg(x_from, y_from, x_to, y_to):
        d.line([(x_from, y_from), (x_to, y_to)], fill=BODY, width=4)
        d.rounded_rectangle([x_to - 2, y_to - 2, x_to + 3, y_to + 2],
                            radius=2, fill=CREAM)

    if phase == "arch":
        # rear haunch high, spine sloping down to the front paws
        d.ellipse([48, 34, 72, 58], fill=BODY)
        d.polygon([(24, 50), (52, 38), (66, 40), (66, 56), (30, 62)], fill=BODY)
        d.rectangle([40, 44, 60, 45], fill=BODY_LIGHT)
        d.rectangle([34, 56, 62, 58], fill=BODY_DARK)
        leg(58, 54, 60, 70)
        leg(53, 54, 55, 70)
        leg(28, 56, 12, 68)
        leg(31, 56, 17, 70)
        # head low between the paws, serene
        d.rounded_rectangle([6, 40, 26, 58], radius=8, fill=BODY)
        d.ellipse([6, 50, 26, 62], fill=BODY)
        for cx, dy in [(12, -1), (19, -2)]:
            d.polygon([(cx + 4, 42 + dy), (cx - 2, 33 + dy), (cx + 6, 44 + dy)], fill=BODY)
            d.ellipse([cx - 3, 34 + dy, cx + 1, 38 + dy], fill=MARBLE)
        d.rectangle([6, 45, 26, 48], fill=BOOK_RED)  # headband
        d.rectangle([17, 48, 21, 49], fill=EYE)      # closed, breathing out
        d.rectangle([22, 53, 25, 56], fill=CREAM)
        d.rectangle([24, 51, 25, 52], fill=NOSE)
        draw_blob(d, 58, 44, [(0, 0, 2.6), (3, 2, 2.0), (-1, 4, 1.8)], MARBLE)
        # tail straight up, proud
        for i, (x, y) in enumerate(_bezier((62, 36), (68, 28), (72, 20), (68, 12), n=14)):
            t = i / 14
            r = 2.6 if t < 0.7 else 2.2
            ring = (0.35 <= t <= 0.5) or (0.62 <= t <= 0.77) or t > 0.9
            col = MARBLE if ring else BODY
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
            if not ring:
                d.rectangle([int(x - 1), int(y - r), int(x), int(y - r)],
                            fill=BODY_LIGHT)
    elif phase == "dip":
        # front tall, back swayed down, face to the sky
        d.rounded_rectangle([30, 46, 66, 62], radius=10, fill=BODY)
        d.rectangle([34, 47, 62, 48], fill=BODY_LIGHT)
        d.rectangle([34, 58, 62, 60], fill=BODY_DARK)
        leg(34, 54, 32, 70)
        leg(38, 54, 36, 70)
        d.ellipse([50, 44, 72, 62], fill=BODY)
        leg(58, 56, 60, 70)
        leg(62, 56, 64, 70)
        d.rounded_rectangle([38, 24, 58, 44], radius=8, fill=BODY)
        d.ellipse([38, 36, 58, 48], fill=BODY)
        for cx, dy in [(44, -1), (51, 0)]:
            d.polygon([(cx - 4, 28 + dy), (cx, 19 + dy), (cx + 4, 28 + dy)], fill=BODY)
            d.ellipse([cx - 2, 20 + dy, cx + 2, 24 + dy], fill=MARBLE)
        d.rectangle([38, 28, 58, 31], fill=BOOK_RED)  # headband
        d.rectangle([44, 30, 46, 34], fill=EYE)
        d.rectangle([44, 31, 45, 31], fill=BODY)      # serene half-closed
        d.rectangle([48, 38, 53, 40], fill=CREAM)
        d.rectangle([52, 36, 53, 37], fill=NOSE)
        draw_blob(d, 44, 30, HEAD_MARBLE, MARBLE)
        draw_blob(d, 58, 52, [(0, 0, 2.4), (3, 2, 2.0)], MARBLE)
        for i, (x, y) in enumerate(_bezier((64, 56), (70, 60), (74, 56), (76, 48), n=12)):
            t = i / 12
            r = 2.6 if t < 0.7 else 2.2
            ring = (0.35 <= t <= 0.5) or t > 0.8
            d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
        d.rectangle([58, 26, 60, 27], fill=WHISKER)
        d.rectangle([59, 29, 61, 29], fill=WHISKER)
    elif phase == "reach":
        # sitting tall, one paw arcing overhead to the far side
        d.ellipse([52, 50, 74, 70], fill=BODY)                # folded haunch
        d.rounded_rectangle([38, 40, 62, 66], radius=10, fill=BODY)
        d.rectangle([42, 41, 58, 42], fill=BODY_LIGHT)
        d.rectangle([42, 62, 58, 64], fill=BODY_DARK)
        leg(44, 62, 42, 70)                                   # crossed legs
        leg(56, 62, 60, 70)
        # head tilted into the reach
        d.rounded_rectangle([40, 22, 62, 42], radius=8, fill=BODY)
        d.ellipse([40, 34, 62, 46], fill=BODY)
        for cx, dy in [(46, -1), (55, 0)]:
            d.polygon([(cx - 4, 26 + dy), (cx, 17 + dy), (cx + 4, 26 + dy)], fill=BODY)
            d.ellipse([cx - 2, 18 + dy, cx + 2, 22 + dy], fill=MARBLE)
        d.rectangle([40, 26, 60, 29], fill=BOOK_RED)  # headband
        d.rectangle([45, 30, 49, 34], fill=EYE)
        d.rectangle([45, 30, 49, 30], fill=BODY)      # serene reach squint
        d.rectangle([52, 37, 57, 39], fill=CREAM)
        d.rectangle([57, 35, 58, 36], fill=NOSE)
        draw_blob(d, 46, 28, HEAD_MARBLE, MARBLE)
        # reaching arm: shoulder arcs over the head to the far side
        for i, (x, y) in enumerate(_bezier((52, 42), (58, 32), (68, 26), (74, 30), n=10)):
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=BODY)
        d.ellipse([72, 27, 78, 33], fill=CREAM)       # paw past the far ear
        # tail wrapped around the front paws
        for i, (x, y) in enumerate(_bezier((60, 62), (58, 68), (48, 70), (38, 66), n=12)):
            t = i / 12
            r = 2.4 if t < 0.8 else 2.0
            ring = (0.3 <= t <= 0.45) or (0.6 <= t <= 0.75)
            d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
    else:  # settle
        return draw_cat({"sway": 2.0, "gaze": "center"})
    return im


def draw_cat_camera(state, var=0):
    """Best Take's own mascot doing the thing: raises the camera to the
    eye, squints through the viewfinder, FLASH (with a white pop), lowers
    it to review the shot on the back screen, celebrates the keep.
    state: 'down' | 'aim' | 'squint' | 'flash' | 'review' | 'celebrate'.
    var alternates burst size / squint eye."""
    poses = {
        "down": {"gaze": "down", "head_dy": 1},
        "aim": {"gaze": "center", "sway": 0.0},
        "squint": {"gaze": "center", "sway": 0.0, "mouth": True},
        "flash": {"gaze": "center", "sway": 0.0},
        "review": {"gaze": "down", "happy": True, "head_dy": 1},
        "celebrate": {"both_paws": True, "happy": True, "sway": 2.5,
                      "ear_r_dx": 1},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)

    # Smile wherever the mouth is visible (the raised camera covers it).
    if state in ("down", "review", "celebrate"):
        hd = 1 if state != "celebrate" else 0
        for px, py in [(44, 43), (45, 44), (46, 44), (47, 44), (48, 43)]:
            d.rectangle([px, py + hd, px, py + hd], fill=EYE)

    if state in ("aim", "squint", "flash"):
        # Camera raised: covers the muzzle, eyes peek over the top rim.
        cx = 49
        d.rounded_rectangle([37, 31, 61, 45], radius=3, fill=MARBLE)
        d.rectangle([43, 28, 55, 31], fill=MARBLE)          # viewfinder hump
        d.rectangle([56, 28, 58, 30], fill=NOSE)            # shutter button
        d.ellipse([cx - 7, 33, cx + 7, 45], fill=MARBLE)    # lens barrel
        d.ellipse([cx - 5, 35, cx + 5, 44], fill=SCREEN)    # lens glass
        d.rectangle([cx - 2, 37, cx - 1, 38], fill=CREAM)   # lens glint
        # paws gripping both sides
        d.rounded_rectangle([33, 33, 40, 41], radius=2, fill=CREAM)
        d.rounded_rectangle([58, 33, 65, 41], radius=2, fill=CREAM)
        # viewfinder squint: one eye closes while aiming
        if state == "squint":
            if var == 0:
                d.rectangle([40, 29, 44, 29], fill=BODY)
                d.rectangle([40, 30, 44, 33], fill=EYE)
            else:
                d.rectangle([51, 29, 55, 29], fill=BODY)
                d.rectangle([51, 30, 55, 33], fill=EYE)
        if state == "flash":
            r1 = 6 if var == 0 else 8
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                d.rectangle([cx + dx * r1 - 1, 38 + dy * r1 - 1,
                             cx + dx * r1 + 1, 38 + dy * r1 + 1], fill=YOLK)
            d.ellipse([cx - r1, 38 - r1, cx + r1, 38 + r1],
                      fill=(255, 252, 235, 255))
            d.rectangle([30, 16, 31, 20], fill=YOLK)
            d.rectangle([68, 16, 69, 20], fill=YOLK)
    else:
        # Lowered / at rest: we see the camera's back with the preview.
        cx = 49
        top = 48 if state == "review" else 52
        d.rounded_rectangle([cx - 12, top, cx + 12, top + 15], radius=3,
                            fill=MARBLE)
        d.rectangle([cx - 9, top + 3, cx + 2, top + 12], fill=SCREEN)
        # the shot on the preview: sky, hill, sun
        d.rectangle([cx - 8, top + 5, cx - 1, top + 8], fill=SWEAT)
        d.rectangle([cx - 8, top + 9, cx - 1, top + 11], fill=LETTUCE)
        d.rectangle([cx - 3, top + 5, cx - 2, top + 6], fill=YOLK)
        d.rectangle([cx + 5, top + 4, cx + 9, top + 7], fill=BODY_DARK)  # dial
        d.rounded_rectangle([cx - 9, top + 13, cx + 9, top + 15], radius=1,
                            fill=BODY_DARK)
        if state == "review":
            d.rectangle([cx + 4, top + 9, cx + 10, top + 10], fill=CREAM)
    return im


def glitch_frame(im, bands, ghost=False, invert=False):
    """Retro glitch: horizontal slice shifts + optional ghost/invert flicker."""
    a = np.array(im)
    out = a.copy()
    for y0, h, dx in bands:
        out[y0:y0 + h] = np.roll(a[y0:y0 + h], dx, axis=1)
    if ghost:
        g = a.copy()
        g[:, :, 3] = (g[:, :, 3] * 0.4).astype(np.uint8)
        shifted = np.zeros_like(a)
        shifted[:, 3:] = g[:, :-3, :]
        base_a = out[:, :, 3:4].astype(np.float32) / 255
        ghost_a = shifted[:, :, 3:4].astype(np.float32) / 255 * (1 - base_a)
        rgb = (out[:, :, :3] * base_a + shifted[:, :, :3] * ghost_a).astype(np.uint8)
        alpha = np.maximum(out[:, :, 3], shifted[:, :, 3])
        out = np.dstack([rgb, alpha])
    if invert:
        mask = out[:, :, 3] > 0
        out[:, :, :3][mask] = (255 - out[:, :, :3][mask].astype(np.int16)).astype(np.uint8)
    return Image.fromarray(out)
