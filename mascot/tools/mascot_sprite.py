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
    """Morning yoga, side view facing right. 'arch': chest low between
    forward-stretched paws, rear high, tail straight up. 'dip': front tall,
    back dipped, face to the sky. 'settle' sits easy between flows."""
    im = new_canvas()
    d = ImageDraw.Draw(im)

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
        # head low between the paws
        d.rounded_rectangle([6, 40, 26, 58], radius=8, fill=BODY)
        d.ellipse([6, 50, 26, 62], fill=BODY)
        for cx, dy in [(12, -1), (19, -2)]:
            d.polygon([(cx + 4, 42 + dy), (cx - 2, 33 + dy), (cx + 6, 44 + dy)], fill=BODY)
            d.ellipse([cx - 3, 34 + dy, cx + 1, 38 + dy], fill=MARBLE)
        d.rectangle([6, 45, 26, 48], fill=BOOK_RED)  # headband
        d.rectangle([16, 47, 20, 51], fill=EYE)
        d.rectangle([22, 53, 25, 56], fill=CREAM)
        d.rectangle([24, 51, 25, 52], fill=NOSE)
        draw_blob(d, 58, 44, [(0, 0, 2.6), (3, 2, 2.0), (-1, 4, 1.8)], MARBLE)
        # tail straight up, proud
        for i, (x, y) in enumerate(_bezier((62, 36), (68, 28), (72, 20), (68, 12), n=14)):
            t = i / 14
            r = 2.6 if t < 0.7 else 2.2
            ring = (0.35 <= t <= 0.5) or (0.62 <= t <= 0.77) or t > 0.9
            d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
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
        d.rectangle([42, 30, 46, 34], fill=EYE)
        d.rectangle([44, 31, 45, 32], fill=WHISKER)
        d.rectangle([48, 38, 53, 40], fill=CREAM)
        d.rectangle([52, 36, 53, 37], fill=NOSE)
        draw_blob(d, 44, 30, HEAD_MARBLE, MARBLE)
        draw_blob(d, 58, 52, [(0, 0, 2.4), (3, 2, 2.0)], MARBLE)
        for i, (x, y) in enumerate(_bezier((64, 56), (70, 60), (74, 56), (76, 48), n=12)):
            t = i / 12
            r = 2.6 if t < 0.7 else 2.2
            ring = (0.35 <= t <= 0.5) or t > 0.8
            d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
        d.rectangle([58, 26, 60, 27], fill=WHISKER)  # whiskers up with the face
        d.rectangle([59, 29, 61, 29], fill=WHISKER)
    else:  # settle
        im = draw_cat({"sway": 2.0, "gaze": "center"})
        return im
    return im


def draw_cat_phone(state):
    """Answering the phone: the retro brick cell rings on the floor, gets
    held to the ear, chats with a gesticulating free paw, hears SHOCKING
    news, recovers, waves bye. States: ring/pick/chat1/chat2/shock/bye."""
    poses = {
        "ring": {"gaze": "down", "ear_r_dx": 1, "sway": 2.5},
        "pick": {"mouth": True, "head_dy": -1, "gaze": "center"},
        "chat1": {"paw_side": "l", "paw_raise": -1, "mouth": True, "gaze": "up",
                  "sway": 2.0, "ear_r_dx": 1},
        "chat2": {"paw_side": "l", "paw_raise": 1, "mouth": True, "gaze": "center",
                  "sway": -2.0},
        "shock": {"shock": True, "paw_side": "l", "paw_raise": 0, "gaze": "right",
                  "squash": 1},
        "bye": {"paw_side": "l", "paw_raise": 2, "happy": True, "mouth": True,
                "sway": 2.5},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    if state == "ring":
        # brick phone rattling on the floor
        d.rounded_rectangle([58, 72, 66, 78], radius=2, fill=MARBLE)
        d.rectangle([60, 74, 64, 75], fill=SWEAT)
        d.rectangle([64, 70, 65, 71], fill=MARBLE)  # antenna
        d.rectangle([56, 71, 57, 71], fill=BODY_DARK)
        d.rectangle([68, 71, 69, 71], fill=BODY_DARK)
        d.rectangle([62, 67, 63, 67], fill=BODY_DARK)
    else:
        # retro brick cell held to the ear: body, lit screen, buttons, antenna
        d.rounded_rectangle([62, 24, 68, 36], radius=2, fill=MARBLE)
        d.rectangle([64, 27, 67, 29], fill=SWEAT)
        d.rectangle([64, 31, 65, 31], fill=CREAM)
        d.rectangle([66, 31, 67, 31], fill=CREAM)
        d.rectangle([65, 21, 66, 23], fill=MARBLE)
        # paw gripping the phone
        d.rounded_rectangle([59, 32, 66, 38], radius=3, fill=CREAM)
        if state == "pick":
            d.rectangle([46, 42, 47, 44], fill=EYE)  # "hello!"
        if state == "shock":
            d.rectangle([56, 18, 57, 19], fill=SWEAT)
    return im


CONFETTI = [(20, 10, BUN), (74, 6, NOSE), (28, 26, LETTUCE), (68, 22, SWEAT),
            (14, 38, CREAM), (80, 34, PAGE), (58, 2, NOSE), (36, 4, SWEAT)]


def _cap(d, x_off=0, y_off=0, tassel="right"):
    """Mortarboard: flat diamond, center button, swinging tassel."""
    d.polygon([(35 + x_off, 13 + y_off), (48 + x_off, 8 + y_off),
               (61 + x_off, 13 + y_off), (48 + x_off, 18 + y_off)], fill=MARBLE)
    d.rectangle([47 + x_off, 9 + y_off, 49 + x_off, 10 + y_off], fill=CREAM)
    if tassel == "right":
        d.rectangle([55 + x_off, 14 + y_off, 56 + x_off, 19 + y_off], fill=NOSE)
        d.rectangle([55 + x_off, 19 + y_off, 57 + x_off, 21 + y_off], fill=NOSE)
    else:
        d.rectangle([41 + x_off, 14 + y_off, 42 + x_off, 19 + y_off], fill=NOSE)
        d.rectangle([39 + x_off, 19 + y_off, 41 + x_off, 21 + y_off], fill=NOSE)


def draw_cat_grad(state, var=0):
    """Commencement! Gown + gold stole, diploma scroll, mortarboard with a
    swinging tassel. state: 'ready' (proud, diploma in paws) | 'toss' (cap
    in the air, arms up, confetti) | 'cheer' (cap back, celebrating).
    var flips the tassel side / confetti jitter between frames."""
    tassel = "left" if var == 1 else "right"
    im = draw_cat({"gaze": "center", "mouth": state != "ready",
                   "happy": state == "cheer", "both_paws": state != "ready",
                   "sway": 1.5 if state == "ready" else -2.0})
    d = ImageDraw.Draw(im)

    # Gown over the body with a gold stole V.
    d.rounded_rectangle([29, 45, 67, 79], radius=8, fill=MARBLE)
    d.polygon([(39, 47), (45, 47), (49, 68), (43, 68)], fill=NOSE)
    d.polygon([(57, 47), (51, 47), (47, 68), (53, 68)], fill=NOSE)

    if state == "ready":
        # diploma scroll held at the chest
        d.rounded_rectangle([38, 57, 52, 61], radius=1, fill=PAGE)
        d.rectangle([44, 57, 46, 61], fill=BOOK_RED)
        _cap(d, tassel=tassel)
        d.rounded_rectangle([35, 72, 43, 78], radius=3, fill=CREAM)
        d.rounded_rectangle([53, 72, 61, 78], radius=3, fill=CREAM)
    elif state == "toss":
        # cap in the air, scroll raised in one paw, confetti raining
        _cap(d, 0, -9 + var * 2, tassel=tassel)
        d.rounded_rectangle([24, 26, 38, 30], radius=1, fill=PAGE)
        d.rectangle([30, 26, 32, 30], fill=BOOK_RED)
        for x, y, c in CONFETTI:
            dy = var * 3
            d.rectangle([x, y + dy, x + 1, y + 1 + dy], fill=c)
    else:  # cheer
        _cap(d, 0, 1, tassel=tassel)  # caught it, slightly askew
        d.rounded_rectangle([24, 26, 38, 30], radius=1, fill=PAGE)
        d.rectangle([30, 26, 32, 30], fill=BOOK_RED)
        for x, y, c in CONFETTI[:6]:
            dy = (1 - var) * 3
            d.rectangle([x, y + 2 + dy, x + 1, y + 3 + dy], fill=c)
    return im


BALL = (216, 110, 48, 255)  # basketball orange


def draw_basketball(d, x, y, rx=3, ry=3):
    """Basketball with cross seams; wider+flatter when squashed on a bounce."""
    d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=BALL)
    d.rectangle([x, y - ry + 1, x, y + ry - 1], fill=MARBLE)
    d.rectangle([x - rx + 1, y, x + rx - 1, y], fill=MARBLE)


def draw_hoop(d):
    """Mini hoop in the top-right corner: backboard, rim, swaying net."""
    d.rectangle([74, 2, 92, 15], fill=MARBLE)
    d.rectangle([76, 4, 90, 12], fill=(226, 222, 210, 255))
    d.rectangle([80, 6, 86, 10], fill=SCREEN)
    d.rectangle([73, 16, 89, 17], fill=NOSE)  # rim
    for i, x0 in enumerate((76, 81, 86)):
        d.line([(x0, 18), (x0 + 3 if i != 1 else x0 - 1, 23)], fill=CREAM, width=1)
    d.rectangle([79, 23, 86, 23], fill=CREAM)


SNOW = (247, 240, 218, 255)
SNOW_SHADE = (221, 212, 188, 255)


def draw_snowman(d, level, rx_bottom=11):
    """Snowman at the cat's right. level: 1 bottom ball, 2 +middle,
    3 +head, 4 finished (coal face, carrot, stick arms, red scarf)."""
    if level >= 1:
        d.ellipse([74 - rx_bottom, 57, 74 + rx_bottom, 75], fill=SNOW)
        d.rectangle([68, 71, 80, 73], fill=SNOW_SHADE)
    if level >= 2:
        d.ellipse([66, 43, 82, 57], fill=SNOW)
        d.rectangle([74, 48, 74, 52], fill=MARBLE)
        d.rectangle([74, 54, 74, 55], fill=MARBLE)
    if level >= 3:
        d.ellipse([68, 31, 80, 42], fill=SNOW)
    if level >= 4:
        d.rectangle([71, 34, 72, 35], fill=EYE)
        d.rectangle([75, 34, 76, 35], fill=EYE)
        d.polygon([(65, 36), (71, 35), (71, 38)], fill=BALL)  # carrot, facing cat
        d.rectangle([67, 43, 81, 45], fill=BOOK_RED)          # scarf
        d.rectangle([76, 45, 80, 52], fill=BOOK_RED)
        d.line([(66, 47), (58, 40)], fill=MARBLE, width=2)    # stick arms
        d.line([(82, 47), (90, 40)], fill=MARBLE, width=2)


def _note(d, x, y):
    """Little music note floating up."""
    d.ellipse([x, y + 3, x + 2, y + 5], fill=BODY_DARK)
    d.rectangle([x + 2, y, x + 2, y + 3], fill=BODY_DARK)
    d.rectangle([x + 3, y, x + 3, y + 1], fill=BODY_DARK)


def draw_cat_piano(hands, cresc=False, var=0):
    """Seated at the keyboard: both paws on the keys (they visibly press),
    swaying and humming along, notes floating up. hands: 'l' | 'c' | 'r'
    shifts the paws across the keys; cresc = the big finale."""
    gazes = {"l": "left", "c": "center", "r": "right"}
    hands = "c" if hands == "cresc" else hands
    im = draw_cat({
        "head_dy": 1, "gaze": gazes[hands], "mouth": True,
        "sway": 2.0 if hands == "l" else -2.0,
        "happy": cresc or hands == "c",
    })
    d = ImageDraw.Draw(im)

    # Upright piano: red felt strip over the key bed.
    d.rectangle([27, 51, 69, 68], fill=MARBLE)
    d.rectangle([29, 52, 67, 54], fill=BOOK_RED)

    offset = {"l": -6, "c": 0, "r": 6}[hands]
    paw_xs = [40 + offset, 52 + offset]

    # White keys, pressing 1px under whichever paw is on them.
    for i in range(9):
        kx = 30 + i * 4
        pressed = any(px - 3 <= kx <= px + 2 for px in paw_xs)
        ky = 58 if pressed else 57
        d.rectangle([kx, ky, kx + 3, 66], fill=PAGE)
        if pressed:
            d.rectangle([kx, ky, kx + 3, ky + 1], fill=(198, 188, 164, 255))
    # Black keys (skip the E-F / B-C gaps).
    for i in range(8):
        if i not in (2, 6):
            d.rectangle([33 + i * 4, 56, 34 + i * 4, 61], fill=MARBLE)

    # Playing paws on the keys, outlined so they read against the ivory.
    for px in paw_xs:
        d.rounded_rectangle([px - 4, 56, px + 4, 64], radius=3, fill=MARBLE)
        d.rounded_rectangle([px - 3, 57, px + 3, 63], radius=2, fill=CREAM)

    # Notes floating up from the keys, cycling with var.
    if cresc:
        for nx, ny in [(64, 34), (68, 26), (62, 20)]:
            _note(d, nx, ny + (var * 3))
    elif hands == "l":
        _note(d, 64, 38 + (var * 3))
    elif hands == "r":
        _note(d, 68, 32 + (var * 3))
    else:
        _note(d, 64, 36 + (var * 3))
        _note(d, 70, 28 + (var * 3))
    return im


YOLK = (240, 204, 80, 255)
CERAMIC = (226, 222, 210, 255)


def _chef_hat(d):
    """White toque: puffy crown sitting over the ears."""
    d.rounded_rectangle([37, 19, 59, 25], radius=2, fill=PAGE)
    d.ellipse([35, 4, 61, 22], fill=PAGE)
    d.ellipse([38, 0, 52, 12], fill=PAGE)
    d.ellipse([46, 1, 60, 11], fill=PAGE)


def _mixing_bowl(d, batter=True):
    d.rounded_rectangle([34, 62, 62, 77], radius=5, fill=CERAMIC)
    d.rectangle([36, 63, 60, 65], fill=SNOW_SHADE)
    if batter:
        d.rectangle([38, 63, 58, 64], fill=BUN)


def _cake(d, cherry=True):
    """Layer cake with frosting stripes on a plate, cherry on top."""
    d.ellipse([34, 71, 62, 77], fill=CERAMIC)
    d.rounded_rectangle([39, 60, 57, 71], radius=3, fill=BUN)
    d.rectangle([39, 64, 57, 66], fill=PAGE)
    d.rounded_rectangle([41, 55, 55, 61], radius=3, fill=BUN)
    d.ellipse([41, 52, 55, 58], fill=PAGE)
    if cherry:
        d.ellipse([46, 50, 49, 53], fill=NOSE)
        d.rectangle([48, 47, 49, 49], fill=MARBLE)


def draw_cat_baking(stage, var=0):
    """Bake a cake: toque on, crack the egg, stir the bowl, sneak a spoon
    lick, POOF — the cake appears, cherry goes on, sparkles. stage:
    crack | stir | lick | cake | decorate | celebrate."""
    poses = {
        "crack": {"gaze": "down", "mouth": True},
        "stir": {"gaze": "down", "mouth": True, "sway": 2.0 if var else -2.0},
        "lick": {"happy": True, "gaze": "center"},
        "cake": {"both_paws": True, "gaze": "down", "mouth": True},
        "decorate": {"gaze": "down", "paw_side": "l", "paw_raise": 1,
                     "mouth": True, "head_dy": 0},
        "celebrate": {"both_paws": True, "happy": True, "sway": 2.5},
    }
    im = draw_cat(poses[stage])
    d = ImageDraw.Draw(im)
    _chef_hat(d)

    if stage == "crack":
        _mixing_bowl(d)
        if var:
            d.rectangle([43, 52, 45, 55], fill=PAGE)
            d.rectangle([49, 52, 51, 55], fill=PAGE)
            d.ellipse([46, 58, 49, 61], fill=YOLK)  # yolk drops in
        else:
            d.ellipse([45, 50, 49, 55], fill=PAGE)  # whole egg
    elif stage == "stir":
        _mixing_bowl(d)
        sx, sy = [(44, 64), (48, 66), (52, 64)][var % 3]
        d.line([(57, 52), (sx, sy - 3)], fill=MARBLE, width=2)
        d.ellipse([sx - 2, sy - 2, sx + 2, sy], fill=PAGE)
        if var == 1:
            d.rectangle([40, 57, 41, 58], fill=SNOW_SHADE)  # flour poof
    elif stage == "lick":
        _mixing_bowl(d)
        d.line([(50, 58), (46, 46)], fill=MARBLE, width=2)
        d.ellipse([43, 43, 47, 47], fill=PAGE)
        d.rectangle([44, 42, 47, 44], fill=NOSE)  # tongue on the spoon
    elif stage == "decorate":
        _cake(d, cherry=False)
        d.line([(57, 52), (48, 50)], fill=BODY, width=4)
        d.rounded_rectangle([45, 47, 51, 52], radius=2, fill=CREAM)
        if var:
            d.ellipse([46, 50, 49, 53], fill=NOSE)
    else:  # cake / celebrate
        _cake(d)
        if stage == "celebrate":
            for x, y in [(30, 44), (66, 42), (28, 58)]:
                d.rectangle([x, y, x + 1, y + 3], fill=CREAM)
                d.rectangle([x - 1, y + 1, x + 2, y + 2], fill=CREAM)
    return im


FLAME_ORANGE = (226, 120, 40, 255)
FLAME_RED = (208, 68, 40, 255)


def draw_flames(d, size, var, bx=85, by=66):
    """Flickering fire at the cat's right; size 3 (blaze) .. 1 (ember),
    0 = out (nothing). var flips the flicker lean."""
    flick = 1 if var else -1

    def flame(cx, cy, r, h):
        d.polygon([(cx - r, cy + r), (cx + flick, cy - h), (cx + r, cy + r)],
                  fill=FLAME_RED)
        d.ellipse([cx - r, cy - r // 2, cx + r, cy + r], fill=FLAME_ORANGE)
        d.rectangle([cx - 1, cy, cx + 1, cy + 1], fill=YOLK)

    if size >= 3:
        flame(76, 66, 3, 6)
        flame(92, 66, 3, 6)
    if size >= 2:
        flame(85, 63, 4, 12)
    elif size == 1:
        flame(85, 66, 2, 5)


def draw_smoke(d, bx=85):
    """Gray puffs curling up from an extinguished fire."""
    by = 66
    for dx, dy, r, c in [(-2, -12, 3, (120, 116, 110, 255)),
                         (2, -20, 4, (146, 142, 136, 255)),
                         (-1, -29, 5, (170, 166, 160, 255))]:
        d.ellipse([bx + dx - r, by + dy - r, bx + dx + r, by + dy + r], fill=c)


def draw_cat_firefighter(state):
    """Brave kitty: red helmet with a front badge, dark coat with reflective
    yellow bands. state: 'spray' (braced on the hose) | 'ease' (relaxed,
    proud) | 'shock' (the fire came back?!)."""
    poses = {
        "spray": {"gaze": "right", "mouth": True},
        "ease": {"happy": True, "gaze": "right"},
        "shock": {"shock": True, "gaze": "right", "squash": 1},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)
    # Coat with reflective bands.
    d.rounded_rectangle([29, 45, 67, 79], radius=8, fill=MARBLE)
    d.rectangle([31, 51, 65, 53], fill=YOLK)
    d.rectangle([31, 61, 65, 63], fill=YOLK)
    # Red helmet: wide brim, dome, front badge.
    d.ellipse([33, 13, 63, 22], fill=BOOK_RED)
    d.rounded_rectangle([38, 6, 58, 18], radius=4, fill=BOOK_RED)
    d.rectangle([46, 15, 50, 19], fill=CREAM)
    # Paw tips back over the coat hem.
    d.rounded_rectangle([35, 72, 43, 78], radius=3, fill=CREAM)
    d.rounded_rectangle([53, 72, 61, 78], radius=3, fill=CREAM)
    return im


def draw_cat_noodles(state, var=0):
    """Ramen night: slurps noodles from a steaming red bowl with chopsticks,
    bowl empties, pure happiness. state: 'slurp1' (long strand) |
    'slurp2' (short strand) | 'slurp3' (final slurp) | 'done'."""
    poses = {
        "slurp1": {"gaze": "down", "mouth": True, "sway": 1.5},
        "slurp2": {"gaze": "down", "mouth": True, "sway": -1.5},
        "slurp3": {"gaze": "down", "mouth": True, "squash": 1},
        "done": {"both_paws": True, "happy": True, "sway": 2.5},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)

    # Ramen bowl: red with a cream band, broth + noodle hump inside.
    d.rounded_rectangle([34, 60, 62, 77], radius=5, fill=BOOK_RED)
    d.rectangle([36, 61, 60, 63], fill=CREAM)
    if state != "done":
        d.rectangle([38, 60, 58, 62], fill=BUN)

    # Steam wisps rising.
    for sx, sy in [(40, 50), (52, 46), (46, 42)]:
        yy = sy - var * 3
        d.rectangle([sx + var, yy, sx + var, yy + 3], fill=SNOW)

    if state == "done":
        d.rectangle([44, 70, 45, 70], fill=PAGE)
        d.rectangle([50, 71, 50, 71], fill=PAGE)
        return im

    # Noodle strand from the bowl up to the mouth, slurping shorter.
    wig = 1 if var else -1
    strand = {
        "slurp1": [(47, 60), (48 + wig, 56), (46, 52), (47 + wig, 48), (46, 44)],
        "slurp2": [(47, 60), (48 + wig, 55), (46, 50), (46, 45)],
        "slurp3": [(47, 60), (47 + wig, 53), (46, 47)],
    }[state]
    for px, py in strand:
        d.rectangle([px, py, px + 1, py + 1], fill=BUN)

    # Chopsticks in the right paw, angled to the bowl.
    d.line([(58, 50), (50, 62)], fill=MARBLE, width=1)
    d.line([(61, 51), (53, 63)], fill=MARBLE, width=1)
    d.rounded_rectangle([56, 46, 62, 52], radius=2, fill=CREAM)
    return im


def draw_cat_gaming(state, var=0):
    """Gamer mode: hunched over the controller, mashing (jitter + whisker
    rattle), locked-in shock eyes on the clutch, then the WIN — arms up,
    confetti. state: 'mash' | 'intense' | 'win'."""
    poses = {
        "mash": {"gaze": "center", "mouth": var == 0},
        "intense": {"shock": True, "squash": 1},
        "win": {"both_paws": True, "happy": True, "sway": 2.5, "tail_up": True},
    }
    im = draw_cat(poses[state])
    if state == "mash":
        arr = np_roll(im, var * 2 - 1)
        im = arr
    d = ImageDraw.Draw(im)
    if state == "win":
        for x, y, c in [(30, 20, NOSE), (64, 16, SWEAT), (26, 40, YOLK)]:
            d.rectangle([x, y, x + 1, y + 3], fill=c)
            d.rectangle([x - 1, y + 1, x + 2, y + 2], fill=c)
        return im
    # controller: body, d-pad, buttons, gripping paws
    d.rounded_rectangle([41, 58, 57, 66], radius=3, fill=MARBLE)
    d.rectangle([44, 61, 45, 62], fill=CREAM)
    d.rectangle([43, 62, 46, 63], fill=CREAM)
    d.ellipse([52, 60, 55, 63], fill=NOSE)
    d.rounded_rectangle([39, 58, 45, 64], radius=2, fill=CREAM)
    d.rounded_rectangle([53, 58, 59, 64], radius=2, fill=CREAM)
    return im


def np_roll(im, dx):
    """Shift a whole image horizontally (jitter/shake helper)."""
    import numpy as np
    arr = np.array(im)
    out = np.zeros_like(arr)
    if dx >= 0:
        out[:, dx:] = arr[:, :arr.shape[1] - dx]
    else:
        out[:, :arr.shape[1] + dx] = arr[:, -dx:]
    return Image.fromarray(out)


def draw_cat_campfire(state, var=0):
    """Night under the stars: campfire crackling, toasting a marshmallow on
    a stick — golden, charred, eaten. state: 'toast_a' | 'toast_b' |
    'toast_c' (golden) | 'charred' | 'eaten'."""
    poses = {
        "toast_a": {"gaze": "right", "mouth": True, "sway": 1.5},
        "toast_b": {"gaze": "right", "mouth": True, "sway": -1.5},
        "toast_c": {"gaze": "right", "happy": True},
        "charred": {"gaze": "right", "shock": var == 1},
        "eaten": {"happy": True, "sway": 2.5, "mouth": True},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)

    # Starry night.
    for sx, sy in [(10, 6), (20, 14), (66, 8), (88, 18), (60, 4), (14, 30),
                   (90, 30), (30, 4)]:
        d.rectangle([sx, sy, sx, sy], fill=CREAM)

    # Campfire at the right: crossed logs + flickering flames.
    d.rectangle([72, 70, 90, 74], fill=MARBLE)
    d.rectangle([76, 68, 86, 76], fill=MARBLE)
    draw_flames(d, 2, var, bx=79, by=68)

    # Stick with the marshmallow over the flames.
    d.line([(56, 54), (72, 50)], fill=MARBLE, width=2)
    if state != "eaten":
        toast_color = {"toast_a": PAGE, "toast_b": PAGE,
                       "toast_c": BUN, "charred": (110, 92, 74, 255)}[state]
        mx, my = (70, 46) if var == 0 else (71, 47)
        d.rounded_rectangle([mx, my, mx + 4, my + 4], radius=1, fill=toast_color)
        if state == "toast_c":
            d.rectangle([mx + 2, my + 2, mx + 3, my + 3], fill=PATTY)
        if state == "charred" and var:
            d.rectangle([mx - 1, my - 1, mx, my], fill=BODY_DARK)
    else:
        d.rectangle([70, 47, 71, 48], fill=PAGE)  # last crumb
    return im


def draw_cat_dance(state, var=0):
    """Disco night: ball overhead throwing light beams, big side-to-side
    moves with an arm pump, notes in the air. state: 'left' | 'right' |
    'pump' (both arms) | 'spin' (quick swap)."""
    poses = {
        "left": {"paw_side": "l", "paw_raise": 0, "happy": True, "mouth": True,
                 "sway": 2.5, "gaze": "left"},
        "right": {"paw_side": "r", "paw_raise": 0, "happy": True, "mouth": True,
                  "sway": -2.5, "gaze": "right"},
        "pump": {"both_paws": True, "happy": True, "mouth": True, "tail_up": True,
                 "head_dy": -1},
        "spin": {"both_paws": True, "happy": True, "sway": 0.0, "ear_r_dx": 2},
    }
    im = draw_cat(poses[state])
    if state == "left":
        im = np_roll(im, -3)
    elif state == "right":
        im = np_roll(im, 3)
    d = ImageDraw.Draw(im)

    # Disco ball + beams.
    d.rectangle([47, 0, 49, 4], fill=MARBLE)
    d.ellipse([40, 3, 58, 21], fill=MARBLE)
    for fx in (45, 50, 55):
        d.rectangle([fx, 5, fx, 19], fill=(96, 102, 118, 255))
    for fy in (9, 14):
        d.rectangle([42, fy, 56, fy], fill=(96, 102, 118, 255))
    if var:
        d.rectangle([28, 22, 29, 34], fill=CREAM)
        d.rectangle([68, 20, 69, 30], fill=CREAM)
    else:
        d.rectangle([36, 24, 37, 34], fill=CREAM)
        d.rectangle([60, 22, 61, 32], fill=CREAM)
    # notes bouncing at the sides
    _note(d, 24, 40 + (var * 4))
    _note(d, 70, 36 - var * 4)
    return im


def draw_cat_weightlifting(state, var=0):
    """The clean and press, headband on. state: 'ready' (eyeing the bar) |
    'grip' (crouch, paws on the bar) | 'lift' (bar at the chest) |
    'press' (bar overhead) | 'hold' (trembling, sweat) | 'drop' (thud,
    dust, pride). var jitters the hold and swaps the drop bounce."""
    poses = {
        "ready": {"gaze": "down", "headband": True},
        "grip": {"gaze": "down", "squash": 1, "mouth": True, "headband": True},
        "lift": {"gaze": "center", "mouth": True, "head_dy": 1,
                 "headband": True, "sway": 0.0},
        "press": {"both_paws": True, "head_dy": -1, "mouth": True, "gaze": "up",
                  "headband": True},
        "hold": {"both_paws": True, "head_dy": -1, "happy": True, "gaze": "up",
                 "headband": True, "whisker_dy": -1 if var else 1},
        "drop": {"happy": True, "squash": 1, "headband": True, "sway": 2.5},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)

    # Barbell heights through the lift.
    bar_y = {"ready": 72, "grip": 71, "lift": 56,
             "press": 14, "hold": 15 + (var % 2), "drop": 70 + (1 - var)}[state]

    # Bar + plates.
    d.rectangle([33, bar_y, 63, bar_y + 2], fill=MARBLE)
    for px in (29, 60):
        d.rounded_rectangle([px, bar_y - 5, px + 5, bar_y + 7], radius=1,
                            fill=MARBLE)
        d.rectangle([px + 2, bar_y - 3, px + 2, bar_y + 5],
                    fill=(96, 102, 118, 255))

    if state == "grip":
        for ax in (44, 52):
            d.line([(ax, 56), (ax, bar_y)], fill=BODY, width=4)
        d.rounded_rectangle([42, bar_y - 2, 54, bar_y + 4], radius=2, fill=CREAM)
    elif state == "lift":
        for ax, bx in ((44, 36), (52, 60)):
            d.line([(ax, 52), (bx, bar_y + 1)], fill=BODY, width=4)
        d.rounded_rectangle([34, bar_y - 2, 42, bar_y + 3], radius=2, fill=CREAM)
        d.rounded_rectangle([56, bar_y - 2, 64, bar_y + 3], radius=2, fill=CREAM)
    elif state in ("press", "hold"):
        # straight arms up to the bar, paws locked on
        for ax, bx in ((31, 35), (63, 59)):
            d.line([(ax, 28), (bx, bar_y + 2)], fill=BODY, width=4)
        d.rounded_rectangle([33, bar_y - 1, 41, bar_y + 4], radius=2, fill=CREAM)
        d.rounded_rectangle([55, bar_y - 1, 63, bar_y + 4], radius=2, fill=CREAM)
        d.rectangle([45, 43, 46, 45], fill=NOSE)  # effort tongue
        d.rectangle([33, 30, 34, 32], fill=SWEAT)

    if state == "drop" and var:
        for dx in (28, 45, 62):  # impact dust
            d.ellipse([dx, 76, dx + 6, 81], fill=SNOW_SHADE)
    if state == "press" or (state == "hold" and var):
        d.rectangle([36, 26, 37, 28], fill=SWEAT)
    return im


def draw_cat_suit(state, var=0):
    """Fresh: dark suit, white shirt, red tie. state: 'tie' (adjusts the
    knot, both paws) | 'shades' (sunglasses on, strutting) | 'strut_l'/
    'strut_r' (confident sway) | 'point' (finger guns + sparkles)."""
    poses = {
        "tie": {"both_paws": True, "gaze": "down", "sway": 0.0},
        "shades": {"gaze": "right", "mouth": True, "sway": 2.0},
        "strut_l": {"happy": True, "sway": 2.5, "gaze": "left", "paw_dx": 1},
        "strut_r": {"happy": True, "sway": -2.5, "gaze": "right", "paw_dx": -1},
        "point": {"gaze": "right", "mouth": True},
    }
    im = draw_cat(poses[state])
    d = ImageDraw.Draw(im)

    # Suit jacket with a white shirt V and dark lapels.
    d.rounded_rectangle([29, 45, 67, 79], radius=8, fill=MARBLE)
    d.polygon([(40, 45), (48, 45), (48, 58), (40, 52)], fill=PAGE)
    d.polygon([(56, 45), (48, 45), (48, 58), (56, 52)], fill=PAGE)
    lapel = (58, 52, 46, 255)
    d.polygon([(37, 45), (42, 45), (48, 57), (43, 51)], fill=lapel)
    d.polygon([(59, 45), (54, 45), (48, 57), (53, 51)], fill=lapel)
    # The tie: knot + blade.
    d.rectangle([46, 46, 50, 50], fill=BOOK_RED)
    d.polygon([(45, 50), (51, 50), (52, 65), (44, 65)], fill=BOOK_RED)
    d.rectangle([46, 52, 50, 53], fill=(160, 60, 56, 255))

    if state == "tie":
        d.rounded_rectangle([42, 47, 50, 53], radius=2, fill=CREAM)
        d.rounded_rectangle([46, 47, 54, 53], radius=2, fill=CREAM)
    # Sunglasses once the fit is complete.
    if state in ("shades", "strut_l", "strut_r", "point"):
        d.rounded_rectangle([37, 28, 58, 34], radius=2, fill=MARBLE)
        d.rectangle([40, 30, 41, 31], fill=CREAM)
        d.rectangle([52, 30, 53, 31], fill=CREAM)
    if state == "point":
        d.line([(57, 50), (68, 42)], fill=BODY, width=4)
        d.rounded_rectangle([65, 38, 72, 44], radius=2, fill=CREAM)
        for x, y in [(72, 33), (75, 38)]:
            d.rectangle([x, y, x + 1, y + 3], fill=CREAM)
            d.rectangle([x - 1, y + 1, x + 2, y + 2], fill=CREAM)
    return im


BASEBALL = (247, 240, 218, 255)  # white leather, red stitches


def draw_baseball(d, x, y):
    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=BASEBALL)
    d.rectangle([x - 2, y, x - 1, y], fill=NOSE)
    d.rectangle([x + 1, y, x + 2, y], fill=NOSE)


def draw_volleyball(d, x, y):
    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=PAGE)
    d.rectangle([x - 3, y - 1, x + 3, y], fill=SWEAT)
    d.rectangle([x - 1, y - 3, x, y + 3], fill=SWEAT)
    d.rectangle([x + 2, y - 2, x + 3, y - 1], fill=NOSE)


def draw_batter(phase, var=0):
    """Side-view batter facing left, red headband on. phase: 'ready' (bat
    cocked behind) | 'swing1' (bat sweeps down through the zone) |
    'contact' (bat level, ball meeting it) | 'follow' (wrapped around,
    watching it fly)."""
    im = new_canvas()
    d = ImageDraw.Draw(im)
    d.ellipse([44, 68, 76, 74], fill=GROUND_SHADOW)

    def leg(x0, x1):
        d.rectangle([x0, 58, x1, 70], fill=BODY)
        d.rounded_rectangle([x0 - 1, 68, x1 + 1, 72], radius=2, fill=CREAM)

    leg(52, 58)
    leg(64, 70)
    # Body upright, leaning toward the plate (left).
    d.rounded_rectangle([46, 34, 72, 60], radius=8, fill=BODY)
    d.rectangle([50, 36, 68, 37], fill=BODY_LIGHT)
    d.rectangle([50, 56, 68, 58], fill=BODY_DARK)
    draw_blob(d, 62, 46, [(0, 0, 2.6), (3, 3, 2.0)], MARBLE)
    # Tail curling up behind.
    for i, (x, y) in enumerate(_bezier((70, 56), (76, 50), (78, 40), (74, 32), n=12)):
        t = i / 12
        r = 2.4 if t < 0.7 else 2.0
        ring = (0.4 <= t <= 0.6) or t > 0.85
        d.ellipse([x - r, y - r, x + r, y + r], fill=MARBLE if ring else BODY)
    # Head facing left.
    d.rounded_rectangle([40, 20, 62, 40], radius=8, fill=BODY)
    d.ellipse([38, 30, 62, 44], fill=BODY)
    for cx, dy in [(46, -1), (55, 0)]:
        d.polygon([(cx - 4, 24 + dy), (cx, 15 + dy), (cx + 4, 24 + dy)], fill=BODY)
        d.ellipse([cx - 2, 16 + dy, cx + 2, 20 + dy], fill=MARBLE)
    d.rectangle([42, 25, 60, 28], fill=BOOK_RED)  # headband
    d.rectangle([40, 32, 44, 35], fill=EYE)       # eye, locked on the pitch
    d.rectangle([40, 37, 46, 39], fill=CREAM)     # muzzle
    d.rectangle([40, 36, 41, 37], fill=NOSE)
    for wx in (36, 38):
        d.rectangle([wx, 34, wx, 34], fill=WHISKER)
    draw_blob(d, 56, 26, HEAD_MARBLE, MARBLE)

    # Arms + bat through the swing.
    BAT = BUN
    if phase == "ready":
        d.line([(56, 42), (66, 34)], fill=BODY, width=4)
        d.line([(66, 34), (76, 20)], fill=BAT, width=3)
        d.rectangle([75, 18, 77, 20], fill=MARBLE)
        d.rounded_rectangle([54, 40, 60, 45], radius=2, fill=CREAM)
    elif phase == "swing1":
        d.line([(52, 46), (46, 54)], fill=BODY, width=4)
        d.line([(46, 54), (42, 68)], fill=BAT, width=3)
        d.rounded_rectangle([44, 50, 50, 55], radius=2, fill=CREAM)
    elif phase == "contact":
        d.line([(52, 44), (40, 46)], fill=BODY, width=4)
        d.line([(40, 46), (24, 44)], fill=BAT, width=3)
        d.rounded_rectangle([42, 43, 48, 48], radius=2, fill=CREAM)
    else:  # follow
        d.line([(52, 42), (42, 36)], fill=BODY, width=4)
        d.line([(42, 36), (26, 24)], fill=BAT, width=3)
        d.rounded_rectangle([40, 33, 46, 38], radius=2, fill=CREAM)
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
