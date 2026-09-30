"""Action frame generators for the marbled cat mascot.

Each generator returns a list of poses; every pose is a draw_cat /
draw_cat_curled argument dict, except peek which yields (pose, x_offset).
The builder renders each pose twice (2 x 42ms ticks), giving ~12fps stepped
animation per the guide.
"""


def frames_idle():
    """Full of life: tail sway, whisker twitches, looks left and right,
    double ear twitch, breathing, a happy squint moment. Seamless loop."""
    return [
        {"sway": 2.5}, {"sway": 2.5},
        {"gaze": "left", "sway": -2.5},
        {"gaze": "left", "sway": -2.5, "whisker_dy": 1},
        {"gaze": "center", "ear_r_dx": 1, "sway": 2.5},
        {},
        {"gaze": "right", "sway": -2.5},
        {"gaze": "right", "ear_r_dx": 1, "whisker_dy": -1, "sway": -2.5},
        {"gaze": "center"},
        {"body_dy": 1}, {"body_dy": 1, "head_dy": 1},
        {"body_dy": 1, "head_dy": 1}, {"body_dy": 1, "head_dy": 1},
        {"blink": True}, {"blink": True},
        {"happy": True, "sway": 2.5}, {"happy": True, "sway": -2.5},
        {"sway": 1.0},
    ]


def frames_tv():
    """Movie night with real channels: the cat's own show (a tiny gold cat
    walking a stage), an aquarium with a swimming goldfish, a rocket launch
    that jump-scares it (glitched), the weather on the news, channel zaps in
    between. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_tv, glitch_frame

    seq = [
        ("watch", "catshow"), ("watch", "catshow"), ("watch", "catshow"),
        ("lean", "catshow"), ("lean", "catshow"), ("laugh", "catshow"),
        ("watch", "static"),
        ("watch", "fish"), ("watch", "fish"), ("watch", "fish"),
        ("lean", "fish"), ("lean", "fish"),
        ("watch", "static"),
        ("watch", "rocket"), ("watch", "rocket"), ("watch", "rocket"),
        ("scare", "rocket"), ("scare", "rocket"), ("scare", "rocket"),
        ("watch", "news"), ("watch", "news"), ("lean", "news"),
        ("laugh", "news"), ("watch", "bars"), ("watch", "bars"),
    ]
    out = []
    for i, (s, c) in enumerate(seq):
        im = draw_cat_tv(s, c, t=i)
        if s == "scare":
            im = glitch_frame(im, [(30, 8, 5), (56, 10, -6)], invert=True)
        out.append(im)
    return out


CAT_ON_BELT = -8  # paste offset: cat body centered over the treadmill belt


def frames_treadmill():
    """Gym session: warm-up jog, run, all-out sprint (sweat, tongue out),
    exhausted cool-down, then stops and pants on the dead belt. Seamless
    loop; returns Images. Belt scroll speed follows the workout level and
    the console bars light up with it. Returns composed Images."""
    import numpy as np
    from PIL import ImageDraw

    from mascot_sprite import (BODY_DARK, MARBLE, NOSE, SWEAT, draw_cat_running,
                               draw_cat_tired, draw_treadmill, new_canvas)

    def scene(level, belt, state=None, phase=None):
        im = new_canvas()
        d = ImageDraw.Draw(im)
        draw_treadmill(d, belt, level)
        if phase is not None:
            # effort rides on the sprite itself: tongue hangs from the mouth
            # and the sweat drop rides the head bob
            tongue = 2 if level == 0 else (1 if level == 3 else 0)
            cat = draw_cat_running(phase, 0.0, tongue=tongue, sweat=level >= 2,
                                   headband=True)
            im.paste(cat, (CAT_ON_BELT, 0), cat)
        else:
            cat = draw_cat_tired(state, headband=True)
            im.paste(cat, (CAT_ON_BELT, 0), cat)
        return im

    seq = []
    belt = 0
    phase = 0
    # Warm-up jog: slow belt, easy strides held two ticks.
    for _ in range(2):
        for p in range(6):
            belt += 1
            seq.append(scene(1, belt, phase=p))
            seq.append(scene(1, belt, phase=p))
            phase += 1
    # Run.
    for _ in range(2):
        for p in range(6):
            belt += 2
            seq.append(scene(2, belt, phase=p))
            phase += 1
    # All-out sprint.
    for _ in range(3):
        for p in range(6):
            belt += 3
            seq.append(scene(3, belt, phase=p))
            phase += 1
    # Exhausted cool-down, belt slowing.
    for _ in range(2):
        for p in range(6):
            belt += 1
            seq.append(scene(0, belt, phase=p))
            phase += 1
    # Dead belt. Pant.
    for r in range(3):
        belt += 1 if r == 0 else 0
        seq.append(scene(0, belt, state="pant_a"))
        seq.append(scene(0, belt, state="pant_b"))
    return seq


def frames_flexing():
    """Strength set: easy stance, throw the double-biceps, hold the max with
    sparkles, drop, repeat. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_flex
    seq = ["ready", "ready", "ready",
           "flex", "flex", "flex", "max", "max", "max",
           "flex", "flex", "max", "max",
           "ready", "ready", "ready"]
    return [draw_cat_flex(s, headband=True) for s in seq]


def frames_stretching():
    """Flexibility flow: deep arch (chest low, tail up), hold, sway into the
    dip (face to the sky), hold, settle, repeat. Seamless loop; returns
    Images. The holds get a tiny 1px breathing bob."""
    import numpy as np
    from PIL import Image

    from mascot_sprite import draw_cat_stretch, new_canvas

    def bob(im, dy):
        if not dy:
            return im
        arr = np.array(im)
        out = np.zeros_like(arr)
        h = arr.shape[0]
        out[max(0, dy):min(h, h + dy)] = arr[max(0, -dy):min(h, h - dy)]
        return Image.fromarray(out)

    seq = [
        ("arch", 0), ("arch", 0), ("arch", 1), ("arch", 0),
        ("dip", 0), ("dip", 0), ("dip", -1), ("dip", 0),
        ("arch", 0), ("arch", 1), ("arch", 0), ("arch", 0),
        ("settle", 0), ("settle", 0), ("settle", 0), ("settle", 0),
    ]
    return [bob(draw_cat_stretch(s), dy) for s, dy in seq]


def frames_phone():
    """The call: ring ring on the floor, pick up — hello! — chats with the
    free paw waving, hears something SHOCKING, recovers, waves bye, and the
    phone starts ringing again. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_phone
    seq = ["ring", "ring", "ring", "ring", "pick", "pick",
           "chat1", "chat2", "chat1", "chat2", "chat1",
           "shock", "shock", "chat2", "chat1", "chat2",
           "bye", "bye"]
    return [draw_cat_phone(s) for s in seq]


def frames_graduating():
    """Commencement: stands proud with the diploma (tassel swinging side to
    side), hurls the cap skyward through the confetti, catches it and cheers.
    Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_grad
    seq = [("ready", 0)] * 3 + [("ready", 1)] * 3 + [
        ("toss", 0), ("toss", 1), ("toss", 0), ("toss", 1),
        ("cheer", 0), ("cheer", 1), ("cheer", 0), ("cheer", 1),
        ("cheer", 0), ("ready", 0),
    ]
    return [draw_cat_grad(s, v) for s, v in seq]


def frames_basketball():
    """Streetball: dribbles beside it (arm push, ball squash on the bounce),
    rises into a shot, the ball arcs to the corner hoop — swish — net sway,
    sparkle, back to dribbling. Seamless loop; returns Images."""
    from PIL import ImageDraw

    from mascot_sprite import (BODY, CREAM, MARBLE, draw_basketball, draw_cat,
                               draw_hoop, new_canvas)

    def scene(tag, bx, by):
        im = new_canvas()
        d = ImageDraw.Draw(im)
        draw_hoop(d)
        poses = {
            "up": {"gaze": "down", "mouth": True, "sway": 2.0, "headband": True},
            "down": {"gaze": "down", "squash": 1, "sway": -2.0, "headband": True},
            "shoot": {"both_paws": True, "mouth": True, "gaze": "up",
                      "head_dy": -1, "headband": True},
            "fly": {"both_paws": True, "mouth": True, "gaze": "right",
                    "head_dy": -1, "headband": True},
            "score": {"both_paws": True, "happy": True, "gaze": "right",
                      "headband": True},
        }
        im_cat = draw_cat(poses[tag])
        im.paste(im_cat, (0, 0), im_cat)
        d = ImageDraw.Draw(im)
        if tag == "down":
            # pushing arm reaching down to the bounce
            d.line([(57, 52), (63, 61)], fill=BODY[:3] + (255,), width=4)
            d.rounded_rectangle([60, 58, 66, 63], radius=2, fill=CREAM)
        sq = tag == "down"
        draw_basketball(d, bx, by, rx=4 if sq else 3, ry=2 if sq else 3)
        if tag == "score":
            # swish sparkle at the rim
            d.rectangle([88, 13, 89, 16], fill=CREAM)
            d.rectangle([86, 14, 91, 15], fill=CREAM)
        return im

    seq = [
        ("up", 63, 45), ("down", 63, 60), ("up", 63, 45), ("down", 63, 60),
        ("up", 63, 45), ("down", 63, 60),
        ("shoot", 67, 34), ("shoot", 72, 27),
        ("fly", 78, 21), ("fly", 81, 18),
        ("score", 81, 19), ("score", 81, 24), ("score", 81, 19), ("score", 81, 24),
    ]
    return [scene(*item) for item in seq]


def frames_snowman():
    """Winter build: rolls a snowball, pats it out, stacks the middle and
    head (plop-puffs), plants the carrot nose, celebrates under falling
    snow, then starts a fresh one. Seamless loop; returns Images."""
    import numpy as np
    from PIL import ImageDraw

    from mascot_sprite import (BALL, BODY, CREAM, MARBLE, SNOW, SNOW_SHADE,
                               draw_cat, draw_snowman, new_canvas)

    SNOW_DOTS = [(12, 0), (34, 5), (56, 2), (88, 7), (22, 9), (80, 12)]

    def scene(stage, var, idx):
        im = new_canvas()
        d = ImageDraw.Draw(im)

        levels = {"roll": 0, "pat": 1, "place_mid": 2, "pat2": 2,
                  "place_head": 3, "carrot": 3, "done": 4}
        level = levels[stage]
        rx = 8 if stage == "roll" else 11
        draw_snowman(d, level, rx_bottom=rx)

        poses = {
            "roll": {"gaze": "right", "mouth": True, "tail_up": True},
            "pat": {"gaze": "right", "mouth": True, "tail_up": True},
            "place_mid": {"both_paws": True, "gaze": "right", "mouth": True,
                          "tail_up": True},
            "pat2": {"gaze": "right", "mouth": True, "tail_up": True},
            "place_head": {"both_paws": True, "gaze": "right", "mouth": True,
                           "tail_up": True},
            "carrot": {"gaze": "right", "paw_side": "l", "paw_raise": 1,
                       "mouth": True, "head_dy": -1, "tail_up": True},
            "done": {"both_paws": True, "happy": True, "gaze": "right",
                     "sway": 2.5, "ear_r_dx": 1, "tail_up": True},
        }
        im_cat = draw_cat(poses[stage])
        im.paste(im_cat, (0, 0), im_cat)
        d = ImageDraw.Draw(im)

        # reach/push/pat arm overlays toward the snowman
        if stage == "roll":
            d.line([(57, 54), (63, 63)], fill=BODY, width=4)
            d.rounded_rectangle([60, 60, 66, 65], radius=2, fill=CREAM)
        elif stage in ("pat", "pat2"):
            top = 58 if stage == "pat" else 43
            d.line([(57, 54), (64, top + 2)], fill=BODY, width=4)
            d.rounded_rectangle([61, top, 67, top + 5], radius=2, fill=CREAM)
        elif stage == "carrot":
            d.line([(57, 52), (66, 38)], fill=BODY, width=4)
            d.rounded_rectangle([63, 35, 69, 40], radius=2, fill=CREAM)
            if var:  # carrot appears on the second beat
                d.polygon([(65, 36), (71, 35), (71, 38)], fill=BALL)

        # plop puffs where a ball just landed
        if stage.startswith("place") and var:
            spot = (74, 50) if stage == "place_mid" else (74, 37)
            for px, py in [(spot[0] - 7, spot[1] - 4), (spot[0] + 6, spot[1] - 5),
                           (spot[0], spot[1] - 7)]:
                d.ellipse([px - 2, py - 2, px + 2, py + 1], fill=SNOW_SHADE)

        # snowfall, stepped
        fall = (idx * 5 + var * 3) % 96
        for sx, sy in SNOW_DOTS:
            y = (sy + fall) % 96
            d.rectangle([sx, y, sx + 1, y + 1], fill=SNOW)
        return im

    stages = [
        ("roll", 0), ("roll", 1), ("roll", 0), ("roll", 1),
        ("pat", 0), ("pat", 1), ("pat", 0),
        ("place_mid", 0), ("place_mid", 1),
        ("pat2", 0), ("pat2", 1), ("pat2", 0),
        ("place_head", 0), ("place_head", 1),
        ("carrot", 0), ("carrot", 1), ("carrot", 0),
        ("done", 0), ("done", 1), ("done", 0), ("done", 1), ("done", 0),
    ]
    return [scene(stage, var, i) for i, (stage, var) in enumerate(stages)]


def frames_rain():
    """Cozy-melancholy: the cat shelters under a red umbrella while rain
    streaks down on both sides. Its tail pokes out and drips, a stray drop
    lands right on its nose (mini shock), it shakes the water off, then
    decides to enjoy the pitter-patter. Seamless loop; returns Images."""
    import numpy as np
    from PIL import ImageDraw

    from mascot_sprite import (BODY, BOOK_RED, CREAM, MARBLE, SWEAT, draw_cat,
                               new_canvas)

    RAIN_X = [4, 12, 19, 76, 84, 91, 8, 80]  # only outside the umbrella span

    def drop(d, x, y, length=3):
        d.rectangle([x, y, x, y + length], fill=SWEAT)

    def scene(state, var, idx):
        im = new_canvas()
        d = ImageDraw.Draw(im)

        poses = {
            "hold": {"gaze": "center", "sway": 1.5},
            "drip": {"gaze": "right", "sway": 2.0, "ear_r_dx": 1},
            "nose": {"gaze": "down", "shock": var == 1},
            "shake": {"whisker_dy": -1 if var == 0 else 1,
                      "ear_r_dx": -2 if var == 0 else 2, "sway": -2.5},
            "cozy": {"happy": True, "sway": 2.5},
        }
        im_cat = draw_cat(poses[state])
        im.paste(im_cat, (0, 0), im_cat)
        d = ImageDraw.Draw(im)

        # tail/canopy-edge drip building under the rim
        if state == "drip":
            dy = 19 + var * 4
            drop(d, 76, dy)
            if var:
                drop(d, 76, 40)
                d.rectangle([76, 54, 76, 55], fill=SWEAT)  # lands on the tail

        # umbrella: red dome + scalloped rim, ribs, pole on the grip side
        d.pieslice([36, -6, 84, 38], 180, 360, fill=BOOK_RED)
        for cx in (48, 60, 72):
            d.ellipse([cx - 4, 14, cx + 4, 20], fill=BOOK_RED)  # scallops
        d.rectangle([49, 3, 50, 15], fill=CREAM)  # ribs
        d.rectangle([70, 3, 71, 15], fill=CREAM)
        d.rectangle([59, 16, 61, 56], fill=MARBLE)  # pole beside the head
        d.line([(57, 52), (60, 47)], fill=BODY, width=4)
        d.rounded_rectangle([56, 43, 63, 50], radius=2, fill=CREAM)

        # wind-slanted drop sneaks under the rim and boops the nose
        if state == "nose":
            if var == 0:
                drop(d, 70, 24, length=2)
                drop(d, 60, 33, length=2)
            else:
                drop(d, 47, 40, length=2)
                d.rectangle([45, 44, 48, 44], fill=SWEAT)  # splash on nose

        # shake-off flings droplets
        if state == "shake":
            for dx, dy in [(28, 26), (66, 24), (24, 44), (70, 42)]:
                d.rectangle([dx + var * 2, dy, dx + 1 + var * 2, dy + 2], fill=SWEAT)

        # rain streaks (sides only) + ground splashes, stepped
        fall = (idx * 6 + var * 4) % 88
        for i, rx in enumerate(RAIN_X):
            y = (i * 11 + fall) % 88
            drop(d, rx, y, length=3 + (i % 2))
        for sx in (6, 16, 79, 89):
            sy = 76 + (idx + sx) % 3
            d.rectangle([sx - 1, sy, sx + 2, sy], fill=SWEAT)
        return im

    seq = [
        ("hold", 0), ("hold", 1), ("hold", 0), ("hold", 1),
        ("drip", 0), ("drip", 1), ("drip", 0),
        ("hold", 0), ("hold", 1),
        ("nose", 0), ("nose", 1),
        ("shake", 0), ("shake", 1), ("shake", 0),
        ("cozy", 0), ("cozy", 1), ("cozy", 0), ("cozy", 1),
    ]
    return [scene(s, v, i) for i, (s, v) in enumerate(seq)]


def frames_piano():
    """Recital: paws wander the keys (pressed keys dip under them), swaying
    and humming, notes floating up — then the crescendo with a triple-note
    burst. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_piano
    seq = [
        ("l", False, 0), ("l", False, 1),
        ("r", False, 0), ("r", False, 1),
        ("l", False, 0), ("c", False, 0), ("r", False, 0), ("c", False, 1),
        ("cresc", True, 0), ("cresc", True, 1),
        ("c", False, 0), ("c", False, 1),
        ("l", False, 0), ("c", False, 1),
    ]
    return [draw_cat_piano(h, cresc=c, var=v) for h, c, v in seq]


def frames_baking():
    """Bake day: toque on, crack the egg, stir the bowl, sneak a spoon lick,
    POOF — the cake appears, cherry on top, sparkle celebration. Seamless
    loop; returns Images."""
    from mascot_sprite import draw_cat_baking
    seq = [
        ("crack", 0), ("crack", 1),
        ("stir", 0), ("stir", 1), ("stir", 2), ("stir", 0), ("stir", 1),
        ("lick", 0), ("lick", 0), ("lick", 1),
        ("cake", 0), ("cake", 1),
        ("decorate", 0), ("decorate", 1),
        ("celebrate", 0), ("celebrate", 1), ("celebrate", 0),
    ]
    return [draw_cat_baking(s, v) for s, v in seq]


def frames_firefighter():
    """Fire duty: braced on the hose, hosing the blaze down to embers to
    smoke — then the fire relights and it's SHOCKED. Seamless loop; returns
    Images."""
    import numpy as np
    from PIL import ImageDraw

    from mascot_sprite import (CREAM, MARBLE, SNOW_SHADE, SWEAT,
                               draw_cat_firefighter, draw_flames, draw_smoke,
                               new_canvas)

    ARC = [(68, 52), (74, 49), (80, 50), (85, 55), (88, 61)]

    def scene(state, var, idx):
        im = new_canvas()
        d = ImageDraw.Draw(im)

        size = {"blaze": 3, "spray": 2, "ember": 1, "out": 0, "relight": 2}[state]
        if state == "out":
            draw_smoke(d, 85, )
        else:
            draw_flames(d, size, var)

        im_cat = draw_cat_firefighter("spray" if state != "out" and state != "relight"
                                      else ("ease" if state == "out" else "shock"))
        im.paste(im_cat, (0, 0), im_cat)
        d = ImageDraw.Draw(im)

        # Hose from the braced paws to the nozzle (light gray on the dark coat).
        d.line([(36, 60), (56, 58)], fill=SNOW_SHADE, width=5)
        d.rounded_rectangle([68, 52, 75, 61], radius=2, fill=SNOW_SHADE)
        d.rectangle([73, 54, 75, 59], fill=MARBLE)  # nozzle tip
        d.rounded_rectangle([51, 55, 58, 61], radius=2, fill=CREAM)  # grip paws
        d.rounded_rectangle([57, 56, 63, 61], radius=2, fill=CREAM)

        # Water arc while spraying.
        if state in ("blaze", "spray", "ember"):
            for i in range(5):
                drop = ARC[(i + idx) % len(ARC)]
                d.rectangle([drop[0], drop[1] + var, drop[0] + 1,
                             drop[1] + 2 + var], fill=SWEAT)
            d.rectangle([ARC[-1][0], ARC[-1][1] + 2, ARC[-1][0] + 3,
                         ARC[-1][1] + 4], fill=SWEAT)  # splash on the fire
        return im

    stages = [
        ("blaze", 0), ("blaze", 1), ("blaze", 0), ("blaze", 1),
        ("spray", 0), ("spray", 1), ("spray", 0), ("spray", 1),
        ("ember", 0), ("ember", 1), ("ember", 0), ("ember", 1),
        ("out", 0), ("out", 1), ("out", 0), ("out", 1),
        ("relight", 0), ("relight", 1),
    ]
    return [scene(s, v, i) for i, (s, v) in enumerate(stages)]


def frames_discovery():
    """The discovery: naps under the tree, an apple falls and bonks its
    head — and the apple is ELECTRIFIED. Wakes up shocked, holds the
    sparking apple up, lightbulb moment. Seamless loop; returns Images."""
    import numpy as np
    from PIL import ImageDraw

    from mascot_sprite import (BALL, BODY_DARK, CREAM, EYE, LETTUCE, MARBLE,
                               SWEAT, YOLK, draw_cat, draw_cat_curled,
                               new_canvas)
    CAT_X = -12  # curled/sitting cat shifted under the canopy

    def apple(d, x, y, spark=False):
        d.rectangle([x + 1, y - 1, x + 1, y - 1], fill=MARBLE)  # stem
        d.ellipse([x, y, x + 3, y + 3], fill=BALL)
        if spark:
            for sx, sy in [(x - 2, y - 1), (x + 4, y), (x - 1, y + 4),
                           (x + 4, y + 3), (x + 2, y - 3)]:
                d.rectangle([sx, sy, sx, sy + 1], fill=SWEAT)
            d.rectangle([x - 1, y + 1, x, y + 1], fill=YOLK)

    def scene(state, var):
        im = new_canvas()
        d = ImageDraw.Draw(im)

        # Tree: trunk on the right, leafy canopy overhead.
        d.rectangle([70, 14, 88, 84], fill=BODY_DARK)
        d.rectangle([77, 18, 79, 84], fill=MARBLE)
        d.ellipse([26, -16, 92, 26], fill=LETTUCE)
        d.ellipse([52, -10, 92, 18], fill=LETTUCE)
        d.rectangle([35, 18, 37, 19], fill=MARBLE)  # apple hanging over the head
        d.ellipse([34, 19, 38, 23], fill=BALL)

        if state == "sleep":
            zz = [[(58, 40, 2)], [(60, 36, 2)], [(62, 32, 3)],
                  [(65, 27, 3)], [(68, 21, 4)], [(70, 15, 4)]][var % 6]
            cat = draw_cat_curled({"zz": zz})
            im.paste(cat, (CAT_X, 0), cat)
        elif state == "fall":
            cat = draw_cat_curled({"zz": [(60, 36, 2)]})
            im.paste(cat, (CAT_X, 0), cat)
            apple(d, 34, 20 + var * 9)
        elif state == "bonk":
            cat = draw_cat_curled({"zz": []})
            im.paste(cat, (CAT_X, 0), cat)
            apple(d, 32, 28)
            for sx, sy in [(28, 30), (38, 28), (26, 33)]:
                d.rectangle([sx, sy, sx + 1, sy + 1], fill=YOLK)
        elif state == "wake":
            cat = draw_cat({"shock": True, "gaze": "up", "mouth": True})
            im.paste(cat, (CAT_X, 0), cat)
            apple(d, 50, 60 + var * 3)
        elif state == "spark":
            cat = draw_cat({"paw_side": "r", "paw_raise": 0, "mouth": True,
                            "gaze": "center", "shock": var == 1})
            im.paste(cat, (CAT_X, 0), cat)
            apple(d, 48, 21, spark=True)
        else:  # eureka
            cat = draw_cat({"both_paws": True, "happy": True, "gaze": "up",
                            "tail_up": True})
            im.paste(cat, (CAT_X, 0), cat)
            apple(d, 48 + CAT_X + 12, 16, spark=True)
            # the idea bulb
            d.rounded_rectangle([44, 2, 52, 9], radius=2, fill=YOLK)
            d.rectangle([45, 9, 51, 11], fill=MARBLE)
            d.rectangle([45, 4, 46, 5], fill=CREAM)
            for rx, ry in [(40, 0), (56, 2), (42, 12), (54, 12)]:
                d.rectangle([rx, ry, rx + 1, ry + 1], fill=YOLK)
        return im

    seq = [
        ("sleep", 0), ("sleep", 1), ("sleep", 2), ("sleep", 3),
        ("sleep", 4), ("sleep", 5),
        ("fall", 0), ("fall", 1),
        ("bonk", 0), ("bonk", 1),
        ("wake", 0), ("wake", 1),
        ("spark", 0), ("spark", 1),
        ("eureka", 0), ("eureka", 1), ("eureka", 0), ("eureka", 1),
    ]
    return [scene(s, v) for s, v in seq]


def frames_plane():
    """Weekend aviator: putters through the sky in a tiny red plane —
    spinning propeller, scarf fluttering, clouds gliding by, gentle bob.
    Seamless loop (clouds wrap on the exact loop distance); returns Images."""
    from PIL import ImageDraw

    from mascot_sprite import (BODY, BOOK_RED, CREAM, MARBLE, SNOW, SCREEN,
                               draw_cat, new_canvas)

    CLOUDS = [(6, 16, 1.0), (38, 36, 0.7), (58, 8, 0.85)]
    BOB = [0, 1, 2, 1]

    def cloud(d, x, y, s):
        d.ellipse([x, y, x + 18 * s, y + 8 * s], fill=SNOW)
        d.ellipse([x + 6 * s, y - 5 * s, x + 16 * s, y + 4 * s], fill=SNOW)

    def frame(i):
        dy = BOB[i % 4]
        im = new_canvas()

        # Sky: clouds wrap on a 72px grid — exactly the per-loop scroll.
        d = ImageDraw.Draw(im)
        for x0, y, s in CLOUDS:
            x = (x0 - i * 3) % 72
            for off in (0, 72):
                cloud(d, x + off - 4, y, s)

        # Cat raised into the cockpit; cropped at the fuselage line so the
        # legs can never dangle below the plane.
        cy = dy - 12
        cat = draw_cat({"gaze": "right", "mouth": i % 8 < 6,
                        "happy": i % 8 >= 6})
        cat = cat.crop((0, 0, 96, 64))
        im.paste(cat, (-4, cy), cat)
        d = ImageDraw.Draw(im)

        # Aviator scarf fluttering behind the neck.
        fl = 1 if i % 2 else -1
        d.rectangle([30 + fl, 25 + cy, 37 + fl, 27 + cy], fill=BOOK_RED)
        d.rectangle([23 - fl, 27 + cy, 29 - fl, 29 + cy], fill=BOOK_RED)

        # Chunky little plane.
        d.polygon([(8, 36 + cy), (20, 36 + cy), (18, 22 + cy), (10, 26 + cy)],
                  fill=BOOK_RED)  # tail fin
        d.rounded_rectangle([12, 36 + cy, 76, 64 + cy], radius=10, fill=BOOK_RED)
        d.rectangle([14, 44 + cy, 74, 46 + cy], fill=CREAM)  # stripe
        d.rounded_rectangle([24, 46 + cy, 54, 56 + cy], radius=4,
                            fill=(150, 50, 40, 255))  # wing
        # control stick + paw above the panel
        d.rectangle([58, 28 + cy, 60, 38 + cy], fill=MARBLE)
        d.line([(48, 34 + cy), (57, 31 + cy)], fill=BODY, width=4)
        d.rounded_rectangle([53, 28 + cy, 60, 34 + cy], radius=2, fill=CREAM)
        # nose cowl + spinning propeller (alternating blade blur)
        d.rounded_rectangle([62, 34 + cy, 76, 62 + cy], radius=4, fill=MARBLE)
        d.ellipse([70, 46 + cy, 76, 52 + cy], fill=MARBLE)
        if i % 2:
            d.rectangle([68, 28 + cy, 72, 68 + cy], fill=SNOW)
        else:
            d.rectangle([56, 46 + cy, 86, 50 + cy], fill=SNOW)
        return im

    return [frame(i) for i in range(24)]


def frames_noodles():
    """Ramen night: slurps the strand shorter and shorter until the bowl is
    empty — pure happiness. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_noodles
    seq = [("slurp1", 0), ("slurp1", 1), ("slurp2", 0), ("slurp2", 1),
           ("slurp3", 0), ("slurp3", 1),
           ("done", 0), ("done", 1), ("done", 0), ("done", 1)]
    return [draw_cat_noodles(s, v) for s, v in seq]


def frames_gaming():
    """Gamer mode: mashing the controller with full-body jitter, locked-in
    shock eyes for the clutch, then the WIN with confetti. Seamless loop."""
    from mascot_sprite import draw_cat_gaming
    seq = [("mash", 0), ("mash", 1), ("mash", 0), ("mash", 1),
           ("intense", 0), ("intense", 1), ("intense", 0), ("intense", 1),
           ("mash", 0), ("mash", 1),
           ("win", 0), ("win", 1), ("win", 0), ("win", 1), ("win", 0), ("win", 1)]
    return [draw_cat_gaming(s, v) for s, v in seq]


def frames_campfire():
    """Night under the stars: toasts a marshmallow over the campfire until
    it's perfectly golden — a little too charred — then eats it. Seamless."""
    from mascot_sprite import draw_cat_campfire
    seq = [("toast_a", 0), ("toast_a", 1), ("toast_b", 0), ("toast_b", 1),
           ("toast_c", 0), ("toast_c", 1), ("toast_c", 0),
           ("charred", 0), ("charred", 1),
           ("eaten", 0), ("eaten", 1), ("eaten", 0), ("eaten", 1), ("eaten", 0)]
    return [draw_cat_campfire(s, v) for s, v in seq]


def frames_dance():
    """Disco night: ball throwing beams, big side-to-side moves with an arm
    pump and notes in the air. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_dance
    seq = [("left", 0), ("left", 1), ("right", 0), ("right", 1),
           ("left", 0), ("right", 1), ("left", 1), ("right", 0),
           ("pump", 0), ("pump", 1), ("pump", 0), ("pump", 1),
           ("spin", 0), ("spin", 1), ("pump", 0), ("pump", 1)]
    return [draw_cat_dance(s, v) for s, v in seq]


def frames_weightlifting():
    """The clean and press: stares down the bar, grips, cleans it to the
    chest, presses overhead, holds the tremble, drops it with a dust thud
    and drinks in the glory. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_weightlifting
    seq = [("ready", 0), ("ready", 1), ("ready", 0),
           ("grip", 0), ("grip", 1),
           ("lift", 0), ("lift", 1), ("lift", 0),
           ("press", 0), ("press", 1),
           ("hold", 0), ("hold", 1), ("hold", 0), ("hold", 1),
           ("drop", 0), ("drop", 1), ("drop", 0), ("drop", 1)]
    return [draw_cat_weightlifting(s, v) for s, v in seq]


def frames_suit():
    """Fresh: adjusts the tie, the sunglasses come down, struts side to
    side, ends on finger guns with sparkles. Seamless loop; returns
    Images."""
    from mascot_sprite import draw_cat_suit
    seq = [("tie", 0), ("tie", 1), ("tie", 0), ("tie", 1),
           ("shades", 0), ("shades", 1), ("shades", 0), ("shades", 1),
           ("strut_l", 0), ("strut_r", 1), ("strut_l", 0), ("strut_r", 1),
           ("point", 0), ("point", 1), ("point", 0), ("point", 1)]
    return [draw_cat_suit(s, v) for s, v in seq]


def frames_baseball():
    """At the plate: the pitch comes in, one level swing — CRACK — and the
    ball rockets off to the sky while the batter watches it go. Seamless
    loop; returns Images."""
    from PIL import ImageDraw

    from mascot_sprite import (CREAM, YOLK, draw_batter, draw_baseball,
                               new_canvas)

    def scene(phase, bx=None, by=None):
        im = draw_batter(phase)
        d = ImageDraw.Draw(im)
        if bx is not None:
            draw_baseball(d, bx, by)
            if phase == "contact":
                d.rectangle([22, 40, 26, 41], fill=YOLK)
                d.rectangle([24, 38, 25, 43], fill=YOLK)
            if bx and bx > 40:  # speed lines on the leave
                d.rectangle([bx - 12, by, bx - 7, by], fill=CREAM)
                d.rectangle([bx - 14, by + 3, bx - 10, by + 3], fill=CREAM)
        return im

    seq = [
        ("ready", None, None), ("ready", None, None),
        ("ready", 8, 46), ("ready", 16, 46),
        ("swing1", 26, 46),
        ("contact", 32, 46), ("contact", 34, 45),
        ("follow", 48, 38), ("follow", 66, 28),
        ("follow", 84, 18),
        ("follow", None, None), ("follow", None, None),
    ]
    return [scene(*s) for s in seq]


def frames_volleyball():
    """Bump, set, SPIKE: receives the ball off the forearms, watches it
    rise, jumps and hammers it down. Seamless loop; returns Images."""
    from PIL import ImageDraw

    from mascot_sprite import (BODY, CREAM, draw_cat, draw_volleyball,
                               new_canvas)

    def scene(phase, bx, by, var):
        im = new_canvas()
        d = ImageDraw.Draw(im)
        airborne = phase == "spike"
        dy = -5 if airborne else 0
        pose = {
            "bump": {"gaze": "down", "squash": 1, "headband": True},
            "risen": {"both_paws": True, "gaze": "up", "mouth": True,
                      "headband": True},
            "spike": {"both_paws": True, "mouth": True, "gaze": "down",
                      "headband": True, "shadow": False},
            "reset": {"gaze": "up", "headband": True},
        }[phase]
        cat = draw_cat(pose)
        if airborne:
            d.ellipse([28, 74, 68, 80], fill=(20, 18, 16, 56))  # ground shadow
        im.paste(cat, (0, dy), cat)
        d = ImageDraw.Draw(im)

        if phase == "bump":
            # joined forearms under the ball
            d.line([(42, 52), (48, 60)], fill=BODY, width=4)
            d.line([(54, 52), (48, 60)], fill=BODY, width=4)
            d.rounded_rectangle([43, 58, 53, 64], radius=2, fill=CREAM)
            draw_volleyball(d, bx, by)
        elif phase == "risen":
            draw_volleyball(d, bx, by)
        elif phase == "spike":
            # hammer arm swinging down at the ball
            d.line([(56, 24 + dy), (bx - 2, by - 2)], fill=BODY, width=4)
            d.rounded_rectangle([bx - 6, by - 5, bx, by + 1], radius=2, fill=CREAM)
            draw_volleyball(d, bx, by)
        else:  # reset: ball drops back in from above
            draw_volleyball(d, 48, 6 + var * 4)
        return im

    seq = [
        ("reset", 48, 8, 0),
        ("bump", 48, 60, 0), ("bump", 48, 58, 1),
        ("risen", 48, 42, 0), ("risen", 54, 30, 1), ("risen", 58, 20, 0),
        ("spike", 62, 28, 0), ("spike", 68, 38, 1),
        ("bump", 48, 60, 0),
    ]
    return [scene(*s) for s in seq]


def frames_camera():
    """Best Take in one loop: raises the camera, squints through the
    viewfinder, FLASH, lowers it to review the shot on the screen, then
    celebrates the keep. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_camera
    seq = [("down", 0), ("down", 1),
           ("aim", 0), ("aim", 1), ("squint", 0), ("squint", 1),
           ("flash", 0), ("flash", 1),
           ("review", 0), ("review", 1), ("review", 0),
           ("celebrate", 0), ("celebrate", 1), ("celebrate", 0)]
    return [draw_cat_camera(s, v) for s, v in seq]


def frames_grooming():
    """Paw-lick groom with tongue darts, a cheek wipe, then perks up.
    Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_groom
    seq = ["settle", "raise", "lick1", "lick2", "lick1", "lick2",
           "wipe", "wipe", "raise", "settle", "perk", "perk", "settle", "settle"]
    return [draw_cat_groom(s) for s in seq]


def frames_boing():
    """Excited bounce: crouch, spring with paws and tail flying, hang time,
    land with a squash, shake it off. Seamless loop; returns Images."""
    import numpy as np
    from PIL import Image

    from mascot_sprite import draw_cat_boing

    def at(state, dy=0):
        im = draw_cat_boing(state)
        if not dy:
            return im
        arr = np.array(im)
        out = np.zeros_like(arr)
        h = arr.shape[0]
        out[max(0, dy):min(h, h + dy)] = arr[max(0, -dy):min(h, h - dy)]
        return Image.fromarray(out)

    seq = [
        ("settle", 0), ("crouch", 0), ("crouch", 0),
        ("up", 0), ("hang", -4), ("hang", -6), ("hang", -4), ("up", -1),
        ("land", 0), ("land", 0),
        ("shake_a", 0), ("shake_b", 0), ("shake_a", 0), ("shake_b", 0),
        ("settle", 0), ("settle", 0), ("settle", 0),
    ]
    return [at(s, dy) for s, dy in seq]


def frames_wave():
    """Anticipation crouch, paw up, three stepped wags, settle. Loopable."""
    return [
        {},
        {"squash": 1, "mouth": True},
        {"paw_raise": 0, "mouth": True, "head_dy": -1, "gaze": "right"},
        {"paw_raise": -1, "mouth": True, "head_dy": -1, "gaze": "right"},
        {"paw_raise": 1, "mouth": True, "head_dy": -1, "gaze": "right", "sway": 2.5},
        {"paw_raise": -1, "mouth": True, "head_dy": -1, "gaze": "right", "sway": -2.5},
        {"paw_raise": 1, "mouth": True, "head_dy": -1, "gaze": "right", "happy": True},
        {"paw_raise": -1, "mouth": True, "head_dy": -1, "gaze": "right", "happy": True},
        {"paw_raise": 1, "mouth": True, "head_dy": -1, "gaze": "right", "happy": True},
        {"paw_raise": 0, "mouth": True, "head_dy": -1, "happy": True},
        {"mouth": True, "blink": True},
        {},
    ]


def frames_float():
    """Weightless bob; tail and ears lag, legs dangle, eyes drift around.
    Loopable."""
    offs = [0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1]
    gazes = ["center", "up", "up", "left", "center", "down",
             "center", "down", "down", "right", "right", "center"]
    poses = []
    for dy, g in zip(offs, gazes):
        poses.append({
            "body_dy": dy,
            "head_dy": dy - 1,
            "sway": 1.5 if dy < 0 else -1.5,
            "ear_r_dx": -1 if dy < 0 else 0,
            "mouth": True,
            "gaze": g,
            "paw_dx": -1 if dy < 0 else 1,
            "shadow": False,
        })
    return poses


def frames_sleep():
    """Curled coil, slow breath, Zzz glyphs cycling small to large. Loopable."""
    zz_cycle = [
        [(58, 40, 2)], [(60, 36, 2)], [(62, 32, 3)],
        [(65, 27, 3)], [(68, 21, 4)], [(70, 15, 4)],
        [(70, 15, 4)], [], [], [], [], [],
    ]
    poses = []
    for i in range(12):
        poses.append({
            "breath": 1 if i in (6, 7, 8, 9) else 0,
            "zz": zz_cycle[i],
        })
    return poses


GLITCH_BANDS = [
    [(20, 6, 4), (52, 4, -5)],
    [(36, 8, -4)],
    [(18, 5, 3), (58, 7, 4)],
    [(44, 6, -3)],
]


def frames_glitch():
    """Idle poses with retro slice-glitches, a ghost frame and an invert.
    Heart eyes never glitch — that moment stays exclusive to eating."""
    from mascot_sprite import draw_cat, glitch_frame

    poses = []
    for p in frames_idle():
        p = dict(p)
        if p.pop("heart_eyes", False):
            p["happy"] = True
        poses.append(p)
    imgs = [draw_cat(p) for p in poses]
    hits = {2, 3, 6, 9, 10}
    out = []
    for i, im in enumerate(imgs):
        if i in hits:
            bands = GLITCH_BANDS[i % len(GLITCH_BANDS)]
            im = glitch_frame(im, bands,
                              ghost=(i == 3),
                              invert=(i == 6))
        out.append(im)
    return out


def frames_running():
    """Zoomies and consequences: nose-first sprint across (facing right),
    off-screen, a beat — then it re-enters from the right on the tired walk
    of shame: slouched, tongue lolling, sweat drop, one huffing pant-stop
    mid-frame, then shuffles off left. One-shot. Yields tagged tuples:
    ('sprint', phase, ox[, speed]) | ('tired', state, ox) | ('empty',).
    Sprint frames hold one tick (~24fps); tired walk holds two (slow)."""
    enter = [-95, -85, -75, -65]
    sprint = [-58, -51, -44, -37, -30, -23, -16, -9, -2]     # ~7px per tick:
    exit_ = [5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89]  # legs read
    seq = []
    phase = 0
    for ox in enter * 2:
        seq.append(("sprint", phase % 6, ox))
        phase += 1
    for ox in sprint:
        seq.append(("sprint", phase % 6, ox, 1.25))
        phase += 1
    for ox in exit_:
        seq.append(("sprint", phase % 6, ox, 1.25))
        phase += 1
    seq += [("empty",)] * 8  # it's resting. it earned this.

    # The tired walk of shame, right to left (mirrored sprite).
    walk = ["walk0", "walk1"]
    k = 0
    for ox in (98, 91, 84, 77, 70, 63):
        seq.append(("tired", walk[k % 2], ox))
        seq.append(("tired", walk[k % 2], ox))
        k += 1
    # Pant-stop in view: huffing, sweat sliding, tongue out.
    for _ in range(3):
        seq.append(("tired", "pant_a", 56))
        seq.append(("tired", "pant_b", 56))
    for ox in range(50, -66, -8):
        seq.append(("tired", walk[k % 2], ox))
        k += 1
    seq += [("empty",)] * 4
    return seq


def frames_zapped():
    """Electric curiosity: reaches for a bolt, ZAP (glitch shock, fur puffed),
    dazed smoke, then goes back for more. Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_zapped, glitch_frame

    seq = ["reach", "reach", "reach", "tease",
           "zap_a", "zap_b", "zap_c", "zap_b", "zap_c",
           "dazed", "dazed", "dazed"]
    glitch = {
        "zap_a": dict(bands=[(18, 10, 5), (40, 8, -6), (58, 6, 3)], invert=True),
        "zap_b": dict(bands=[(24, 6, -4), (52, 10, 4), (36, 4, 6)]),
        "zap_c": dict(bands=[(30, 5, 3), (48, 5, -3)]),
    }
    out = []
    for s in seq:
        im = draw_cat_zapped(s)
        g = glitch.get(s)
        if g:
            im = glitch_frame(im, g["bands"], invert=g.get("invert", False))
        out.append(im)
    return out


def frames_chasing():
    """A toy mouse skitters across (dart-dart-pause), a beat, then the cat
    blasts after it in the gallop and both clear the right edge. One-shot;
    returns Images held one tick each (~24fps)."""
    from PIL import ImageDraw

    from mascot_sprite import draw_cat_running, draw_mouse, new_canvas

    def scene(mouse_x=None, mouse_hop=False, cat=None):
        im = new_canvas()
        d = ImageDraw.Draw(im)
        if mouse_x is not None:
            my = 62 if (mouse_hop and (mouse_x // 8) % 2 == 0) else 64
            draw_mouse(d, mouse_x, my)
        if cat is not None:
            phase, cx = cat
            c = draw_cat_running(phase)
            im.paste(c, (cx, 0), c)
        return im

    seq = []
    phase = 0
    # Mouse enters and darts across with rodent pauses.
    for mx in [-16, -8, 0, 8, 16, 16, 24, 32, 32, 40, 48, 48, 56, 64, 64, 72, 80, 88, 96]:
        seq.append(scene(mouse_x=mx, mouse_hop=True))
    seq.append(scene())  # beat: empty frame between prey and predator
    # Cat blasts through after it, ~8px per tick so the stride stays readable.
    for cx in [-90, -82, -74, -66, -58, -50, -42, -34, -26, -18, -10, -2,
               6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86]:
        seq.append(scene(cat=(phase % 6, cx)))
        phase += 1
    seq.append(scene())
    return seq


def frames_reading():
    """Cozy read: eyes on the book, ear twitch, a page turn, a glance up.
    Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_reading

    seq = [
        dict(), dict(), dict(ear_r_dx=1), dict(), dict(),
        dict(look_up=True, gaze="center"), dict(), dict(),
        dict(turn=1), dict(turn=2), dict(gaze="down"),
        dict(look_up=True, mouth=True, gaze="center"), dict(blink=True), dict(),
    ]
    return [draw_cat_reading(**p) for p in seq]


def frames_eating():
    """Chicken sandwich (favorite food) held in both paws: sniff it, then
    raise-chomp-chew through four bites, crumbs and a happy heart at the end.
    Seamless loop; returns Images."""
    from mascot_sprite import draw_cat_eating

    seq = [
        ("look", "hold"), ("look", "hold"),
        (0, "hold"), (0, "chomp"), (0, "chew"), (0, "chew"),
        (1, "chomp"), (1, "chew"), (1, "chew"),
        (2, "chomp"), (2, "chew"), (2, "chew"),
        (3, "chomp"), (3, "chew"), (3, "chew"), (3, "chew"),
        ("gone", "hold"), ("gone", "hold"), ("gone", "hold"),
    ]
    return [draw_cat_eating(s, p) for s, p in seq]


def frames_peek():
    """Corner peek: slides in from the left edge, waves, slides back out.
    Hold offset -40 keeps roughly the cat's right half inside the frame,
    so it reads as leaning in from the edge (guide's corner-peek rule).
    One-shot (not a loop). Yields (pose, x_offset) pairs."""
    enter = [-58, -52, -46, -43, -40]
    exit_ = [-46, -52, -58]
    seq = []
    for ox in enter:
        seq.append(({"ear_r_dx": 1 if ox == -40 else 0}, ox))
    for k in range(10):
        wag = -1 if k % 2 == 0 else 1
        seq.append(({"paw_raise": wag, "mouth": True, "head_dy": -1}, -40))
    seq.append(({"blink": True}, -40))
    seq.append(({}, -40))
    for ox in exit_:
        seq.append(({}, ox))
    return seq
