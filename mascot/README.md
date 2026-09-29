# Marbled Cat — Animated Pixel Mascot

A hand-built animated pixel mascot for Mily: a golden marbled tabby with
black swirl blotches, black square eyes, whiskers, and a black-ringed tail.
The marketing angle writes itself — the marbled cat is famously fast and
impossible to photograph, and Mily exists to catch exactly that shot.

Built locally (no external AI service): the sprite is drawn frame-by-frame in
Python, animated at a stepped ~6fps cadence (sport bursts ~12fps), and shipped through the same
transparent-asset pipeline as the
[Notion E2E guide](https://app.notion.com/p/Animated-Pixel-Mascot-E2E-Build-Guide-39395decbef981e9babfd075f9f87ab8)
(Steps 3–4). No Higgsfield account needed.

## See it

Open `preview.html` in any browser — every action plays as a transparent APNG
on light, dark, and checkerboard backgrounds.

## The asset pack (`assets/`)

Thirty-five actions, five outputs each:

| Action | Loop | What it does |
| --- | --- | --- |
| `idle` | seamless | breathing, blink, ear twitch, tail sway |
| `waving` | seamless | anticipation crouch, 3 stepped paw waves |
| `floating` | seamless | weightless bob, tail/ears lag a step |
| `sleeping` | seamless | curled coil, slow breath, cycling Zzz |
| `glitching` | seamless | retro slice glitches, ghost frame, palette pop |
| `peeking` | one-shot | corner peek: slides in from the left edge, waves, exits |
| `running` | one-shot | zoomies: ~24fps gallop burst across, then the tired walk of shame back — slouched, tongue out, sweat drop, one huffing pant-stop |
| `zapped` | seamless | touches a lightning bolt → glitch shock (inverted flash, fur puffed) → dazed smoke → goes back for more |
| `chasing` | one-shot | toy mouse darts across, beat, the cat blasts after it |
| `reading` | seamless | small red hardcover held close, page turn, glance up |
| `eating` | seamless | holds the chicken sandwich up in both paws and chomps it at the mouth, four bites, crumbs + heart |
| `tv` | seamless | channel zapping with real shows: a tiny-cat stage series, an aquarium channel, a rocket launch that jump-scares it (glitched), weather on the news |
| `treadmill` | seamless | gym session on a scrolling belt: warm-up jog, run, all-out sprint (sweat, tongue), bonk, pants on the dead belt while the console dims |
| `flexing` | seamless | strength set: throws the double-biceps with bicep bumps, sparkle max-effort hold |
| `stretching` | seamless | yoga flow: deep arch (chest low, tail straight up) swaying into the dip, face to the sky |
| `phone` | seamless | retro brick cell rings on the floor, held to the ear — hello! — chats with the free paw, hears something shocking, waves bye |
| `graduating` | seamless | cap and gown with gold stole, diploma scroll, tassel swinging side to side, cap toss through the confetti, catch and cheer |
| `basketball` | seamless | dribbles with a pushing arm and ball squash, rises into a shot — swish through the corner hoop, net sway and sparkle |
| `snowman` | seamless | winter build: rolls a snowball, pats it out, stacks middle and head with plop-puffs, plants the carrot, celebrates under falling snow |
| `rain` | seamless | shelters under a red umbrella while rain streaks down; a wind-blown drop boops the nose (shock!), shake-off, then enjoys the pitter-patter |
| `piano` | seamless | seated at the keyboard: paws wander and press, sways and hums, notes float up, ends on a triple-note crescendo |
| `baking` | seamless | toque on: cracks the egg, stirs the bowl, sneaks a spoon lick, POOF — layer cake appears, cherry on top, sparkle celebration |
| `firefighter` | seamless | red helmet, reflective coat, braced on the hose — douses the blaze to embers to smoke, then the fire relights and it's SHOCKED |
| `discovery` | seamless | naps under the tree, an apple falls and bonks its head — the apple is electrified! Wakes, holds it up sparking, lightbulb moment |
| `plane` | seamless | weekend aviator: putters through the sky in a tiny red plane — spinning prop, scarf in the wind, clouds gliding by |
| `noodles` | seamless | ramen night: slurps the strand shorter and shorter with chopsticks, bowl empty — pure happiness |
| `gaming` | seamless | hunched over the controller mashing (full-body jitter), shock eyes for the clutch, WIN with confetti and tail up |
| `campfire` | seamless | night under the stars: toasts a marshmallow over the campfire until golden, slightly charred, eats it |
| `dance` | seamless | disco night: ball throwing light beams, big side-to-side moves, arm pump, notes bouncing |
| `weightlifting` | seamless | the clean and press in a headband: grip, clean to the chest, press overhead, trembling sweat hold, drop with a dust thud |
| `suit` | seamless | FRESH: adjusts the red tie, sunglasses slide on, confident strut, finger guns with sparkles |
| `baseball` | one-shot | at the plate: the pitch comes in, one level swing — CRACK — ball rockets off with speed lines while the batter watches it fly |
| `volleyball` | seamless | bump off the forearms, watch it rise, jump and SPIKE it down — serve comes back in |
| `grooming` | seamless | paw lick with tongue darts, cheek wipe, perks up |
| `boing` | seamless | excited bounce: crouch, spring with paws and tail flying, shake it off |

The cat's eyes carry a moving glint — it looks left, right, up and down
across the actions — plus happy ^ squints, whisker twitches, dangling legs
while floating, and a vertical excitement tail for the boing. Heart eyes
appear exactly once: finishing the chicken sandwich. Athletic actions
(`treadmill`, `basketball`, `flexing`, `stretching`, `boing`)
wear a red sport headband with tie tails (`running` and `chasing` go
bareheaded). Shading follows a
top-left light — sheen on crowns and tails, shaded undersides, and a soft
contact shadow that grounds the cat (and lifts away when it's airborne).

## Switching animations smoothly

Two ways:

- **Live**: the Mixer at the top of `preview.html` — click any action and the
  big stage crossfades to it (~350ms), no reload.
- **Offline**: render a transition between any two actions, or chain a whole
  story into one reel:

  ```sh
  python3 tools/mix.py --from waving --to tv --fade 8 --out mix.gif --apng
  python3 tools/mix.py --chain baking,eating,sleeping --fade 8 --out reel.gif
  ```

  The blend is premultiplied-alpha, so the fade never halos.

Directional one-shots (`running`, `chasing`, `peeking`) also ship mirrored as
`<action>_r` — the dash and chase go right-to-left, and `peeking_r` leans in
from the frame's right edge. Same 5 outputs each.

Per action: `ACTION.png` (APNG full 576px), `ACTION_160.png` (APNG 160px),
`ACTION.gif` + `ACTION_160.gif` (1-bit alpha, last resort), `ACTION.webm`
(VP9 alpha, for video editors). APNG is the one to use on web/Notion;
WebM for Premiere/DaVinci/CapCut.

The corner peek is staged for the guide's placement rule: put the asset's left
edge against the video frame's left edge and keep the cat ~10–15% of frame width.

## Rebuild

```sh
# master still + RGBA frames (mascot/master, mascot/frames)
python3 tools/build_mascot.py

# audit + encode each action into assets/ (plus mirrored _r twins)
for a in idle waving floating sleeping glitching peeking running zapped chasing reading eating grooming boing tv treadmill flexing stretching phone graduating basketball snowman rain piano baking firefighter discovery plane noodles gaming campfire dance weightlifting suit baseball volleyball; do
  python3 tools/convert_transparent.py encode --frames frames/$a --name $a --out assets/$a
done
for a in running chasing peeking; do
  python3 tools/convert_transparent.py encode --frames frames/$a --name ${a}_r --out assets/${a}_r --flip
done
```

## Design rules baked in (from the guide)

- One flat background color when a master still is exported
  (`master/master_still.png`, off-white `#f2f3ee`) so keying stays clean.
- Stepped-animation feel: every pose is held 2 ticks of 80ms (poses
  ~6fps; 1-tick sport bursts ~12fps). No tweening, no motion blur, camera
  static.
- Premultiplied alpha before any resize (no halo when scaling down to 160px).
- Every frame is audited: zero opaque pixels may sit within tolerance of the
  key colors (off-white / white / green). Encoding refuses to ship a failing set.

## Higgsfield path (optional, not required)

No account needed for any of the above. If you ever want the AI-video look,
`PROMPTS.md` holds the ready-to-paste Higgsfield prompts, and

```sh
python3 tools/convert_transparent.py convert --input ~/Downloads --out assets_hf
```

runs the guide's exact chroma-key pipeline (green-screen or flat-light mode,
2px fringe dilation/erosion, audit with auto-tighten) over any `hf_*.mp4`
clips dropped in `~/Downloads`.

## Character sheet

- Coat: golden `#E8B458`, shade `#C6903C`
- Marble/rings/ear tips: black `#231E1A`
- Accents: cream muzzle/chest/paws `#F5EEDA`, whiskers `#F7F0DA`, rose nose
- 96×96 logical grid, ×6 nearest-neighbor export (576px full size)
- Engine: `tools/mascot_sprite.py` · actions: `tools/mascot_actions.py`
