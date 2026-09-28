# Higgsfield prompts (optional upgrade path)

The mascot ships fully local — these are only needed if you later want the
AI-video versions of the same character. The design rules that matter:
generate everything on ONE flat background, reuse the master still as the
`start_image` for every clip, and keep the three retro phrases in every
animation prompt. Save prompts the day you run them (Higgsfield keeps ~90
days of history).

## Step 1 — master still (nano_banana_pro, count 4, 1:1)

> Minimalist 2D 8-bit pixel art character: a small marbled cat made of large
> clean pixels, warm golden-yellow coat with black marbled swirl blotches, a
> long thick black-ringed tail, two black square eyes, a cream muzzle and
> chest patch, and white whiskers. Flat solid [#f2f3ee off-white] background,
> nothing else in frame, character centered. Hard pixel grid, crisp squares,
> no gradients, no outlines, no shadows.

## Step 2 — action clips (kling3_0, pro, 1:1, 4s, start_image = master still)

Keep this suffix on every prompt:

> Fast-paced retro aesthetic. 12fps stepped animation. Hard pixel grid, no
> motion blur. Camera static.

- Wave: `The 8-bit marbled cat raises one front paw and waves at the camera, mouth open in a happy grin.`
- Sleep: `The 8-bit marbled cat curls up, closes its eyes and sleeps, tail wrapped around its body, small Zzz floating up.`
- Float: `The 8-bit marbled cat floats gently upward and bobs weightlessly, tail swaying.`
- Glitch: `The 8-bit marbled cat glitches with rapid retro frame jumps, briefly flickering between two poses.`
- Corner peek: `The 8-bit marbled cat peeks from the left edge, only right half visible first, waves at camera and then slides off-left.`

Looping tip: pass the master still again as `end_image` so clips snap back to
idle and repeat cleanly.

## Step 3 — transparency

Don't hand-key anything; run:

```sh
python3 tools/convert_transparent.py convert --input ~/Downloads --out assets_hf
```

(green-screen or flat off-white keying, 2px fringe dilation/erosion, audit,
GIF + APNG + WebM pack named by action).

## Hero 3D variant (from the guide, verbatim template)

> Make the character in <<<image_1>>> In high-quality iPhone-like 3D with a
> black background with extreme premium quality glass-type texture
