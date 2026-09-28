# Best Take — Phase 3: Level B synthesis (3D-aware re-projection)

> **Workspace overview** — two workstreams share this repo.
> - **Phase 1 / Level A**: Swift package `Sources/MilyCore` (production path, `swift test`) + Python `besttake/level_a` + product policy `besttake/orchestrator.py` (least-invasive-first A→B). Status: `docs/PHASE1.md`.
> - **Phase 3 / Level B**: this document.
> - Coordination, ownership map, and merged-suite status: `docs/INTEGRATION.md`. Combined tests: Python `python3 tests/run_tests.py`, Swift `swift test`.

This workspace hosts the **Phase 3** workstream of the Best Take plan: the
**Level B** synthesis stage — *3D face fitting with depth, re-projection, and
fill from other frames*. Roadmap exit criterion: **handles head turns up to
~30° without a detectable edit.**

Phase 1 (capture, tracking, matting, noise matching, artifact/identity
checkers) is a separate workstream; this code consumes it through the
contracts in `besttake/common/contracts.py` and ships working fallbacks so
everything runs standalone today. Integration rule: `besttake/level_a/` is
owned by Phase 1; Phase 3 never imports from it.

## Status

Working end-to-end on synthetic closed-loop data (known ground truth):
fit → re-project → fill → harmonize (color + shading + grain) → verify →
composite-or-reject.

Turn sweep over **6 identities** (skin tone/structure range) with randomized
donor expressions, head pitch, off-center framing and ±12% auto-exposure
jitter; donor at 5°; PSNR of the result vs. the ideal render (same person,
base pose, donor expression) in the ground-truth face region — a prototype
stand-in for the planned blind edit test:

| base yaw | 0° | 10° | 20° | **30°** | 40° |
|---|---|---|---|---|---|
| PSNR dB mean | 28.7 | 28.6 | 28.1 | **27.9** | 27.4 |
| PSNR dB worst identity | 26.7 | 26.7 | 26.5 | **25.6** | 24.4 |
| SSIM mean | 0.93 | 0.93 | 0.92 | **0.91** | 0.90 |
| face coverage (swapped fraction) | 0.67 | 0.67 | 0.66 | **0.63** | 0.59 |

The prototype artifact/identity gates rejected 0 of these swaps; ablation
with corrupted composites (plastic-skin smoothing, noise mismatch, wrong-person
references) shows the gates rejecting all of them. Identity is fitted per
burst frame and median-combined — ~3× lower identity-coefficient error than a
single-frame fit. Fill ablation at 35° yaw: with all three burst frames the
WORST identity improves from 23.9 dB (single turned-away donor) to 26.5 dB
and face coverage rises 0.56 → 0.67. `SwapResult.coverage_sectors` reports
the per-sector map so the capture loop knows what no frame has seen yet
(plan §Smart shutter). Fits use a landmark yaw proxy as a start hint with
an rmse-fallback to the full multi-start — identical accuracy, ~3.7× faster
stage A. Failed detections (NaN landmarks, degenerate fits) are validated
and dropped before they can poison identity or rendering.

## Layout

```
besttake/
  common/            shared types, geometry, z-buffer rasterizer, contracts
    contracts.py     ← the Phase 1 integration surface (implement these)
  face_model/
    canonical.py     SyntheticHeadModel: parametric head (3DMM stand-in)
  level_b/           ← Phase 3 proper
    fitter.py        3D fit: landmarks (multi-start, Huber-robust LM) → depth-anchored refine
    reproject.py     donor→base transfer + visibility (model z-buffer AND captured depth) + blending
    harmonize.py     color transfer, low-frequency shading transfer, grain matching
    pipeline.py      LevelBSwap orchestrator (frontality donor ordering, reject path)
  adapters/
    synthetic_capture.py  renders multi-frame captures with ground truth + occluders
    simple_checks.py      prototype artifact (texture-energy) + identity (palette) gates
product/
  make_picker.py   per-person picker demo: filmstrip, swipe frame-by-frame,
                   hold-to-compare, per-frame synthesis-level labels
tests/                python3 tests/run_tests.py   (merged suite, both workstreams)
eval/turn_sweep.py    multi-identity exit-criterion eval + panels in eval/out/
```

See `COORDINATION.md` for the parallel-workstream handshake (ownership,
shared-surface changelog, current merged-suite status).

## How Level B works

1. **Order donors** — the most frontal frame becomes primary (fit-free yaw
   proxy: nose-tip offset from the inner-canthi midpoint).
2. **Burst identity** — identity is fitted independently on every donor
   frame and combined with a coefficient-wise median (plan §4C: identity
   learned from the person's own burst frames); expressions are then
   refit against the shared identity.
3. **Fit** — each frame's landmarks give pose + expression via
   Levenberg-Marquardt (yaw multi-start for the left/right ambiguity, Huber
   weighting against bad detections); metric depth then anchors scale and
   out-of-plane pose with an occlusion-aware inlier gate.
4. **Re-project** — the base render yields, per pixel, the canonical surface
   point. Its position in the donor frame = canonical point − base expression
   displacement + donor expression displacement, projected with the donor fit
   (anatomical correspondence under a shared identity), sampled bicubically.
   A sample is valid only if the surface faces the donor camera, the donor's
   model z-buffer shows no occluder, **and the donor's captured depth
   agrees** — a hand or drink in front of the face (in the depth map but not
   the model) marks those samples invalid.
5. **Fill** — pixels no donor can see stay base; pixels the primary donor
   cannot see come from other frames of the same person, chosen per pixel by
   view confidence with a Gaussian cross-fade between sources.
6. **Finish** — per-channel color transfer, low-frequency shading transfer
   (masked luminance ratio vs. the base frame — the prototype stand-in for
   learned relighting, plan §risks #2), grain matching (adds only the noise
   deficit vs. the donor's own grain), soft alpha from the matting contract
   (fallback: distance-transform feather — never a hard cutout).
7. **Verify** — the artifact and identity checker contracts gate the result;
   a failed gate leaves the person **unchanged** (method `"rejected"`), never
   a detectable composite. Defaults are the prototype gates in
   `adapters/simple_checks.py`; the Phase 1 trained models drop into the
   same two methods.

## What Phase 1 provides (the contracts)

| Contract | Used for | Default today |
|---|---|---|
| `LandmarkSource` | per-frame observations (pipeline takes them as input) | synthetic adapter |
| `MattingProvider` | soft alpha at the composite step | distance feather |
| `NoiseEstimator` | grain matching σ per frame | per-channel Laplacian MAD |
| `ArtifactChecker` | reject detectable composites (gets blend weight + donor frames) | texture-energy gate, calibrated ±10× margins |
| `IdentityChecker` | reject wrong-person results (references may carry their own face masks) | palette gate (color statistics) |
| `RelightModel` | learned relighting (later) | low-frequency shading transfer |

Pipeline entry point:

```python
swap = LevelBSwap(model, LevelBConfig(mode="pinhole"),
                  matting=..., noise=..., artifact_checker=..., identity_checker=...)
result: SwapResult = swap.run(base_frame, base_obs,
                              [DonorInput(frame, obs), ...])
# result.image / weight / source_map / coverage / fill_fraction / method / checks
```

`method` is `"B"` or `"B+fill"` — this is the per-swap label the per-person
picker UI surfaces.

## Running

```bash
PYTHONPATH=. python3 tests/run_tests.py     # 39 tests
python3 eval/turn_sweep.py                  # exit-criterion table + panels
```

Dependencies: `numpy`, `scipy`, `pillow` (see `requirements.txt`). Pure
numpy/scipy renders are ~1–2 s/frame at 320 px — fine for prototyping; the
production path is Core ML on the Neural Engine (plan §Data).

## Known limitations (deliberate scope cuts)

- **SyntheticHeadModel is a stand-in** for a FLAME-class model with a real
  consented dataset. The downstream code depends only on the `FaceModel`
  interface (mesh, identity/expression modes, landmarks, albedo, hair mask).
- **Silhouette changes are out of scope.** If the donor's expression changes
  the head contour (e.g., jaw drop), base pixels outside the swapped surface
  remain — whole-person swap + background plate is Phase 2's deliverable.
  The eval scenarios therefore use silhouette-preserving expression pairs.
- **No-intrinsics fallback refines the focal from foreshortening** (focal
  free, camera distance fixed), recovering yaw to ~0.1° and the true focal
  from a wrong prior. Pure weak perspective (no intrinsics at all) still
  over-rotates ~10° — avoid it; capture always knows the intrinsics.
- **Shading transfer is low-frequency only**; it corrects illumination level,
  not view-dependent effects (specular, shadow direction). Learned relighting
  (plan §risks #2) plugs into the `RelightModel` contract. An image-space
  residual-alignment polish (grid phase correlation) was tried and rejected:
  expression change between donor and base violates brightness constancy and
  it cost 0.7-1.2 dB. The right fix is joint fit refinement, not warping.
- **The prototype gates are detectors of gross failure, not subtle edits.**
  The artifact gate calibrates texture energy against the donor frames
  (pass band ±2× around the nominal 0.65, measured margins: good 0.64–0.67,
  plastic-skin 0.13–0.19, noise-corrupted 1.8–2.1); the identity gate is a
  palette check that cannot catch intra-palette identity drift. The trained
  composite detector and embedding identity threshold from Phase 1/4 replace
  them behind the same two method signatures.
- **PSNR is a stand-in** for the planned blind "was this edited?" test,
  identity-threshold check, and demographic-fairness splits (plan
  §Principles 4) — those need human preference data and the Phase 1 checkers.
