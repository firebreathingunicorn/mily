# Coordination — Best Take workstreams

Two agents work in this workspace in parallel. This file is the handshake:
read it before changing shared surfaces, update it when you change them.

| Workstream | Owner | Location |
|---|---|---|
| Phase 1 — Level A (capture, tracking, matting, noise, checkers, transplant) | agent 1 | `besttake/level_a/` |
| Phase 3 — Level B (3D fit, re-projection, fill, finishing) | agent 2 | `besttake/level_b/` |
| Orchestrator (least-invasive-first A→B policy) | agent 1 | `besttake/orchestrator.py` |
| Shared contracts, types, geometry, rasterizer | both — change with care | `besttake/common/` |
| Face model (3DMM stand-in) | agent 2 (Phase 3 needs it) | `besttake/face_model/` |
| Synthetic capture adapter (closed-loop GT) | shared — additive changes only | `besttake/adapters/synthetic_capture.py` |
| Per-person picker prototype (plan §Product surfaces) | agent 2 | `product/make_picker.py` |
| Test runner (no pytest): add your module to `MODULES` | shared | `tests/run_tests.py` |

Rule from the plan: Phase 3 never imports `besttake.level_a`; both consume
the protocols in `besttake/common/contracts.py`.

## Run everything

```bash
PYTHONPATH=. python3 tests/run_tests.py      # merged suite (both workstreams)
python3 eval/turn_sweep.py                   # Phase 3 exit-criterion eval
PYTHONPATH=. python3 product/make_picker.py  # regenerate the picker demo
```

## Current merged-suite status (2026-09-28, evening)

49/50. The one failure is in agent 1's workstream:
`test_level_a.test_level_a_swap_small_yaw_beats_base` — `LevelASwap` still
declines the small-yaw case (8° base / 2° donor), so coverage is 0 where the
test expects > 0.2. (The earlier noise-mismatch failure went green once the
`ClassicalArtifactChecker` accept path was fixed.) Diagnosis trail for the
remaining one: the swap returns `method=None` from `_one_person` — check
which internal gate declines at yaw ≤ 8°.

An earlier run also showed transient `NameError: key_points` failures that
disappeared on re-run — consistent with files being edited mid-run. If you
see impossible failures, re-run before debugging.

## Shared-surface changelog (agent 2, Phase 3 — for agent 1's attention)

- `common/types.py` — `Fit` gained an `intrinsics` field; pinhole
  `project()` now returns **pixels** (fx/fy/cx/cy applied), not normalized
  coordinates. Fits without intrinsics must use weak mode.
- `level_b/fitter.py` — `FaceFitter.fit` signature is
  `(landmarks, depth, intrinsics, c_identity, fix_identity)` (the unused
  `c_expression_init`/`image_center` params are gone). `mode='weak'` with
  intrinsics now runs an extra stage that treats the focal as free
  (`weakrefine`) and returns a **pinhole** fit with estimated intrinsics —
  yaw error drops from ~10° to ~0.1° on turned heads. If Level A relied on
  the old weak-perspective over-rotation as a yaw cue, switch to
  `fit.R` via the returned fit or `key_points().yaw_proxy`.
- `common/contracts.py` — `ArtifactChecker.check` gained optional
  `weight=` and `references=` kwargs; `IdentityChecker` references may be
  `(frame, region_mask)` pairs (pose-independent comparison). Additive.
- `level_b/reproject.py` — `reproject` gained keyword-only-in-practice
  `donor_depth=` (occlusion guard against real foreground objects).
- `level_b/fitter.py` — `fit` gained `yaw_hint_deg=`: runs the nearest
  single yaw start (~3.7× faster stage A) and falls back to the full
  multi-start if the hinted fit misses the rmse bar. Returns ValueError
  instead of a garbage fit when observations are non-finite.
- `level_b/pipeline.py` — donors with unusable fits (NaN landmarks,
  rmse > config) are dropped before identity median; `SwapResult` gained
  `coverage_sectors` (3x3 smart-shutter map) and `coverage` is now
  **face-relative** (it was accidentally frame-relative before).
- `tests/run_tests.py` — 55 tests across both workstreams.
- `adapters/synthetic_capture.py` — `shoot`/`make_capture` gained optional
  `center_offset`, `base_pitch`/`donor_pitch`, `occluder`; defaults
  unchanged, so existing calls behave as before.
- `product/` — new picker prototype (see below); no impact on pipelines.

## Picker prototype (swipe frame by frame)

`PYTHONPATH=. python3 product/make_picker.py` renders a synthetic burst
(6 candidate frames, one person), runs Level B per candidate frame, and
writes a self-contained `product/picker_out/index.html`:

- filmstrip of candidate frames; swipe the photo or use ←/→ to step
  frame by frame
- press-and-hold the photo to compare against the original frame
- badge labels the synthesis level per frame (`B`, `B+fill`, `rejected`)
  with the swapped-coverage percentage

In the real product this is the plan's per-person picker; the frames come
from the burst and each carries its `SwapResult.method` label. The
generator is the integration point: replace the synthetic capture +
`LevelBSwap.run` loop with orchestrator output and the same HTML works.
