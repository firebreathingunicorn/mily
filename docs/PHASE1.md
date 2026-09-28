# Phase 1 — Level A, done properly: status

Roadmap phase 1 ships: ring-buffer capture, tracking, learned expression
selection, soft matting, noise matching, artifact checker. Exit criteria:
*beats the best single frame in blind tests; passes the fairness tests.*

Two implementations exist, algorithmically identical:

| | Swift (`Sources/MilyCore`) | Python (`besttake/level_a`) |
|---|---|---|
| Purpose | production path (Core ML / Neural Engine target) | research loop, plugs into Phase 3's pipeline |
| Tests | `swift test` — 17 tests | `tests/test_level_a.py` + `test_orchestrator.py` |
| Status | **17/17 green** | **green** (see `tests/run_tests.py`) |

## What is implemented (plan § → code)

| Plan item | Swift | Python | State |
|---|---|---|---|
| Ring-buffer capture (full-res frames + depth + motion) | `Capture/RingBufferCapture.swift` (iOS), `Capture/CaptureModels.swift` | — | capture scaffold; depth rides on `CaptureFrame`/`Frame` for Level B. Device bring-up pending |
| Per-person identity across frames | `Understanding/VisionFaceAnalyzer.swift` (greedy position + descriptor matching) | — | works within a burst; cross-shot matching is Phase 2 |
| Dense face geometry | Vision landmarks + derived scalars (`Understanding/Annotations.swift`) | `level_a/landmarks.py` | 2D; real 3D fitting is Phase 3's fitter |
| Hair/edges soft matting | Vision person segmentation, alpha resampled (`Understanding/VisionFaceAnalyzer.swift`) | `level_a/matting.py` (feathered ellipse) | **learned matting model pending** — both paths take a provider |
| Expression quality model | `Selection/Scoring.swift` (`ExpressionScorer` protocol + `HeuristicExpressionScorer`) | `level_a/scoring.py` | **learned preference model pending** — protocol slot ready (see `training/`) |
| Personal calibration | not started | — | needs opt-in + data |
| Group coherence + joint frame choice | `GroupPlanner` (base argmax w/ coherence term; per-person donor argmax w/ risk discount) | `level_a/swap.py` donor choice | done at Level A scope |
| Level A direct transplant | `Synthesis/Synthesizer.swift` (`DirectTransplantSynthesizer`) | `level_a/transplant.py` | done, verified in closed-loop tests |
| Noise/grain matching | `Finishing/Finisher.swift` (Immerkær σ + deficit injection) | `level_a/noise.py` + transplant | done; estimator constant verified against synthetic ground truth |
| Color/lighting harmonization (Level A scope: exposure) | `Finisher.swift` (per-channel mean/σ transfer to the base's own face, noise-gated) | `level_a/transplant._color_match` (same constants) | done; burts that re-meter mid-burst no longer paste at the wrong exposure |
| Sharpness matching | `Finisher.swift` (vs. base face region, not background) | `level_a/transplant.py` | done |
| Artifact checker | `Verification/Checks.swift` (`ClassicalArtifactChecker`) | `level_a/checks.py` (reference-residual Laplacian energy + color shift) | **learned classifier pending** — protocol slot ready |
| Identity check | `IdentityChecker` (geometric descriptor, relative distance ≤ 0.08) | `level_a/identity.py` | done at Level A; embeddings swap in one file (`GeometricIdentity`) |
| Provenance / C2PA | not started | — | Phase 4 scope per roadmap |
| Clean background plate | `Understanding/BackgroundPlate.swift` (temporal median of person-free pixels) | — | done; consumed by Phase 2 whole-person swap |
| Smart shutter ("capture until everyone has had a good moment") | `Selection/ShutterAdvisor.swift` — per-person best-moment tracking, `done`/`gaveUp` signals | — | done (unit-tested); device integration pending |
| Per-person picker filmstrip data | `BestTakeReport.scores` (per-person per-frame scores in report JSON) | — | done; picker UI is Phase 3's workstream, consumes this + `SwapResult.method` |
| Tiny-face guard | `GroupPlanner.minInterOcular` (default 10 px) | — | done; untransplantable faces are excluded from planning |
| Deferred processing / on-device | pipeline is synchronous CLI today | — | orchestration for background jobs pending |
| Fairness slice reporting | — | `eval/slices.py` (per-slice pass rates, worst-slice surface, unspecified-bucket never dropped) | done as harness; real slice data needs the consented dataset |

## Level B handoff (consumed by Phase 3)

- `FaceSynthesizer` protocol (`Synthesis/Synthesizer.swift`) is the Swift seam:
  Level B implements `transplant(base:donor:person:) -> TransplantResult?` and
  the pipeline tries synthesizers least-invasive-first.
- Python: `besttake/orchestrator.py` runs A then B (see `docs/INTEGRATION.md`).
- `CaptureFrame.depth` / `Frame.depth` carry LiDAR disparity for the fitter.

## Verification results (synthetic closed loop)

- Transplant correctness: output equals base ⊕ transplant up to finishing
  noise; pasted content tracks the donor frame (sub-pixel warp accounted).
- Grain matching: post-finish σ matches base σ within tolerance.
- Artifact checker: rejects noise-mismatched (5× grain) and color-shifted
  transplants; passes clean ones.
- Identity gate: separates different face proportions (>0.08), keeps the same
  person under camera jitter (<0.02); rejects misassigned identities.
- Planner: picks each person's best frame; rejects a high-scoring donor whose
  head is rolled ~50° (risk-gated).
- Alignment, similarity fit, noise σ, mask utilities: unit-tested.

## Performance

Always benchmark and run in Release — the debug build is ~60× slower:

| Stage (release, 2 people, 640×480, 6-frame burst) | Time |
|---|---|
| Synthetic scene generation | ~107 ms/frame |
| Pipeline (plan → transplant → finish → verify) | ~66 ms |
| Vision analysis (detect + accurate person segmentation, 4 frames concurrent, segmentation at 1024 px) | ~0.6 s for 6 frames |

`mily bench` measures the first two. `swift test -c release` runs the
suite in ~1 min (the same suite in Debug takes ~17).

## Not done yet (honest list)

- Learned expression scorer, learned matting, learned artifact checker: slots
  + training scaffold (`training/`); need the consented dataset first.
- Device bring-up of the ring-buffer capture (front camera full-res + depth).
- C2PA provenance, personal calibration, fairness slice harness on real data.
- Performance: Swift path is CPU-only scalar code at working res; port hot
  loops to Accelerate/Metal before device use.
