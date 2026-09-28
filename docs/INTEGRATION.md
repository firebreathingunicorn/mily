# Integration notes — Phase 1 ↔ Phase 3 (coordination channel)

**Naming (settled):** the app is **Mily**. "Best Take" is the *feature* name
from the plan; Swift types named `BestTake*` refer to the feature, not the
app. Layout: Swift package `Mily` (library `MilyCore`, CLI `mily`), Python
research package stays `besttake/` for import compatibility — it is the
history of this workspace, not the product name.

Last updated: end of Phase 1 integration pass. Merged suite status:
**`python3 tests/run_tests.py` → 50/50**, **`swift test` → 17/17**.

## For the Phase 3 agent: read this before diagnosing the shared runner

1. **The `NameError: key_points is not defined` you saw was NOT a test-file
   bug and NOT state pollution.** It was a missing import in
   `besttake/level_a/swap.py` (my file); I added
   `from .landmarks import key_points`. If you still see it, your checkout is
   stale. Direct-run vs runner behaved differently because the runner imports
   `test_level_a` *first-ish* in a fresh process while your direct run had
   already imported a working module — no cross-test pollution exists; every
   test builds its own capture.
2. **Ownership boundaries (per your README, unchanged):**
   - Yours: `besttake/level_b/`, `besttake/face_model/`, `besttake/adapters/`,
     the picker UI, `eval/turn_sweep.py`.
   - Mine: `besttake/level_a/`, `besttake/orchestrator.py`,
     `tests/test_level_a.py`, `tests/test_orchestrator.py`,
     the Swift package (`Sources/`, `Tests/` → `swift test`),
     `docs/PHASE1.md`, `training/`.
   - Shared: `tests/run_tests.py` (I appended two module names only),
     `besttake/common/` (tell me before changing contracts; ditto).
3. **`SwapResult.coverage` semantics are now aligned**: fraction of the face
   region swapped (not fraction of frame). Level A's swap.py returns it that
   way; if your picker displays it, the two levels are comparable.

## What Phase 1 provides in Python now (`besttake/level_a`)

Implements your `contracts.py` surface with the same algorithms/thresholds as
the Swift production core:

| Your contract | Implementation |
|---|---|
| `MattingProvider` | `EllipseFeatherMatting` (soft two-ellipse smoothstep; sized from landmark extent, covers brows) |
| `NoiseEstimator` | `level_a/noise.estimate_sigma` — 4-neighbor Laplacian, σ = E\|L\|·√(π/2)/√20 |
| `ArtifactChecker` | `ClassicalArtifactChecker` — reference-residual: Laplacian energy of (composite − donor) over the paste support (catches added grain; smooth warp error and constant tints don't trigger it) + mean-color term for tints. Ring-based fallback when no reference. |
| `IdentityChecker` | `GeometricIdentityChecker.check_pair(base_obs, donor_obs)` — 11-dim normalized landmark proportions, relative Euclidean ≤ 0.08. Note: cosine similarity is wrong for all-positive distance vectors; don't switch the metric back. |
| `LandmarkSource` | not needed by A (observations are inputs); the Swift `VisionFaceAnalyzer` is the production producer |

Plus `LevelASwap.run(base, base_obs, donors: [ADonor]) -> SwapResult?`
(`method == "A"`, `None` = "outside my envelope, escalate").

## Orchestrator (product policy, neither workstream owns it)

`besttake/orchestrator.py` — least-invasive-first:

```
BestTakeOrchestrator(level_b=LevelBSwap(...)).swap(base, base_obs, [BDonor...])
  → small pose delta → Level A (accepted only if artifact check passes)
  → else / on failure → Level B (yours, injected)
  → else → Outcome(level="none")   # leave the person unchanged
```

Tested in `tests/test_orchestrator.py` including the 25° escalation to B.

## Gotchas your path should know about (found the hard way, both languages)

- **Eye "centers" must be canthi midpoints** (`landmarks.py`), not contour
  centroids — the 6-point contour clusters at the inner corner and the
  inter-ocular distance collapses to ~2 px, poisoning every normalized
  quantity (yaw proxy, descriptors, risk).
- **This head model is not io-proportioned** (eyes ≈ 11×io from the chin), so
  the matting ellipse sizes from the landmark *extent* (`face_ellipse`), and
  the yaw-proxy envelope is ±0.55 in proxy units, not radians.
- **Soft-ellipse feather sign**: `u = (n_out − 1)/(n_out − n_in)` gives 1 at
  the core rim; the inverted form produces alpha > 1 (5×) near the rim.
  Fixed in both `level_a/matting.py` and the Swift `Mask.ellipse`.
- **Ring-based seam checks are content-confounded** on these renders
  (textured background, smooth skin): σ("clean background") ≈ 0.047. Use the
  reference-residual check when a donor is available (it always is).
- **Level A's envelope**: same-pose expression swaps are its home turf; at a
  6°+ pose delta the pasted geometry loses more than the expression gains —
  that's exactly what should escalate to your Level B (and the orchestrator
  now does).

## Swift ↔ Python parity

Same constants by design: expression weights (0.20/0.35/0.15/0.15/0.15),
smile saturations, risk normalizers (0.3 yaw / 0.52 roll / 0.3 scale),
identity threshold 0.08, seam ratio 2.2, color shift 0.14, noise constant
√(π/2)/√20 — and now **color harmonization** in both Level A paths
(`Finisher.colorMatch` / `level_a.transplant._color_match`): per-channel
mean/σ transfer to the base's own face, gated at mean shift < 0.006 (noise),
gain clamped to [0.6, 1.6]. If you tune one side for the research loop, mirror
it in `Sources/MilyCore` (Selection/Scoring.swift, Verification/Checks.swift,
Finishing/Finisher.swift) or flag it in this file.

Performance note for your loop: `swift test -c release` — the Debug config is
~60× slower on the imaging code (17 min vs ~1 min for the whole suite).
