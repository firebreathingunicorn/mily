# Training scaffolds (learned models — Phase 1 slots)

Three places in the plan call for learned models. The pipeline takes them
through narrow interfaces; nothing else changes when they land. Data does not
exist yet, so these are schemas and wiring contracts, not training code.

## 1. Expression preference model (replaces `HeuristicExpressionScorer`)

- Interface (Swift): `ExpressionScorer.score(frame:person:) -> Float`
  (`Sources/BestTakeCore/Selection/Scoring.swift`);
  Python: `level_a/scoring.score_frame`.
- Input: face crop + landmark scalars; output: scalar quality.
- Data: pairwise human preferences — "which moment of this person is better?"
  (plan §Selection). Store as `preferences.csv`:
  `pair_id, burst_id, person_id, frame_left, frame_right, winner, annotator, fitzpatrick, glasses, beard, head_covering, low_light`
  — the slice columns are mandatory from day one (plan principle 4: results
  are measured across skin tones, glasses, beards, head coverings, low light).
- Loss: pairwise logistic (Bradley–Terry) over a small face encoder; export
  with `coremltools` to `Models/ExpressionScorer.mlmodel`; load via a
  `CoreMLExpressionScorer` implementing the protocol (slot exists).

## 2. Artifact checker (replaces the classical checker)

- Interface: `ArtifactChecker.check` (both languages; contract-compatible with
  Phase 3's `contracts.ArtifactChecker`).
- Data: composites produced by this pipeline (A and B) paired with untouched
  real frames — the checker must tell them apart *within this pipeline's
  distribution*. `manifest.csv`:
  `sample_id, image_path, is_composite, level, person_id, slice columns…`
- Metric to publish per release: false-negative rate under human audit
  (plan §Verification) **per slice**, plus the tracked "share of swaps
  handled by level A" (it should rise as capture improves).
- Export to `Models/ArtifactChecker.mlmodel`; drop-in via the same protocol.

## 3. Learned matting (replaces Vision person segmentation / ellipse feather)

- Interface: `MattingProvider.alpha(frame:obs:) -> (H, W) float [0,1]`
  (contract-compatible with Phase 3).
- Data: consented group-burst captures with trimap-free matting labels from
  the multi-camera rig ground truth; synthetic-render pairs first (the
  `SyntheticCapture` adapter already yields exact masks for pretraining).

## Data collection (plan §Data)

The consented group-burst dataset (many groups, lighting, ages, appearances;
multi-camera rigs for real pose ground truth) gates all three. Record
provenance for every frame at capture time (timestamp, device, exposure) —
the Swift `FrameMetadata` already carries it.

## Fairness evaluation

`slices.py`-style reporting is a release gate: every metric above is reported
per slice; a model that regresses any slice does not ship. (The synthetic
loop validates wiring; only real data can validate fairness.)
