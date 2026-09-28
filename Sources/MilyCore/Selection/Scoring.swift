import Foundation

// MARK: - Expression scoring

/// Scores how good one person's moment is in one frame, in [0, 1].
///
/// The plan replaces hand-tuned eye/smile measurements with a model learned
/// from human side-by-side preferences. `CoreMLExpressionScorer` is that slot:
/// drop in a converted model and the pipeline behavior changes with no other
/// edits. `HeuristicExpressionScorer` is the calibrated stand-in that lets the
/// whole pipeline run, be tested, and be measured before the training data
/// exists — it deliberately uses only scale-invariant geometry.
public protocol ExpressionScorer: Sendable {
    func score(frame: AnnotatedFrame, person: PersonID) -> Float
}

public struct HeuristicExpressionScorer: ExpressionScorer {
    public var eyesWeight: Float = 0.20
    public var smileWeight: Float = 0.35
    public var midWordWeight: Float = 0.15
    public var frontalityWeight: Float = 0.15
    public var sharpnessWeight: Float = 0.15

    public init() {}

    public func score(frame: AnnotatedFrame, person: PersonID) -> Float {
        guard let face = frame.faces[person] else { return 0 }

        // Eyes: 0 closed, saturating at ~0.10 inter-ocular aperture.
        let eyes = clamp01(face.eyeAperture / 0.10)

        // Smile: corners raised relative to mouth mid-line; saturating ~0.10.
        let smile = clamp01(face.smileCurve / 0.10 + 0.5)

        // Mid-word mouth: wide vertical aperture is a bad moment.
        let midWord = 1 - clamp01(face.mouthAperture / 0.18)

        // Facing the camera: yaw and roll proxies. ±0.3 yaw ≈ 30°, ±0.6 roll.
        let frontality = 1 - clamp01(max(abs(face.yawProxy) / 0.3, abs(face.rollProxy) / 0.6))

        // Local sharpness of the face relative to the whole frame.
        let crop = face.bbox.inflated(byFactor: 1.2, toWidth: frame.image.width, height: frame.image.height)
        let faceSharp = frame.image.laplacianVariance(in: crop)
        let global = frame.image.laplacianVariance(
            in: RectI(x: 0, y: 0, width: frame.image.width, height: frame.image.height)
        )
        let sharp = clamp01(0.5 + log2(max(faceSharp, 1e-6) / max(global, 1e-6)) / 2)

        return eyesWeight * eyes
            + smileWeight * smile
            + midWordWeight * midWord
            + frontalityWeight * frontality
            + sharpnessWeight * sharp
    }

    @inline(__always)
    private func clamp01(_ v: Float) -> Float { max(0, min(1, v)) }
}

// MARK: - Swap risk

/// Estimates how risky a donor→base face swap is before doing it, so the
/// planner can avoid composites that would fail verification.
public protocol RiskModel: Sendable {
    func risk(base: FaceGeometry, donor: FaceGeometry) -> Float
}

public struct HeuristicRiskModel: RiskModel {
    /// Risk saturates at these deltas: 30° of yaw difference, 30° of roll
    /// difference, 30% size change.
    public var maxPoseDelta: Float = 0.3
    public var maxRollDelta: Float = 0.52 // ≈ 30°
    public var maxScaleDelta: Float = 0.3
    /// Coverage below this on either side counts as occlusion. An elliptical
    /// head fills ≈ 0.78 of its rectangular bbox even when fully visible, so
    /// the threshold must sit well below that.
    public var occlusionThreshold: Float = 0.55

    public init() {}

    public func risk(base: FaceGeometry, donor: FaceGeometry) -> Float {
        let pose = min(1, abs(base.yawProxy - donor.yawProxy) / maxPoseDelta)
        let roll = min(1, abs(base.rollProxy - donor.rollProxy) / maxRollDelta)
        let scale = min(1, abs(log(max(base.interOcular, 1) / max(donor.interOcular, 1))) / maxScaleDelta)
        let occlusion = max(
            0, (occlusionThreshold - base.faceCoverage) + (occlusionThreshold - donor.faceCoverage)
        )
        return min(1.5, pose + roll + scale + max(0, occlusion))
    }
}

// MARK: - Joint group planning

/// Chooses the base frame and, for each person, whether/where to swap from.
///
/// Strategy (plan §Selection): score every person in every frame, choose the
/// base frame with the best *group* score, then per person take the best
/// donor frame discounted by swap risk. With ≤10 people and ≤60 frames the
/// full scoring matrix is cheap; only risk needs to be learned later.
public struct GroupPlanner: Sendable {
    /// How strongly swap risk discounts a donor frame.
    public var riskWeight: Float = 0.6
    /// Donor must beat keeping the base by at least this much to trigger a swap.
    public var swapGainEpsilon: Float = 0.05
    /// Penalty on score variance across the group when choosing the base
    /// (group coherence: matching energy, not one person grinning alone).
    public var coherenceWeight: Float = 0.10
    /// Faces smaller than this inter-ocular distance (px at working
    /// resolution) are ignored — tiny faces transplant garbage.
    public var minInterOcular: Float = 10.0

    public init() {}

    public struct TakePlan: Sendable {
        public var baseFrame: Int
        /// person → donor frame index (== baseFrame when no swap)
        public var donors: [PersonID: Int]
        /// person → frame → score (for reports and debugging)
        public var scoreMatrix: [PersonID: [Float]]
    }

    private func usable(_ frame: AnnotatedFrame, _ person: PersonID) -> Bool {
        guard let face = frame.faces[person] else { return false }
        return face.interOcular >= minInterOcular
    }

    public func plan(
        frames: [AnnotatedFrame],
        people: [PersonID],
        scorer: ExpressionScorer,
        risk: RiskModel
    ) -> TakePlan {
        var matrix: [PersonID: [Float]] = [:]
        for p in people {
            matrix[p] = frames.map { usable($0, p) ? scorer.score(frame: $0, person: p) : -1 }
        }

        // Base frame: maximize total group score minus score spread.
        var baseFrame = 0
        var bestBase = -Float.greatestFiniteMagnitude
        for f in 0..<frames.count {
            let present = people.filter { usable(frames[f], $0) }
            guard !present.isEmpty else { continue }
            let scores = present.map { matrix[$0]![f] }
            let total = scores.reduce(0, +)
            let mean = total / Float(scores.count)
            let variance = scores.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(scores.count)
            let value = total - coherenceWeight * variance * Float(scores.count)
            if value > bestBase {
                bestBase = value
                baseFrame = f
            }
        }

        // Per-person donor: maximize score − λ·risk, requiring a real gain.
        var donors: [PersonID: Int] = [:]
        for p in people {
            guard usable(frames[baseFrame], p) else {
                // Not visible (or too small) in the base frame: no swap.
                donors[p] = baseFrame
                continue
            }
            var best = (frame: baseFrame, value: matrix[p]![baseFrame])
            for f in 0..<frames.count {
                guard usable(frames[f], p), frames[f].personMasks[p] != nil else { continue }
                let geom = frames[f].faces[p]!
                let value = matrix[p]![f] - riskWeight * risk.risk(base: frames[baseFrame].faces[p]!, donor: geom)
                if value > best.value { best = (f, value) }
            }
            donors[p] = (best.value - matrix[p]![baseFrame] >= swapGainEpsilon) ? best.frame : baseFrame
        }

        return TakePlan(baseFrame: baseFrame, donors: donors, scoreMatrix: matrix)
    }
}
