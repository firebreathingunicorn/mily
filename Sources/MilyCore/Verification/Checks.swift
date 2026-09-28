import Foundation

/// Result of one automated check. Everything the UI labels and the report
/// JSON consumes comes from these.
public struct CheckReport: Codable, Sendable {
    public var name: String
    public var passed: Bool
    public var metrics: [String: Float]

    public init(name: String, passed: Bool, metrics: [String: Float] = [:]) {
        self.name = name
        self.passed = passed
        self.metrics = metrics
    }
}

// MARK: - Artifact checker

/// Runs on every swap; nothing reaches the user unless this passes (plan
/// principle 3). On failure the pipeline falls back (next synthesis level, or
/// leaving the person unchanged).
public protocol ArtifactChecker: Sendable {
    func check(base: AnnotatedFrame, result: TransplantResult, person: PersonID) -> CheckReport
}

/// Classical (model-free) checker for Phase 1: seam noise discontinuity,
/// color shift, and sharpness mismatch. The learned classifier (trained on
/// composites vs. real photos — see python/training/artifact_checker) replaces
/// this via the same protocol.
public struct ClassicalArtifactChecker: ArtifactChecker {

    public struct Thresholds: Sendable {
        /// σ(inside) / σ(ring) above this reads as a seam.
        public var seamNoiseRatio: Float = 2.2
        /// Mean-RGB distance between pasted core and the original base face.
        public var maxColorShift: Float = 0.14
        /// Laplacian-variance ratio bounds between content and surroundings.
        public var sharpnessLow: Float = 0.3
        public var sharpnessHigh: Float = 3.0
        public init() {}
    }

    public var thresholds: Thresholds

    public init(thresholds: Thresholds = Thresholds()) {
        self.thresholds = thresholds
    }

    public func check(base: AnnotatedFrame, result: TransplantResult, person: PersonID) -> CheckReport {
        let region = result.region
        let full = RectI(x: 0, y: 0, width: result.content.width, height: result.content.height)

        // 1) Seam noise: σ inside the pasted core vs σ in the ring outside,
        //    both measured on the *composited* frame so the checker sees what
        //    the user would see.
        var composited = base.image
        composited.pasteRegion(content: result.content, alpha: result.alpha, region: region)

        let core = coreRect(alpha: result.alpha)
        let sigmaIn = NoiseEstimator.sigma(image: composited, region: core ?? region)
        let ring = region.inflated(byFactor: 1.7, toWidth: base.image.width, height: base.image.height)
        let sigmaOut = NoiseEstimator.sigma(image: base.image, region: ring)
        let noiseRatio = sigmaOut > 1e-6 ? sigmaIn / sigmaOut : 1

        // 2) Color shift: pasted core vs the base's own pixels under the same
        //    alpha support (face-to-face, background excluded on both sides).
        var accBase = (Float(0), Float(0), Float(0))
        var accContent = (Float(0), Float(0), Float(0))
        var n: Float = 0
        for j in 0..<result.content.height {
            for i in 0..<result.content.width where result.alpha[i, j] > 0.5 {
                let bx = region.x + i, by = region.y + j
                guard bx >= 0, bx < base.image.width, by >= 0, by < base.image.height else { continue }
                let b = base.image[bx, by]
                accBase.0 += b.0; accBase.1 += b.1; accBase.2 += b.2
                let p = result.content[i, j]
                accContent.0 += p.0; accContent.1 += p.1; accContent.2 += p.2
                n += 1
            }
        }
        let baseMean = n > 0 ? (accBase.0 / n, accBase.1 / n, accBase.2 / n) : base.image.meanRGB(in: region)
        let contentMean = n > 0 ? (accContent.0 / n, accContent.1 / n, accContent.2 / n) : baseMean
        let colorShift = sqrt(
            pow(contentMean.0 - baseMean.0, 2)
            + pow(contentMean.1 - baseMean.1, 2)
            + pow(contentMean.2 - baseMean.2, 2)
        )

        // 3) Sharpness ratio between content and base surroundings.
        let contentSharp = max(result.content.laplacianVariance(in: full), 1e-8)
        let baseSharp = max(base.image.laplacianVariance(in: ring), 1e-8)
        let sharpnessRatio = contentSharp / baseSharp

        let passNoise = noiseRatio <= thresholds.seamNoiseRatio
        let passColor = colorShift <= thresholds.maxColorShift
        let passSharp = sharpnessRatio >= thresholds.sharpnessLow && sharpnessRatio <= thresholds.sharpnessHigh

        return CheckReport(
            name: "classical-artifact",
            passed: passNoise && passColor && passSharp,
            metrics: [
                "seamNoiseRatio": noiseRatio,
                "colorShift": colorShift,
                "sharpnessRatio": sharpnessRatio,
            ]
        )
    }

    private func coreRect(alpha: Mask) -> RectI? {
        var minX = alpha.width, maxX = -1, minY = alpha.height, maxY = -1
        for y in 0..<alpha.height {
            for x in 0..<alpha.width where alpha[x, y] > 0.7 {
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }
        }
        guard maxX > minX, maxY > minY else { return nil }
        return RectI(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}

// MARK: - Identity check

/// The face of the result must still be this person (plan §Verification).
/// At Level A this guards against tracking misassignment (wrong-person swap);
/// at Level C it guards identity drift of generated pixels.
public struct IdentityChecker: Sendable {
    /// Relative descriptor distance above this rejects the swap — for a
    /// same-pose pair. 2D landmark proportions drift as the head turns, so
    /// the threshold adapts to the pose difference between the two frames.
    public var baseMaxDistance: Float = 0.08
    /// Threshold growth per unit of |Δyaw proxy|.
    public var poseSlope: Float = 0.15
    /// Upper bound on the adaptive threshold.
    public var maxDistanceCap: Float = 0.14

    public init() {}

    /// Pose-adaptive limit for one base/donor pair.
    public func limit(base: FaceGeometry, donor: FaceGeometry) -> Float {
        let poseDelta = abs(base.yawProxy - donor.yawProxy)
        return min(maxDistanceCap, baseMaxDistance + poseSlope * poseDelta)
    }

    public func check(base: AnnotatedFrame, donor: AnnotatedFrame, person: PersonID) -> CheckReport {
        guard let gb = base.faces[person], let gd = donor.faces[person] else {
            return CheckReport(name: "identity", passed: false, metrics: [:])
        }
        let d = GeometricIdentity.relativeDistance(
            GeometricIdentity.descriptor(for: gb),
            GeometricIdentity.descriptor(for: gd)
        )
        let tau = limit(base: gb, donor: gd)
        return CheckReport(name: "identity", passed: d <= tau, metrics: ["distance": d, "limit": tau])
    }
}
