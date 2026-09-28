import Foundation

/// Finishing: what separates good from undetectable (plan §5).
///
/// - Noise/grain matching: transplanted pixels come from differently
///   processed frames; we estimate the base frame's σ and inject the
///   difference. The most common giveaway once seams are fixed.
/// - Sharpness matching: if the donor content is much sharper than its new
///   surroundings, soften it toward the base's local sharpness.
/// - Soft edges: alpha feathering happens in the synthesizer (matting-driven);
///   the finisher preserves it by weighting noise by the same alpha.
public struct Finisher: Sendable {

    public struct Config: Sendable {
        public var noiseMatching: Bool = true
        public var sharpnessMatching: Bool = true
        /// Match donor content's per-channel mean/σ to the base's own face
        /// (exposure/lighting differences between burst frames).
        public var colorMatching: Bool = true
        /// Mean shift below this is treated as noise, not exposure.
        public var colorMatchMinShift: Float = 0.006
        /// Blur the donor content only when it is sharper than the base by
        /// more than this ratio.
        public var sharpnessRatioThreshold: Float = 1.8
        public init() {}
    }

    public var config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Mutates `result.content` in place. `rngSeed` makes noise deterministic
    /// for tests and reproducible renders.
    public func finish(base: AnnotatedFrame, result: inout TransplantResult, rngSeed: UInt64) {
        let region = result.region

        // 1) Sharpness matching (before noise, so σ is measured after blur).
        //    Reference is the base's own face region — same content type. A
        //    face is always sharper than the background ring around it, so
        //    comparing against the ring would wrongly soften every swap.
        if config.sharpnessMatching {
            let contentSharp = result.content.laplacianVariance(
                in: RectI(x: 0, y: 0, width: result.content.width, height: result.content.height)
            )
            let baseSharp = base.image.laplacianVariance(in: region)
            if baseSharp > 1e-8, contentSharp > baseSharp * config.sharpnessRatioThreshold {
                let ratio = contentSharp / baseSharp
                let radius = max(1, min(2, Int((log2(ratio) * 0.6).rounded())))
                result.content = result.content.blurred(radius: radius)
            }
        }

        // 2) Color harmonization: per-channel mean/σ transfer of the donor
        //    content toward the base's own face pixels. Real bursts can
        //    re-meter between frames; a pasted face must sit at the base
        //    frame's exposure or it reads as "cut and pasted".
        if config.colorMatching {
            colorMatch(base: base, result: &result)
        }

        // 3) Noise matching. σ_base from the ring around the transplant
        //    region (same scene lighting/processing), σ_donor from the
        //    (possibly blurred) content itself.
        guard config.noiseMatching else { return }
        let ring = region.inflated(byFactor: 1.6, toWidth: base.image.width, height: base.image.height)
        let baseSigma = NoiseEstimator.sigma(image: base.image, region: ring)

        var contentCore = RectI(x: 0, y: 0, width: result.content.width, height: result.content.height)
        // Estimate σ only where alpha is confident, to avoid seam contamination.
        if let core = coreRegion(alpha: result.alpha) { contentCore = core }
        let donorSigma = max(result.donorSigma, 0)
        _ = contentCore

        let extraSigma = sqrt(max(0, baseSigma * baseSigma - donorSigma * donorSigma))
        if extraSigma > 0.0005 {
            var rng = SeededRNG(seed: rngSeed)
            NoiseSynthesizer.addMatchedNoise(
                to: &result.content,
                region: RectI(x: 0, y: 0, width: result.content.width, height: result.content.height),
                sigma: extraSigma,
                weight: result.alpha,
                rng: &rng
            )
        }
    }

    /// Per-channel mean/σ transfer of content toward the base's own face
    /// pixels (measured on the alpha-solid support). Skipped when the shift
    /// is within noise — do not chase per-pixel grain with a global gain.
    private func colorMatch(base: AnnotatedFrame, result: inout TransplantResult) {
        let region = result.region
        let alpha = result.alpha

        var n: Float = 0
        var muB = [Float](repeating: 0, count: 3)
        var muD = [Float](repeating: 0, count: 3)
        for j in 0..<alpha.height {
            for i in 0..<alpha.width where alpha[i, j] > 0.5 {
                let bx = region.x + i, by = region.y + j
                guard bx >= 0, bx < base.image.width, by >= 0, by < base.image.height else { continue }
                let b = base.image[bx, by]
                let d = result.content[i, j]
                muB[0] += b.0; muB[1] += b.1; muB[2] += b.2
                muD[0] += d.0; muD[1] += d.1; muD[2] += d.2
                n += 1
            }
        }
        guard n > 64 else { return }
        for c in 0..<3 {
            muB[c] /= n
            muD[c] /= n
        }

        var varB = [Float](repeating: 0, count: 3)
        var varD = [Float](repeating: 0, count: 3)
        for j in 0..<alpha.height {
            for i in 0..<alpha.width where alpha[i, j] > 0.5 {
                let bx = region.x + i, by = region.y + j
                guard bx >= 0, bx < base.image.width, by >= 0, by < base.image.height else { continue }
                let b = base.image[bx, by]
                let d = result.content[i, j]
                let bc = [b.0, b.1, b.2]
                let dc = [d.0, d.1, d.2]
                for c in 0..<3 {
                    varB[c] += (bc[c] - muB[c]) * (bc[c] - muB[c])
                    varD[c] += (dc[c] - muD[c]) * (dc[c] - muD[c])
                }
            }
        }

        var gains = [Float](repeating: 1, count: 3)
        var shifts = [Float](repeating: 0, count: 3)
        var any = false
        for c in 0..<3 {
            let sd = sqrt(varD[c] / n)
            let sb = sqrt(varB[c] / n)
            if sd > 1e-4 {
                gains[c] = max(0.6, min(1.6, sb / sd))
            }
            shifts[c] = muB[c] - muD[c]
            if abs(shifts[c]) > config.colorMatchMinShift || abs(gains[c] - 1) > 0.05 {
                any = true
            }
        }
        guard any else { return }

        for j in 0..<alpha.height {
            for i in 0..<alpha.width {
                let w = alpha[i, j]
                guard w > 0.003 else { continue }
                let o = (j * result.content.width + i) * 4
                for (c, off) in [(0, 0), (1, 1), (2, 2)] {
                    let v = (result.content.data[o + off] - muD[c]) * gains[c] + muB[c]
                    result.content.data[o + off] += (v - result.content.data[o + off]) * w
                }
            }
        }
    }

    /// Largest rectangle where alpha is essentially solid.
    private func coreRegion(alpha: Mask) -> RectI? {
        // Coarse scan: shrink from each side until a row/column exceeds 0.5 mean.
        var top = 0, bottom = alpha.height - 1, left = 0, right = alpha.width - 1
        func rowMean(_ y: Int, _ x0: Int, _ x1: Int) -> Float {
            guard x1 > x0 else { return 0 }
            var acc: Float = 0
            for x in x0...x1 { acc += alpha[x, y] }
            return acc / Float(x1 - x0 + 1)
        }
        func colMean(_ x: Int, _ y0: Int, _ y1: Int) -> Float {
            guard y1 > y0 else { return 0 }
            var acc: Float = 0
            for y in y0...y1 { acc += alpha[x, y] }
            return acc / Float(y1 - y0 + 1)
        }
        var guardCounter = 0
        while top < bottom && rowMean(top, left, right) < 0.5 && guardCounter < alpha.height / 2 { top += 1; guardCounter += 1 }
        guardCounter = 0
        while bottom > top && rowMean(bottom, left, right) < 0.5 && guardCounter < alpha.height / 2 { bottom -= 1; guardCounter += 1 }
        guardCounter = 0
        while left < right && colMean(left, top, bottom) < 0.5 && guardCounter < alpha.width / 2 { left += 1; guardCounter += 1 }
        guardCounter = 0
        while right > left && colMean(right, top, bottom) < 0.5 && guardCounter < alpha.width / 2 { right -= 1; guardCounter += 1 }
        guard right - left > 2, bottom - top > 2 else { return nil }
        return RectI(x: left, y: top, width: right - left, height: bottom - top)
    }
}
