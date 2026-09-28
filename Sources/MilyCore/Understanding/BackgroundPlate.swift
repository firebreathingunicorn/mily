import Foundation

/// Clean background plate built from the burst: for every pixel, take the
/// temporal median of samples from frames where no person covers that pixel.
/// Whole-person swaps (plan §Synthesis) and uncovered-region fills use this.
public enum BackgroundPlate {

    public struct SourceFrame {
        /// Content offset of this frame relative to the base frame (from
        /// `Alignment.estimateTranslation`): reference(p) ≈ moving(p + offset).
        public var offset: (dx: Int, dy: Int)
        public var frame: AnnotatedFrame

        public init(offset: (dx: Int, dy: Int), frame: AnnotatedFrame) {
            self.offset = offset
            self.frame = frame
        }
    }

    /// Build the plate in base-frame coordinates.
    /// - Parameters:
    ///   - base: the frame whose coordinate system the plate uses
    ///   - sources: all burst frames aligned to the base (the base itself should
    ///     be included with offset (0, 0))
    ///   - coverageThreshold: person alpha below this counts as "pixel free"
    public static func build(
        base: AnnotatedFrame,
        sources: [SourceFrame],
        coverageThreshold: Float = 0.35
    ) -> PixelImage {
        let w = base.image.width, h = base.image.height
        var out = base.image

        // Precompute per-frame combined person coverage in base coordinates.
        let coverage: [Mask] = sources.map { src in
            guard !src.frame.personMasks.isEmpty else {
                return Mask(width: w, height: h)
            }
            // Union of all person masks, sampled at base pixel p ← frame(p + offset).
            var m = Mask(width: w, height: h)
            for pm in src.frame.personMasks.values {
                for y in 0..<h {
                    let sy = y + src.offset.dy
                    guard sy >= 0, sy < h else { continue }
                    for x in 0..<w {
                        let sx = x + src.offset.dx
                        guard sx >= 0, sx < w else { continue }
                        let v = pm[sx, sy]
                        if v > m[x, y] { m[x, y] = v }
                    }
                }
            }
            return m
        }

        // For pixels where the base is itself free, the base pixel is already
        // correct. Only pixels covered in the base need filling from elsewhere.
        var baseCoverage = Mask(width: w, height: h)
        for pm in base.personMasks.values {
            for i in baseCoverage.data.indices {
                if pm.data[i] > baseCoverage.data[i] {
                    baseCoverage.data[i] = pm.data[i]
                }
            }
        }

        for y in 0..<h {
            for x in 0..<w {
                guard baseCoverage[x, y] > coverageThreshold else { continue }
                var samples: [(Float, Float, Float)] = []
                for (si, src) in sources.enumerated() where coverage[si][x, y] <= coverageThreshold {
                    let sx = x + src.offset.dx
                    let sy = y + src.offset.dy
                    guard sx >= 0, sx < w, sy >= 0, sy < h else { continue }
                    let p = src.frame.image[sx, sy]
                    samples.append((p.0, p.1, p.2))
                }
                guard samples.count >= 2 else { continue } // keep base pixel otherwise
                samples.sort { $0.0 < $1.0 }
                let mid = samples[samples.count / 2]
                out[x, y] = (mid.0, mid.1, mid.2, 1)
            }
        }
        return out
    }
}
