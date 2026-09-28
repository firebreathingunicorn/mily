import Foundation

/// Noise estimation and matched-noise synthesis.
///
/// Phase 1 principle: pasted regions come from differently processed frames,
/// and mismatched grain is the most common giveaway. We estimate the base
/// frame's noise level and inject the difference into transplanted content.
public enum NoiseEstimator {

    /// Immerkær-style noise estimation on a luma plane, adapted to the
    /// 4-neighbor Laplacian L = 4I − (N+S+E+W), whose variance under additive
    /// Gaussian noise is 20σ². Uses the MEDIAN of |L| — for the half-normal
    /// distribution of |L|, median = 0.6745·std — which makes the estimate
    /// robust to sparse texture edges that would inflate a mean-based one.
    public static func sigma(gray: [Float], width: Int, height: Int, region: RectI? = nil) -> Float {
        let c = (region ?? RectI(x: 0, y: 0, width: width, height: height))
            .inflated(byPixels: 0, toWidth: width, height: height)
        guard c.width > 2, c.height > 2 else { return 0 }
        @inline(__always) func g(_ x: Int, _ y: Int) -> Float {
            gray[max(0, min(height - 1, y)) * width + max(0, min(width - 1, x))]
        }
        var values: [Float] = []
        values.reserveCapacity((c.width - 2) * (c.height - 2))
        for y in (c.y + 1)..<(c.maxY - 1) {
            for x in (c.x + 1)..<(c.maxX - 1) {
                let lap = 4 * g(x, y)
                    - g(x - 1, y) - g(x + 1, y)
                    - g(x, y - 1) - g(x, y + 1)
                values.append(abs(lap))
            }
        }
        guard !values.isEmpty else { return 0 }
        values.sort()
        let median = values[values.count / 2]
        return median / (0.6745 * 4.4721)
    }

    public static func sigma(image: PixelImage, region: RectI? = nil) -> Float {
        sigma(gray: image.gray, width: image.width, height: image.height, region: region)
    }
}

public enum NoiseSynthesizer {

    /// Add Gaussian noise weighted by `weight` (0 = none, 1 = full σ),
    /// restricted to `region`. Used to match donor grain to the base frame.
    public static func addMatchedNoise(
        to image: inout PixelImage,
        region: RectI,
        sigma: Float,
        weight: Mask,
        rng: inout SeededRNG
    ) {
        guard sigma > 0 else { return }
        let c = region.clamped(toWidth: image.width, height: image.height)
        for j in c.y..<c.maxY {
            for i in c.x..<c.maxX {
                let w = weight[i, j]
                guard w > 0.003 else { continue }
                let n = rng.nextGaussian() * sigma * w
                let o = (j * image.width + i) * 4
                image.data[o] += n
                image.data[o + 1] += n
                image.data[o + 2] += n
            }
        }
    }
}
