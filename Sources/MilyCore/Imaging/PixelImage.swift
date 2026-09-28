import Foundation

/// Straight (non-premultiplied) RGBA image stored as interleaved Float in [0, 1].
/// Float everywhere keeps compositing math simple; capture paths convert from
/// 8-bit at load time and quantize only on save.
public struct PixelImage: Sendable {
    public var width: Int
    public var height: Int
    public var data: [Float] // RGBA interleaved

    public init(width: Int, height: Int, fill: (Float, Float, Float, Float) = (0, 0, 0, 0)) {
        self.width = width
        self.height = height
        var d = [Float](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            d[i * 4 + 0] = fill.0
            d[i * 4 + 1] = fill.1
            d[i * 4 + 2] = fill.2
            d[i * 4 + 3] = fill.3
        }
        self.data = d
    }

    public init(width: Int, height: Int, data: [Float]) {
        precondition(data.count == width * height * 4, "PixelImage data size mismatch")
        self.width = width
        self.height = height
        self.data = data
    }

    @inline(__always)
    public subscript(x: Int, y: Int) -> (Float, Float, Float, Float) {
        get {
            let o = (y * width + x) * 4
            return (data[o], data[o + 1], data[o + 2], data[o + 3])
        }
        set {
            let o = (y * width + x) * 4
            data[o] = newValue.0
            data[o + 1] = newValue.1
            data[o + 2] = newValue.2
            data[o + 3] = newValue.3
        }
    }

    @inline(__always)
    public func sampleClamped(x: Int, y: Int) -> (Float, Float, Float, Float) {
        let cx = max(0, min(width - 1, x))
        let cy = max(0, min(height - 1, y))
        return self[cx, cy]
    }

    public var gray: [Float] {
        var g = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let o = i * 4
            g[i] = 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2]
        }
        return g
    }

    public func crop(_ r: RectI) -> PixelImage {
        let c = r.clamped(toWidth: width, height: height)
        var out = PixelImage(width: c.width, height: c.height)
        for j in 0..<c.height {
            let srcStart = ((c.y + j) * width + c.x) * 4
            let n = c.width * 4
            out.data.replaceSubrange(j * n..<j * n + n, with: data[srcStart..<(srcStart + n)])
        }
        return out
    }

    /// Composite `src` over self using `alpha` (same dims as self), restricted to `region`.
    public mutating func compositeOver(region: RectI, from src: PixelImage, alpha: Mask) {
        let c = region.clamped(toWidth: width, height: height)
        precondition(src.width == width && src.height == height, "src must match dst dims")
        for j in c.y..<c.maxY {
            for i in c.x..<c.maxX {
                let a = alpha[i, j]
                guard a > 0 else { continue }
                let s = src[i, j]
                let o = (j * width + i) * 4
                data[o] = s.0 * a + data[o] * (1 - a)
                data[o + 1] = s.1 * a + data[o + 1] * (1 - a)
                data[o + 2] = s.2 * a + data[o + 2] * (1 - a)
                // alpha channel: opaque sources dominate
                data[o + 3] = max(data[o + 3], s.3 * a)
            }
        }
    }

    /// Composite region-sized `content`/`alpha` onto self at `region`.
    public mutating func pasteRegion(content: PixelImage, alpha: Mask, region: RectI) {
        let c = region.clamped(toWidth: width, height: height)
        precondition(content.width >= c.width && content.height >= c.height, "content smaller than region")
        for j in 0..<c.height {
            for i in 0..<c.width {
                let a = alpha[i, j]
                guard a > 0 else { continue }
                let s = content[i, j]
                let o = ((c.y + j) * width + c.x + i) * 4
                data[o] = s.0 * a + data[o] * (1 - a)
                data[o + 1] = s.1 * a + data[o + 1] * (1 - a)
                data[o + 2] = s.2 * a + data[o + 2] * (1 - a)
                data[o + 3] = max(data[o + 3], s.3 * a)
            }
        }
    }

    /// Box-filter downsample by integer factor (dims divided, remainder dropped).
    public func downsampled(factor: Int) -> PixelImage {
        precondition(factor >= 1)
        guard factor > 1 else { return self }
        let nw = width / factor, nh = height / factor
        guard nw > 0, nh > 0 else { return PixelImage(width: 1, height: 1, fill: (0, 0, 0, 1)) }
        var out = PixelImage(width: nw, height: nh)
        let inv = 1 / Float(factor * factor)
        for j in 0..<nh {
            for i in 0..<nw {
                var r: Float = 0, g: Float = 0, b: Float = 0, a: Float = 0
                for v in 0..<factor {
                    for u in 0..<factor {
                        let o = ((j * factor + v) * width + i * factor + u) * 4
                        r += data[o]; g += data[o + 1]; b += data[o + 2]; a += data[o + 3]
                    }
                }
                let o2 = (j * nw + i) * 4
                out.data[o2] = r * inv
                out.data[o2 + 1] = g * inv
                out.data[o2 + 2] = b * inv
                out.data[o2 + 3] = a * inv
            }
        }
        return out
    }

    // MARK: Statistics

    public func meanRGB(in region: RectI) -> (Float, Float, Float) {
        let c = region.clamped(toWidth: width, height: height)
        guard c.width > 0, c.height > 0 else { return (0, 0, 0) }
        var r: Float = 0, g: Float = 0, b: Float = 0
        for j in c.y..<c.maxY {
            for i in c.x..<c.maxX {
                let o = (j * width + i) * 4
                r += data[o]; g += data[o + 1]; b += data[o + 2]
            }
        }
        let n = Float(c.width * c.height)
        return (r / n, g / n, b / n)
    }

    /// Variance of the Laplacian (sharpness proxy) over the region's gray values.
    public func laplacianVariance(in region: RectI) -> Float {
        let c = region.inflated(byPixels: 1, toWidth: width, height: height)
        guard c.width > 2, c.height > 2 else { return 0 }
        var vals: [Float] = []
        vals.reserveCapacity(c.width * c.height)
        for j in (c.y + 1)..<(c.maxY - 1) {
            for i in (c.x + 1)..<(c.maxX - 1) {
                let c0 = luma(x: i, y: j)
                let lap = 4 * c0
                    - luma(x: i - 1, y: j) - luma(x: i + 1, y: j)
                    - luma(x: i, y: j - 1) - luma(x: i, y: j + 1)
                vals.append(lap)
            }
        }
        guard !vals.isEmpty else { return 0 }
        let mean = vals.reduce(0, +) / Float(vals.count)
        var vr: Float = 0
        for v in vals { let d = v - mean; vr += d * d }
        return vr / Float(vals.count)
    }

    @inline(__always)
    private func luma(x: Int, y: Int) -> Float {
        let o = (y * width + x) * 4
        return 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2]
    }

    // MARK: Filtering

    /// 3-pass box blur ≈ Gaussian on RGB channels, whole image.
    public func blurred(radius: Int) -> PixelImage {
        guard radius > 0 else { return self }
        var img = self
        for _ in 0..<3 { img = img.boxPass(radius: radius) }
        return img
    }

    private func boxPass(radius: Int) -> PixelImage {
        var tmp = [Float](repeating: 0, count: width * height * 4)
        var out = PixelImage(width: width, height: height)
        let d = Float(2 * radius + 1)
        let w = width, h = height, n = 4
        for y in 0..<h {
            var a0: Float = 0, a1: Float = 0, a2: Float = 0, a3: Float = 0
            for x in -radius...radius {
                let p = sampleClamped(x: x, y: y)
                a0 += p.0; a1 += p.1; a2 += p.2; a3 += p.3
            }
            for x in 0..<w {
                let o = (y * w + x) * n
                tmp[o] = a0 / d; tmp[o + 1] = a1 / d; tmp[o + 2] = a2 / d; tmp[o + 3] = a3 / d
                let add = sampleClamped(x: x + radius + 1, y: y)
                let sub = sampleClamped(x: x - radius, y: y)
                a0 += add.0 - sub.0; a1 += add.1 - sub.1; a2 += add.2 - sub.2; a3 += add.3 - sub.3
            }
        }
        for x in 0..<w {
            var a0: Float = 0, a1: Float = 0, a2: Float = 0, a3: Float = 0
            for y in -radius...radius {
                let p = sampleClamped(x: x, y: max(0, min(h - 1, y)))
                a0 += p.0; a1 += p.1; a2 += p.2; a3 += p.3
            }
            for y in 0..<h {
                let o = (y * w + x) * n
                out.data[o] = a0 / d; out.data[o + 1] = a1 / d; out.data[o + 2] = a2 / d; out.data[o + 3] = a3 / d
                let add = sampleClamped(x: x, y: y + radius + 1)
                let sub = sampleClamped(x: x, y: y - radius)
                a0 += add.0 - sub.0; a1 += add.1 - sub.1; a2 += add.2 - sub.2; a3 += add.3 - sub.3
            }
        }
        return out
    }
}

@inline(__always)
private func k(_ p: (Float, Float, Float, Float), _ i: Int) -> Float {
    switch i {
    case 0: return p.0
    case 1: return p.1
    case 2: return p.2
    default: return p.3
    }
}
