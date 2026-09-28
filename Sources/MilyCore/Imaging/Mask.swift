import Foundation

/// Single-channel float mask in [0, 1]. Used for soft person alpha, feather
/// weights and checker internals. Values are straight (non-premultiplied).
public struct Mask: Sendable {
    public var width: Int
    public var height: Int
    public var data: [Float]

    public init(width: Int, height: Int, fill: Float = 0) {
        self.width = width
        self.height = height
        self.data = [Float](repeating: fill, count: width * height)
    }

    public init(width: Int, height: Int, data: [Float]) {
        precondition(data.count == width * height, "Mask data size mismatch")
        self.width = width
        self.height = height
        self.data = data
    }

    @inline(__always)
    public subscript(x: Int, y: Int) -> Float {
        get { data[y * width + x] }
        set { data[y * width + x] = newValue }
    }

    @inline(__always)
    public func sampleClamped(x: Int, y: Int) -> Float {
        let cx = max(0, min(width - 1, x))
        let cy = max(0, min(height - 1, y))
        return data[cy * width + cx]
    }

    public func crop(_ r: RectI) -> Mask {
        let c = r.clamped(toWidth: width, height: height)
        var out = Mask(width: c.width, height: c.height)
        for j in 0..<c.height {
            let srcRow = (c.y + j) * width + c.x
            out.data.replaceSubrange(j * c.width..<(j + 1) * c.width, with: data[srcRow..<(srcRow + c.width)])
        }
        return out
    }

    /// Paste `src` into self at offset, taking max (union) or sum-clamped.
    public mutating func overlayMax(_ src: Mask, atX x0: Int, y0: Int) {
        for j in 0..<src.height {
            let y = y0 + j
            guard y >= 0, y < height else { continue }
            for i in 0..<src.width {
                let x = x0 + i
                guard x >= 0, x < width else { continue }
                let v = src.data[j * src.width + i]
                if v > data[y * width + x] { data[y * width + x] = v }
            }
        }
    }

    // MARK: Filtering

    /// One separable box pass, horizontal then vertical.
    private func boxPass(radius: Int) -> Mask {
        guard radius > 0 else { return self }
        var tmp = [Float](repeating: 0, count: width * height)
        var out = Mask(width: width, height: height)
        let d = Float(2 * radius + 1)
        for y in 0..<height {
            var acc: Float = 0
            for x in -radius...radius { acc += sampleClamped(x: x, y: y) }
            for x in 0..<width {
                tmp[y * width + x] = acc / d
                acc += sampleClamped(x: x + radius + 1, y: y) - sampleClamped(x: x - radius, y: y)
            }
        }
        for x in 0..<width {
            var acc: Float = 0
            for y in -radius...radius { acc += tmp[max(0, min(height - 1, y)) * width + x] }
            for y in 0..<height {
                out.data[y * width + x] = acc / d
                let yAdd = y + radius + 1, ySub = y - radius
                acc += tmp[max(0, min(height - 1, yAdd)) * width + x]
                    - tmp[max(0, min(height - 1, ySub)) * width + x]
            }
        }
        return out
    }

    /// 3× box blur ≈ Gaussian. `radius` is the per-pass radius.
    public func blurred(radius: Int) -> Mask {
        guard radius > 0 else { return self }
        var m = self
        for _ in 0..<3 { m = m.boxPass(radius: radius) }
        return m
    }

    public func dilated(radius: Int) -> Mask {
        guard radius > 0 else { return self }
        var tmp = [Float](repeating: 0, count: width * height)
        var out = Mask(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                var m: Float = 0
                for k in -radius...radius { m = max(m, sampleClamped(x: x + k, y: y)) }
                tmp[y * width + x] = m
            }
        }
        for y in 0..<height {
            for x in 0..<width {
                var m: Float = 0
                for k in -radius...radius { m = max(m, tmp[max(0, min(height - 1, y + k)) * width + x]) }
                out.data[y * width + x] = m
            }
        }
        return out
    }

    public func eroded(radius: Int) -> Mask {
        var inv = Mask(width: width, height: height, data: data.map { 1 - $0 })
        inv = inv.dilated(radius: radius)
        inv.data = inv.data.map { 1 - $0 }
        return inv
    }

    public func clamped01() -> Mask {
        var m = self
        for i in m.data.indices { m.data[i] = max(0, min(1, m.data[i])) }
        return m
    }

    public func multiplied(_ other: Mask) -> Mask {
        var m = self
        for i in m.data.indices { m.data[i] *= other.data[i] }
        return m
    }

    public var mean: Float {
        data.reduce(0, +) / Float(max(1, data.count))
    }

    public func mean(in region: RectI) -> Float {
        let c = region.clamped(toWidth: width, height: height)
        guard c.width > 0, c.height > 0 else { return 0 }
        var acc: Float = 0
        for j in c.y..<c.maxY {
            for i in c.x..<c.maxX { acc += data[j * width + i] }
        }
        return acc / Float(c.width * c.height)
    }

    // MARK: Construction

    /// Soft elliptical blob: solid 1 inside the core ellipse
    /// (radius − feather), 0 outside (radius + feather), smoothstep between.
    public static func ellipse(
        center: Point2,
        radiusX: Float,
        radiusY: Float,
        feather: Float,
        width: Int,
        height: Int,
        value: Float = 1
    ) -> Mask {
        var m = Mask(width: width, height: height)
        let f = max(1, feather)
        let inX = max(1.0, radiusX - f), inY = max(1.0, radiusY - f)
        let outX = radiusX + f, outY = radiusY + f
        let x0 = max(0, Int(center.x - outX))
        let x1 = min(width - 1, Int(center.x + outX) + 1)
        let y0 = max(0, Int(center.y - outY))
        let y1 = min(height - 1, Int(center.y + outY) + 1)
        guard x0 <= x1, y0 <= y1 else { return m } // ellipse entirely off-canvas
        for y in y0...y1 {
            for x in x0...x1 {
                let dx = Float(x) - center.x, dy = Float(y) - center.y
                let nIn = hypot(dx / inX, dy / inY)
                if nIn <= 1 {
                    m[x, y] = value
                    continue
                }
                let nOut = hypot(dx / outX, dy / outY)
                if nOut >= 1 { continue }
                // Between the ellipses: u = 1 at the core rim, 0 at the outer.
                let u = (nOut - 1) / (nOut - nIn)
                let s = u * u * (3 - 2 * u)
                m[x, y] = value * s
            }
        }
        return m
    }

    /// Area-average resample to a new size.
    public func resampledTo(width nw: Int, height nh: Int) -> Mask {
        guard nw > 0, nh > 0 else { return Mask(width: max(1, nw), height: max(1, nh)) }
        var out = Mask(width: nw, height: nh)
        let sx = Float(width) / Float(nw)
        let sy = Float(height) / Float(nh)
        for j in 0..<nh {
            let fy0 = Float(j) * sy
            let fy1 = Float(j + 1) * sy
            let iy0 = Int(fy0), iy1 = max(iy0 + 1, Int(fy1.rounded(.up)))
            for i in 0..<nw {
                let fx0 = Float(i) * sx
                let fx1 = Float(i + 1) * sx
                let ix0 = Int(fx0), ix1 = max(ix0 + 1, Int(fx1.rounded(.up)))
                var acc: Float = 0
                var count: Float = 0
                for y in iy0..<min(iy1, height) {
                    for x in ix0..<min(ix1, width) {
                        acc += data[y * width + x]
                        count += 1
                    }
                }
                out[i, j] = count > 0 ? acc / count : 0
            }
        }
        return out
    }
}
