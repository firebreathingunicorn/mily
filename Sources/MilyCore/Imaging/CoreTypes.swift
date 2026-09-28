import Foundation

public typealias PersonID = String

// MARK: - Geometry

/// Integer rectangle in pixel coordinates, origin top-left, y-down.
public struct RectI: Equatable, Sendable {
    public var x: Int
    public var y: Int
    public var width: Int
    public var height: Int

    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public var maxX: Int { x + width }
    public var maxY: Int { y + height }
    public var centerX: Float { Float(x) + Float(width) / 2 }
    public var centerY: Float { Float(y) + Float(height) / 2 }

    public func clamped(toWidth w: Int, height h: Int) -> RectI {
        let x0 = max(0, min(x, w))
        let y0 = max(0, min(y, h))
        let x1 = max(0, min(maxX, w))
        let y1 = max(0, min(maxY, h))
        return RectI(x: x0, y: y0, width: max(0, x1 - x0), height: max(0, y1 - y0))
    }

    /// Scale around the center by `factor`, then clamp to image bounds.
    public func inflated(byFactor factor: Float, toWidth w: Int, height h: Int) -> RectI {
        let cx = centerX, cy = centerY
        let hw = Float(width) * factor / 2
        let hh = Float(height) * factor / 2
        let r = RectI(
            x: Int((cx - hw).rounded()),
            y: Int((cy - hh).rounded()),
            width: Int((hw * 2).rounded()),
            height: Int((hh * 2).rounded())
        )
        return r.clamped(toWidth: w, height: h)
    }

    public func inflated(byPixels p: Int, toWidth w: Int, height h: Int) -> RectI {
        RectI(x: x - p, y: y - p, width: width + 2 * p, height: height + 2 * p)
            .clamped(toWidth: w, height: h)
    }

    public func contains(x px: Int, y py: Int) -> Bool {
        px >= x && px < maxX && py >= y && py < maxY
    }
}

/// 2D point in pixel coordinates (y-down).
public struct Point2: Equatable, Sendable {
    public var x: Float
    public var y: Float
    public init(_ x: Float, _ y: Float) {
        self.x = x
        self.y = y
    }
    public func distance(to other: Point2) -> Float {
        hypot(x - other.x, y - other.y)
    }
}

// MARK: - Similarity transform

/// 2D similarity transform: dst = M(src) where
///   x' = a*x - b*y + tx
///   y' = b*x + a*y + ty
public struct SimilarityTransform: Sendable {
    public var a: Float
    public var b: Float
    public var tx: Float
    public var ty: Float

    public init(a: Float, b: Float, tx: Float, ty: Float) {
        self.a = a
        self.b = b
        self.tx = tx
        self.ty = ty
    }

    public static let identity = SimilarityTransform(a: 1, b: 0, tx: 0, ty: 0)

    public var scale: Float { hypot(a, b) }
    public var rotation: Float { atan2(b, a) }

    public func apply(x: Float, y: Float) -> (Float, Float) {
        (a * x - b * y + tx, b * x + a * y + ty)
    }

    public func apply(_ p: Point2) -> Point2 {
        let q = apply(x: p.x, y: p.y)
        return Point2(q.0, q.1)
    }

    /// Inverse transform (M⁻¹), assuming scale > 0.
    public var inverted: SimilarityTransform {
        let d = a * a + b * b
        guard d > 1e-12 else { return .identity }
        let ia = a / d
        let ib = -b / d
        return SimilarityTransform(a: ia, b: ib, tx: -(ia * tx - ib * ty), ty: -(ib * tx + ia * ty))
    }

    /// Least-squares fit of a similarity transform mapping `src` onto `dst`.
    /// Returns nil if there are fewer than 2 points or degenerate spread.
    public static func fit(src: [Point2], dst: [Point2]) -> SimilarityTransform? {
        guard src.count == dst.count, src.count >= 2 else { return nil }
        let n = Float(src.count)
        var msx: Float = 0, msy: Float = 0, mdx: Float = 0, mdy: Float = 0
        for i in 0..<src.count {
            msx += src[i].x; msy += src[i].y
            mdx += dst[i].x; mdy += dst[i].y
        }
        msx /= n; msy /= n; mdx /= n; mdy /= n

        var numA: Float = 0, numB: Float = 0, den: Float = 0
        for i in 0..<src.count {
            let cx = src[i].x - msx, cy = src[i].y - msy
            let dx = dst[i].x - mdx, dy = dst[i].y - mdy
            numA += cx * dx + cy * dy
            numB += cx * dy - cy * dx
            den += cx * cx + cy * cy
        }
        guard den > 1e-8 else { return nil }
        let a = numA / den
        let b = numB / den
        return SimilarityTransform(a: a, b: b, tx: mdx - (a * msx - b * msy), ty: mdy - (b * msx + a * msy))
    }
}

// MARK: - Deterministic RNG (for reproducible noise in tests and finishing)

/// SplitMix64-based seeded RNG. Deterministic across runs on the same platform.
public struct SeededRNG: Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func nextUnit() -> Float {
        Float(nextUInt64() >> 40) / Float(1 << 24)
    }

    /// Standard normal via Box-Muller.
    public mutating func nextGaussian() -> Float {
        let u1 = max(nextUnit(), 1e-9)
        let u2 = nextUnit()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}
