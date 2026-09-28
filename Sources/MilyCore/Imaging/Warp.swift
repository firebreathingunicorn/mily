import Foundation

/// Image warping and resampling.
///
/// The resampling entry points take a coordinate mapping closure rather than a
/// concrete transform type. Level A passes a fitted similarity transform;
/// Level B (3D-aware re-projection) can pass a perspective or per-pixel
/// deformation without touching the sampling code.
public enum Warp {

    /// Resample `src` into a full `dstWidth × dstHeight` canvas. For each
    /// destination pixel `q`, sample `src` at `dstToSrc(q)` with bilinear
    /// interpolation; out-of-bounds samples come back transparent.
    public static func resample(
        src: PixelImage,
        dstWidth: Int,
        dstHeight: Int,
        dstToSrc: (Float, Float) -> (Float, Float)
    ) -> PixelImage {
        var out = PixelImage(width: dstWidth, height: dstHeight)
        for y in 0..<dstHeight {
            for x in 0..<dstWidth {
                let s = dstToSrc(Float(x), Float(y))
                let v = bilinear(src: src, sx: s.0, sy: s.1)
                let o = (y * dstWidth + x) * 4
                out.data[o] = v.0
                out.data[o + 1] = v.1
                out.data[o + 2] = v.2
                out.data[o + 3] = v.3
            }
        }
        return out
    }

    /// Resample only `region` of the destination canvas; result is region-sized.
    /// This is the hot path for face transplants (small crop, not full frame).
    public static func resampleRegion(
        src: PixelImage,
        region: RectI,
        dstToSrc: (Float, Float) -> (Float, Float)
    ) -> PixelImage {
        let c = region.clamped(toWidth: max(src.width, 1), height: max(src.height, 1))
        var out = PixelImage(width: c.width, height: c.height)
        for j in 0..<c.height {
            for i in 0..<c.width {
                let s = dstToSrc(Float(c.x + i), Float(c.y + j))
                let v = bilinear(src: src, sx: s.0, sy: s.1)
                let o = (j * c.width + i) * 4
                out.data[o] = v.0
                out.data[o + 1] = v.1
                out.data[o + 2] = v.2
                out.data[o + 3] = v.3
            }
        }
        return out
    }

    /// Mask variant of `resampleRegion` (person alpha warped with the same
    /// mapping as the pixels, so edges stay in registration).
    public static func resampleMaskRegion(
        src: Mask,
        region: RectI,
        dstToSrc: (Float, Float) -> (Float, Float)
    ) -> Mask {
        let c = region.clamped(toWidth: max(src.width, 1), height: max(src.height, 1))
        var out = Mask(width: c.width, height: c.height)
        for j in 0..<c.height {
            for i in 0..<c.width {
                let s = dstToSrc(Float(c.x + i), Float(c.y + j))
                out[i, j] = bilinearMask(src: src, sx: s.0, sy: s.1)
            }
        }
        return out
    }

    @inline(__always)
    static func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a * (1 - t) + b * t }

    @inline(__always)
    static func bilinear(src: PixelImage, sx: Float, sy: Float) -> (Float, Float, Float, Float) {
        let w = Float(src.width), h = Float(src.height)
        guard sx >= 0, sy >= 0, sx <= w - 1, sy <= h - 1 else { return (0, 0, 0, 0) }
        let x0 = Int(sx), y0 = Int(sy)
        let x1 = min(x0 + 1, src.width - 1), y1 = min(y0 + 1, src.height - 1)
        let fx = sx - Float(x0), fy = sy - Float(y0)
        let p00 = src[x0, y0], p10 = src[x1, y0], p01 = src[x0, y1], p11 = src[x1, y1]
        let top = (mix(p00.0, p10.0, fx), mix(p00.1, p10.1, fx), mix(p00.2, p10.2, fx), mix(p00.3, p10.3, fx))
        let bot = (mix(p01.0, p11.0, fx), mix(p01.1, p11.1, fx), mix(p01.2, p11.2, fx), mix(p01.3, p11.3, fx))
        return (mix(top.0, bot.0, fy), mix(top.1, bot.1, fy), mix(top.2, bot.2, fy), mix(top.3, bot.3, fy))
    }

    @inline(__always)
    static func bilinearMask(src: Mask, sx: Float, sy: Float) -> Float {
        let w = Float(src.width), h = Float(src.height)
        guard sx >= 0, sy >= 0, sx <= w - 1, sy <= h - 1 else { return 0 }
        let x0 = Int(sx), y0 = Int(sy)
        let x1 = min(x0 + 1, src.width - 1), y1 = min(y0 + 1, src.height - 1)
        let fx = sx - Float(x0), fy = sy - Float(y0)
        let top = mix(src[x0, y0], src[x1, y0], fx)
        let bot = mix(src[x0, y1], src[x1, y1], fx)
        return mix(top, bot, fy)
    }
}
