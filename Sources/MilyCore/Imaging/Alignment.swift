import Foundation

/// Global frame alignment for bursts of the same scene (handheld camera).
///
/// Phase 1 (Level A) needs translation only: rotation and per-face geometry
/// are handled locally by the landmark-fitted similarity transform in the
/// synthesizer. A coarse-to-fine pyramid SSD search is robust for the kinds
/// of motion bursts contain and avoids FFT edge effects.
///
/// Recoverable range ≈ (coarseRadius + fineRadius) · 2^levels pixels.
public enum Alignment {

    /// Result: moving-frame content offset (subpixel). reference(p) ≈
    /// moving(p + d); to warp `moving` into reference coordinates, sample
    /// moving at (p + d). The integer pyramid optimum is refined with a 1D
    /// parabola fit through the SSD at its four neighbors.
    public static func estimateTranslation(
        reference: PixelImage,
        moving: PixelImage,
        maxShift: Int = 80,
        coarseRadius: Int = 6,
        fineRadius: Int = 2
    ) -> (dx: Float, dy: Float) {
        precondition(reference.width == moving.width && reference.height == moving.height)
        let levelCount = pyramidLevelCount(maxShift: maxShift, width: reference.width, height: reference.height)
        let refPyr = pyramid(reference, levels: levelCount)
        let movPyr = pyramid(moving, levels: levelCount)
        guard refPyr.count == movPyr.count, !refPyr.isEmpty else { return (0, 0) }

        // Search in level coordinates; scale up by 2 when descending to the
        // next (2× finer) level.
        var lx = 0, ly = 0
        for l in stride(from: refPyr.count - 1, through: 0, by: -1) {
            let radius = l == refPyr.count - 1 ? coarseRadius : fineRadius
            let best = bestShiftLevel(
                refGray: refPyr[l].g, movGray: movPyr[l].g,
                w: refPyr[l].w, h: refPyr[l].h,
                aroundLX: lx, aroundLY: ly, radius: radius
            )
            lx = best.dx
            ly = best.dy
            if l > 0 { lx *= 2; ly *= 2 }
        }

        // Subpixel refinement at full resolution.
        let refGray = refPyr[0].g, movGray = movPyr[0].g
        let w = refPyr[0].w, h = refPyr[0].h
        let cCenter = ssd(refGray: refGray, movGray: movGray, w: w, h: h, cx: lx, cy: ly)
        let dxOff = subpixelOffset(
            cMinus: ssd(refGray: refGray, movGray: movGray, w: w, h: h, cx: lx - 1, cy: ly),
            cCenter: cCenter,
            cPlus: ssd(refGray: refGray, movGray: movGray, w: w, h: h, cx: lx + 1, cy: ly)
        )
        let dyOff = subpixelOffset(
            cMinus: ssd(refGray: refGray, movGray: movGray, w: w, h: h, cx: lx, cy: ly - 1),
            cCenter: cCenter,
            cPlus: ssd(refGray: refGray, movGray: movGray, w: w, h: h, cx: lx, cy: ly + 1)
        )
        return (Float(lx) + dxOff, Float(ly) + dyOff)
    }

    /// Offset in [-0.5, 0.5] of the parabola minimum through three
    /// equally spaced SSD samples; 0 when the fit is degenerate.
    static func subpixelOffset(cMinus: Float, cCenter: Float, cPlus: Float) -> Float {
        let denom = cMinus - 2 * cCenter + cPlus
        guard denom > 1e-12 else { return 0 } // not a proper minimum
        let offset = 0.5 * (cMinus - cPlus) / denom
        return max(-0.5, min(0.5, offset))
    }

    static func ssd(refGray: [Float], movGray: [Float], w: Int, h: Int, cx: Int, cy: Int) -> Float {
        let m = max(2, abs(cx) + abs(cy) + 2)
        let x0 = m, x1 = w - m
        let y0 = m, y1 = h - m
        guard x1 - x0 >= 4, y1 - y0 >= 4 else { return .greatestFiniteMagnitude }
        var cost: Float = 0
        for y in stride(from: y0, to: y1, by: 2) {
            let rowRef = y * w
            let rowMov = (y + cy) * w + cx
            for x in stride(from: x0, to: x1, by: 2) {
                let d = refGray[rowRef + x] - movGray[rowMov + x]
                cost += d * d
            }
        }
        return cost
    }

    static func pyramidLevelCount(maxShift: Int, width: Int, height: Int) -> Int {
        var l = 0
        while (1 << l) * 4 <= maxShift && min(width, height) >> l >= 48 { l += 1 }
        return min(l, 5)
    }

    static func pyramid(_ img: PixelImage, levels: Int) -> [(g: [Float], w: Int, h: Int)] {
        var out: [(g: [Float], w: Int, h: Int)] = [(img.gray, img.width, img.height)]
        for _ in 1...max(0, levels) {
            let prev = out.last!
            let nw = prev.w / 2, nh = prev.h / 2
            guard nw >= 12, nh >= 12 else { break }
            var g = [Float](repeating: 0, count: nw * nh)
            for y in 0..<nh {
                for x in 0..<nw {
                    let o0 = (2 * y) * prev.w + 2 * x
                    let o1 = o0 + 1
                    let o2 = o0 + prev.w
                    let o3 = o2 + 1
                    g[y * nw + x] = 0.25 * (prev.g[o0] + prev.g[o1] + prev.g[o2] + prev.g[o3])
                }
            }
            out.append((g, nw, nh))
        }
        return out
    }

    /// SSD over the level's interior (sampled every 2 px), level coordinates.
    /// The interior shrinks to keep every candidate window — which can be
    /// centered up to |around| + radius off the origin — inside the level.
    static func bestShiftLevel(
        refGray: [Float], movGray: [Float],
        w: Int, h: Int,
        aroundLX: Int, aroundLY: Int, radius: Int
    ) -> (dx: Int, dy: Int) {
        let bx = abs(aroundLX) + radius + 1
        let by = abs(aroundLY) + radius + 1
        let x0 = bx, x1 = w - bx
        let y0 = by, y1 = h - by
        guard x1 - x0 >= 4, y1 - y0 >= 4 else { return (aroundLX, aroundLY) }

        var best = (dx: aroundLX, dy: aroundLY)
        var bestCost = Float.greatestFiniteMagnitude
        for cy in (aroundLY - radius)...(aroundLY + radius) {
            for cx in (aroundLX - radius)...(aroundLX + radius) {
                var cost: Float = 0
                for y in stride(from: y0, to: y1, by: 2) {
                    let rowRef = y * w
                    let rowMov = (y + cy) * w + cx
                    for x in stride(from: x0, to: x1, by: 2) {
                        let d = refGray[rowRef + x] - movGray[rowMov + x]
                        cost += d * d
                    }
                }
                if cost < bestCost {
                    bestCost = cost
                    best = (cx, cy)
                }
            }
        }
        return best
    }
}
