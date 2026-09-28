import Foundation

/// Small image composition utilities for review output.
public enum ImageOps {

    /// Base | result side by side with a dark divider — the artifact for
    /// blind "is this edited?" review and the CLI's --compare output.
    public static func sideBySide(_ a: PixelImage, _ b: PixelImage, gap: Int = 10) -> PixelImage {
        precondition(a.width == b.width && a.height == b.height, "sideBySide expects same-size images")
        let w = a.width, h = a.height
        var out = PixelImage(width: w * 2 + gap, height: h, fill: (0.08, 0.08, 0.09, 1))
        for y in 0..<h {
            let srcRow = y * w * 4
            out.data.replaceSubrange((y * out.width) * 4..<(y * out.width + w) * 4,
                                     with: a.data[srcRow..<(srcRow + w * 4)])
            let bStart = (y * out.width + w + gap) * 4
            out.data.replaceSubrange(bStart..<(bStart + w * 4),
                                     with: b.data[srcRow..<(srcRow + w * 4)])
        }
        return out
    }
}
