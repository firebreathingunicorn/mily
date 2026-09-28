import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// CGImage <-> PixelImage conversion and file IO (JPEG/PNG/HEIC).
public enum ImageIO {

    /// Decode an image file, optionally downsampling so the long edge is
    /// `maxLongEdge` (0 = full resolution). Applies EXIF orientation.
    public static func load(path: URL, maxLongEdge: Int = 0) throws -> PixelImage {
        let srcOptions: [CFString: Any] = [kCGImageSourceShouldCache: false]
        guard let src = CGImageSourceCreateWithURL(path as CFURL, srcOptions as CFDictionary) else {
            throw IOError.cannotOpen(path)
        }
        var thumbOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        if maxLongEdge > 0 {
            thumbOptions[kCGImageSourceThumbnailMaxPixelSize] = maxLongEdge
        }
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, thumbOptions as CFDictionary) else {
            throw IOError.cannotDecode(path)
        }
        return fromCGImage(cg)
    }

    public static func save(_ image: PixelImage, to path: URL, quality: Double = 0.92) throws {
        let cg = toCGImage(image)
        let type: UTType
        switch path.pathExtension.lowercased() {
        case "png": type = .png
        case "heic", "heif": type = .heic
        default: type = .jpeg
        }
        guard let dest = CGImageDestinationCreateWithURL(
            path as CFURL, type.identifier as CFString, 1, nil
        ) else {
            throw IOError.cannotWrite(path)
        }
        let opts: [CFString: Any] = type == .png
            ? [:]
            : [kCGImageDestinationLossyCompressionQuality: quality]
        CGImageDestinationAddImage(dest, cg, opts as CFDictionary)
        guard CGImageDestinationFinalize(dest) else {
            throw IOError.cannotWrite(path)
        }
    }

    // MARK: Conversion

    /// High-quality downscale, no-op when the image is already within bounds.
    public static func resized(_ cg: CGImage, maxLongEdge: Int) -> CGImage {
        guard maxLongEdge > 0 else { return cg }
        let longEdge = max(cg.width, cg.height)
        guard longEdge > maxLongEdge else { return cg }
        let scale = CGFloat(maxLongEdge) / CGFloat(longEdge)
        let w = max(1, Int((CGFloat(cg.width) * scale).rounded()))
        let h = max(1, Int((CGFloat(cg.height) * scale).rounded()))
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return cg }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage() ?? cg
    }

    public static func fromCGImage(_ cg: CGImage) -> PixelImage {
        let w = cg.width, h = cg.height
        var ctxData = [UInt8](repeating: 0, count: w * h * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let ctx = ctxData.withUnsafeMutableBytes { ptr -> CGContext? in
            CGContext(
                data: ptr.baseAddress,
                width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        }
        guard let context = ctx else {
            // Fall back to a solid image if the context cannot be created.
            return PixelImage(width: w, height: h, fill: (0, 0, 0, 1))
        }
        context.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var out = PixelImage(width: w, height: h)
        let n = w * h
        for i in 0..<n {
            let s = i * 4
            let a = Float(ctxData[s + 3]) / 255
            var r = Float(ctxData[s]) / 255
            var g = Float(ctxData[s + 1]) / 255
            var b = Float(ctxData[s + 2]) / 255
            // Unpremultiply (loaded CGImages are premultiplied).
            if a > 0.0001 {
                r /= a; g /= a; b /= a
            } else {
                r = 0; g = 0; b = 0
            }
            out.data[s] = min(1, r)
            out.data[s + 1] = min(1, g)
            out.data[s + 2] = min(1, b)
            out.data[s + 3] = a
        }
        return out
    }

    public static func toCGImage(_ image: PixelImage) -> CGImage {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) {
            let o = i * 4
            let a = max(0, min(1, image.data[o + 3]))
            bytes[o] = UInt8((max(0, min(1, image.data[o])) * 255).rounded())
            bytes[o + 1] = UInt8((max(0, min(1, image.data[o + 1])) * 255).rounded())
            bytes[o + 2] = UInt8((max(0, min(1, image.data[o + 2])) * 255).rounded())
            bytes[o + 3] = UInt8((a * 255).rounded())
        }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let ctx = bytes.withUnsafeMutableBytes { ptr -> CGContext? in
            CGContext(
                data: ptr.baseAddress,
                width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        }
        return ctx!.makeImage()!
    }
}

public enum IOError: Error, CustomStringConvertible {
    case cannotOpen(URL)
    case cannotDecode(URL)
    case cannotWrite(URL)

    public var description: String {
        switch self {
        case .cannotOpen(let u): return "cannot open \(u.path)"
        case .cannotDecode(let u): return "cannot decode \(u.path)"
        case .cannotWrite(let u): return "cannot write \(u.path)"
        }
    }
}
