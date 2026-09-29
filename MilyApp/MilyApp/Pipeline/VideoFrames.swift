import AVFoundation
import CoreGraphics

/// Samples evenly spaced frames from a short clip (the first ~4 s) — a video
/// gives each person far more moments to choose from than a photo burst.
enum VideoFrames {
    static func extract(from url: URL, count: Int, maxLongEdge: Int) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        let span = min(max(duration, 0.1), 4.0)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: maxLongEdge, height: maxLongEdge)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var images: [CGImage] = []
        for i in 0..<count {
            let t = CMTime(seconds: span * Double(i) / Double(max(count - 1, 1)), preferredTimescale: 600)
            if let (image, _) = try? await generator.image(at: t) {
                images.append(image)
            }
        }
        guard images.count >= 2 else {
            throw NSError(domain: "Mily", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "The clip is too short to find moments in."])
        }
        return images
    }
}
