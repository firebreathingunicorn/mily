import SwiftUI
import ImageIO

/// Mily's mascot states, backed by the animated PNGs from `mascot/assets`.
enum MascotState: String, CaseIterable {
    case idle
    case baking      // processing — Mily is "baking" the fix
    case boing       // success
    case rain        // couldn't safely fix
    case chasing     // loader: the fast cat running flat out (seamless loop)
    case peeking     // shutter swat

    var fileName: String {
        switch self {
        case .idle: return "idle_160"
        case .baking: return "baking_160"
        case .boing: return "boing_160"
        case .rain: return "rain_160"
        case .chasing: return "treadmill_160"
        case .peeking: return "peeking_160"
        }
    }

    static func image(named name: String) -> UIImage? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "png") else {
            return nil
        }
        return AnimatedPNG.imageAtURL(url)
    }
}

/// Decodes an animated PNG (APNG) into a repeating `UIImage.animatedImage`.
/// ImageIO enumerates APNG subframes like GIF frames; per-frame delays come
/// from the `fcTL` chunk (surfaced as `APNGDelayTime`).
enum AnimatedPNG {
    static func imageAtURL(_ url: URL) -> UIImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let frameCount = CGImageSourceGetCount(source)
        guard frameCount > 1 else {
            // Static fallback.
            guard let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
            return UIImage(cgImage: cg)
        }

        var frames: [UIImage] = []
        frames.reserveCapacity(frameCount)
        var duration = 0.0
        for i in 0..<frameCount {
            guard let cg = CGImageSourceCreateImageAtIndex(source, i, nil) else { continue }
            frames.append(UIImage(cgImage: cg))
            if let frameProps = CGImageSourceCopyPropertiesAtIndex(source, i, nil) as? [CFString: Any],
               let png = frameProps[kCGImagePropertyPNGDictionary] as? [CFString: Any],
               let delay = png[kCGImagePropertyAPNGDelayTime] as? Double {
                duration += delay
            } else {
                duration += 0.08
            }
        }
        guard !frames.isEmpty else { return nil }
        if frames.count == 1 { return frames[0] }
        return UIImage.animatedImage(with: frames, duration: max(duration, 0.4))
    }
}

/// Loader variety: every fix, Mily is up to something different while she
/// works. Only seamless loops (one-shots like running/chasing leave the frame empty).
enum LoaderAnimation {
    static let names = "treadmill baking noodles tv reading piano gaming dance campfire plane eating snowman rain firefighter basketball volleyball weightlifting stretching phone graduating suit flexing grooming discovery zapped floating".split(separator: " ").map { "\($0)_160" }
    static func random() -> String { names.randomElement() ?? "treadmill_160" }
}

/// Plays any bundled mascot animation by file name.
struct MascotClip: View {
    let fileName: String
    var size: CGFloat = 160
    var body: some View {
        if let image = MascotState.image(named: fileName) {
            Image(uiImage: image).resizable().interpolation(.none).frame(width: size, height: size)
        }
    }
}

/// The mascot, sized and looping.
struct MascotView: View {
    let state: MascotState
    var size: CGFloat = 160

    var body: some View {
        Group {
            if let image = Self.cached(state) {
                Image(uiImage: image)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: size, height: size)
            } else {
                Image(systemName: "face.smiling")
                    .font(.system(size: size * 0.6))
                    .frame(width: size, height: size)
            }
        }
        .accessibilityLabel("Mily mascot, \(state.rawValue)")
    }

    private static var cache: [MascotState: UIImage] = [:]
    private static let lock = NSLock()
    private static func cached(_ state: MascotState) -> UIImage? {
        lock.lock()
        defer { lock.unlock() }
        if let known = cache[state] { return known }
        let decoded = MascotState.image(named: state.fileName)
        cache[state] = decoded
        return decoded
    }
}
