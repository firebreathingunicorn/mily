import SwiftUI
import MilyCore

/// Drives one Mily fix end to end: burst in → `VisionFaceAnalyzer` →
/// `BestTakePipeline` → before/after images + per-person report.
///
/// Heavy work runs detached from the MainActor; `MilyCore` types are all
/// `Sendable` value types. The demo burst uses the same `SyntheticScene`
/// two-person setup as `mily demo-burst`, so the app always has a working
/// path — including for the submission video.
@MainActor
final class FixSession: ObservableObject {
    enum Stage: Equatable {
        case idle
        case analyzing
        case synthesizing
        case done
        case failed(String)

        var isBusy: Bool {
            self == .analyzing || self == .synthesizing
        }
    }

    struct FixResult {
        let after: UIImage
        let before: UIImage
        let people: [PersonReport]
        let isDemo: Bool
        /// Pixel size of the working frames (face boxes are in this space).
        let imageSize: CGSize
        /// person → face box in the base frame (tap targets).
        let faceBoxes: [PersonID: CGRect]
        /// person → frames offered in the face picker, best first.
        let candidates: [PersonID: [Int]]
        /// person → frame currently used for them.
        let chosen: [PersonID: Int]
    }

    @Published var stage: Stage = .idle
    @Published var result: FixResult?
    /// True while a face choice is being re-composited (fast; no Vision pass).
    @Published var recompositing = false

    private var workTask: Task<Void, Never>?
    /// Analyzed frames of the current burst — kept so tapping a face and
    /// picking another moment only re-runs synthesis (~tens of ms).
    private var frames: [AnnotatedFrame] = []
    private var overrides: [PersonID: Int] = [:]
    private var isDemo = false
    private var cropCache: [String: UIImage] = [:]

    enum BurstInput {
        case annotated([AnnotatedFrame])
        case raw([CaptureFrame])
    }

    func runDemo() {
        guard !stage.isBusy else { return }
        stage = .analyzing
        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            let frames = Self.makeDemoBurst()
            await self?.analyze(input: .annotated(frames), isDemo: true)
        }
    }

    func run(images: [UIImage]) {
        guard !stage.isBusy, !images.isEmpty else { return }
        stage = .analyzing
        let payload = images.compactMap { $0.cgImage }
        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.analyze(input: .raw(Self.captureFrames(payload)), isDemo: false)
        }
    }

    /// A short video is the best input: dozens of moments per person instead
    /// of a handful of stills — the thing the Camera app can't give you.
    func run(videoURL: URL) {
        guard !stage.isBusy else { return }
        stage = .analyzing
        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let cgs = try await VideoFrames.extract(from: videoURL, count: FixConfig.videoFrameCount,
                                                        maxLongEdge: FixConfig.workingLongEdge)
                await self?.analyze(input: .raw(Self.captureFrames(cgs)), isDemo: false)
            } catch {
                await MainActor.run {
                    self?.stage = .failed("Couldn't read that video: \(error.localizedDescription)")
                }
            }
        }
    }

    func cancel() {
        workTask?.cancel()
        workTask = nil
        result = nil
        stage = .idle
    }

    /// The user picked `frame` for `person` in the face picker.
    func choose(_ frame: Int, for person: PersonID) {
        guard !recompositing, result != nil else { return }
        overrides[person] = frame
        recompositing = true
        let frames = self.frames, overrides = self.overrides, isDemo = self.isDemo
        Task.detached(priority: .userInitiated) { [weak self] in
            let built = Self.composite(frames: frames, overrides: overrides, isDemo: isDemo)
            await MainActor.run {
                self?.result = built
                self?.recompositing = false
            }
        }
    }

    /// Face crop of `person` in `frame`, for the picker strip (cached).
    func faceCrop(_ person: PersonID, frame: Int) -> UIImage? {
        let key = "\(person)#\(frame)"
        if let hit = cropCache[key] { return hit }
        guard frames.indices.contains(frame), let face = frames[frame].faces[person] else { return nil }
        let img = frames[frame].image
        let b = face.bbox
        let pad = b.width / 3
        let x = max(0, b.x - pad), y = max(0, b.y - pad)
        let r = RectI(x: x, y: y,
                      width: min(img.width - x, b.width + 2 * pad),
                      height: min(img.height - y, b.height + 2 * pad))
        guard r.width > 4, r.height > 4 else { return nil }
        let ui = UIImage(cgImage: ImageIO.toCGImage(img.crop(r)))
        cropCache[key] = ui
        return ui
    }

    // MARK: - Pipeline

    private nonisolated static func captureFrames(_ cgs: [CGImage]) -> [CaptureFrame] {
        cgs.enumerated().map { index, cg in
            CaptureFrame(image: ImageIO.fromCGImage(ImageIO.resized(cg, maxLongEdge: FixConfig.workingLongEdge)),
                         metadata: FrameMetadata(timestamp: Double(index)))
        }
    }

    private nonisolated func analyze(input: BurstInput, isDemo: Bool) async {
        let busy = await MainActor.run { self.stage.isBusy }
        guard busy, !Task.isCancelled else { return }
        do {
            let annotated: [AnnotatedFrame]
            switch input {
            case .annotated(let frames):
                annotated = frames
            case .raw(let frames):
                annotated = try VisionFaceAnalyzer().annotate(frames: frames)
            }
            guard !Task.isCancelled else { return }
            await MainActor.run { if self.stage == .analyzing { self.stage = .synthesizing } }

            guard annotated.contains(where: { !$0.faces.isEmpty }) else {
                await MainActor.run {
                    self.stage = .failed("Mily couldn't find any faces — try a closer shot of the group.")
                }
                return
            }
            let built = Self.composite(frames: annotated, overrides: [:], isDemo: isDemo)
            await MainActor.run {
                self.frames = annotated
                self.overrides = [:]
                self.isDemo = isDemo
                self.cropCache = [:]
                self.result = built
                self.stage = .done
            }
        } catch {
            await MainActor.run {
                self.stage = .failed("Something went wrong while analyzing: \(error.localizedDescription)")
            }
        }
    }

    private nonisolated static func composite(frames: [AnnotatedFrame], overrides: [PersonID: Int],
                                              isDemo: Bool) -> FixResult {
        let (output, report) = BestTakePipeline().run(frames: frames, overrides: overrides)
        let base = frames[report.baseFrame]
        var boxes: [PersonID: CGRect] = [:]
        for (p, face) in base.faces {
            boxes[p] = CGRect(x: face.bbox.x, y: face.bbox.y, width: face.bbox.width, height: face.bbox.height)
        }
        var chosen: [PersonID: Int] = [:]
        for person in report.people { chosen[person.person] = person.swapped ? person.donorFrame : report.baseFrame }
        return FixResult(
            after: UIImage(cgImage: ImageIO.toCGImage(output)),
            before: UIImage(cgImage: ImageIO.toCGImage(base.image)),
            people: report.people,
            isDemo: isDemo,
            imageSize: CGSize(width: base.image.width, height: base.image.height),
            faceBoxes: boxes,
            candidates: report.candidates,
            chosen: chosen
        )
    }

    // MARK: - Demo burst (mirrors `mily demo-burst`)

    /// Two people, 6 frames: Ada smiles big at frame 2 but blinks at 3 and is
    /// mid-word at 5; Bo blinks at 1 and smiles at 4. Mily should combine the
    /// best of both into the base frame.
    private nonisolated static func makeDemoBurst() -> [AnnotatedFrame] {
        let scale: Float = 1.6 // render at 1024×768 from the 640×480 layout
        var framesConfig: [[SyntheticFace]] = []
        // 12 moments, like frames pulled from a short clip: smiles, blinks,
        // frowns and mid-word mouths spread across both people.
        let n = 12
        for f in 0..<n {
            let aSmile: Float = [0.03, 0.05, 0.11, 0.06, -0.05, 0.02, 0.09, -0.04, 0.03, 0.12, 0.04, 0.02][f]
            let aEyes: Float = [0.10, 0.10, 0.10, 0.02, 0.09, 0.10, 0.10, 0.10, 0.01, 0.10, 0.10, 0.08][f]
            let bSmile: Float = [0.02, 0.04, -0.05, 0.03, 0.10, 0.06, -0.03, 0.11, 0.02, 0.05, 0.08, 0.03][f]
            let bEyes: Float = [0.10, 0.02, 0.10, 0.10, 0.10, 0.09, 0.10, 0.10, 0.10, 0.01, 0.10, 0.10][f]
            framesConfig.append([
                SyntheticFace(
                    name: "personA",
                    headCenter: Point2(190 * scale, 170 * scale),
                    interOcular: 34 * scale,
                    eyeAperture: aEyes,
                    smileCurve: aSmile,
                    mouthAperture: (f == 5 || f == 10) ? 0.16 : 0.02
                ),
                SyntheticFace(
                    name: "personB",
                    headCenter: Point2(450 * scale, 200 * scale),
                    interOcular: 28 * scale,
                    eyeAperture: bEyes,
                    smileCurve: bSmile,
                    mouthAperture: 0.02,
                    skin: (0.78, 0.58, 0.44),
                    shirt: (0.5, 0.26, 0.22)
                ),
            ])
        }
        let offsets = (0..<n).map { (dx: ($0 * 3) % 7 - 3, dy: ($0 * 5) % 5 - 2) }
        return SyntheticScene.burst(
            width: Int(640 * scale),
            height: Int(480 * scale),
            frames: framesConfig,
            offsets: offsets,
            noiseSigma: 0.02,
            seed: 11
        )
    }
}

/// Friendly display names for demo person IDs; real-photo IDs pass through.
extension FixSession {
    static func displayName(for person: PersonID) -> String {
        switch person {
        case "personA": return "Ada"
        case "personB": return "Bo"
        default: return person
        }
    }
}

/// Pipeline resolution + export settings. Plain enum so both the MainActor
/// session and nonisolated code can read them.
enum FixConfig {
    /// Working resolution for the pipeline. Balances Vision quality and
    /// speed on device; Pro saves at this full quality, free saves downscaled.
    static let workingLongEdge = 1280
    static let freeSaveLongEdge = 1024
    /// Frames sampled from an imported video clip.
    static let videoFrameCount = 16
}

/// Saves the result into the user's photo library.
struct PhotoSaver {
    static func save(_ image: UIImage, isPro: Bool) -> UIImage {
        let output: UIImage
        if isPro {
            output = image
        } else {
            // Free tier: downscaled save.
            if let cg = image.cgImage {
                output = UIImage(cgImage: ImageIO.resized(cg, maxLongEdge: FixConfig.freeSaveLongEdge))
            } else {
                output = image
            }
        }
        UIImageWriteToSavedPhotosAlbum(output, nil, nil, nil)
        return output
    }
}
