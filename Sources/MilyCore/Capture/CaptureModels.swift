import Foundation

/// Depth buffer accompanying a frame. Level A does not need it, but the
/// capture contract records it so Level B (3D-aware re-projection) can consume
/// it without changing the frame model.
public struct DepthMap: Sendable {
    public var width: Int
    public var height: Int
    public var values: [Float]
    /// true = disparity (inverse depth), false = metric depth.
    public var isDisparity: Bool
    /// Metric reliability hint in [0, 1] (LiDAR confidence / dual-camera cost).
    public var confidence: Float

    public init(width: Int, height: Int, values: [Float], isDisparity: Bool, confidence: Float = 1) {
        self.width = width
        self.height = height
        self.values = values
        self.isDisparity = isDisparity
        self.confidence = confidence
    }
}

/// Per-frame capture metadata used by selection and finishing.
public struct FrameMetadata: Sendable {
    public var timestamp: Double
    public var exposureDuration: Double?
    public var iso: Float?
    /// Whether the frame came from the merged multi-frame capture (HDR+-style
    /// noise reduction) rather than a single raw frame.
    public var isMerged: Bool

    public init(timestamp: Double, exposureDuration: Double? = nil, iso: Float? = nil, isMerged: Bool = false) {
        self.timestamp = timestamp
        self.exposureDuration = exposureDuration
        self.iso = iso
        self.isMerged = isMerged
    }
}

/// One frame of a capture session: pixels plus everything scene understanding
/// and synthesis need downstream.
public struct CaptureFrame: Sendable {
    public var image: PixelImage
    public var metadata: FrameMetadata
    public var depth: DepthMap?

    public init(image: PixelImage, metadata: FrameMetadata, depth: DepthMap? = nil) {
        self.image = image
        self.metadata = metadata
        self.depth = depth
    }
}

/// Source of burst frames. The live camera (ring buffer) implements this on
/// device; `FileBurstSource` reads saved bursts for development, evaluation
/// and library suggestions.
public protocol BurstSource {
    func loadFrames() throws -> [CaptureFrame]
}

/// Loads a burst from a directory of image files (JPEG/PNG/HEIC), sorted by
/// filename. This is the development path for Phase 1: it lets the whole
/// pipeline run against real captures without the camera stack.
public struct FileBurstSource: BurstSource {
    public var directory: URL
    public var workingLongEdge: Int

    public init(directory: URL, workingLongEdge: Int = 2048) {
        self.directory = directory
        self.workingLongEdge = workingLongEdge
    }

    public func loadFrames() throws -> [CaptureFrame] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            throw IOError.cannotOpen(directory)
        }
        let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tiff"]
        let files = entries
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw IOError.cannotDecode(directory.appendingPathComponent("(no image files)"))
        }
        return try files.enumerated().map { index, url in
            let img = try ImageIO.load(path: url, maxLongEdge: workingLongEdge)
            return CaptureFrame(
                image: img,
                metadata: FrameMetadata(timestamp: Double(index))
            )
        }
    }
}
