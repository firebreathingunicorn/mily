import Foundation

// MARK: - Face geometry

/// Everything scene understanding knows about one person in one frame.
/// Pixel coordinates are in the frame's pixel space, y-down.
public struct FaceGeometry: Sendable {
    /// Face bounding box (from detection).
    public var bbox: RectI

    /// Canonical landmark points for transform fitting: eye corners, brows,
    /// nose, mouth ring, face contour. More points = more stable similarity fit.
    public var landmarks: [Point2]

    // Key points (derived from `landmarks` by analyzers).
    public var leftEye: Point2
    public var rightEye: Point2
    public var noseTip: Point2
    public var mouthLeft: Point2
    public var mouthRight: Point2
    public var chin: Point2

    // Expression scalars, normalized by inter-ocular distance so they are
    // scale- and translation-invariant. Computed by the analyzer; the
    // synthetic test scene sets ground truth directly.
    /// Vertical eye opening / inter-ocular. ~0.05–0.12 for open eyes.
    public var eyeAperture: Float
    /// Vertical mouth gap / inter-ocular. ~0.02 closed, >0.15 mid-word.
    public var mouthAperture: Float
    /// (mouth mid y − mean corner y) / mouth width. Positive = smile (y-down).
    public var smileCurve: Float
    /// Mouth width / inter-ocular. Wider = stronger smile.
    public var mouthWidthRatio: Float

    // Pose proxies. Level A only needs coarse pose risk; Level B replaces
    // these with real 3D pose from depth + fitted 3D face model.
    /// ≈ (noseTip.x − eyeMid.x) / inter-ocular. 0 frontal, grows with yaw.
    public var yawProxy: Float
    /// Eye-line angle in radians.
    public var rollProxy: Float

    /// Inter-ocular distance in pixels (scale proxy).
    public var interOcular: Float

    /// Mean person-mask coverage over the face bbox (1 = unoccluded).
    public var faceCoverage: Float

    public init(
        bbox: RectI,
        landmarks: [Point2],
        leftEye: Point2,
        rightEye: Point2,
        noseTip: Point2,
        mouthLeft: Point2,
        mouthRight: Point2,
        chin: Point2,
        eyeAperture: Float,
        mouthAperture: Float,
        smileCurve: Float,
        mouthWidthRatio: Float,
        yawProxy: Float,
        rollProxy: Float,
        interOcular: Float,
        faceCoverage: Float = 1
    ) {
        self.bbox = bbox
        self.landmarks = landmarks
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.noseTip = noseTip
        self.mouthLeft = mouthLeft
        self.mouthRight = mouthRight
        self.chin = chin
        self.eyeAperture = eyeAperture
        self.mouthAperture = mouthAperture
        self.smileCurve = smileCurve
        self.mouthWidthRatio = mouthWidthRatio
        self.yawProxy = yawProxy
        self.rollProxy = rollProxy
        self.interOcular = interOcular
        self.faceCoverage = faceCoverage
    }

    public var eyeMid: Point2 {
        Point2((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2)
    }

    public var mouthMid: Point2 {
        Point2((mouthLeft.x + mouthRight.x) / 2, (mouthLeft.y + mouthRight.y) / 2)
    }
}

// MARK: - Geometric identity descriptor

/// Normalized inter-landmark distance vector. Level A uses it as the identity
/// gate: it must agree between the donor frame and the base frame or the swap
/// is rejected (wrong-person guard).
///
/// The plan calls for face embeddings (`VNGenerateFaceEmbeddingsRequest`,
/// iOS 17+/macOS 14+) plus body re-identification; that swap is a one-file
/// change here — everything downstream consumes a `[Float]` descriptor and a
/// cosine distance.
public enum GeometricIdentity {

    public static func descriptor(for face: FaceGeometry) -> [Float] {
        let s = max(face.interOcular, 1e-6)
        func d(_ a: Point2, _ b: Point2) -> Float { a.distance(to: b) / s }
        return [
            d(face.noseTip, face.leftEye),
            d(face.noseTip, face.rightEye),
            d(face.mouthLeft, face.leftEye),
            d(face.mouthRight, face.rightEye),
            d(face.mouthLeft, face.mouthRight),
            d(face.chin, face.noseTip),
            d(face.chin, face.mouthLeft),
            d(face.chin, face.mouthRight),
            d(face.chin, face.leftEye),
            d(face.chin, face.rightEye),
            d(face.mouthMid, face.eyeMid),
        ]
    }

    public static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 1 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        guard na > 0, nb > 0 else { return 1 }
        let cosv = dot / (sqrt(na) * sqrt(nb))
        return 1 - max(-1, min(1, cosv))
    }

    /// Relative Euclidean distance — the identity metric actually used by the
    /// gate. Cosine is too forgiving for all-positive distance vectors (a
    /// different face scoring within a few percent on every component still
    /// has near-identical direction); relative Euclidean responds to the
    /// per-component proportion changes that distinguish people.
    public static func relativeDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 1 }
        var diff: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count {
            let d = a[i] - b[i]
            diff += d * d
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let scale = 0.5 * (sqrt(na) + sqrt(nb))
        guard scale > 1e-9 else { return 1 }
        return sqrt(diff) / scale
    }
}

// MARK: - Annotated frame

/// A capture frame plus per-person understanding results. The pipeline core
/// operates on these; analyzers (Vision, synthetic test scenes) produce them,
/// which keeps the core testable without a camera or a neural net.
public struct AnnotatedFrame: Sendable {
    public var image: PixelImage
    public var metadata: FrameMetadata
    public var depth: DepthMap?
    /// Per-person face geometry (people missing in this frame are absent).
    public var faces: [PersonID: FaceGeometry]
    /// Per-person soft alpha masks (same dims as `image`).
    public var personMasks: [PersonID: Mask]

    public init(
        image: PixelImage,
        metadata: FrameMetadata = FrameMetadata(timestamp: 0),
        depth: DepthMap? = nil,
        faces: [PersonID: FaceGeometry] = [:],
        personMasks: [PersonID: Mask] = [:]
    ) {
        self.image = image
        self.metadata = metadata
        self.depth = depth
        self.faces = faces
        self.personMasks = personMasks
    }

    public func personIDs() -> [PersonID] {
        Array(faces.keys).sorted()
    }
}
