import Foundation
import Vision
import CoreVideo

/// Person matting via Vision's person segmentation: soft alpha covering hair
/// and body edges. Requested at `.accurate`; the returned alpha is resampled
/// to the frame's working resolution.
public enum PersonMatting {

    public static func segment(cgImage: CGImage, targetWidth: Int, targetHeight: Int) throws -> Mask {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])
        guard let obs = request.results?.first as? VNPixelBufferObservation else {
            return Mask(width: targetWidth, height: targetHeight)
        }
        return mask(from: obs.pixelBuffer, targetWidth: targetWidth, targetHeight: targetHeight)
    }

    public static func mask(
        from pixelBuffer: CVPixelBuffer, targetWidth: Int, targetHeight: Int
    ) -> Mask {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let base = CVPixelBufferGetBaseAddress(pixelBuffer)!
        var raw = Mask(width: w, height: h)

        switch format {
        case kCVPixelFormatType_OneComponent8:
            let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let ptr = base.assumingMemoryBound(to: UInt8.self)
            for y in 0..<h {
                for x in 0..<w {
                    raw[x, y] = Float(ptr[y * stride + x]) / 255
                }
            }
        case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DisparityFloat32:
            let stride = CVPixelBufferGetBytesPerRow(pixelBuffer) / MemoryLayout<Float>.size
            let ptr = base.assumingMemoryBound(to: Float.self)
            for y in 0..<h {
                for x in 0..<w {
                    raw[x, y] = ptr[y * stride + x]
                }
            }
        default:
            break // unknown format: empty mask
        }

        return raw.resampledTo(width: targetWidth, height: targetHeight)
    }
}

/// Face analysis + cross-frame identity tracking using the classic Vision API.
///
/// Frames are analyzed concurrently (each frame gets its own requests and
/// handler); the identity matcher then walks the frames in order. Person
/// segmentation runs on a 1024-px-downscaled input — matting alpha is
/// low-frequency, and this keeps the accurate-quality request cheap.
///
/// Identity within a burst: greedy nearest-neighbor matching on position (bbox
/// center distance in units of bbox size) plus the geometric descriptor. This
/// is the Level A tracker; the plan's body re-identification plugs in at the
/// same seam for cross-shot matching in Phase 2.
public final class VisionFaceAnalyzer {

    /// Long-edge cap for the person-segmentation input.
    public var segmentationLongEdge: Int = 1024
    /// Simultaneous Vision analyses.
    public var maxConcurrentFrames: Int = 4

    public init() {}

    /// Annotate a burst in order. Identities are assigned "person0", "person1",
    /// … in order of first appearance, then matched across frames.
    public func annotate(frames: [CaptureFrame]) throws -> [AnnotatedFrame] {
        // 1) Per-frame analysis, concurrent.
        var raw = [RawAnalysis?](repeating: nil, count: frames.count)
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = maxConcurrentFrames
        let errorBox = ErrorBox()
        let group = DispatchGroup()
        for (index, frame) in frames.enumerated() {
            group.enter()
            queue.addOperation {
                defer { group.leave() }
                do {
                    raw[index] = try self.analyzeFrame(frame)
                } catch {
                    errorBox.set(error)
                }
            }
        }
        group.wait()
        if let error = errorBox.error { throw error }

        // 2) Sequential identity matching in burst order.
        var annotated: [AnnotatedFrame] = []
        var reference: [PersonID: FaceGeometry] = [:]
        for (index, analysis) in raw.enumerated() {
            guard let analysis else { continue }
            var faces: [PersonID: FaceGeometry] = [:]
            var assigned: [PersonID] = []
            for geom in analysis.geometries {
                let id = matchOrAssign(geom: geom, reference: reference, assigned: assigned)
                assigned.append(id)
                faces[id] = geom
                if reference[id] == nil { reference[id] = geom }
            }
            var personMasks: [PersonID: Mask] = [:]
            for (id, geom) in faces {
                let influence = faceInfluenceMask(geom: geom, width: analysis.alpha.width, height: analysis.alpha.height)
                personMasks[id] = analysis.alpha.multiplied(influence)
            }
            annotated.append(AnnotatedFrame(
                image: analysis.image,
                metadata: frames[index].metadata,
                depth: frames[index].depth,
                faces: faces,
                personMasks: personMasks
            ))
        }

        // 3) Burst-relative pose refinement: the raw nose-offset proxy is
        //    noisy; foreshortening of the inter-ocular distance against the
        //    person's own burst maximum is much more accurate.
        PoseRefiner.refine(frames: &annotated)
        return annotated
    }

    private struct RawAnalysis {
        var image: PixelImage
        var alpha: Mask
        /// Geometries in detection order; identities assigned later.
        var geometries: [FaceGeometry]
    }

    private final class ErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var boxed: Error?
        func set(_ e: Error) {
            lock.lock(); if boxed == nil { boxed = e }; lock.unlock()
        }
        var error: Error? {
            lock.lock(); defer { lock.unlock() }
            return boxed
        }
    }

    private func analyzeFrame(_ frame: CaptureFrame) throws -> RawAnalysis {
        let cg = ImageIO.toCGImage(frame.image)
        let detections = try detectFaces(cg: cg)
        let segCG = ImageIO.resized(cg, maxLongEdge: segmentationLongEdge)
        let alpha = try PersonMatting.segment(
            cgImage: segCG,
            targetWidth: frame.image.width,
            targetHeight: frame.image.height
        )

        var geometries: [FaceGeometry] = []
        for obs in detections {
            if let geom = geometry(from: obs, imageWidth: frame.image.width,
                                   imageHeight: frame.image.height, personAlpha: alpha) {
                geometries.append(geom)
            }
        }
        return RawAnalysis(image: frame.image, alpha: alpha, geometries: geometries)
    }

    // MARK: - Detection

    private func detectFaces(cg: CGImage) throws -> [VNFaceObservation] {
        let detect = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        try handler.perform([detect])
        return detect.results ?? []
    }

    // MARK: - Geometry extraction

    private func geometry(
        from obs: VNFaceObservation,
        imageWidth: Int,
        imageHeight: Int,
        personAlpha: Mask
    ) -> FaceGeometry? {
        let bb = obs.boundingBox // normalized, origin bottom-left
        let bx = Float(bb.origin.x) * Float(imageWidth)
        let byNorm = Float(bb.origin.y)
        let bw = Float(bb.size.width) * Float(imageWidth)
        let bh = Float(bb.size.height) * Float(imageHeight)

        guard let lm = obs.landmarks else { return nil }

        // pointsInImage gives image-pixel coordinates, origin bottom-left (y-up).
        let imgSize = CGSize(width: imageWidth, height: imageHeight)
        func regionPoints(_ region: VNFaceLandmarkRegion2D?) -> [Point2] {
            guard let region else { return [] }
            return region.pointsInImage(imageSize: imgSize).map {
                Point2(Float($0.x), Float(imageHeight) - Float($0.y))
            }
        }

        let leftEyePts = regionPoints(lm.leftEye)
        let rightEyePts = regionPoints(lm.rightEye)
        let nosePts = regionPoints(lm.nose)
        let lipsPts = regionPoints(lm.outerLips)
        let contourPts = regionPoints(lm.faceContour)
        let medianPts = regionPoints(lm.medianLine)

        guard leftEyePts.count >= 4, rightEyePts.count >= 4, lipsPts.count >= 6 else { return nil }

        func centroid(_ pts: [Point2]) -> Point2 {
            let sx = pts.reduce(Float(0)) { $0 + $1.x }
            let sy = pts.reduce(Float(0)) { $0 + $1.y }
            return Point2(sx / Float(pts.count), sy / Float(pts.count))
        }

        let leftEye = centroid(leftEyePts)
        let rightEye = centroid(rightEyePts)
        let interOcular = max(leftEye.distance(to: rightEye), 1)

        // Mouth corners: left/right extremes of the outer lip ring.
        let mouthLeft = lipsPts.min { $0.x < $1.x } ?? lipsPts[0]
        let mouthRight = lipsPts.max { $0.x < $1.x } ?? lipsPts[0]
        let mouthTop = lipsPts.min { $0.y < $1.y } ?? lipsPts[0]
        let mouthBottom = lipsPts.max { $0.y < $1.y } ?? lipsPts[0]

        // Nose tip: lowest point of the median line, else nose region centroid.
        let noseTip = medianPts.max { $0.y < $1.y } ?? (nosePts.isEmpty ? mouthTop : centroid(nosePts))
        let chin = contourPts.max { $0.y < $1.y } ?? noseTip

        // Expression scalars.
        let eyeApertureL = verticalSpread(leftEyePts) / interOcular
        let eyeApertureR = verticalSpread(rightEyePts) / interOcular
        let eyeAperture = (eyeApertureL + eyeApertureR) / 2
        let mouthWidth = max(mouthLeft.distance(to: mouthRight), 1)
        let mouthMidY = (mouthTop.y + mouthBottom.y) / 2
        let cornerY = (mouthLeft.y + mouthRight.y) / 2
        let smileCurve = (mouthMidY - cornerY) / mouthWidth // y-down: corners above mid = smile
        let mouthAperture = abs(mouthBottom.y - mouthTop.y) / interOcular

        // Pose proxies.
        let eyeMid = Point2((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2)
        let yawProxy = (noseTip.x - eyeMid.x) / interOcular
        let rollProxy = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x)

        let bbox = RectI(
            x: Int(bx), y: Int((1 - byNorm - Float(bb.size.height)) * Float(imageHeight)),
            width: Int(bw), height: Int(bh)
        ).clamped(toWidth: imageWidth, height: imageHeight)

        // Occlusion: person-mask coverage over the face box.
        let faceCoverage = personAlpha.mean(in: bbox)

        let landmarks = leftEyePts + rightEyePts + nosePts + lipsPts + contourPts + medianPts

        return FaceGeometry(
            bbox: bbox,
            landmarks: landmarks,
            leftEye: leftEye, rightEye: rightEye,
            noseTip: noseTip,
            mouthLeft: mouthLeft, mouthRight: mouthRight,
            chin: chin,
            eyeAperture: eyeAperture,
            mouthAperture: mouthAperture,
            smileCurve: smileCurve,
            mouthWidthRatio: mouthWidth / interOcular,
            yawProxy: yawProxy,
            rollProxy: rollProxy,
            interOcular: interOcular,
            faceCoverage: faceCoverage
        )
    }

    private func verticalSpread(_ pts: [Point2]) -> Float {
        guard let minY = pts.map(\.y).min(), let maxY = pts.map(\.y).max() else { return 0 }
        return max(0, maxY - minY)
    }

    /// Smooth influence region around one person's face/head: 1 near the face,
    /// falling off over ~1 bbox beyond it. Multiplied into the whole-scene
    /// person alpha to produce that person's mask.
    private func faceInfluenceMask(geom: FaceGeometry, width: Int, height: Int) -> Mask {
        let cx = geom.eyeMid.x
        let cy = (geom.eyeMid.y + geom.chin.y) / 2
        let rx = Float(max(geom.bbox.width, 24)) * 2.0
        let ry = Float(max(geom.bbox.height, 24)) * 2.6
        return Mask.ellipse(
            center: Point2(cx, cy),
            radiusX: rx, radiusY: ry,
            feather: max(24, rx * 0.35),
            width: width, height: height
        )
    }

    // MARK: - Cross-frame identity

    private func matchOrAssign(
        geom: FaceGeometry, reference: [PersonID: FaceGeometry], assigned: [PersonID]
    ) -> PersonID {
        var best: (id: PersonID, cost: Float)?
        for (id, ref) in reference where !assigned.contains(id) {
            let center = Point2(Float(geom.bbox.x) + Float(geom.bbox.width) / 2,
                                Float(geom.bbox.y) + Float(geom.bbox.height) / 2)
            let refCenter = Point2(Float(ref.bbox.x) + Float(ref.bbox.width) / 2,
                                   Float(ref.bbox.y) + Float(ref.bbox.height) / 2)
            let size = Float(max(geom.bbox.width, 1))
            let positionCost = center.distance(to: refCenter) / size
            let descriptorCost = GeometricIdentity.cosineDistance(
                GeometricIdentity.descriptor(for: geom),
                GeometricIdentity.descriptor(for: ref)
            ) * 4 // descriptor dominates once faces differ
            let cost = positionCost + descriptorCost
            if best == nil || cost < best!.cost { best = (id, cost) }
        }
        // Accept the match only if it is unambiguous; otherwise treat as new.
        if let b = best, b.cost < 1.5 {
            return b.id
        }
        return "person\(reference.count)"
    }
}
