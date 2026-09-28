import Foundation
import CoreGraphics

/// Synthetic burst generator: renders stylized people (head, hair, eyes, brows,
/// nose, mouth) over a textured background with controllable expression,
/// pose, scale, position and sensor noise, and returns `AnnotatedFrame`s with
/// ground-truth geometry and soft person masks.
///
/// This is the test fixture for the whole Level A pipeline: selection,
/// transplant, finishing and verification can all be exercised without a
/// camera, Vision, or any trained model. It also powers `besttake demo-burst`
/// so the CLI can be tried end to end.
public struct SyntheticFace: Sendable {
    public var name: PersonID
    public var headCenter: Point2
    public var interOcular: Float
    /// Radians, positive = clockwise in image (y-down) space.
    public var roll: Float
    /// Vertical eye opening / interOcular. 0.02 closed, ~0.10 open.
    public var eyeAperture: Float
    /// Corner lift / mouth width. Positive = smile.
    public var smileCurve: Float
    /// Vertical mouth gap / interOcular. 0.02 closed, >0.15 open.
    public var mouthAperture: Float
    /// Mouth width / interOcular.
    public var mouthWidthRatio: Float
    /// Vertical distance from eye line to nose tip / interOcular.
    public var noseDropRatio: Float
    /// Vertical distance from eye line to mouth center / interOcular.
    public var mouthDropRatio: Float
    public var skin: (Float, Float, Float)
    public var hair: (Float, Float, Float)
    public var shirt: (Float, Float, Float)

    public init(
        name: PersonID,
        headCenter: Point2,
        interOcular: Float,
        roll: Float = 0,
        eyeAperture: Float = 0.10,
        smileCurve: Float = 0.0,
        mouthAperture: Float = 0.02,
        mouthWidthRatio: Float = 0.42,
        noseDropRatio: Float = 0.60,
        mouthDropRatio: Float = 1.05,
        skin: (Float, Float, Float) = (0.93, 0.78, 0.65),
        hair: (Float, Float, Float) = (0.15, 0.11, 0.09),
        shirt: (Float, Float, Float) = (0.30, 0.34, 0.46)
    ) {
        self.name = name
        self.headCenter = headCenter
        self.interOcular = interOcular
        self.roll = roll
        self.eyeAperture = eyeAperture
        self.smileCurve = smileCurve
        self.mouthAperture = mouthAperture
        self.mouthWidthRatio = mouthWidthRatio
        self.noseDropRatio = noseDropRatio
        self.mouthDropRatio = mouthDropRatio
        self.skin = skin
        self.hair = hair
        self.shirt = shirt
    }

    public var eyeMid: Point2 { headCenter }
    public var noseTip: Point2 { Point2(headCenter.x, headCenter.y + interOcular * noseDropRatio) }
    public var chin: Point2 { Point2(headCenter.x, headCenter.y + interOcular * 1.5) }
    public var mouthCenter: Point2 { Point2(headCenter.x, headCenter.y + interOcular * mouthDropRatio) }

    public var eyeLeft: Point2 { rotated(Point2(headCenter.x - interOcular * 0.5, headCenter.y)) }
    public var eyeRight: Point2 { rotated(Point2(headCenter.x + interOcular * 0.5, headCenter.y)) }
    public var mouthWidth: Float { mouthWidthRatio * interOcular }

    func rotated(_ p: Point2) -> Point2 {
        let c = cos(roll), s = sin(roll)
        let dx = p.x - headCenter.x, dy = p.y - headCenter.y
        return Point2(headCenter.x + c * dx - s * dy, headCenter.y + s * dx + c * dy)
    }
}

public enum SyntheticScene {

    /// Render a single annotated frame.
    /// - Parameters:
    ///   - globalOffset: integer scene translation for this frame (camera
    ///     jitter). Both people and background stripes move together, so
    ///     `Alignment.estimateTranslation` can recover it.
    public static func renderFrame(
        width: Int = 640,
        height: Int = 480,
        faces facesInput: [SyntheticFace],
        globalOffset: (dx: Int, dy: Int) = (0, 0),
        backgroundSeed: UInt64 = 7,
        noiseSigma: Float = 0.012,
        seed: UInt64 = 42,
        timestamp: Double = 0
    ) -> AnnotatedFrame {
        let base = background(width: width, height: height, offset: globalOffset, seed: backgroundSeed)
        // Camera jitter moves everything: shift people by the same offset.
        let faces = facesInput.map { f -> SyntheticFace in
            var f = f
            f.headCenter = Point2(
                f.headCenter.x + Float(globalOffset.dx),
                f.headCenter.y + Float(globalOffset.dy)
            )
            return f
        }
        let ss = 2
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: width * ss, height: height * ss,
            bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            fatalError("SyntheticScene: cannot create CGContext")
        }
        ctx.scaleBy(x: CGFloat(ss), y: CGFloat(ss))
        ctx.interpolationQuality = .high
        ctx.setShouldAntialias(true)
        ctx.draw(ImageIO.toCGImage(base), in: CGRect(x: 0, y: 0, width: width, height: height))

        var masks: [PersonID: Mask] = [:]
        for face in faces {
            drawPerson(ctx: ctx, face: face, canvasW: width, canvasH: height)
        }
        for face in faces {
            masks[face.name] = personMask(face: face, width: width, height: height, ss: ss, colorSpace: colorSpace)
        }

        guard let rendered = ctx.makeImage() else { fatalError("SyntheticScene: render failed") }
        // Downsample the 2× supersampled canvas back to the target size.
        var image = ImageIO.fromCGImage(rendered).downsampled(factor: ss)

        // Sensor noise on top of the quantized render.
        if noiseSigma > 0 {
            var rng = SeededRNG(seed: seed)
            for i in image.data.indices {
                image.data[i] = max(0, min(1, image.data[i] + rng.nextGaussian() * noiseSigma))
            }
        }

        var out = AnnotatedFrame(
            image: image,
            metadata: FrameMetadata(timestamp: timestamp),
            faces: [:],
            personMasks: masks
        )
        for face in faces {
            out.faces[face.name] = geometry(for: face, width: width, height: height, mask: masks[face.name]!)
        }
        return out
    }

    /// Render a burst with per-frame face configs, camera jitter and noise.
    public static func burst(
        width: Int = 640,
        height: Int = 480,
        frames: [[SyntheticFace]],
        offsets: [(dx: Int, dy: Int)]? = nil,
        noiseSigma: Float = 0.012,
        backgroundSeed: UInt64 = 7,
        seed: UInt64 = 42
    ) -> [AnnotatedFrame] {
        var rng = SeededRNG(seed: seed)
        let offs = offsets ?? frames.map { _ in (0, 0) }
        return frames.enumerated().map { i, faces in
            renderFrame(
                width: width, height: height,
                faces: faces,
                globalOffset: offs[i],
                backgroundSeed: backgroundSeed,
                noiseSigma: noiseSigma,
                seed: rng.nextUInt64(),
                timestamp: Double(i)
            )
        }
    }

    // MARK: - Geometry from parameters (ground truth, matches the renderer)

    static func geometry(for face: SyntheticFace, width: Int, height: Int, mask: Mask) -> FaceGeometry {
        let io = face.interOcular
        let hc = face.headCenter

        // Head contour: 16 points around the head ellipse, rolled.
        var contour: [Point2] = []
        let headCx = hc.x, headCy = hc.y + io * 0.15
        let headRx = io * 1.05, headRy = io * 1.35
        for k in 0..<16 {
            let a = Float(k) / 16 * 2 * .pi
            let px = headCx + cos(a) * headRx
            let py = headCy + sin(a) * headRy
            contour.append(face.rotated(Point2(px, py)))
        }

        // Eye ring points (4 per eye) for vertical spread measurement.
        func eyeRing(_ c: Point2) -> [Point2] {
            let rx = io * 0.17, ry = max(io * face.eyeAperture / 2, io * 0.005)
            return [
                Point2(c.x - rx, c.y), Point2(c.x + rx, c.y),
                Point2(c.x, c.y - ry), Point2(c.x, c.y + ry),
            ]
        }
        let leftRing = eyeRing(face.eyeLeft)
        let rightRing = eyeRing(face.eyeRight)

        let mw = face.mouthWidth
        let mc = face.mouthCenter
        let ap = max(face.mouthAperture * io, io * 0.015)
        let cornerL = Point2(mc.x - mw / 2, mc.y - face.smileCurve * mw)
        let cornerR = Point2(mc.x + mw / 2, mc.y - face.smileCurve * mw)
        let mouthTop = face.rotated(Point2(mc.x, mc.y - ap / 2))
        let mouthBottom = face.rotated(Point2(mc.x, mc.y + ap / 2))

        let bbox = RectI(
            x: Int(headCx - headRx * 1.1), y: Int(headCy - headRy * 1.1),
            width: Int(headRx * 2.2), height: Int(headRy * 2.2)
        ).clamped(toWidth: width, height: height)

        return FaceGeometry(
            bbox: bbox,
            landmarks: contour + leftRing + rightRing + [face.noseTip, cornerL, cornerR, mouthTop, mouthBottom],
            leftEye: face.eyeLeft,
            rightEye: face.eyeRight,
            noseTip: face.rotated(face.noseTip),
            mouthLeft: face.rotated(cornerL),
            mouthRight: face.rotated(cornerR),
            chin: face.rotated(face.chin),
            eyeAperture: face.eyeAperture,
            mouthAperture: face.mouthAperture,
            smileCurve: face.smileCurve,
            mouthWidthRatio: face.mouthWidthRatio,
            yawProxy: 0,
            rollProxy: face.roll,
            interOcular: io,
            faceCoverage: mask.mean(in: bbox)
        )
    }

    // MARK: - Background

    static func background(width: Int, height: Int, offset: (dx: Int, dy: Int), seed: UInt64) -> PixelImage {
        var img = PixelImage(width: width, height: height, fill: (0, 0, 0, 1))
        let top: (Float, Float, Float) = (0.38, 0.44, 0.55)
        let bottom: (Float, Float, Float) = (0.16, 0.19, 0.27)
        for y in 0..<height {
            let t = Float(y) / Float(max(1, height - 1))
            let rt: Float = top.0 * (1 - t) + bottom.0 * t
            let gt: Float = top.1 * (1 - t) + bottom.1 * t
            let bt: Float = top.2 * (1 - t) + bottom.2 * t
            let stripePhase = Float(y + offset.dy) * 0.02
            for x in 0..<width {
                // Vertical gradient + diagonal stripes (alignment signal).
                let stripe = 0.5 + 0.5 * sin(Float(x + offset.dx) * 0.11 + stripePhase)
                let s: Float = 0.06 * stripe
                let o = (y * width + x) * 4
                img.data[o] = rt + s
                img.data[o + 1] = gt + s
                img.data[o + 2] = bt + s
            }
        }
        return img
    }

    // MARK: - Person rendering (CoreGraphics, y flipped)

    static func drawPerson(ctx: CGContext, face: SyntheticFace, canvasW: Int, canvasH: Int) {
        let io = face.interOcular
        let hc = face.headCenter

        func col(_ c: (Float, Float, Float)) -> CGColor {
            CGColor(srgbRed: CGFloat(c.0), green: CGFloat(c.1), blue: CGFloat(c.2), alpha: 1)
        }
        func ellipseRect(cx: Float, cy: Float, rx: Float, ry: Float) -> CGRect {
            CGRect(x: CGFloat(cx - rx), y: CGFloat(canvasH) - CGFloat(cy + ry),
                   width: CGFloat(rx * 2), height: CGFloat(ry * 2))
        }
        func pt(_ p: Point2) -> CGPoint {
            CGPoint(x: CGFloat(p.x), y: CGFloat(canvasH) - CGFloat(p.y))
        }

        // Torso rotation: only the head group rotates with `roll`.
        ctx.setFillColor(col(face.skin))
        ctx.fill(CGRect(x: CGFloat(hc.x - io * 0.45), y: CGFloat(canvasH) - CGFloat(hc.y + io * 2.6),
                        width: CGFloat(io * 0.9), height: CGFloat(io * 1.4))) // neck
        ctx.setFillColor(col(face.shirt))
        ctx.fillEllipse(in: ellipseRect(cx: hc.x, cy: hc.y + io * 3.0, rx: io * 2.3, ry: io * 1.9))

        // Head group (rotated).
        ctx.saveGState()
        ctx.translateBy(x: CGFloat(hc.x), y: CGFloat(canvasH) - CGFloat(hc.y))
        ctx.rotate(by: CGFloat(-face.roll))
        ctx.translateBy(x: CGFloat(-hc.x), y: -(CGFloat(canvasH) - CGFloat(hc.y)))

        // Hair behind/above head.
        ctx.setFillColor(col(face.hair))
        ctx.fillEllipse(in: ellipseRect(cx: hc.x, cy: hc.y - io * 0.30, rx: io * 1.28, ry: io * 1.5))
        // Face.
        ctx.setFillColor(col(face.skin))
        ctx.fillEllipse(in: ellipseRect(cx: hc.x, cy: hc.y + io * 0.15, rx: io * 1.05, ry: io * 1.35))

        // Eyes: sclera + iris; lids via reduced ry. Drawn in analytic
        // coordinates inside the rotated head group.
        for dx in [-0.5 as Float, 0.5] {
            let c = Point2(hc.x + io * dx, hc.y)
            let rx = io * 0.17
            let ry = max(io * face.eyeAperture / 2, io * 0.005)
            ctx.setFillColor(CGColor(srgbRed: 0.98, green: 0.98, blue: 0.96, alpha: 1))
            ctx.fillEllipse(in: ellipseRect(cx: c.x, cy: c.y, rx: rx, ry: ry))
            let iris = min(io * 0.06, ry * 0.9)
            ctx.setFillColor(CGColor(srgbRed: 0.18, green: 0.12, blue: 0.08, alpha: 1))
            ctx.fillEllipse(in: ellipseRect(cx: c.x, cy: c.y, rx: iris, ry: iris))
            // Brow.
            ctx.setFillColor(col(face.hair))
            ctx.fill(CGRect(x: CGFloat(c.x - io * 0.22), y: CGFloat(canvasH) - CGFloat(c.y - io * 0.16),
                            width: CGFloat(io * 0.44), height: CGFloat(io * 0.07)))
        }

        // Nose (small line).
        ctx.setStrokeColor(col((face.skin.0 * 0.82, face.skin.1 * 0.82, face.skin.2 * 0.82)))
        ctx.setLineWidth(CGFloat(io * 0.05))
        let nose = pt(face.noseTip)
        ctx.move(to: CGPoint(x: nose.x, y: nose.y + CGFloat(io * 0.18)))
        ctx.addLine(to: nose)
        ctx.strokePath()

        // Mouth: lens shape via two quadratic curves.
        let mw = face.mouthWidth
        let mc = face.mouthCenter
        let ap = max(face.mouthAperture * io, io * 0.015)
        let cl = Point2(mc.x - mw / 2, mc.y - face.smileCurve * mw)
        let cr = Point2(mc.x + mw / 2, mc.y - face.smileCurve * mw)
        let path = CGMutablePath()
        path.move(to: pt(cl))
        path.addQuadCurve(to: pt(cr), control: pt(Point2(mc.x, mc.y - ap)))
        path.addQuadCurve(to: pt(cl), control: pt(Point2(mc.x, mc.y + ap)))
        ctx.setFillColor(CGColor(srgbRed: 0.45, green: 0.18, blue: 0.17, alpha: 1))
        ctx.addPath(path)
        ctx.fillPath()

        ctx.restoreGState()
    }

    /// Render one person's silhouette (union of body shapes) as a soft mask.
    static func personMask(face: SyntheticFace, width: Int, height: Int, ss: Int, colorSpace: CGColorSpace) -> Mask {
        guard let ctx = CGContext(
            data: nil, width: width * ss, height: height * ss,
            bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return Mask(width: width, height: height)
        }
        ctx.scaleBy(x: CGFloat(ss), y: CGFloat(ss))
        ctx.setShouldAntialias(true)
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))

        let io = face.interOcular
        let hc = face.headCenter

        func ellipseRect(cx: Float, cy: Float, rx: Float, ry: Float) -> CGRect {
            CGRect(x: CGFloat(cx - rx), y: CGFloat(height) - CGFloat(cy + ry),
                   width: CGFloat(rx * 2), height: CGFloat(ry * 2))
        }

        ctx.fill(CGRect(x: CGFloat(hc.x - io * 0.45), y: CGFloat(height) - CGFloat(hc.y + io * 2.6),
                        width: CGFloat(io * 0.9), height: CGFloat(io * 1.4)))
        ctx.fillEllipse(in: ellipseRect(cx: hc.x, cy: hc.y + io * 3.0, rx: io * 2.3, ry: io * 1.9))

        ctx.saveGState()
        ctx.translateBy(x: CGFloat(hc.x), y: CGFloat(height) - CGFloat(hc.y))
        ctx.rotate(by: CGFloat(-face.roll))
        ctx.translateBy(x: CGFloat(-hc.x), y: -(CGFloat(height) - CGFloat(hc.y)))
        ctx.fillEllipse(in: ellipseRect(cx: hc.x, cy: hc.y - io * 0.30, rx: io * 1.28, ry: io * 1.5))
        ctx.fillEllipse(in: ellipseRect(cx: hc.x, cy: hc.y + io * 0.15, rx: io * 1.05, ry: io * 1.35))
        ctx.restoreGState()

        guard let cg = ctx.makeImage() else { return Mask(width: width, height: height) }
        // Downsample the 2× render before reading, so the mask covers the
        // full canvas.
        let img = ImageIO.fromCGImage(cg).downsampled(factor: ss)
        var mask = Mask(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                mask[x, y] = img[x, y].0
            }
        }
        // Soft edges, mimicking matting alpha falloff at the hairline.
        return mask.blurred(radius: 1).clamped01()
    }
}
