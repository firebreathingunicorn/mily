import XCTest
@testable import MilyCore

/// Accuracy upgrades: subpixel alignment, burst-relative pose refinement,
/// pose-adaptive identity gating, robust noise estimation.
final class AccuracyTests: XCTestCase {

    // MARK: Subpixel alignment

    func testAlignmentRecoversSubpixelShift() throws {
        let ref = SyntheticScene.renderFrame(faces: [
            SyntheticFace(name: "p0", headCenter: Point2(200, 180), interOcular: 32, smileCurve: 0.05),
        ], noiseSigma: 0.01, seed: 5)
        // mov(x) = ref(x − d): content displaced by d = (2.5, −1.5).
        let d = (dx: Float(2.5), dy: Float(-1.5))
        let mov = Warp.resample(src: ref.image, dstWidth: ref.image.width, dstHeight: ref.image.height) { x, y in
            (x - d.dx, y - d.dy)
        }
        let est = Alignment.estimateTranslation(reference: ref.image, moving: mov)
        XCTAssertEqual(est.dx, d.dx, accuracy: 0.4, "dx \(est.dx)")
        XCTAssertEqual(est.dy, d.dy, accuracy: 0.4, "dy \(est.dy)")
    }

    func testSubpixelOffsetParabola() {
        // Perfect parabola c(t) = (t−0.3)²: minimum at +0.3.
        let c: (Float) -> Float = { ($0 - 0.3) * ($0 - 0.3) }
        let off = Alignment.subpixelOffset(cMinus: c(-1), cCenter: c(0), cPlus: c(1))
        XCTAssertEqual(off, 0.3, accuracy: 1e-4)
        // Degenerate (flat) → 0.
        XCTAssertEqual(Alignment.subpixelOffset(cMinus: 5, cCenter: 5, cPlus: 5), 0)
    }

    // MARK: Burst-relative pose refinement

    private func geometry(io: Float, yawProxy: Float) -> FaceGeometry {
        // Minimal geometry; only the pose fields matter to the refiner.
        let c = Point2(200, 180)
        return FaceGeometry(
            bbox: RectI(x: 180, y: 160, width: 40, height: 40),
            landmarks: [],
            leftEye: Point2(c.x - io / 2, c.y), rightEye: Point2(c.x + io / 2, c.y),
            noseTip: Point2(c.x + yawProxy * io, c.y + io * 0.6),
            mouthLeft: Point2(c.x - io * 0.2, c.y + io), mouthRight: Point2(c.x + io * 0.2, c.y + io),
            chin: Point2(c.x, c.y + io * 1.5),
            eyeAperture: 0.1, mouthAperture: 0.02, smileCurve: 0.02, mouthWidthRatio: 0.42,
            yawProxy: yawProxy, rollProxy: 0, interOcular: io, faceCoverage: 1
        )
    }

    func testPoseRefinerUsesForeshortening() {
        var f0 = AnnotatedFrame(image: PixelImage(width: 8, height: 8))
        var f1 = AnnotatedFrame(image: PixelImage(width: 8, height: 8))
        var f2 = AnnotatedFrame(image: PixelImage(width: 8, height: 8))
        var f3 = AnnotatedFrame(image: PixelImage(width: 8, height: 8))
        f0.faces["P"] = geometry(io: 30, yawProxy: 0.10)   // frontal reference
        f1.faces["P"] = geometry(io: 28, yawProxy: 0.35)   // cos⁻¹(28/30) ≈ 0.333
        f2.faces["P"] = geometry(io: 30, yawProxy: -0.05)  // frontal
        f3.faces["P"] = geometry(io: 15, yawProxy: 0.20)   // extreme: ratio 0.5 < minRatio

        var frames = [f0, f1, f2, f3]
        PoseRefiner.refine(frames: &frames)

        XCTAssertEqual(frames[0].faces["P"]!.yawProxy, 0, accuracy: 0.05, "frontal reference → yaw 0")
        XCTAssertEqual(frames[1].faces["P"]!.yawProxy, acos(28.0 / 30.0), accuracy: 0.02,
                       "yaw from foreshortening")
        XCTAssertLessThan(abs(frames[2].faces["P"]!.yawProxy), 0.05)
        // Extreme ratio: keeps the proxy magnitude (sign-corrected), not acos garbage.
        XCTAssertEqual(frames[3].faces["P"]!.yawProxy, 0.20, accuracy: 0.02)
    }

    // MARK: Pose-adaptive identity threshold

    func testIdentityLimitAdaptsToPoseDelta() {
        let checker = IdentityChecker()
        let frontal = geometry(io: 30, yawProxy: 0.0)
        let turned = geometry(io: 30, yawProxy: 0.4)
        XCTAssertEqual(checker.limit(base: frontal, donor: frontal), 0.08, accuracy: 1e-6)
        let adaptive = checker.limit(base: frontal, donor: turned)
        XCTAssertEqual(adaptive, min(0.14, 0.08 + 0.15 * 0.4), accuracy: 1e-6)
        // Capped.
        let far = geometry(io: 30, yawProxy: 1.0)
        XCTAssertEqual(checker.limit(base: frontal, donor: far), 0.14, accuracy: 1e-6)
    }

    // MARK: Robust noise estimation

    func testNoiseSigmaRobustToTextureEdges() {
        // Two textured stripes crossing a noisy flat field: the mean-based
        // estimator reads the edges as noise; the median-based one ignores them.
        let w = 300, h = 300
        var img = PixelImage(width: w, height: h, fill: (0.5, 0.5, 0.5, 1))
        for y in 0..<h where abs(y - h / 2) < 6 {
            for x in 0..<w { img[x, y] = (0.9, 0.1, 0.2, 1) } // high-contrast stripe
        }
        var rng = SeededRNG(seed: 42)
        let trueSigma: Float = 0.02
        for i in stride(from: 0, to: img.data.count, by: 4) {
            let n = rng.nextGaussian() * trueSigma
            img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n
        }
        let est = NoiseEstimator.sigma(image: img)
        XCTAssertEqual(est, trueSigma, accuracy: trueSigma * 0.25,
                       "median estimator must ignore the stripe (got \(est))")
    }
}
