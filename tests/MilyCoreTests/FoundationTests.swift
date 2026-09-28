import XCTest
@testable import MilyCore

final class FoundationTests: XCTestCase {

    // MARK: Similarity transform

    func testSimilarityFitTranslation() {
        let src = [Point2(10, 20), Point2(40, 20), Point2(25, 60), Point2(30, 35)]
        let dst = src.map { Point2($0.x + 12, $0.y - 7) }
        let t = SimilarityTransform.fit(src: src, dst: dst)
        XCTAssertNotNil(t)
        XCTAssertEqual(t!.scale, 1, accuracy: 1e-4)
        XCTAssertEqual(t!.tx, 12, accuracy: 1e-3)
        XCTAssertEqual(t!.ty, -7, accuracy: 1e-3)
    }

    func testSimilarityFitScaleAndRotation() {
        let src = [Point2(1, 0), Point2(0, 1), Point2(-1, 0), Point2(0, -1), Point2(1, 1)]
        // dst = 2× rotate 90° (y-down: (x,y) → (−y, x))
        let dst = src.map { Point2(-$0.y * 2, $0.x * 2) }
        let t = SimilarityTransform.fit(src: src, dst: dst)
        XCTAssertNotNil(t)
        XCTAssertEqual(t!.scale, 2, accuracy: 1e-4)
        for (s, d) in zip(src, dst) {
            let m = t!.apply(s)
            XCTAssertEqual(m.x, d.x, accuracy: 1e-3)
            XCTAssertEqual(m.y, d.y, accuracy: 1e-3)
        }
    }

    func testSimilarityInverseRoundtrip() throws {
        let t = SimilarityTransform(a: 1.3, b: 0.4, tx: -15, ty: 8)
        let inv = t.inverted
        let p = Point2(33, -12)
        let q = inv.apply(t.apply(p))
        XCTAssertEqual(q.x, p.x, accuracy: 1e-3)
        XCTAssertEqual(q.y, p.y, accuracy: 1e-3)
    }

    // MARK: Noise estimation

    func testNoiseSigmaFlatImage() {
        let w = 200, h = 200
        var img = PixelImage(width: w, height: h, fill: (0.5, 0.5, 0.5, 1))
        var rng = SeededRNG(seed: 99)
        let trueSigma: Float = 0.02
        // Luma (grayscale) noise: same sample on all three channels, which is
        // what the estimator sees after luma conversion.
        for i in stride(from: 0, to: img.data.count, by: 4) {
            let n = rng.nextGaussian() * trueSigma
            img.data[i] += n
            img.data[i + 1] += n
            img.data[i + 2] += n
        }
        let est = NoiseEstimator.sigma(image: img)
        XCTAssertEqual(est, trueSigma, accuracy: trueSigma * 0.2)
    }

    // MARK: Seeded RNG determinism

    func testSeededRNGDeterministic() {
        var a = SeededRNG(seed: 7), b = SeededRNG(seed: 7)
        for _ in 0..<100 {
            XCTAssertEqual(a.nextGaussian(), b.nextGaussian(), accuracy: 1e-9)
        }
    }

    // MARK: Warp

    func testWarpRegionTranslation() {
        // 20×20: value = x/19
        var src = PixelImage(width: 20, height: 20, fill: (0, 0, 0, 1))
        for y in 0..<20 {
            for x in 0..<20 {
                src[x, y] = (Float(x) / 19, Float(x) / 19, Float(x) / 19, 1)
            }
        }
        let region = RectI(x: 0, y: 0, width: 10, height: 10)
        // dst(x) samples src(x + 5): content shifts left by 5.
        let out = Warp.resampleRegion(src: src, region: region) { x, y in (x + 5, y) }
        let (r, _, _, _) = out[0, 0]
        XCTAssertEqual(r, 5.0 / 19, accuracy: 1e-6)
        let (r2, _, _, _) = out[9, 4]
        XCTAssertEqual(r2, 14.0 / 19, accuracy: 1e-6)
    }

    // MARK: Alignment

    func testAlignmentRecoversShift() {
        let f0 = SyntheticScene.renderFrame(faces: [
            SyntheticFace(name: "p0", headCenter: Point2(200, 180), interOcular: 32, smileCurve: 0.05),
        ], noiseSigma: 0.01, seed: 5)
        let f1 = SyntheticScene.renderFrame(faces: [
            SyntheticFace(name: "p0", headCenter: Point2(200, 180), interOcular: 32, smileCurve: 0.05),
        ], globalOffset: (6, -3), noiseSigma: 0.01, seed: 6)
        let shift = Alignment.estimateTranslation(reference: f0.image, moving: f1.image)
        XCTAssertEqual(shift.dx, 6, accuracy: 1)
        XCTAssertEqual(shift.dy, -3, accuracy: 1)
    }

    // MARK: Mask utilities

    func testEllipseMaskCoreAndFeather() {
        let m = Mask.ellipse(
            center: Point2(50, 50), radiusX: 30, radiusY: 30, feather: 10,
            width: 100, height: 100
        )
        XCTAssertEqual(m[50, 50], 1, accuracy: 1e-6)
        XCTAssertGreaterThan(m[50, 77], 0.2) // inside falloff
        XCTAssertLessThan(m[50, 92], 0.05) // far outside
        XCTAssertEqual(m.mean(in: RectI(x: 0, y: 0, width: 100, height: 100)) > 0, true)
    }

    func testMaskResampleAreaAverage() {
        var m = Mask(width: 4, height: 4)
        for j in 0..<2 { for i in 0..<4 { m[i, j] = 1 } }
        let r = m.resampledTo(width: 2, height: 2)
        XCTAssertEqual(r[0, 0], 1, accuracy: 1e-6) // top half full
        XCTAssertEqual(r[0, 1], 0, accuracy: 1e-6) // bottom half empty
    }
}
