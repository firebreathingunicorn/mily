import XCTest
@testable import MilyCore

/// End-to-end Level A pipeline over a synthetic two-person burst.
final class PipelineTests: XCTestCase {

    /// Burst design:
    ///   person A: peak smile at 2, blinks at 1 and 4, mid-word at 5
    ///   person B: peak smile at 4, blinks at 1, mid-word at 3
    ///   camera jitters a few pixels per frame; sensor noise σ = 0.02
    func makeBurst() -> [AnnotatedFrame] {
        func faceA(_ f: Int) -> SyntheticFace {
            SyntheticFace(
                name: "A", headCenter: Point2(180, 165), interOcular: 34,
                eyeAperture: (f == 1 || f == 4) ? 0.01 : 0.10,
                smileCurve: f == 2 ? 0.13 : 0.02,
                mouthAperture: f == 5 ? 0.16 : 0.02
            )
        }
        func faceB(_ f: Int) -> SyntheticFace {
            SyntheticFace(
                name: "B", headCenter: Point2(445, 205), interOcular: 27,
                eyeAperture: f == 1 ? 0.01 : 0.10,
                smileCurve: f == 4 ? 0.12 : 0.02,
                mouthAperture: f == 3 ? 0.16 : 0.02,
                noseDropRatio: 0.75, mouthDropRatio: 1.2,
                skin: (0.80, 0.60, 0.46), shirt: (0.52, 0.25, 0.20)
            )
        }
        let frames = (0..<6).map { f -> [SyntheticFace] in [faceA(f), faceB(f)] }
        let offsets = (0..<6).map { i -> (dx: Int, dy: Int) in
            (dx: [0, 3, -2, 4, -3, 2][i], dy: [0, -2, 3, -1, 2, -2][i])
        }
        return SyntheticScene.burst(frames: frames, offsets: offsets, noiseSigma: 0.02, seed: 1234)
    }

    func testEndToEndSwap() {
        let frames = makeBurst()
        let pipeline = BestTakePipeline()
        let (output, report) = pipeline.run(frames: frames)

        XCTAssertEqual(report.baseFrame, 2, "A's peak frame should win the group base")

        guard let b = report.people.first(where: { $0.person == "B" }) else {
            return XCTFail("missing report for person B")
        }
        XCTAssertTrue(b.swapped, "B should be swapped from frame 4; report: \(b)")
        XCTAssertEqual(b.donorFrame, 4)
        XCTAssertEqual(b.level, "A")
        XCTAssertTrue(b.checksPassed)
        XCTAssertNotNil(b.identityDistance)

        guard let a = report.people.first(where: { $0.person == "A" }) else {
            return XCTFail("missing report for person A")
        }
        XCTAssertFalse(a.swapped, "A's base frame is already their best moment")

        // Picker filmstrip data: per-person per-frame scores in the report.
        XCTAssertEqual(report.scores["B"]?.count, frames.count)
        XCTAssertEqual(report.scores["A"]?.count, frames.count)
        let bScores = report.scores["B"]!
        XCTAssertEqual(bScores.firstIndex(of: bScores.max()!), b.donorFrame, "B's filmstrip peak must be the chosen donor")

        // The output must actually differ from the base frame inside B's face
        // region, and the pasted result must reconstruct the transplant:
        // output = base ⊕ (content, alpha) up to finishing noise. Proximity
        // to the raw donor frame is looser because the fitted similarity can
        // carry a sub-pixel component, which bilinear resampling smooths.
        let geomB = frames[2].faces["B"]!
        let core = coreRect(geom: geomB, w: output.width, h: output.height)
        let synth = DirectTransplantSynthesizer()
        let transplant = synth.transplant(base: frames[2], donor: frames[4], person: "B")!
        let diffVsBase = meanAbsDiff(output, frames[2].image, in: core)
        XCTAssertGreaterThan(diffVsBase, 0.004, "output unchanged in B's face region")

        var expected = frames[2].image
        expected.pasteRegion(content: transplant.content, alpha: transplant.alpha, region: transplant.region)
        let diffVsExpected = meanAbsDiff(output, expected, in: transplant.region.inflated(byPixels: 4, toWidth: output.width, height: output.height))
        XCTAssertLessThan(diffVsExpected, 0.015, "output must equal base ⊕ transplant up to finishing noise (got \(diffVsExpected))")

        let diffVsDonorWeighted = alphaWeightedDiff(output, frames[4].image, alpha: transplant.alpha, region: transplant.region)
        XCTAssertLessThan(diffVsDonorWeighted, 0.05, "pasted content should track the donor frame (got \(diffVsDonorWeighted))")
    }

    func testNoiseMatchingBringsGrainToBaseLevel() {
        let frames = makeBurst()
        let synthesizer = DirectTransplantSynthesizer()
        guard var result = synthesizer.transplant(base: frames[2], donor: frames[4], person: "B") else {
            return XCTFail("transplant failed")
        }

        let geomB = frames[2].faces["B"]!
        let core = coreRect(geom: geomB, w: frames[2].image.width, h: frames[2].image.height)

        let finisher = Finisher()
        finisher.finish(base: frames[2], result: &result, rngSeed: 77)

        // Compare pasted core σ against the base frame's σ there.
        var composited = frames[2].image
        composited.pasteRegion(content: result.content, alpha: result.alpha, region: result.region)
        let sigmaAfter = NoiseEstimator.sigma(image: composited, region: core)
        let sigmaBase = NoiseEstimator.sigma(image: frames[2].image, region: core)
        XCTAssertEqual(sigmaAfter, sigmaBase, accuracy: sigmaBase * 0.45,
                       "grain after noise matching (\(sigmaAfter)) should track base (\(sigmaBase))")
    }

    func testArtifactCheckerRejectsBadTransplants() {
        let frames = makeBurst()
        let synthesizer = DirectTransplantSynthesizer()
        let checker = ClassicalArtifactChecker()

        // 1) Noise mismatch: donor content with 5× the grain and full hard alpha.
        guard let good = synthesizer.transplant(base: frames[2], donor: frames[4], person: "B") else {
            return XCTFail("transplant failed")
        }
        var noisy = good
        var rng = SeededRNG(seed: 3)
        for i in noisy.content.data.indices where i % 4 != 3 {
            noisy.content.data[i] += rng.nextGaussian() * 0.1
        }
        noisy.alpha = Mask(width: good.alpha.width, height: good.alpha.height, fill: 1)
        let reportNoisy = checker.check(base: frames[2], result: noisy, person: "B")
        XCTAssertFalse(reportNoisy.passed, "noise-mismatched transplant must be rejected: \(reportNoisy.metrics)")

        // 2) Color shift: donor content tinted far from the base scene.
        guard let good2 = synthesizer.transplant(base: frames[2], donor: frames[4], person: "B") else {
            return XCTFail("transplant failed")
        }
        var tinted = good2
        for i in stride(from: 0, to: tinted.content.data.count, by: 4) {
            tinted.content.data[i] = min(1, tinted.content.data[i] + 0.3)
        }
        tinted.alpha = Mask(width: good2.alpha.width, height: good2.alpha.height, fill: 1)
        let reportTinted = checker.check(base: frames[2], result: tinted, person: "B")
        XCTAssertFalse(reportTinted.passed, "color-shifted transplant must be rejected: \(reportTinted.metrics)")

        // The unmodified transplant passes.
        let reportGood = checker.check(base: frames[2], result: good, person: "B")
        XCTAssertTrue(reportGood.passed, "clean transplant must pass: \(reportGood.metrics)")
    }

    func testIdentityCheckRejectsWrongPerson() {
        let frames = makeBurst()
        let checker = IdentityChecker()
        // Same person in both frames → must pass.
        XCTAssertTrue(checker.check(base: frames[2], donor: frames[4], person: "B").passed)

        // Wrong-person case: frame 4's B face misassigned as A (a stand-in for
        // a tracking error). The identity gate must reject the swap.
        var misassigned = frames[4]
        misassigned.faces["A"] = frames[4].faces["B"]
        XCTAssertFalse(checker.check(base: frames[2], donor: misassigned, person: "A").passed)
    }

    func testColorMatchingCorrectsExposureOffset() {
        let frames = makeBurst()
        // Donor frame shot ~0.08 brighter (camera re-metered mid-burst).
        var donor = frames[4]
        donor.image = brightened(donor.image, by: 0.08)

        let synth = DirectTransplantSynthesizer()

        // With color matching: pasted core mean must sit at the base's level.
        var matched = synth.transplant(base: frames[2], donor: donor, person: "B")!
        Finisher().finish(base: frames[2], result: &matched, rngSeed: 5)
        let shiftMatched = supportMeanShift(frames[2], matched)
        XCTAssertLessThan(abs(shiftMatched), 0.02, "color match failed to correct exposure (shift \(shiftMatched))")

        // Without: the offset must survive (proves the test can see it).
        var offConfig = Finisher.Config()
        offConfig.colorMatching = false
        var unmatched = synth.transplant(base: frames[2], donor: donor, person: "B")!
        Finisher(config: offConfig).finish(base: frames[2], result: &unmatched, rngSeed: 5)
        let shiftUnmatched = supportMeanShift(frames[2], unmatched)
        XCTAssertGreaterThan(shiftUnmatched, 0.04, "expected visible offset without color match")
    }

    private func brightened(_ image: PixelImage, by amount: Float) -> PixelImage {
        var out = image
        for i in stride(from: 0, to: out.data.count, by: 4) {
            out.data[i] = min(1, out.data[i] + amount)
            out.data[i + 1] = min(1, out.data[i + 1] + amount)
            out.data[i + 2] = min(1, out.data[i + 2] + amount)
        }
        return out
    }

    /// Mean RGB distance between pasted content and the base's own pixels,
    /// over the alpha-solid support.
    private func supportMeanShift(_ base: AnnotatedFrame, _ result: TransplantResult) -> Float {
        var n: Float = 0
        var accB = (Float(0), Float(0), Float(0))
        var accD = (Float(0), Float(0), Float(0))
        for j in 0..<result.alpha.height {
            for i in 0..<result.alpha.width where result.alpha[i, j] > 0.5 {
                let b = base.image[result.region.x + i, result.region.y + j]
                let d = result.content[i, j]
                accB.0 += b.0; accB.1 += b.1; accB.2 += b.2
                accD.0 += d.0; accD.1 += d.1; accD.2 += d.2
                n += 1
            }
        }
        guard n > 0 else { return 0 }
        let mb = (accB.0 / n, accB.1 / n, accB.2 / n)
        let md = (accD.0 / n, accD.1 / n, accD.2 / n)
        return sqrt(pow(mb.0 - md.0, 2) + pow(mb.1 - md.1, 2) + pow(mb.2 - md.2, 2))
    }

    // MARK: Helpers

    func coreRect(geom: FaceGeometry, w: Int, h: Int) -> RectI {
        let io = geom.interOcular
        let c = geom.eyeMid
        let r = RectI(
            x: Int(c.x - io * 0.9), y: Int(c.y - io * 0.9),
            width: Int(io * 1.8), height: Int(io * 2.2)
        ).clamped(toWidth: w, height: h)
        return r
    }

    func meanAbsDiff(_ a: PixelImage, _ b: PixelImage, in region: RectI) -> Float {
        precondition(a.width == b.width && a.height == b.height)
        var acc: Float = 0
        var n: Float = 0
        for y in region.y..<region.maxY {
            for x in region.x..<region.maxX {
                let p = a[x, y], q = b[x, y]
                acc += abs(p.0 - q.0) + abs(p.1 - q.1) + abs(p.2 - q.2)
                n += 3
            }
        }
        return n > 0 ? acc / n : 0
    }

    /// Mean |a − b| weighted by the paste alpha, over the transplant region:
    /// measures "where we pasted, how close is the output to the donor".
    func alphaWeightedDiff(_ a: PixelImage, _ b: PixelImage, alpha: Mask, region: RectI) -> Float {
        var acc: Float = 0
        var w: Float = 0
        for j in 0..<alpha.height {
            for i in 0..<alpha.width {
                let at = alpha[i, j]
                let x = region.x + i, y = region.y + j
                guard x >= 0, x < a.width, y >= 0, y < a.height else { continue }
                let p = a[x, y], q = b[x, y]
                acc += at * (abs(p.0 - q.0) + abs(p.1 - q.1) + abs(p.2 - q.2))
                w += at * 3
            }
        }
        return w > 0 ? acc / w : 0
    }
}
