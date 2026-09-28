import XCTest
@testable import MilyCore

final class SelectionTests: XCTestCase {

    /// Two people with distinct expression peaks: the planner must pick the
    /// frame where the group total is best as base, and per-person donors at
    /// their own best frames.
    func testPlannerPicksPeaks() {
        func frame(_ f: Int) -> AnnotatedFrame {
            let aSmile: Float = f == 2 ? 0.13 : 0.02
            let aEyes: Float = (f == 1 || f == 4) ? 0.01 : 0.10
            let bSmile: Float = f == 4 ? 0.12 : 0.02
            let bEyes: Float = f == 1 ? 0.01 : 0.10
            let bAperture: Float = f == 3 ? 0.16 : 0.02
            return SyntheticScene.renderFrame(faces: [
                SyntheticFace(
                    name: "A", headCenter: Point2(180, 170), interOcular: 34,
                    eyeAperture: aEyes, smileCurve: aSmile
                ),
                SyntheticFace(
                    name: "B", headCenter: Point2(440, 200), interOcular: 27,
                    eyeAperture: bEyes, smileCurve: bSmile, mouthAperture: bAperture,
                    noseDropRatio: 0.75, mouthDropRatio: 1.2
                ),
            ], noiseSigma: 0.015, seed: UInt64(100 + f))
        }
        let frames = (0..<6).map(frame)
        let planner = GroupPlanner()
        let plan = planner.plan(
            frames: frames, people: ["A", "B"],
            scorer: HeuristicExpressionScorer(), risk: HeuristicRiskModel()
        )

        // Base: frame 2 has A's peak + B neutral; frame 4 has B's peak but A
        // blinking — total should favor 2 (or at least place B's donor at 4).
        XCTAssertEqual(plan.baseFrame, 2)
        XCTAssertEqual(plan.donors["A"], 2) // already the best; no swap
        XCTAssertEqual(plan.donors["B"], 4) // swap from B's peak
    }

    /// A donor whose head is rolled far from the base must be rejected by the
    /// risk model even though its expression scores higher.
    func testRiskPenaltyRejectsRolledDonor() {
        func frame(_ f: Int) -> AnnotatedFrame {
            let smile: Float = f == 3 ? 0.13 : 0.02
            let roll: Float = f == 3 ? 0.9 : 0.0 // huge head tilt in the "best" frame
            return SyntheticScene.renderFrame(faces: [
                SyntheticFace(
                    name: "A", headCenter: Point2(200, 180), interOcular: 32,
                    roll: roll, eyeAperture: 0.10, smileCurve: smile
                ),
            ], noiseSigma: 0.015, seed: UInt64(200 + f))
        }
        let frames = (0..<5).map(frame)
        let plan = GroupPlanner().plan(
            frames: frames, people: ["A"],
            scorer: HeuristicExpressionScorer(), risk: HeuristicRiskModel()
        )
        // Base should be a neutral-pose frame, and frame 3 must NOT be the
        // donor despite the higher expression score.
        XCTAssertNotEqual(plan.baseFrame, 3)
        XCTAssertNotEqual(plan.donors["A"], 3)
    }

    /// The shutter fires only once EVERY seen person has had a good moment,
    /// and reports each person's best frame for the capture session.
    func testShutterAdvisorSignalsDoneWhenAllGood() {
        func frame(_ f: Int) -> AnnotatedFrame {
            let aSmile: Float = f == 2 ? 0.13 : 0.0
            let bSmile: Float = f == 4 ? 0.13 : 0.0
            return SyntheticScene.renderFrame(faces: [
                SyntheticFace(name: "A", headCenter: Point2(180, 170), interOcular: 34,
                              eyeAperture: 0.10, smileCurve: aSmile),
                SyntheticFace(name: "B", headCenter: Point2(440, 200), interOcular: 27,
                              eyeAperture: 0.10, smileCurve: bSmile),
            ], noiseSigma: 0.015, seed: UInt64(300 + f))
        }
        // Explicit bar between "neutral" (~0.83 on this scorer) and "strong
        // smile" (~0.93): the default is scorer-relative and configurable.
        var config = ShutterAdvisor.Config()
        config.goodMomentThreshold = 0.87
        var advisor = ShutterAdvisor(config: config)
        var signals: [ShutterAdvisor.Signal] = []
        for f in 0..<6 {
            signals.append(advisor.update(frame: frame(f), scorer: HeuristicExpressionScorer()))
        }
        // Nothing done before B's peak; done exactly at B's peak frame.
        for f in 0..<4 {
            XCTAssertEqual(signals[f], .keepGoing, "frame \(f)")
        }
        guard case .done(let best) = signals[4] else {
            return XCTFail("expected done at frame 4, got \(signals[4])")
        }
        XCTAssertEqual(best["A"], 2)
        XCTAssertEqual(best["B"], 4)
    }

    /// A person who never reaches the bar → the advisor gives up at maxFrames.
    func testShutterAdvisorGivesUp() {
        func frame(_ f: Int) -> AnnotatedFrame {
            SyntheticScene.renderFrame(faces: [
                // Perpetually blinking person: never a good moment.
                SyntheticFace(name: "A", headCenter: Point2(200, 180), interOcular: 32,
                              eyeAperture: 0.01, smileCurve: 0.0),
            ], noiseSigma: 0.015, seed: UInt64(400 + f))
        }
        var config = ShutterAdvisor.Config()
        config.maxFrames = 5
        config.goodMomentThreshold = 0.87
        var advisor = ShutterAdvisor(config: config)
        var last: ShutterAdvisor.Signal = .keepGoing
        for _ in 0..<5 {
            last = advisor.update(frame: frame(0), scorer: HeuristicExpressionScorer())
        }
        guard case .gaveUp(let best) = last else {
            return XCTFail("expected gaveUp, got \(last)")
        }
        // The identical frames tie, so the best frame stays the first seen.
        XCTAssertEqual(best["A"], 0)
    }

    /// Faces too small to transplant safely are excluded from planning.
    func testPlannerIgnoresTinyFaces() {
        func frame(_ f: Int) -> AnnotatedFrame {
            SyntheticScene.renderFrame(faces: [
                SyntheticFace(name: "A", headCenter: Point2(180, 170), interOcular: 34,
                              eyeAperture: 0.10, smileCurve: f == 3 ? 0.13 : 0.02),
                // Tiny background face: great expression in frame 3, but far
                // below the transplantable size.
                SyntheticFace(name: "C", headCenter: Point2(520, 380), interOcular: 6,
                              eyeAperture: 0.10, smileCurve: f == 3 ? 0.14 : 0.02),
            ], noiseSigma: 0.015, seed: UInt64(500 + f))
        }
        let frames = (0..<5).map(frame)
        let plan = GroupPlanner().plan(
            frames: frames, people: ["A", "C"],
            scorer: HeuristicExpressionScorer(), risk: HeuristicRiskModel()
        )
        XCTAssertEqual(plan.donors["A"], 3, "normal-size person should swap to their peak")
        XCTAssertEqual(plan.donors["C"], plan.baseFrame, "tiny face must not be swapped")
        XCTAssertEqual(plan.scoreMatrix["C"]?.max() ?? -2, Float(-1), accuracy: 1e-6, "tiny face excluded from scoring")
    }

    /// The geometric identity descriptor separates people with different face
    /// proportions and keeps the same person together across camera jitter.
    func testIdentityDescriptorSeparatesPeople() {
        let a = SyntheticScene.renderFrame(faces: [
            SyntheticFace(name: "A", headCenter: Point2(200, 180), interOcular: 32),
        ], noiseSigma: 0.01, seed: 1)
        let b = SyntheticScene.renderFrame(faces: [
            SyntheticFace(name: "B", headCenter: Point2(200, 180), interOcular: 32,
                          mouthWidthRatio: 0.55, noseDropRatio: 0.78, mouthDropRatio: 1.22),
        ], noiseSigma: 0.01, seed: 2)
        let aJittered = SyntheticScene.renderFrame(faces: [
            SyntheticFace(name: "A", headCenter: Point2(200, 180), interOcular: 32),
        ], globalOffset: (9, -4), noiseSigma: 0.01, seed: 3)

        let descA = GeometricIdentity.descriptor(for: a.faces["A"]!)
        let descB = GeometricIdentity.descriptor(for: b.faces["B"]!)
        let descAJ = GeometricIdentity.descriptor(for: aJittered.faces["A"]!)

        let dDifferent = GeometricIdentity.relativeDistance(descA, descB)
        let dSame = GeometricIdentity.relativeDistance(descA, descAJ)
        XCTAssertGreaterThan(dDifferent, 0.08, "different face proportions must exceed identity threshold")
        XCTAssertLessThan(dSame, 0.02, "same person under camera jitter must stay well below threshold")
    }
}
