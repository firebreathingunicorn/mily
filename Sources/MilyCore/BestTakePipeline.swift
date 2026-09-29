import Foundation

/// Per-person outcome, serialized into the report JSON shown alongside the
/// result (the picker's "which level / which frame" labels).
public struct PersonReport: Codable, Sendable {
    public var person: PersonID
    public var baseFrame: Int
    public var donorFrame: Int
    public var swapped: Bool
    public var baseScore: Float
    public var donorScore: Float
    public var level: String
    public var checksPassed: Bool
    public var checkMetrics: [String: Float]
    public var identityDistance: Float?
    public var fallbackReason: String?
}

public struct BestTakeReport: Codable, Sendable {
    public var baseFrame: Int
    public var frameCount: Int
    public var people: [PersonReport]
    /// Per-person expression scores for every frame — the filmstrip data the
    /// per-person picker renders (which frame was best, how close the rest).
    public var scores: [PersonID: [Float]]
    /// Per person: frames they can be swapped in from, best first — expression
    /// score discounted by pose/size risk vs. the base, pose-unsafe frames
    /// (risk ≥ `maxChoiceRisk`) dropped. Drives the tap-a-face picker. The base
    /// frame is always included so "keep original" is a choice.
    public var candidates: [PersonID: [Int]] = [:]
}

/// Level A end to end: plan → synthesize → finish → verify → assemble.
/// On any verification failure the person is left unchanged in the base frame
/// (least-invasive-first, and a real pixel beats a generated one).
public struct BestTakePipeline: Sendable {

    public var scorer: ExpressionScorer
    public var risk: RiskModel
    public var planner: GroupPlanner
    public var synthesizer: FaceSynthesizer
    public var finisher: Finisher
    public var checker: ArtifactChecker
    public var identity: IdentityChecker
    public var rngSeed: UInt64
    /// Frames riskier than this (pose/roll/size delta vs. the base) are not
    /// offered in the picker — they would fail verification or look wrong.
    public var maxChoiceRisk: Float = 1.0

    public init(
        scorer: ExpressionScorer = HeuristicExpressionScorer(),
        risk: RiskModel = HeuristicRiskModel(),
        planner: GroupPlanner = GroupPlanner(),
        synthesizer: FaceSynthesizer = DirectTransplantSynthesizer(),
        finisher: Finisher = Finisher(),
        checker: ArtifactChecker = ClassicalArtifactChecker(),
        identity: IdentityChecker = IdentityChecker(),
        rngSeed: UInt64 = 0xB577_1CE5
    ) {
        self.scorer = scorer
        self.risk = risk
        self.planner = planner
        self.synthesizer = synthesizer
        self.finisher = finisher
        self.checker = checker
        self.identity = identity
        self.rngSeed = rngSeed
    }

    /// - Parameter overrides: person → frame the user picked in the face picker;
    ///   replaces the planner's automatic donor choice for that person.
    public func run(frames: [AnnotatedFrame], overrides: [PersonID: Int] = [:]) -> (output: PixelImage, report: BestTakeReport) {
        precondition(!frames.isEmpty, "pipeline needs at least one frame")

        var people = Set<PersonID>()
        for f in frames { people.formUnion(f.faces.keys) }
        let ordered = people.sorted()

        var plan = planner.plan(frames: frames, people: ordered, scorer: scorer, risk: risk)
        for (person, frame) in overrides where frames.indices.contains(frame) {
            plan.donors[person] = frame
        }
        let candidates = choiceCandidates(frames: frames, people: ordered, plan: plan)
        var output = frames[plan.baseFrame].image

        var reports: [PersonReport] = []
        var seed = rngSeed

        for person in ordered {
            let baseFrame = plan.baseFrame
            let donorFrame = plan.donors[person] ?? baseFrame
            let baseScore = plan.scoreMatrix[person]?[baseFrame] ?? 0
            let donorScore = plan.scoreMatrix[person]?[donorFrame] ?? 0

            guard donorFrame != baseFrame,
                  let result = synthesizer.transplant(base: frames[baseFrame], donor: frames[donorFrame], person: person) else {
                reports.append(PersonReport(
                    person: person, baseFrame: baseFrame, donorFrame: baseFrame,
                    swapped: false, baseScore: baseScore, donorScore: baseScore,
                    level: "none", checksPassed: true, checkMetrics: [:],
                    identityDistance: nil, fallbackReason: "no beneficial donor"
                ))
                continue
            }

            var finished = result
            seed &+= 0x9E3779B9
            finisher.finish(base: frames[baseFrame], result: &finished, rngSeed: seed)

            let artifact = checker.check(base: frames[baseFrame], result: finished, person: person)
            let idReport = identity.check(base: frames[baseFrame], donor: frames[donorFrame], person: person)

            if artifact.passed && idReport.passed {
                output.pasteRegion(content: finished.content, alpha: finished.alpha, region: finished.region)
                reports.append(PersonReport(
                    person: person, baseFrame: baseFrame, donorFrame: donorFrame,
                    swapped: true, baseScore: baseScore, donorScore: donorScore,
                    level: finished.level.rawValue, checksPassed: true,
                    checkMetrics: artifact.metrics,
                    identityDistance: idReport.metrics["distance"], fallbackReason: nil
                ))
            } else {
                let reason = !idReport.passed ? "identity check failed" : "artifact check failed"
                reports.append(PersonReport(
                    person: person, baseFrame: baseFrame, donorFrame: donorFrame,
                    swapped: false, baseScore: baseScore, donorScore: donorScore,
                    level: "none", checksPassed: false,
                    checkMetrics: artifact.metrics,
                    identityDistance: idReport.metrics["distance"], fallbackReason: reason
                ))
            }
        }

        return (output, BestTakeReport(baseFrame: plan.baseFrame, frameCount: frames.count, people: reports, scores: plan.scoreMatrix, candidates: candidates))
    }

    private func choiceCandidates(frames: [AnnotatedFrame], people: [PersonID],
                                  plan: GroupPlanner.TakePlan) -> [PersonID: [Int]] {
        var out: [PersonID: [Int]] = [:]
        for p in people {
            guard let baseFace = frames[plan.baseFrame].faces[p] else { continue }
            var ranked: [(frame: Int, value: Float)] = []
            for (f, frame) in frames.enumerated() {
                guard let face = frame.faces[p], frame.personMasks[p] != nil || f == plan.baseFrame else { continue }
                let r = f == plan.baseFrame ? 0 : risk.risk(base: baseFace, donor: face)
                guard r < maxChoiceRisk else { continue }
                let score = plan.scoreMatrix[p]?[f] ?? 0
                ranked.append((f, score - planner.riskWeight * r))
            }
            out[p] = ranked.sorted { $0.value > $1.value }.map(\.frame)
        }
        return out
    }
}
