import Foundation

/// Smart shutter: keep capturing until every person has had a good moment,
/// then signal done (plan §Product surfaces).
///
/// Feed each annotated frame as it arrives; the advisor tracks every seen
/// person's best moment so far and fires when all of them clear the quality
/// bar. Works with any `ExpressionScorer` — the heuristic stand-in today, the
/// learned preference model later.
public struct ShutterAdvisor: Sendable {

    public struct Config: Sendable {
        /// Best-moment score each person must reach. On the heuristic scorer's
        /// scale, ~0.85 means "open eyes + strong smile + frontal".
        public var goodMomentThreshold: Float = 0.85
        /// Signal done only after at least this many frames (avoid firing on
        /// a lucky first frame before the group is even framed).
        public var minFrames: Int = 2
        /// Stop advising (capture too long); the shutter falls back to done.
        public var maxFrames: Int = 90
        public init() {}
    }

    public enum Signal: Equatable, Sendable {
        case keepGoing
        /// Every seen person has a good moment; associated = their best frames.
        case done(bestFrames: [PersonID: Int])
        case gaveUp(bestFrames: [PersonID: Int])
    }

    public var config: Config

    public private(set) var framesSeen: Int = 0
    public private(set) var bestScore: [PersonID: Float] = [:]
    public private(set) var bestFrame: [PersonID: Int] = [:]

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Feed one frame; returns the shutter signal.
    public mutating func update(frame: AnnotatedFrame, scorer: ExpressionScorer) -> Signal {
        framesSeen += 1
        for (person, _) in frame.faces {
            let s = scorer.score(frame: frame, person: person)
            if s > (bestScore[person] ?? -1) {
                bestScore[person] = s
                bestFrame[person] = framesSeen - 1
            }
        }

        let best = bestFrame
        if framesSeen >= config.maxFrames {
            return .gaveUp(bestFrames: best)
        }
        guard framesSeen >= config.minFrames, !bestScore.isEmpty else {
            return .keepGoing
        }
        let allGood = bestScore.values.allSatisfy { $0 >= config.goodMomentThreshold }
        return allGood ? .done(bestFrames: best) : .keepGoing
    }
}
