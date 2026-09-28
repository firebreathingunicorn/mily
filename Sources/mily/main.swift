import Foundation
import MilyCore

// besttake CLI
//
//   besttake demo-burst --out <dir> [--frames N] [--seed N]
//       Write a synthetic two-person burst (no camera needed) for trying the
//       pipeline end to end.
//
//   besttake process <burst-dir> --out <jpg> [--report <json>] [--long-edge N]
//       Run Level A on a directory of burst images; write the composite and a
//       per-person JSON report (base/donor frame, level, checks).
//
//   besttake eval <root> [--out <csv>]
//       Run `process` over every subdirectory (one burst each) and emit a CSV
//       for blind-test preparation and slice metrics.

func usage() -> Never {
    print("""
    usage:
      besttake demo-burst --out <dir> [--frames N] [--seed N]
      besttake process <burst-dir> --out <jpg> [--report <json>] [--long-edge N]
                          [--analyzer auto|vision|sidecar] [--compare]
      besttake eval <root> [--out <csv>] [--long-edge N]
      besttake bench [--frames N]

    process analyzes a burst with Vision (real photos) or, when the directory
    contains a demo-burst sidecar.json, with the recorded ground truth.
    """)
    exit(2)
}

func argValue(_ args: [String], _ flag: String, _ default: String? = nil) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return `default` }
    return args[i + 1]
}

/// Production analysis path: Vision faces + landmarks + person matting.
func analyzeWithVision(_ dir: URL, _ longEdge: Int) throws -> [AnnotatedFrame] {
    let frames = try FileBurstSource(directory: dir, workingLongEdge: longEdge).loadFrames()
    print("loaded \(frames.count) frames; analyzing (Vision: faces, landmarks, person matting)…")
    let annotated = try VisionFaceAnalyzer().annotate(frames: frames)
    for (i, f) in annotated.enumerated() {
        let ids = f.personIDs().joined(separator: ", ")
        print("  frame \(i): \(f.faces.count) face(s) [\(ids)]")
    }
    if annotated.allSatisfy({ $0.faces.isEmpty }) {
        print("note: Vision found no faces. Synthetic renders need a sidecar (demo-burst writes one); real photos use --analyzer vision/auto.")
    }
    return annotated
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { usage() }
args = Array(args.dropFirst())

switch command {
case "demo-burst":
    let out = URL(fileURLWithPath: argValue(args, "--out", "demo-burst")!)
    let frameCount = Int(argValue(args, "--frames", "6")!) ?? 6
    let seed = UInt64(argValue(args, "--seed", "11")!) ?? 11
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

    // Person A peaks at frame 2 (big smile), person B at frame 4 (eyes open +
    // smile while A blinks). Camera jitters a few pixels per frame.
    var frames: [[SyntheticFace]] = []
    for f in 0..<frameCount {
        let aSmile: Float = f == 2 ? 0.11 : 0.03
        let aEyes: Float = f == 3 ? 0.02 : 0.10 // A blinks at 3
        let bSmile: Float = f == 4 ? 0.10 : 0.02
        let bEyes: Float = f == 1 ? 0.02 : 0.10 // B blinks at 1
        frames.append([
            SyntheticFace(
                name: "personA",
                headCenter: Point2(190, 170),
                interOcular: 34,
                eyeAperture: aEyes,
                smileCurve: aSmile,
                mouthAperture: f == 5 ? 0.16 : 0.02 // A mid-word at 5
            ),
            SyntheticFace(
                name: "personB",
                headCenter: Point2(450, 200),
                interOcular: 28,
                eyeAperture: bEyes,
                smileCurve: bSmile,
                mouthAperture: 0.02,
                skin: (0.78, 0.58, 0.44),
                shirt: (0.5, 0.26, 0.22)
            ),
        ])
    }
    let offsets = (0..<frameCount).map { (dx: ($0 * 3) % 7 - 3, dy: ($0 * 5) % 5 - 2) }
    let burst = SyntheticScene.burst(frames: frames, offsets: offsets, noiseSigma: 0.02, seed: seed)

    for (i, frame) in burst.enumerated() {
        let url = out.appendingPathComponent(String(format: "frame_%02d.png", i))
        try ImageIO.save(frame.image, to: url)
    }
    // Ground-truth sidecar: Vision's real-face detector does not fire on the
    // stylized renders, so `process` reads geometry from here (analyzer auto).
    try SyntheticSidecar.write(frames: frames, offsets: offsets, width: 640, height: 480, directory: out)
    print("wrote \(burst.count) frames to \(out.path)")

case "process":
    guard let dir = args.first, !dir.hasPrefix("--") else { usage() }
    let burstDir = URL(fileURLWithPath: dir)
    let out = URL(fileURLWithPath: argValue(args, "--out", "best-take.jpg")!)
    let longEdge = Int(argValue(args, "--long-edge", "2048")!) ?? 2048
    let analyzer = argValue(args, "--analyzer", "auto")!

    print("loading frames from \(burstDir.path)…")
    let annotated: [AnnotatedFrame]
    switch analyzer {
    case "vision":
        annotated = try analyzeWithVision(burstDir, longEdge)
    case "sidecar":
        guard let frames = try SyntheticSidecar.load(directory: burstDir) else {
            print("no sidecar.json in \(burstDir.path)"); exit(1)
        }
        annotated = frames
    default: // auto: ground-truth sidecar if present, else Vision
        if let frames = try SyntheticSidecar.load(directory: burstDir) {
            print("using ground-truth sidecar (\(SyntheticSidecar.fileName))")
            annotated = frames
        } else {
            annotated = try analyzeWithVision(burstDir, longEdge)
        }
    }

    print("planning + synthesizing (Level A)…")
    let pipeline = BestTakePipeline()
    let (output, report) = pipeline.run(frames: annotated)

    try ImageIO.save(output, to: out)
    if args.contains("--compare") {
        let base = annotated[report.baseFrame].image
        let cmp = ImageOps.sideBySide(base, output)
        let cmpURL = out.deletingPathExtension().appendingPathExtension("compare.jpg")
        try ImageIO.save(cmp, to: cmpURL)
        print("wrote \(cmpURL.path)")
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let json = try encoder.encode(report)
    let reportURL = URL(fileURLWithPath: argValue(args, "--report", out.deletingPathExtension().appendingPathExtension("json").path)!)
    try json.write(to: reportURL)
    print("wrote \(out.path)")
    print("wrote \(reportURL.path)")
    print(String(data: json, encoding: .utf8) ?? "")

case "eval":
    guard let root = args.first, !root.hasPrefix("--") else { usage() }
    let rootURL = URL(fileURLWithPath: root)
    let longEdge = Int(argValue(args, "--long-edge", "2048")!) ?? 2048
    let csvPath = argValue(args, "--out", "eval.csv")!

    var rows = ["burst,person,baseFrame,donorFrame,swapped,level,checksPassed,baseScore,donorScore,identityDistance,fallback"]
    let fm = FileManager.default
    let bursts = (try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil))?
        .filter { (try? fm.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil).filter { $0.pathExtension.lowercased().hasSuffix(("jpg" as NSString).pathExtension.lowercased()) || true })?.isEmpty == false }
        ?? []
    let burstDirs = bursts.filter { dir in
        (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
            .contains { ["jpg", "jpeg", "png", "heic"].contains($0.pathExtension.lowercased()) } == true
    }.sorted { $0.lastPathComponent < $1.lastPathComponent }

    for burstDir in burstDirs {
        do {
            let annotated: [AnnotatedFrame]
            if let frames = try SyntheticSidecar.load(directory: burstDir) {
                annotated = frames
            } else {
                annotated = try analyzeWithVision(burstDir, longEdge)
            }
            let (output, report) = BestTakePipeline().run(frames: annotated)
            let outDir = burstDir.appendingPathComponent("result", isDirectory: true)
            try? fm.createDirectory(at: outDir, withIntermediateDirectories: true)
            try ImageIO.save(output, to: outDir.appendingPathComponent("best-take.jpg"))
            for p in report.people {
                rows.append([
                    burstDir.lastPathComponent, p.person,
                    String(p.baseFrame), String(p.donorFrame),
                    p.swapped ? "1" : "0", p.level,
                    p.checksPassed ? "1" : "0",
                    String(format: "%.3f", p.baseScore),
                    String(format: "%.3f", p.donorScore),
                    p.identityDistance.map { String(format: "%.4f", $0) } ?? "",
                    p.fallbackReason ?? "",
                ].joined(separator: ","))
            }
            print("\(burstDir.lastPathComponent): base=\(report.baseFrame), \(report.people.count) person report(s)")
        } catch {
            print("\(burstDir.lastPathComponent): FAILED — \(error)")
        }
    }
    try rows.joined(separator: "\n").write(toFile: csvPath, atomically: true, encoding: .utf8)
    print("wrote \(csvPath)")

case "bench":
    let frameCount = Int(argValue(args, "--frames", "6")!) ?? 6
    func faceA(_ f: Int) -> SyntheticFace {
        SyntheticFace(name: "A", headCenter: Point2(180, 165), interOcular: 34,
            eyeAperture: (f == 1 || f == 4) ? 0.01 : 0.10,
            smileCurve: f == 2 ? 0.13 : 0.02,
            mouthAperture: f == 5 ? 0.16 : 0.02)
    }
    func faceB(_ f: Int) -> SyntheticFace {
        SyntheticFace(name: "B", headCenter: Point2(445, 205), interOcular: 27,
            eyeAperture: f == 1 ? 0.01 : 0.10, smileCurve: f == 4 ? 0.12 : 0.02,
            mouthAperture: f == 3 ? 0.16 : 0.02,
            noseDropRatio: 0.75, mouthDropRatio: 1.2,
            skin: (0.80, 0.60, 0.46), shirt: (0.52, 0.25, 0.20))
    }
    let framesCfg = (0..<frameCount).map { f -> [SyntheticFace] in [faceA(f), faceB(f)] }
    let offsets = (0..<frameCount).map { i -> (dx: Int, dy: Int) in
        (dx: [0, 3, -2, 4, -3, 2][i % 6], dy: [0, -2, 3, -1, 2, -2][i % 6])
    }
    var t0 = Date()
    let burst = SyntheticScene.burst(frames: framesCfg, offsets: offsets, noiseSigma: 0.02, seed: 1234)
    print(String(format: "scene: %d frames in %.2fs (%.0f ms/frame)", frameCount, -t0.timeIntervalSinceNow, -t0.timeIntervalSinceNow * 1000 / Double(frameCount)))

    t0 = Date()
    let (_, report) = BestTakePipeline().run(frames: burst)
    print(String(format: "pipeline: %.0f ms (base frame %d, %d person(s), %d swapped)",
                 -t0.timeIntervalSinceNow * 1000, report.baseFrame,
                 report.people.count, report.people.filter(\.swapped).count))

default:
    usage()
}
