import XCTest
@testable import MilyCore

/// Background plate: pixels a person covers in the base frame must be filled
/// with clean background from frames where they moved away.
final class PlateTests: XCTestCase {

    func testPlateFillsPersonRegion() {
        let left = SyntheticFace(name: "A", headCenter: Point2(150, 200), interOcular: 34)
        let right = SyntheticFace(name: "A", headCenter: Point2(430, 200), interOcular: 34)
        let frames = [
            SyntheticScene.renderFrame(faces: [left], noiseSigma: 0.012, seed: 1),
            SyntheticScene.renderFrame(faces: [left], noiseSigma: 0.012, seed: 2),
            SyntheticScene.renderFrame(faces: [right], noiseSigma: 0.012, seed: 3),
            SyntheticScene.renderFrame(faces: [right], noiseSigma: 0.012, seed: 4),
        ]
        let sources = frames.map { BackgroundPlate.SourceFrame(offset: (0, 0), frame: $0) }
        let plate = BackgroundPlate.build(base: frames[0], sources: sources)

        // Ground-truth background (same function the renderer uses).
        let gt = SyntheticScene.background(width: 640, height: 480, offset: (0, 0), seed: 7)

        // Region where the person stood in the base frame.
        let region = RectI(x: 118, y: 170, width: 66, height: 66)
        var acc: Float = 0
        var n: Float = 0
        for y in region.y..<region.maxY {
            for x in region.x..<region.maxX {
                let p = plate[x, y], q = gt[x, y]
                acc += abs(p.0 - q.0) + abs(p.1 - q.1) + abs(p.2 - q.2)
                n += 3
            }
        }
        let meanDiff = acc / n
        XCTAssertLessThan(meanDiff, 0.03, "plate should be clean background at the old person location (got \(meanDiff))")

        // And untouched background must stay identical to the base frame.
        let corner = RectI(x: 520, y: 20, width: 60, height: 60)
        var acc2: Float = 0
        for y in corner.y..<corner.maxY {
            for x in corner.x..<corner.maxX {
                let p = plate[x, y], q = frames[0].image[x, y]
                acc2 += abs(p.0 - q.0) + abs(p.1 - q.1) + abs(p.2 - q.2)
            }
        }
        XCTAssertLessThan(acc2 / Float(corner.width * corner.height * 3), 1e-6)
    }
}
