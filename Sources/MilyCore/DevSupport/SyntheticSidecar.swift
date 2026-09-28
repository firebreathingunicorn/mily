import Foundation
import CoreGraphics

/// Ground-truth sidecar for synthetic bursts.
///
/// Vision's face detector is trained on real faces and does not fire on the
/// stylized scene renders, so `demo-burst` writes this sidecar next to the
/// PNGs and `process` consumes it (analyzer "auto"). This mirrors the Python
/// research loop, which likewise feeds ground-truth landmarks. Production
/// analysis of real photos stays on `VisionFaceAnalyzer`.
public enum SyntheticSidecar {

    struct Face: Codable {
        var name: String
        var x: Float
        var y: Float
        var interOcular: Float
        var roll: Float
        var eyeAperture: Float
        var smileCurve: Float
        var mouthAperture: Float
        var mouthWidthRatio: Float
        var noseDropRatio: Float
        var mouthDropRatio: Float

        init(_ f: SyntheticFace) {
            name = f.name
            x = f.headCenter.x
            y = f.headCenter.y
            interOcular = f.interOcular
            roll = f.roll
            eyeAperture = f.eyeAperture
            smileCurve = f.smileCurve
            mouthAperture = f.mouthAperture
            mouthWidthRatio = f.mouthWidthRatio
            noseDropRatio = f.noseDropRatio
            mouthDropRatio = f.mouthDropRatio
        }

        var synthetic: SyntheticFace {
            SyntheticFace(
                name: name,
                headCenter: Point2(x, y),
                interOcular: interOcular,
                roll: roll,
                eyeAperture: eyeAperture,
                smileCurve: smileCurve,
                mouthAperture: mouthAperture,
                mouthWidthRatio: mouthWidthRatio,
                noseDropRatio: noseDropRatio,
                mouthDropRatio: mouthDropRatio
            )
        }
    }

    struct Frame: Codable {
        var offset: [Int]
        var faces: [Face]
    }

    struct Sidecar: Codable {
        var width: Int
        var height: Int
        var frames: [Frame]
    }

    public static let fileName = "sidecar.json"

    public static func write(
        frames: [[SyntheticFace]],
        offsets: [(dx: Int, dy: Int)],
        width: Int,
        height: Int,
        directory: URL
    ) throws {
        let sc = Sidecar(
            width: width,
            height: height,
            frames: frames.enumerated().map { i, faces in
                Frame(offset: [offsets[i].dx, offsets[i].dy],
                      faces: faces.map { Face($0) })
            }
        )
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(sc).write(to: directory.appendingPathComponent(fileName))
    }

    /// Load the annotated frames for a synthetic burst: images from the
    /// numbered PNGs, geometry and person masks rebuilt deterministically
    /// from the sidecar parameters.
    public static func load(directory: URL) throws -> [AnnotatedFrame]? {
        let scURL = directory.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: scURL.path) else { return nil }
        let data = try Data(contentsOf: scURL)
        let sc = try JSONDecoder().decode(Sidecar.self, from: data)

        return try sc.frames.enumerated().map { i, frame -> AnnotatedFrame in
            let imgURL = directory.appendingPathComponent(String(format: "frame_%02d.png", i))
            let image = try ImageIO.load(path: imgURL)
            let faces = frame.faces.map { $0.synthetic }
            var masks: [PersonID: Mask] = [:]
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            for face in faces {
                masks[face.name] = SyntheticScene.personMask(
                    face: face, width: image.width, height: image.height, ss: 2, colorSpace: colorSpace
                )
            }
            var out = AnnotatedFrame(
                image: image,
                metadata: FrameMetadata(timestamp: Double(i)),
                faces: [:],
                personMasks: masks
            )
            for face in faces {
                out.faces[face.name] = SyntheticScene.geometry(
                    for: face, width: image.width, height: image.height, mask: masks[face.name]!
                )
            }
            return out
        }
    }
}
