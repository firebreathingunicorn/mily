import Foundation

/// Burst-relative pose refinement.
///
/// The raw yaw proxy — (nose.x − eyeMid.x) / inter-ocular — is noisy: it
/// depends on landmark noise and on each face model's nose prominence. But
/// there is a much stronger signal available *within a burst*: a head turning
/// away from the camera foreshortens the apparent inter-ocular distance as
/// cos(yaw). So the burst-wide maximum of inter-ocular distance is that
/// person's frontal reference, and each frame's |yaw| ≈ acos(io / ioMax).
/// The sign still comes from the nose-offset proxy.
///
/// This makes pose deltas — what swap risk and frontality consume — far more
/// accurate, because both frames are measured against the same reference
/// instead of against noisy per-frame geometry.
public enum PoseRefiner {

    /// Below this io/ioMax ratio acos is dominated by noise; keep the
    /// proxy's magnitude (sign-corrected) instead.
    public static var minRatio: Float = 0.55

    public static func refine(frames: inout [AnnotatedFrame]) {
        let people = Set(frames.flatMap { $0.faces.keys })
        for person in people {
            // Frontal reference: the widest the person's eyes were seen apart.
            var ioMax: Float = 0
            for frame in frames {
                if let face = frame.faces[person], face.interOcular > ioMax {
                    ioMax = face.interOcular
                }
            }
            guard ioMax > 1e-4 else { continue }

            for fi in frames.indices {
                guard let face = frames[fi].faces[person] else { continue }
                let ratio = min(1, face.interOcular / ioMax)
                let magnitude: Float
                if ratio >= minRatio {
                    magnitude = acos(ratio)
                } else {
                    magnitude = min(abs(face.yawProxy), Float.pi / 3)
                }
                let sign: Float = face.yawProxy < 0 ? -1 : (face.yawProxy > 0 ? 1 : 0)
                frames[fi].faces[person]?.yawProxy = sign * magnitude
            }
        }
    }
}
