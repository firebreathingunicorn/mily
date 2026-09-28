import Foundation

/// Synthesis level used for a swap, following the plan's least-invasive-first
/// ladder. The pipeline records it per person and shows it in the UI picker.
public enum SynthesisLevel: String, Codable, Sendable {
    /// Direct pixel transplant: everything shown is a real pixel of this
    /// person from this capture session.
    case directTransplant = "A"
    /// 3D-aware re-projection (Phase 3).
    case reprojection = "B"
    /// Limited generative repair (Phase 4).
    case generativeRepair = "C"
}

/// Output of one face transplant, in base-frame coordinates. `content` and
/// `alpha` are region-sized; `region` places them on the base frame.
public struct TransplantResult: Sendable {
    public var content: PixelImage
    public var alpha: Mask
    public var region: RectI
    public var level: SynthesisLevel
    /// Noise σ measured on the donor content before finishing, so the finisher
    /// can compute the grain difference without re-warping.
    public var donorSigma: Float

    public init(content: PixelImage, alpha: Mask, region: RectI, level: SynthesisLevel, donorSigma: Float) {
        self.content = content
        self.alpha = alpha
        self.region = region
        self.level = level
        self.donorSigma = donorSigma
    }
}

/// The synthesis seam.
///
/// Level A implements direct transplant. Phase 3 adds `ReprojectionSynthesizer`
/// (3D face fitting + depth + re-render at base pose) and Phase 4 adds limited
/// reenactment — both implement this same protocol and slot into the pipeline
/// unchanged. The pipeline tries synthesizers least-invasive-first and falls
/// back to "leave the person unchanged" when verification fails.
public protocol FaceSynthesizer: Sendable {
    var level: SynthesisLevel { get }
    func transplant(base: AnnotatedFrame, donor: AnnotatedFrame, person: PersonID) -> TransplantResult?
}

// MARK: - Level A: direct transplant

/// Aligns the whole frame via landmark geometry, warps the donor's face into
/// the base pose with a fitted similarity transform, and applies the donor's
/// soft person alpha weighted by a feathered face ellipse. Every pasted pixel
/// is a real pixel of the donor frame (principle: least invasive that works).
public struct DirectTransplantSynthesizer: FaceSynthesizer {

    public var level: SynthesisLevel { .directTransplant }

    /// Face bbox inflation for the transplant region.
    public var regionInflation: Float = 1.55
    /// Feather as a fraction of the ellipse's smaller radius.
    public var featherFraction: Float = 0.30

    public init() {}

    public func transplant(base: AnnotatedFrame, donor: AnnotatedFrame, person: PersonID) -> TransplantResult? {
        guard let gb = base.faces[person], let gd = donor.faces[person],
              let donorMask = donor.personMasks[person] else { return nil }

        // Face-to-face similarity fit (donor → base).
        guard let t = SimilarityTransform.fit(src: gd.landmarks, dst: gb.landmarks),
              t.scale > 0.2, t.scale < 5 else { return nil }
        let inverse = t.inverted
        let map: (Float, Float) -> (Float, Float) = { x, y in inverse.apply(x: x, y: y) }

        let region = gb.bbox
            .inflated(byFactor: regionInflation, toWidth: base.image.width, height: base.image.height)
        guard region.width > 8, region.height > 8 else { return nil }

        let content = Warp.resampleRegion(src: donor.image, region: region, dstToSrc: map)
        let warpedMask = Warp.resampleMaskRegion(src: donorMask, region: region, dstToSrc: map)

        // Feathered face ellipse, in REGION-LOCAL coordinates. Center between
        // the eyes and the chin; radii proportional to inter-ocular distance.
        let io = gb.interOcular
        let center = Point2(
            gb.eyeMid.x - Float(region.x),
            gb.eyeMid.y + io * 0.75 - Float(region.y)
        )
        let rx = io * 1.15
        let ry = io * 1.95
        let feather = max(6, min(rx, ry) * featherFraction)
        let featherMask = Mask.ellipse(
            center: center,
            radiusX: rx, radiusY: ry,
            feather: feather,
            width: region.width, height: region.height
        )

        let alpha = warpedMask.multiplied(featherMask).clamped01()

        // Donor noise σ measured on the donor face crop in donor coordinates.
        let donorCrop = gd.bbox.inflated(byFactor: 1.2, toWidth: donor.image.width, height: donor.image.height)
        let donorSigma = NoiseEstimator.sigma(image: donor.image, region: donorCrop)

        return TransplantResult(
            content: content,
            alpha: alpha,
            region: region,
            level: .directTransplant,
            donorSigma: donorSigma
        )
    }
}
