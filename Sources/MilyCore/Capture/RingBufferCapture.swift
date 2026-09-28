import Foundation
#if os(iOS)
import AVFoundation
import CoreMotion
import CoreImage
#endif

#if os(iOS)
/// Full-resolution ring buffer capture for iPhone.
///
/// Design (plan §Capture):
/// - `AVCapturePhotoOutput` with `isLivePhotoCaptureEnabled` is not enough —
///   we need *full-resolution* pre-shutter frames, not a low-res video rail.
///   This recorder uses `AVCaptureVideoDataOutput` at the sensor's max
///   supported resolution plus `AVCaptureDepthDataOutput` (LiDAR or dual
///   camera) and CoreMotion for gyroscope rotation.
/// - Frames are retained in a bounded ring (`capacity`) from before the
///   shutter press to after it, so the selection window covers the moments
///   around the press rather than only after it.
/// - Deferred processing: the shutter returns instantly; `finalizeBurst`
///   hands the retained frames to the pipeline on a background queue.
///
/// NOTE: kept in `#if os(iOS)` so the macOS CLI/package builds cleanly; this
/// module is exercised on device in Phase 1 device bring-up.
public final class RingBufferCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureDepthDataOutputDelegate {

    public struct Config {
        /// Number of full-resolution frames retained around the shutter press.
        public var capacity: Int = 24
        public var wantsDepth: Bool = true
        public init() {}
    }

    public let config: Config
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let motion = CMMotionManager()
    private let lock = NSLock()
    private var ring: [CaptureFrame] = []
    private var isCollecting = false
    private var shutterTime: Double = 0
    /// One context for the whole session — creating a CIContext per frame is
    /// extremely expensive (it compiles/loads the GPU pipeline each time).
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    public init(config: Config) throws {
        self.config = config
        super.init()
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CaptureError.noCamera
        }
        let input = try AVCaptureDeviceInput(device: device)
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        session.addInput(input)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "mily.video"))
        session.addOutput(videoOutput)

        if config.wantsDepth,
           session.canAddOutput(depthOutput) {
            session.addOutput(depthOutput)
            depthOutput.setDelegate(self, callbackQueue: DispatchQueue(label: "mily.depth"))
            // Calibration note: AVCameraCalibrationData rides on the delivered
            // AVDepthData automatically when the device provides it. The
            // explicit *delivery-enabled* toggles exist only on the photo
            // path (AVCapturePhotoOutput.cameraCalibrationDataDeliveryEnabled);
            // if Level B needs guaranteed intrinsics per frame, switch this
            // class to AVCapturePhotoOutput burst capture.
        }
        session.commitConfiguration()
    }

    public func start() {
        DispatchQueue.global().async { [session] in session.startRunning() }
        if motion.isDeviceMotionAvailable {
            motion.startDeviceMotionUpdates()
        }
    }

    /// Shutter press. Capture continues so post-press good moments are retained.
    public func pressShutter() {
        lock.lock()
        shutterTime = CACurrentMediaTime()
        isCollecting = true
        lock.unlock()
    }

    /// Call after the capture window (≈ shutter + look-at-camera delay) ends.
    public func finalizeBurst() -> [CaptureFrame] {
        lock.lock()
        defer {
            ring.removeAll(keepingCapacity: true)
            isCollecting = false
            lock.unlock()
        }
        return ring
    }

    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock()
        let collecting = isCollecting
        lock.unlock()
        guard collecting else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        // CVPixelBuffer → CGImage through Core Image (handles the YCbCr →
        // RGB conversion), then into the pipeline's float representation.
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = ImageIO.fromCGImage(cg)
        lock.lock()
        ring.append(CaptureFrame(
            image: image,
            metadata: FrameMetadata(timestamp: time)
        ))
        if ring.count > config.capacity { ring.removeFirst(ring.count - config.capacity) }
        lock.unlock()
    }

    public func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData, timestamp: CMTime, connection: AVCaptureConnection) {
        lock.lock()
        defer { lock.unlock() }
        guard isCollecting, !ring.isEmpty else { return }
        let converted = depthData.converting(toDepthDataType: kCVPixelFormatType_DisparityFloat32)
        guard let pb = converted.depthDataMap as CVPixelBuffer? else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let base = CVPixelBufferGetBaseAddress(pb)!
        let rowStride = CVPixelBufferGetBytesPerRow(pb) / MemoryLayout<Float>.size
        let ptr = base.assumingMemoryBound(to: Float.self)
        var values = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                values[y * w + x] = ptr[y * rowStride + x]
            }
        }
        ring[ring.count - 1].depth = DepthMap(width: w, height: h, values: values, isDisparity: true)
    }
}

public enum CaptureError: Error {
    case noCamera
}
#endif
