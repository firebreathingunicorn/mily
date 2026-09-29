import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import CoreTransferable

/// Camera-style home: black viewfinder, a big shutter button that opens the
/// burst picker, the last result as the corner thumbnail (like Camera's
/// "last photo"), and a small heart for the optional tip. Mily the marbled
/// cat only appears for small moments — swatting the shutter, and as the
/// loader while a fix runs.
struct RootView: View {
    @EnvironmentObject private var store: MilyStore
    @StateObject private var session = FixSession()

    @State private var pickedItems: [PhotosPickerItem] = []
    @State private var showPicker = false
    @State private var showTip = false
    @State private var showProcessing = false
    @State private var showResult = false
    @State private var loadError: String?
    @State private var swatting = false

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                VStack(spacing: 0) {
                    topBar
                    viewfinder
                    modeLabel
                    shutterRow
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .preferredColorScheme(.dark)
            .navigationDestination(isPresented: $showResult) { ResultView() }
            .photosPicker(isPresented: $showPicker, selection: $pickedItems,
                          maxSelectionCount: 8, matching: .any(of: [.videos, .images]))
            .sheet(isPresented: $showTip) { PaywallHost() }
            .fullScreenCover(isPresented: $showProcessing) { ProcessingView(session: session) }
            .alert("Couldn't load photos", isPresented: .init(
                get: { loadError != nil },
                set: { if !$0 { loadError = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(loadError ?? "")
            }
            .onChange(of: pickedItems) { _, items in
                guard !items.isEmpty else { return }
                Task { await loadPicked() }
            }
            .onChange(of: session.stage) { _, stage in
                switch stage {
                case .done:
                    // Dismiss the cover first; pushing while it animates out drops the push.
                    showProcessing = false
                    Task {
                        try? await Task.sleep(nanoseconds: 600_000_000)
                        showResult = true
                    }
                case .failed(let message):
                    showProcessing = false
                    loadError = message
                default:
                    break
                }
            }
            .onAppear {
                let args = ProcessInfo.processInfo.arguments
                if args.contains("-autoDemo") { startDemo() }
                if args.contains("-autoPaywall") { showTip = true }
                // Debug/testing: run a video file directly (skips the picker).
                if let i = args.firstIndex(of: "-videoPath"), i + 1 < args.count {
                    showProcessing = true
                    session.run(videoURL: URL(fileURLWithPath: args[i + 1]))
                }
            }
        }
        .environmentObject(session)
    }

    // MARK: - Layout

    private var topBar: some View {
        HStack {
            Text("MILY")
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .tracking(3)
                .foregroundStyle(.yellow)
            Spacer()
            Button { showTip = true } label: {
                Image(systemName: store.isPro ? "heart.fill" : "heart")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(store.isPro ? .pink : .white)
                    .frame(width: 36, height: 36)
                    .background(.white.opacity(0.12), in: Circle())
            }
            .accessibilityLabel("Support Mily")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    private var viewfinder: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 4)
                .fill(Color(white: 0.08))
            // Rule-of-thirds grid, like Camera's grid overlay.
            GeometryReader { geo in
                Path { p in
                    for i in 1...2 {
                        let x = geo.size.width * CGFloat(i) / 3
                        let y = geo.size.height * CGFloat(i) / 3
                        p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: geo.size.height))
                        p.move(to: CGPoint(x: 0, y: y)); p.addLine(to: CGPoint(x: geo.size.width, y: y))
                    }
                }
                .stroke(.white.opacity(0.12), lineWidth: 0.5)
            }
            VStack(spacing: 10) {
                Image(systemName: "person.3.sequence")
                    .font(.system(size: 40, weight: .light))
                    .foregroundStyle(.white.opacity(0.55))
                Text("Someone blinked?")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
                Text("Pick a short video (or a few photos)\nof your group. Tap any face to pick\ntheir best moment.")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.6))
                Button("Try a demo burst") { startDemo() }
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.yellow)
                    .padding(.top, 4)
                    .disabled(session.stage.isBusy)
            }
            .padding(24)
        }
        .padding(.horizontal, 0)
        .frame(maxHeight: .infinity)
    }

    private var modeLabel: some View {
        HStack(spacing: 22) {
            Text("PHOTO").foregroundStyle(.white.opacity(0.45))
            Text("BEST TAKE").foregroundStyle(.yellow)
            Text("PORTRAIT").foregroundStyle(.white.opacity(0.45))
        }
        .font(.system(size: 13, weight: .semibold))
        .tracking(1)
        .padding(.vertical, 14)
    }

    private var shutterRow: some View {
        HStack {
            lastThumbnail
            Spacer()
            shutter
            Spacer()
            Color.clear.frame(width: 52, height: 52)
        }
        .padding(.horizontal, 32)
        .padding(.bottom, 28)
    }

    @ViewBuilder
    private var lastThumbnail: some View {
        Button {
            if session.result != nil { showResult = true }
        } label: {
            Group {
                if let after = session.result?.after {
                    Image(uiImage: after).resizable().scaledToFill()
                } else {
                    Color(white: 0.15)
                }
            }
            .frame(width: 52, height: 52)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.white.opacity(0.3), lineWidth: 1))
        }
        .disabled(session.result == nil)
        .accessibilityLabel("Last best take")
    }

    /// The shutter: tapping it lets Mily pop in and swat it, then the burst picker opens.
    private var shutter: some View {
        Button(action: swatShutter) {
            ZStack {
                Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78)
                Circle().fill(.white).frame(width: 64, height: 64)
                    .scaleEffect(swatting ? 0.82 : 1)
            }
        }
        .buttonStyle(.plain)
        .overlay(alignment: .topTrailing) {
            if swatting {
                MascotView(state: .peeking, size: 84)
                    .offset(x: 46, y: -58)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .disabled(session.stage.isBusy)
        .accessibilityLabel("Choose burst photos")
    }

    // MARK: - Actions

    private func swatShutter() {
        withAnimation(.spring(response: 0.25, dampingFraction: 0.5)) { swatting = true }
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        Task {
            try? await Task.sleep(nanoseconds: 450_000_000)
            withAnimation(.easeOut(duration: 0.2)) { swatting = false }
            showPicker = true
        }
    }

    private func startDemo() {
        guard !session.stage.isBusy else { return }
        showProcessing = true
        session.runDemo()
    }

    private func loadPicked() async {
        // One video → sample its moments; otherwise treat the picks as a burst.
        if let item = pickedItems.first, item.supportedContentTypes.contains(where: { $0.conforms(to: .movie) }) {
            pickedItems = []
            do {
                if let movie = try await item.loadTransferable(type: PickedMovie.self) {
                    showProcessing = true
                    session.run(videoURL: movie.url)
                }
            } catch {
                loadError = "That video couldn't be loaded: \(error.localizedDescription)"
            }
            return
        }
        var images: [UIImage] = []
        for item in pickedItems {
            do {
                if let data = try await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    images.append(image)
                }
            } catch {
                loadError = "One photo couldn't be loaded: \(error.localizedDescription)"
            }
        }
        pickedItems = []
        if images.count >= 2 {
            showProcessing = true
            session.run(images: images)
        } else if loadError == nil {
            loadError = "Pick at least 2 burst photos (up to 8)."
        }
    }
}

/// A picked video, copied into our temp directory so AVFoundation can read it.
struct PickedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return PickedMovie(url: dest)
        }
    }
}
