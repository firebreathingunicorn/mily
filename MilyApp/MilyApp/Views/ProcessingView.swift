import SwiftUI

/// The loader. Mily is a marbled cat — famously fast and hard to photograph —
/// so while a fix runs she chases down everyone's best moment.
struct ProcessingView: View {
    @ObservedObject var session: FixSession
    @Environment(\.dismiss) private var dismiss
    @State private var clip = LoaderAnimation.random()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                Spacer()
                MascotClip(fileName: clip, size: 200)
                Text(stageTitle)
                    .font(.headline)
                    .foregroundStyle(.white)
                    .contentTransition(.opacity)
                    .animation(.easeInOut, value: stageTitle)
                Text("Mily's on it. Fast cat, faster fixes.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.55))
                Spacer()
                Button("Cancel") {
                    session.cancel()
                    dismiss()
                }
                .font(.body)
                .foregroundStyle(.yellow)
                .padding(.bottom, 28)
            }
        }
        .preferredColorScheme(.dark)
    }

    private var stageTitle: String {
        switch session.stage {
        case .analyzing: return "Finding every face…"
        case .synthesizing: return "Picking each person's best take…"
        default: return "Working…"
        }
    }
}
