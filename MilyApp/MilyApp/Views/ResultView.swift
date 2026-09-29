import SwiftUI
import MilyCore

/// Photos-style viewer: black background, full-bleed photo, bottom toolbar.
/// Press and hold the photo to see the original (like Photos' compare).
struct ResultView: View {
    @EnvironmentObject private var session: FixSession
    @Environment(\.dismiss) private var dismiss

    @State private var showingBefore = false
    @State private var showInfo = false
    @State private var saved = false
    @State private var celebrate = false
    /// Person whose face was tapped — shows their moments strip.
    @State private var selected: PersonID?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let result = session.result {
                VStack(spacing: 0) {
                    header(result)
                    photo(result)
                    if let person = selected {
                        FaceStrip(person: person, result: result) { selected = nil }
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    } else {
                        toolbar(result)
                    }
                }
            } else {
                Text("Nothing to show yet.").foregroundStyle(.white.opacity(0.6))
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showInfo) {
            if let result = session.result { InfoSheet(people: result.people) }
        }
        .onAppear {
            celebrate = true
            Task {
                try? await Task.sleep(nanoseconds: 1_800_000_000)
                withAnimation { celebrate = false }
            }
        }
    }

    private func header(_ result: FixSession.FixResult) -> some View {
        HStack {
            Button { dismiss() } label: {
                Image(systemName: "chevron.left").font(.system(size: 20, weight: .semibold))
            }
            Spacer()
            VStack(spacing: 1) {
                Text(showingBefore ? "Original" : "Best Take").font(.subheadline.weight(.semibold))
                Text(summary(result)).font(.caption2).foregroundStyle(.white.opacity(0.55))
            }
            Spacer()
            Image(systemName: "chevron.left").opacity(0)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
    }

    private func photo(_ result: FixSession.FixResult) -> some View {
        Image(uiImage: showingBefore ? result.before : result.after)
            .resizable()
            .scaledToFit()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in showingBefore = true }
                    .onEnded { _ in showingBefore = false }
            )
            .overlay { faceTargets(result) }
            .overlay(alignment: .bottomTrailing) {
                // Small success moment: Mily bounces in the corner, then leaves.
                if celebrate && result.people.contains(where: \.swapped) {
                    MascotView(state: .boing, size: 72)
                        .padding(12)
                        .transition(.opacity)
                }
            }
            .accessibilityLabel(showingBefore ? "Original photo" : "Best take")
            .accessibilityHint("Press and hold to compare with the original")
    }

    /// Tap targets over each face, mapped from image pixels into the
    /// aspect-fit photo frame.
    private func faceTargets(_ result: FixSession.FixResult) -> some View {
        GeometryReader { geo in
            let scale = min(geo.size.width / result.imageSize.width, geo.size.height / result.imageSize.height)
            let ox = (geo.size.width - result.imageSize.width * scale) / 2
            let oy = (geo.size.height - result.imageSize.height * scale) / 2
            ForEach(Array(result.faceBoxes.keys).sorted(), id: \.self) { person in
                let b = result.faceBoxes[person]!
                let rect = CGRect(x: ox + b.minX * scale, y: oy + b.minY * scale,
                                  width: b.width * scale, height: b.height * scale).insetBy(dx: -6, dy: -6)
                Button {
                    withAnimation(.spring(response: 0.3)) { selected = selected == person ? nil : person }
                } label: {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.white.opacity(0.001))
                        .overlay(RoundedRectangle(cornerRadius: 10)
                            .stroke(selected == person ? Color.yellow : .white.opacity(showingBefore ? 0 : 0.35),
                                    lineWidth: selected == person ? 2 : 1))
                }
                .buttonStyle(.plain)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
                .accessibilityLabel("Choose a moment for \(FixSession.displayName(for: person))")
            }
        }
    }

    private func toolbar(_ result: FixSession.FixResult) -> some View {
        HStack {
            ShareLink(item: Image(uiImage: result.after),
                      preview: SharePreview("Best take", image: Image(uiImage: result.after))) {
                Image(systemName: "square.and.arrow.up")
            }
            Spacer()
            Button { save(result) } label: {
                Image(systemName: saved ? "checkmark.circle.fill" : "square.and.arrow.down")
            }
            .disabled(saved)
            .accessibilityLabel(saved ? "Saved" : "Save to Photos")
            Spacer()
            Button { showInfo = true } label: { Image(systemName: "info.circle") }
                .accessibilityLabel("What Mily changed")
        }
        .font(.system(size: 22))
        .foregroundStyle(.white)
        .padding(.horizontal, 36)
        .padding(.vertical, 16)
    }

    private func summary(_ result: FixSession.FixResult) -> String {
        let fixed = result.people.filter(\.swapped).count
        return fixed == 0 ? "Tap a face to pick their moment" : "\(fixed) of \(result.people.count) fixed · tap a face to choose"
    }

    private func save(_ result: FixSession.FixResult) {
        let image = result.after
        Task.detached(priority: .userInitiated) {
            _ = PhotoSaver.save(image, isPro: true) // always full quality — Mily is free
            await MainActor.run { saved = true }
        }
    }
}

/// Every usable moment of one person, best first. Tapping one re-composites
/// the photo with that face (pose-unsafe frames are never offered).
private struct FaceStrip: View {
    let person: PersonID
    let result: FixSession.FixResult
    let onDone: () -> Void
    @EnvironmentObject private var session: FixSession

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text(FixSession.displayName(for: person)).font(.subheadline)
                if session.recompositing { ProgressView().controlSize(.mini).tint(.white) }
                Spacer()
                Button("Done", action: onDone).foregroundStyle(.yellow)
            }
            .padding(.horizontal, 18)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(result.candidates[person] ?? [], id: \.self) { frame in
                        let isChosen = result.chosen[person] == frame
                        Button {
                            UISelectionFeedbackGenerator().selectionChanged()
                            session.choose(frame, for: person)
                        } label: {
                            Group {
                                if let crop = session.faceCrop(person, frame: frame) {
                                    Image(uiImage: crop).resizable().scaledToFill()
                                } else {
                                    Color(white: 0.15)
                                }
                            }
                            .frame(width: 64, height: 64)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(isChosen ? Color.yellow : .clear, lineWidth: 2))
                        }
                        .accessibilityLabel("Moment \(frame + 1)\(isChosen ? ", selected" : "")")
                    }
                }
                .padding(.horizontal, 18)
            }
        }
        .foregroundStyle(.white)
        .padding(.vertical, 12)
    }
}

/// Photos' "i" panel: what Mily did to each person.
private struct InfoSheet: View {
    let people: [PersonReport]

    var body: some View {
        NavigationStack {
            List(people, id: \.person) { person in
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: person.swapped ? "checkmark.circle.fill" : "minus.circle")
                        .foregroundStyle(person.swapped ? Color.green : Color.secondary)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(FixSession.displayName(for: person.person)).font(.subheadline.weight(.semibold))
                        Text(person.swapped
                             ? "Best moment taken from photo \(person.donorFrame + 1)."
                             : "Left as-is — \(person.fallbackReason ?? "no safe fix"). A real photo beats a risky edit.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("What Mily changed")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium])
    }
}
