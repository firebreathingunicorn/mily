import SwiftUI
import RevenueCat
import RevenueCatUI

/// Chooses the paywall implementation:
/// - RevenueCat mode → RevenueCatUI's `PaywallView` (offering-driven).
/// - Local StoreKit mode (judges cloning the repo) → the bundled paywall
///   below, purchasing `mily_lifetime` directly through StoreKit 2.
struct PaywallHost: View {
    @EnvironmentObject private var store: MilyStore
    @Environment(\.dismiss) private var dismiss

    @State private var rcHasOffering: Bool?

    var body: some View {
        Group {
            switch store.mode {
            case .revenueCat:
                if rcHasOffering == false {
                    LocalPaywallView()
                } else {
                    // RevenueCatUI's offering-driven paywall.
                    PaywallView(displayCloseButton: true)
                        .onAppear { Task { await checkOfferings() } }
                        .interactiveDismissDisabled(false)
                }
            case .localStoreKit:
                LocalPaywallView()
            }
        }
    }

    /// If the RevenueCat project has no offering yet (fresh project), fall
    /// back to the local paywall instead of RevenueCatUI's error state.
    private func checkOfferings() async {
        guard rcHasOffering == nil else { return }
        do {
            let offerings = try await Purchases.shared.offerings()
            let has = (offerings.current?.availablePackages.isEmpty == false)
            rcHasOffering = has
        } catch {
            rcHasOffering = false
        }
    }
}

/// Mily's own paywall, used in local StoreKit mode (and as a fallback when a
/// RevenueCat project has no offering configured yet).
struct LocalPaywallView: View {
    @EnvironmentObject private var store: MilyStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                MascotView(state: .boing, size: 140)
                    .padding(.top, 32)

                VStack(spacing: 6) {
                    Text("Support Mily")
                        .font(.system(size: 36, weight: .heavy, design: .rounded))
                    Text("Mily is free — every fix, full quality, no limits. If it saved a photo you love, you can chip in.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                VStack(alignment: .leading, spacing: 14) {
                    feature("heart.fill", "Keeps Mily free and on-device")
                    feature("pawprint.fill", "Unlocks a secret cat animation")
                    feature("checkmark.circle", "One-time tip — no subscription")
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
                .padding(.horizontal, 24)

                Button {
                    Task { await store.purchasePro() }
                } label: {
                    HStack {
                        if store.isPro {
                            Label("Thank you! ♥", systemImage: "heart.fill")
                        } else {
                            Text("Tip \(store.localPrice ?? "$6.99")")
                        }
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .tint(.pink)
                .disabled(store.isPro)
                .padding(.horizontal, 24)

                Button("Restore purchases") {
                    Task { await store.restorePurchases() }
                }
                .font(.footnote)

                if let error = store.storeError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                }

                Text(store.mode == .localStoreKit
                     ? "Sandbox testing via StoreKit configuration — no real charge."
                     : "Processed securely through the App Store.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.bottom, 24)
            }
        }
        .background(Color(.systemGroupedBackground))
        .overlay(alignment: .topTrailing) {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.secondary)
                    .padding(16)
            }
        }
    }

    private func feature(_ icon: String, _ title: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.purple)
                .frame(width: 28)
            Text(title)
                .font(.body.weight(.medium))
            Spacer()
        }
    }
}
