import Foundation
import RevenueCat
import StoreKit

/// Purchase layer for Mily.
///
/// Two modes:
/// - **RevenueCat** (default for a shipped build): `purchases-ios` powers the
///   lifetime non-consumable → `pro` entitlement, with RevenueCatUI's paywall
///   for the purchase UI. Drop a real `appl_…` key into
///   `Constants.revenueCatAPIKey` (or the `RevenueCatAPIKey` Info.plist key)
///   to enable it.
/// - **Local StoreKit** (fallback): when no RevenueCat key is configured —
///   e.g. judges cloning this repo — the same product is purchased directly
///   through StoreKit 2 against the bundled `mily.storekit` configuration,
///   so the full flow (paywall → sandbox purchase → Pro unlocked) works in
///   the simulator with zero setup.
@MainActor
final class MilyStore: ObservableObject {
    static let lifetimeProductID = "mily_lifetime"
    static let proEntitlementID = "pro"
    static let freeFixLimit = 3

    enum StoreMode {
        case revenueCat
        case localStoreKit
    }

    let mode: StoreMode

    @Published private(set) var isPro = false
    @Published private(set) var fixesUsed: Int
    @Published private(set) var localPrice: String?
    @Published private(set) var storeError: String?

    private var updatesTask: Task<Void, Never>?

    nonisolated static var apiKey: String {
        let fromPlist = Bundle.main.object(forInfoDictionaryKey: "RevenueCatAPIKey") as? String
        let key = fromPlist?.isEmpty == false ? fromPlist! : Constants.revenueCatAPIKey
        let isPlaceholder = key.isEmpty || key.contains("REPLACE")
        return isPlaceholder ? "" : key
    }

    init() {
        fixesUsed = UserDefaults.standard.integer(forKey: "mily.fixesUsed")

        if !Self.apiKey.isEmpty {
            mode = .revenueCat
            let config = Configuration.Builder(withAPIKey: Self.apiKey)
                .with(usesStoreKit2IfAvailable: true)
                .build()
            Purchases.logLevel = .warn
            Purchases.configure(with: config)
            updatesTask = Task { [weak self] in
                for await info in Purchases.shared.customerInfoStream {
                    await self?.apply(customerInfo: info)
                }
            }
        } else {
            mode = .localStoreKit
            updatesTask = Task { [weak self] in
                for await update in Transaction.updates {
                    if case .verified(let transaction) = update {
                        await transaction.finish()
                    }
                    await self?.refreshEntitlements()
                }
            }
        }

        Task { await refreshEntitlements() }
        Task { await loadLocalPrice() }
    }

    deinit {
        updatesTask?.cancel()
    }

    var freeFixesLeft: Int {
        max(Self.freeFixLimit - fixesUsed, 0)
    }

    var canFixNow: Bool {
        true  // fixing is always free; the purchase is an optional tip
    }

    /// Records one completed fix (used by the free quota; Pro is unlimited).
    func recordFix() {
        fixesUsed += 1
        UserDefaults.standard.set(fixesUsed, forKey: "mily.fixesUsed")
    }

    func purchasePro() async {
        storeError = nil
        switch mode {
        case .revenueCat:
            // Purchases are driven by RevenueCatUI's PaywallView; this path
            // covers a direct purchase from the fallback UI.
            do {
                let products = try await Purchases.shared.products([Self.lifetimeProductID])
                guard let product = products.first else {
                    storeError = "Product unavailable — check the RevenueCat offering."
                    return
                }
                let result = try await Purchases.shared.purchase(product: product)
                await apply(customerInfo: result.customerInfo)
            } catch let error as ErrorCode where error == .purchaseCancelledError {
                break
            } catch {
                storeError = error.localizedDescription
            }
        case .localStoreKit:
            await purchaseLocally()
        }
    }

    func restorePurchases() async {
        storeError = nil
        switch mode {
        case .revenueCat:
            do {
                let info = try await Purchases.shared.restorePurchases()
                await apply(customerInfo: info)
            } catch {
                storeError = error.localizedDescription
            }
        case .localStoreKit:
            do {
                try await AppStore.sync()
                await refreshEntitlements()
            } catch {
                storeError = error.localizedDescription
            }
        }
    }

    // MARK: - StoreKit 2 (local mode)

    private func purchaseLocally() async {
        do {
            guard let product = try await Product.products(for: [Self.lifetimeProductID]).first else {
                storeError = "mily_lifetime is missing from the StoreKit configuration."
                return
            }
            switch try await product.purchase() {
            case .success(let verification):
                if case .verified(let transaction) = verification {
                    await transaction.finish()
                    await refreshEntitlements()
                } else {
                    storeError = "Purchase could not be verified."
                }
            case .userCancelled:
                break
            case .pending:
                storeError = "Purchase is pending approval."
            @unknown default:
                break
            }
        } catch {
            if let storeKitError = error as? StoreKitError, case .userCancelled = storeKitError {
                return
            }
            storeError = error.localizedDescription
        }
    }

    private func refreshEntitlements() async {
        switch mode {
        case .revenueCat:
            do {
                await apply(customerInfo: try await Purchases.shared.customerInfo())
            } catch {
                // Offline: keep the last known entitlement state.
            }
        case .localStoreKit:
            var owned = false
            for await entitlement in Transaction.currentEntitlements {
                if case .verified(let transaction) = entitlement,
                   transaction.productID == Self.lifetimeProductID,
                   transaction.revocationDate == nil {
                    owned = true
                }
            }
            isPro = owned
        }
    }

    // MARK: - RevenueCat

    private func apply(customerInfo: CustomerInfo) {
        isPro = customerInfo.entitlements[Self.proEntitlementID]?.isActive == true
    }

    private func loadLocalPrice() async {
        guard localPrice == nil else { return }
        do {
            let products = try await Product.products(for: [Self.lifetimeProductID])
            localPrice = products.first?.displayPrice
        } catch {
            localPrice = nil
        }
    }
}

enum Constants {
    /// Paste your RevenueCat **Apple** API key (appl_…) here, or set the
    /// `RevenueCatAPIKey` row in MilyApp/Support/Info.plist. While this is a
    /// placeholder the app runs in local StoreKit mode against
    /// `mily.storekit` (sandbox, no RevenueCat project needed).
    static let revenueCatAPIKey = "appl_REPLACE_WITH_YOUR_REVENUECAT_KEY"
}
