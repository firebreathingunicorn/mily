import SwiftUI

@main
struct MilyApp: App {
    @StateObject private var store = MilyStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(store)
                // Thin, blocky type everywhere — echoes the pixel-cat mascot.
                .fontDesign(.monospaced)
                .fontWeight(.light)
        }
    }
}
