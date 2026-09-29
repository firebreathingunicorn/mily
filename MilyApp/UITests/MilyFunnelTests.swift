import XCTest
import StoreKitTest

/// End-to-end smoke flow for Mily's free→Pro funnel, driven entirely through
/// the UI against the bundled StoreKit sandbox configuration:
///
/// demo burst → result + per-person chips → three free fixes used →
/// paywall → sandbox purchase of `mily_lifetime` → Pro unlocked.
final class MilyFunnelTests: XCTestCase {
    var storeSession: SKTestSession?

    override func setUpWithError() throws {
        continueAfterFailure = false
        // Talks to the same mily.storekit config bundled with the test target.
        storeSession = try SKTestSession(configurationFileNamed: "mily")
        storeSession?.disableDialogs = true
        storeSession?.clearTransactions()
    }

    override func tearDownWithError() throws {
        storeSession?.clearTransactions()
    }

    func testDemoBurstToProUnlockFunnel() throws {
        let app = XCUIApplication()
        app.launch()

        // Three free fixes: run the demo burst each time and confirm the
        // result screen reports both people.
        for run in 1...3 {
            runDemoBurst(app)
            XCTAssertTrue(app.staticTexts["Your best take"].waitForExistence(timeout: 120),
                          "result screen did not appear on run \(run)")

            XCTAssertTrue(app.staticTexts["Ada"].exists || app.staticTexts["Bo"].exists,
                          "per-person chips missing on run \(run)")
            XCTAssertTrue(app.buttons["Save to Photos"].waitForExistence(timeout: 10))

            app.navigationBars.buttons.firstMatch.tap()
            XCTAssertTrue(app.staticTexts["One tap. Everybody's best take."].waitForExistence(timeout: 10))
        }

        // Free quota is exhausted → next fix attempt must open the paywall.
        let demoButton = app.buttons["Try the demo burst"]
        XCTAssertTrue(demoButton.waitForExistence(timeout: 10))
        demoButton.tap()

        XCTAssertTrue(app.staticTexts["Mily Pro"].waitForExistence(timeout: 10),
                      "paywall did not appear after free fixes ran out")

        let purchaseButton = app.buttons["Unlock Mily Pro — $6.99 once"]
        XCTAssertTrue(purchaseButton.waitForExistence(timeout: 15),
                      "purchase button with sandbox price missing")
        purchaseButton.tap()

        // disableDialogs auto-confirms; Pro state should flip.
        XCTAssertTrue(app.staticTexts["You're Pro ✓"].waitForExistence(timeout: 20),
                      "purchase did not unlock Pro")

        app.buttons["xmark.circle.fill"].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["Mily Pro"].waitForExistence(timeout: 10),
                      "home screen should show the Pro badge after purchase")
    }

    private func runDemoBurst(_ app: XCUIApplication) {
        let button = app.buttons["Try the demo burst"]
        XCTAssertTrue(button.waitForExistence(timeout: 10))
        button.tap()
    }
}
