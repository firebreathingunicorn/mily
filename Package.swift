// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "Mily",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MilyCore", targets: ["MilyCore"]),
        .executable(name: "mily", targets: ["mily"]),
    ],
    targets: [
        .target(
            name: "MilyCore",
            path: "Sources/MilyCore"
        ),
        .executableTarget(
            name: "mily",
            dependencies: ["MilyCore"],
            path: "Sources/mily"
        ),
        .testTarget(
            name: "MilyCoreTests",
            dependencies: ["MilyCore"],
            // lowercase to match the real folder: this repo must also build
            // on case-sensitive volumes (Linux CI), where Tests/ != tests/
            path: "tests/MilyCoreTests"
        ),
    ]
)
