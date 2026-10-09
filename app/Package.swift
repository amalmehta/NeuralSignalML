// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "NeuralSignalML",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "NeuralSignalML", targets: ["NeuralSignalML"]),
    ],
    targets: [
        // Pure DSP (filters, spectrograms, display decimation) - no UI, unit tested.
        .target(name: "SignalCore"),
        .executableTarget(name: "NeuralSignalML", dependencies: ["SignalCore"]),
        .testTarget(name: "SignalCoreTests", dependencies: ["SignalCore"]),
    ]
)
