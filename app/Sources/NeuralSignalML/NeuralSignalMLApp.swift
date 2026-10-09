import SwiftUI
import UniformTypeIdentifiers

@main
struct NeuralSignalMLApp: App {
    @StateObject private var model = AppModel()
    @State private var showFeedback = false

    init() {
        Defaults.register()
        // When launched as a bare executable (swift run), still behave like a regular app.
        NSApplication.shared.setActivationPolicy(.regular)
    }

    var body: some Scene {
        WindowGroup("Neural Signal ML") {
            ContentView(showFeedback: $showFeedback)
                .environmentObject(model)
                .frame(minWidth: 1000, minHeight: 640)
                .task { await model.refresh() }
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Recording…") { openPanel() }.keyboardShortcut("o")
                Button("Reload Recordings and Models") { Task { await model.refresh() } }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
            }
            CommandGroup(after: .help) {
                Button("Send Feedback…") { showFeedback = true }
            }
        }

        Settings {
            SettingsView()
                .environmentObject(model)
        }
    }

    @MainActor private func openPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open a recording"
        panel.message = "BCI Competition IV 2a (.mat) or Neural Latents Benchmark (.nwb)"
        panel.allowedContentTypes = ["mat", "nwb"].compactMap { UTType(filenameExtension: $0) }
        panel.allowsOtherFileTypes = true
        if panel.runModal() == .OK, let url = panel.url {
            Task { await model.open(url: url) }
        }
    }
}
