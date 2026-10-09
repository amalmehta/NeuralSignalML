import SwiftUI

/// All preferences in one window (⌘,).
struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralSettings().tabItem { Label("General", systemImage: "gearshape") }
            DisplaySettings().tabItem { Label("Display", systemImage: "waveform") }
            SpectrogramSettings().tabItem { Label("Spectrogram", systemImage: "chart.bar.xaxis") }
        }
        .frame(width: 520)
        .padding(20)
    }
}

struct GeneralSettings: View {
    @EnvironmentObject var model: AppModel
    @AppStorage(SettingsKey.projectFolder) var projectFolder = Defaults.projectFolder
    @AppStorage(SettingsKey.pythonPath) var pythonPath = ""
    @State private var check: String?

    var body: some View {
        Form {
            Section {
                HStack {
                    TextField("Project folder", text: $projectFolder)
                    Button("Choose…") {
                        let p = NSOpenPanel()
                        p.canChooseDirectories = true; p.canChooseFiles = false
                        if p.runModal() == .OK, let u = p.url { projectFolder = u.path }
                    }
                }
                TextField("Python", text: $pythonPath, prompt: Text("<project folder>/.venv/bin/python"))
            } footer: {
                Text("The app runs the Python pipeline in this folder to read recordings and run decoders. Leave Python empty to use the project's .venv.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Test Connection") {
                    Task {
                        check = "Checking…"
                        do {
                            let recs = try await model.bridge.recordings()
                            let ms = try await model.bridge.models()
                            check = "OK: \(recs.count) recordings, \(ms.count) decoders."
                            await model.refresh()
                        } catch { check = error.localizedDescription }
                    }
                }
                if let c = check { Text(c).font(.caption).foregroundStyle(.secondary).lineLimit(3) }
            }
        }
    }
}

struct DisplaySettings: View {
    @EnvironmentObject var model: AppModel
    @AppStorage(SettingsKey.windowSeconds) var windowSeconds = 10.0
    @AppStorage(SettingsKey.filterLow) var low = 8.0
    @AppStorage(SettingsKey.filterHigh) var high = 30.0
    @AppStorage(SettingsKey.notchOn) var notchOn = false
    @AppStorage(SettingsKey.notchHz) var notchHz = 50.0
    @AppStorage(SettingsKey.zeroPhase) var zeroPhase = false
    @AppStorage(SettingsKey.showEOG) var showEOG = true
    @AppStorage(SettingsKey.rateSmoothingMs) var smoothing = 50.0

    var body: some View {
        Form {
            Picker("Seconds shown", selection: $windowSeconds) {
                ForEach([2.0, 5.0, 10.0, 20.0, 60.0], id: \.self) { Text("\(Int($0)) s").tag($0) }
            }
            Section("EEG filter") {
                HStack {
                    TextField("Low cut (Hz)", value: $low, format: .number)
                    TextField("High cut (Hz)", value: $high, format: .number)
                }
                Toggle("Mains notch", isOn: $notchOn)
                Picker("Notch frequency", selection: $notchHz) {
                    Text("50 Hz (Europe; BCI 2a was recorded in Graz)").tag(50.0)
                    Text("60 Hz (Americas)").tag(60.0)
                }
                .disabled(!notchOn)
                Toggle("Zero-phase filtering", isOn: $zeroPhase)
                Text("Off matches the pipeline: a causal filter, the kind an online BCI must use. On runs it forwards and backwards so peaks line up with the raw trace, for display only.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Show EOG channels", isOn: $showEOG)
            }
            Section("Spikes") {
                Stepper(value: $smoothing, in: 10...200, step: 10) { Text("Rate smoothing σ: \(Int(smoothing)) ms") }
            }
        }
        .onChange(of: low) { model.refilter() }
        .onChange(of: high) { model.refilter() }
        .onChange(of: notchOn) { model.refilter() }
        .onChange(of: notchHz) { model.refilter() }
        .onChange(of: zeroPhase) { model.refilter() }
        .onChange(of: smoothing) { model.refilter() }
    }
}

struct SpectrogramSettings: View {
    @AppStorage(SettingsKey.spectrogramWindow) var fftWindow = 256
    @AppStorage(SettingsKey.spectrogramMaxHz) var maxHz = 45.0

    var body: some View {
        Form {
            Picker("FFT window", selection: $fftWindow) {
                ForEach([64, 128, 256, 512], id: \.self) { Text("\($0) samples").tag($0) }
            }
            Text("Longer windows give finer frequency resolution and blurrier timing. At 250 Hz, 256 samples ≈ 1 s ≈ 1 Hz resolution.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Highest frequency shown", selection: $maxHz) {
                ForEach([30.0, 45.0, 60.0, 100.0, 125.0], id: \.self) { Text("\(Int($0)) Hz").tag($0) }
            }
        }
    }
}
