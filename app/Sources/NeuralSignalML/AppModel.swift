import SignalCore
import SwiftUI

/// App state: the recording list, the loaded recording, its filtered copy, and decoder results.
@MainActor
final class AppModel: ObservableObject {
    @Published var recordings: [RecordingInfo] = []
    @Published var openedFiles: [RecordingInfo] = []
    @Published var selection: RecordingInfo.ID?
    @Published var loaded: LoadedRecording?
    @Published var filtered: [[Float]] = []
    @Published var rates: RateMatrix?
    @Published var status: String?
    @Published var error: String?
    @Published var models: [DecoderInfo] = []
    @Published var decodeResult: DecodeResult?
    @Published var decoding = false
    @Published var viewStart: Double = 0
    @Published var spectrogramChannel = 7  // C3 for EEG

    private var filterTask: Task<Void, Never>?

    var bridge: PythonBridge {
        let d = UserDefaults.standard
        return PythonBridge(projectFolder: URL(fileURLWithPath: d.string(forKey: SettingsKey.projectFolder) ?? Defaults.projectFolder),
                            pythonPath: d.string(forKey: SettingsKey.pythonPath) ?? "")
    }

    var filterSpec: FilterSpec {
        let d = UserDefaults.standard
        return FilterSpec(low: d.double(forKey: SettingsKey.filterLow), high: d.double(forKey: SettingsKey.filterHigh),
                          notch: d.bool(forKey: SettingsKey.notchOn) ? d.double(forKey: SettingsKey.notchHz) : nil,
                          zeroPhase: d.bool(forKey: SettingsKey.zeroPhase))
    }

    var allRecordings: [RecordingInfo] { recordings + openedFiles }

    func refresh() async {
        status = "Looking for recordings…"
        error = nil
        do {
            recordings = try await bridge.recordings()
            models = try await bridge.models()
            status = nil
            if selection == nil, let first = recordings.first { await select(first.id) }
        } catch {
            status = nil
            self.error = error.localizedDescription
        }
    }

    func open(url: URL) async {
        let info = RecordingInfo.fromFile(url)
        if !allRecordings.contains(info) { openedFiles.append(info) }
        await select(info.id)
    }

    func select(_ id: RecordingInfo.ID?) async {
        selection = id
        guard let id, let info = allRecordings.first(where: { $0.id == id }) else { return }
        if loaded?.info.id == id { return }
        loaded = nil
        filtered = []
        rates = nil
        decodeResult = nil
        error = nil
        status = "Loading \(info.title)…"
        do {
            let rec = try await bridge.load(info)
            loaded = rec
            viewStart = rec.meta.events.first.map { max(0, $0.time_s - 3) } ?? 0
            spectrogramChannel = rec.isSpikes ? 0 : min(7, rec.meta.n_channels - 1)
            status = nil
            refilter()
        } catch {
            status = nil
            self.error = error.localizedDescription
        }
    }

    /// Re-run the filter (EEG) or rate smoothing (spikes) off the main thread.
    func refilter() {
        guard let rec = loaded else { return }
        filterTask?.cancel()
        let spec = filterSpec
        let sigmaMs = UserDefaults.standard.double(forKey: SettingsKey.rateSmoothingMs)
        filterTask = Task {
            if rec.isSpikes {
                let r = await Task.detached(priority: .userInitiated) {
                    RateMatrix(spikes: rec.spikeTimes, duration: rec.duration, binS: 0.01, sigmaMs: sigmaMs)
                }.value
                if !Task.isCancelled { rates = r; filtered = rec.channels }
            } else {
                let out = await Task.detached(priority: .userInitiated) {
                    rec.channels.map { SignalFilter.apply($0, fs: rec.meta.fs, spec: spec) }
                }.value
                if !Task.isCancelled { filtered = out }
            }
        }
    }

    func modelsForLoaded() -> [DecoderInfo] {
        guard let rec = loaded else { return [] }
        let subject = rec.meta.id.prefix(3)  // "A01"
        return models.filter { $0.kind == rec.meta.kind }.sorted { a, b in
            // Models whose held-out target is this subject first, then by name.
            let am = a.target == subject, bm = b.target == subject
            return am != bm ? am : (a.name, a.protocol_, a.target) < (b.name, b.protocol_, b.target)
        }
    }

    func decode(with model: DecoderInfo) async {
        guard let rec = loaded else { return }
        decoding = true
        error = nil
        defer { decoding = false }
        do {
            decodeResult = try await bridge.decode(path: rec.info.path, model: model.path)
        } catch {
            self.error = error.localizedDescription
        }
    }
}

/// Smoothed firing rates (Hz) for every unit on a regular 10 ms grid - the spike "filtered" view.
struct RateMatrix: Sendable {
    let binS: Double
    let nBins: Int
    let rates: [[Float]]   // per unit

    init(spikes: [[Float]], duration: Double, binS: Double, sigmaMs: Double) {
        self.binS = binS
        nBins = Int(duration / binS) + 1
        let sigma = max(sigmaMs / 1000 / binS, 0.5)
        let half = Int(ceil(3 * sigma))
        var kernel = (-half...half).map { Float(exp(-0.5 * pow(Double($0) / sigma, 2))) }
        let ksum = kernel.reduce(0, +)
        kernel = kernel.map { $0 / ksum / Float(binS) }
        let n = nBins
        rates = spikes.map { times in
            var counts = [Float](repeating: 0, count: n)
            for t in times { let i = Int(Double(t) / binS); if i >= 0 && i < n { counts[i] += 1 } }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n where counts[i] > 0 {
                for (k, w) in kernel.enumerated() {
                    let j = i + k - half
                    if j >= 0 && j < n { out[j] += counts[i] * w }
                }
            }
            return out
        }
    }
}
