import Foundation
import SwiftUI

/// One update from `neuralsignal app stream`: spikes give a velocity, EEG gives class probabilities.
struct LiveSample: Decodable {
    let t: Double
    let compute_ms: Double
    let sent: Double
    var trial: Int?
    var held_out: Bool?
    var `true`: [Double]?
    var pred: [Double]?
    var probs: [Double]?
}

struct LiveSummary: Decodable {
    let updates: Int
    let compute_ms_median: Double
    let compute_ms_p95: Double
    let compute_ms_max: Double
}

/// Runs a decoder online through the Python helper and plays its output back at the
/// recording's own pace (or faster), the way a user would experience it.
@MainActor
final class LiveSession: ObservableObject {
    @Published var running = false
    @Published var shown: [LiveSample] = []       // samples whose time has come, most recent last
    @Published var clock: Double = 0              // recording time being shown
    @Published var computeMs: [Double] = []
    @Published var transportMs: [Double] = []
    @Published var summary: LiveSummary?
    @Published var error: String?
    var speed: Double = 1

    private var pending: [LiveSample] = []
    private var process: Process?
    private var timer: Timer?
    private var lastTick = Date()
    private var buffer = Data()

    func start(bridge: PythonBridge, path: String, model: String) {
        stop()
        shown = []; pending = []; computeMs = []; transportMs = []; summary = nil; error = nil; clock = 0
        let python = bridge.pythonPath.isEmpty ? bridge.projectFolder.appendingPathComponent(".venv/bin/python").path : bridge.pythonPath
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = ["-m", "neuralsignal", "app", "stream", "--path", path, "--model", model]
        p.currentDirectoryURL = bridge.projectFolder
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            let now = Date().timeIntervalSince1970
            Task { @MainActor in self?.ingest(data, receivedAt: now) }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                if proc.terminationStatus != 0, self?.summary == nil { self?.error = "Decoder stopped (exit \(proc.terminationStatus))." }
            }
        }
        do { try p.run() } catch { self.error = error.localizedDescription; return }
        process = p
        running = true
        lastTick = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 1 / 30, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        process?.terminate(); process = nil
        running = false
    }

    private func ingest(_ data: Data, receivedAt now: Double) {
        buffer.append(data)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let line = buffer.subdata(in: buffer.startIndex..<nl)
            buffer.removeSubrange(buffer.startIndex...nl)
            if let s = try? JSONDecoder().decode(LiveSample.self, from: line) {
                pending.append(s)
                computeMs.append(s.compute_ms)
                transportMs.append(max(0, (now - s.sent) * 1000))
                if clock == 0 { clock = s.t - 0.05 }
            } else if let sum = try? JSONDecoder().decode(LiveSummary.self, from: line) {
                summary = sum
            } else if let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any], let e = obj["error"] as? String {
                error = e
            }
        }
    }

    /// Advance the playback clock; jump over gaps between trials so playback never stalls.
    private func tick() {
        let now = Date()
        let dt = now.timeIntervalSince(lastTick)
        lastTick = now
        guard let next = pending.first else {
            if summary != nil { running = false; timer?.invalidate() }
            return
        }
        clock = max(clock + dt * speed, min(next.t - 0.05, clock + 1))
        var moved = false
        while let s = pending.first, s.t <= clock {
            shown.append(s); pending.removeFirst(); moved = true
        }
        if moved && shown.count > 4000 { shown.removeFirst(shown.count - 4000) }
    }

    static func percentile(_ x: [Double], _ q: Double) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted()
        return s[min(s.count - 1, Int(Double(s.count - 1) * q))]
    }
}
