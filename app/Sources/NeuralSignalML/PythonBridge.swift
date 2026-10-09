import Foundation

/// Runs `python -m neuralsignal app <command>` in the project folder and decodes its one-line JSON reply.
/// Python only reads file formats and runs trained models; all drawing, filtering and
/// spectrograms happen natively in the app.
struct PythonBridge {
    var projectFolder: URL
    var pythonPath: String

    struct BridgeError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    func run(_ args: [String]) async throws -> Data {
        let python = pythonPath.isEmpty ? projectFolder.appendingPathComponent(".venv/bin/python").path : pythonPath
        guard FileManager.default.isExecutableFile(atPath: python) else {
            throw BridgeError(message: "Python not found at \(python). Set the project folder in Settings (⌘,) and run the setup in docs/INSTRUCTIONS.md.")
        }
        let folder = projectFolder
        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let p = Process()
                p.executableURL = URL(fileURLWithPath: python)
                p.arguments = ["-m", "neuralsignal", "app"] + args
                p.currentDirectoryURL = folder
                var env = ProcessInfo.processInfo.environment
                env["PYTHONPATH"] = folder.path
                p.environment = env
                let out = Pipe(), err = Pipe()
                p.standardOutput = out
                p.standardError = err
                do {
                    try p.run()
                    let data = out.fileHandleForReading.readDataToEndOfFile()
                    let errData = err.fileHandleForReading.readDataToEndOfFile()
                    p.waitUntilExit()
                    if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let msg = obj["error"] as? String {
                        cont.resume(throwing: BridgeError(message: msg))
                    } else if p.terminationStatus != 0 || data.isEmpty {
                        let tail = String(decoding: errData.suffix(800), as: UTF8.self)
                        cont.resume(throwing: BridgeError(message: "Python helper failed (exit \(p.terminationStatus)).\n\(tail)"))
                    } else {
                        cont.resume(returning: data)
                    }
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    func recordings() async throws -> [RecordingInfo] {
        struct R: Decodable { let recordings: [RecordingInfo] }
        return try JSONDecoder().decode(R.self, from: try await run(["recordings"])).recordings
    }

    func models() async throws -> [DecoderInfo] {
        struct R: Decodable { let models: [DecoderInfo] }
        return try JSONDecoder().decode(R.self, from: try await run(["models"])).models
    }

    func decode(path: String, model: String) async throws -> DecodeResult {
        try JSONDecoder().decode(DecodeResult.self, from: try await run(["decode", "--path", path, "--model", model]))
    }

    /// Export to a cache folder (re-used while the source file is unchanged) and read it back.
    func load(_ info: RecordingInfo) async throws -> LoadedRecording {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Neural Signal ML/\(info.id)")
        let metaURL = caches.appendingPathComponent("meta.json")
        let srcDate = (try? FileManager.default.attributesOfItem(atPath: info.path)[.modificationDate] as? Date) ?? .distantPast
        let cacheDate = (try? FileManager.default.attributesOfItem(atPath: metaURL.path)[.modificationDate] as? Date) ?? .distantPast
        if cacheDate < srcDate {
            _ = try await run(["export", "--path", info.path, "--out", caches.path])
        }
        let meta = try JSONDecoder().decode(RecordingMeta.self, from: Data(contentsOf: metaURL))
        let flat = try readFloats(caches.appendingPathComponent("signals.f32"))
        let channels = (0..<meta.n_channels).map { c in Array(flat[(c * meta.n_samples)..<((c + 1) * meta.n_samples)]) }
        var spikes: [[Float]] = []
        if let counts = meta.spike_counts {
            let all = try readFloats(caches.appendingPathComponent("spikes.f32"))
            var i = 0
            for n in counts { spikes.append(Array(all[i..<(i + n)])); i += n }
        }
        return LoadedRecording(info: info, meta: meta, channels: channels, spikeTimes: spikes)
    }

    private func readFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

extension RecordingInfo {
    /// A recording opened from a file rather than picked from the data folder.
    static func fromFile(_ url: URL) -> RecordingInfo {
        let kind = url.pathExtension.lowercased() == "nwb" ? "spikes" : "eeg"
        let stem = url.deletingPathExtension().lastPathComponent
        return RecordingInfo(id: kind == "spikes" ? "MC_Maze_Medium" : stem, kind: kind,
                             title: url.lastPathComponent, group: "Opened files", path: url.path)
    }
}
