import SignalCore
import SwiftUI

/// Time-frequency view of one channel over the visible window (plus padding so the edges are valid).
struct SpectrogramView: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    @AppStorage(SettingsKey.windowSeconds) var windowSeconds = 10.0
    @AppStorage(SettingsKey.spectrogramWindow) var fftWindow = 256
    @AppStorage(SettingsKey.spectrogramMaxHz) var maxHz = 45.0
    @State private var useFiltered = false
    @State private var image: CGImage?
    @State private var range: (Float, Float) = (0, 1)
    @State private var spec: Spectrogram?

    var channelNames: [String] {
        rec.isSpikes ? rec.meta.channels + (0..<(rec.meta.n_units ?? 0)).map { "unit \($0) firing rate" } : rec.meta.channels
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 14) {
                Picker("Channel", selection: $model.spectrogramChannel) {
                    ForEach(Array(channelNames.enumerated()), id: \.offset) { i, n in Text(n).tag(i) }
                }
                .frame(width: 240)
                if !rec.isSpikes { Toggle("Use filtered signal", isOn: $useFiltered).toggleStyle(.checkbox) }
                Picker("FFT window", selection: $fftWindow) {
                    ForEach([64, 128, 256, 512], id: \.self) { Text("\($0) samples").tag($0) }
                }
                .frame(width: 200)
                Spacer()
                Text(resolutionText).font(.caption).foregroundStyle(.secondary)
            }
            GeometryReader { geo in
                HStack(spacing: 4) {
                    freqAxis(height: geo.size.height - 18)
                    VStack(spacing: 2) {
                        ZStack {
                            if let img = image {
                                Image(decorative: img, scale: 1).resizable().interpolation(.medium)
                            } else {
                                Color(nsColor: .textBackgroundColor)
                                Text("Signal too short for this window").foregroundStyle(.secondary)
                            }
                            cueOverlay
                        }
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        HStack {
                            Text(String(format: "%.1f s", model.viewStart)); Spacer()
                            Text(String(format: "%.1f s", model.viewStart + windowSeconds))
                        }
                        .font(.caption2).foregroundStyle(.secondary).frame(height: 14)
                    }
                    colorbar(height: geo.size.height - 18)
                }
            }
            Text(rec.isSpikes
                 ? "Spike-rate and velocity spectra show how fast activity changes. Most movement-related power is below 5 Hz."
                 : "Look for the 8–13 Hz mu rhythm over C3/C4 dropping a second or so after a cue (dashed lines). That drop is what CSP picks up.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .task(id: key) { await recompute() }
    }

    var key: String { "\(model.viewStart)-\(windowSeconds)-\(fftWindow)-\(model.spectrogramChannel)-\(useFiltered)-\(maxHz)-\(model.filtered.count)-\(model.rates != nil)" }

    var resolutionText: String {
        guard let s = spec else { return "" }
        return String(format: "frequency resolution %.2f Hz, time step %.0f ms", s.freqResolution, s.frameStep * 1000)
    }

    var cueOverlay: some View {
        GeometryReader { geo in
            ForEach(rec.meta.events.filter { $0.time_s > model.viewStart && $0.time_s < model.viewStart + windowSeconds }, id: \.time_s) { e in
                Path { p in
                    let x = CGFloat((e.time_s - model.viewStart) / windowSeconds) * geo.size.width
                    p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: geo.size.height))
                }
                .stroke(.white.opacity(0.8), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
        }
    }

    func freqAxis(height: CGFloat) -> some View {
        let shownMax = rec.isSpikes ? min(maxHz, 20) : maxHz
        return VStack {
            ForEach(Array(stride(from: shownMax, through: 0, by: -shownMax / 5)), id: \.self) { f in
                Text("\(Int(f)) Hz").font(.caption2).foregroundStyle(.secondary)
                if f > 0 { Spacer() }
            }
        }
        .frame(width: 42, height: height)
    }

    func colorbar(height: CGFloat) -> some View {
        VStack(spacing: 2) {
            Text(String(format: "%.0f dB", range.1)).font(.caption2)
            LinearGradient(colors: (0...6).reversed().map { Colormap.magma(Double($0) / 6) }, startPoint: .top, endPoint: .bottom)
                .frame(width: 12)
            Text(String(format: "%.0f dB", range.0)).font(.caption2)
        }
        .foregroundStyle(.secondary)
        .frame(width: 44, height: height)
    }

    /// Pull the samples for the selected channel, pad so the first/last frames cover the window, run the STFT.
    func recompute() async {
        let c = model.spectrogramChannel
        let fs: Double
        let source: [Float]
        if rec.isSpikes && c >= 2 {
            guard let r = model.rates, c - 2 < r.rates.count else { return }
            fs = 1 / r.binS
            source = r.rates[c - 2]
        } else {
            fs = rec.meta.fs
            source = (!rec.isSpikes && useFiltered && c < model.filtered.count) ? model.filtered[c] : rec.channels[min(c, rec.channels.count - 1)]
        }
        let n = 1 << Int(ceil(log2(Double(fftWindow))))
        let pad = n / 2
        let i0 = max(0, Int(model.viewStart * fs) - pad), i1 = min(source.count, Int((model.viewStart + windowSeconds) * fs) + pad)
        guard i1 - i0 > n else { image = nil; return }
        let slice = Array(source[i0..<i1])
        let hop = max(1, (i1 - i0 - n) / 600)
        let start = Double(i0) / fs
        let shownMax = rec.isSpikes ? min(maxHz, 20) : maxHz
        let result = await Task.detached(priority: .userInitiated) { () -> (Spectrogram, CGImage?, (Float, Float))? in
            guard let s = Spectrogram.compute(slice, fs: fs, window: n, hop: hop, startTime: start) else { return nil }
            let rows = min(s.nFreqs, Int(shownMax / s.freqResolution) + 1)
            let vals = (0..<rows).flatMap { f in (0..<s.nFrames).map { s.value(freqIndex: f, frame: $0) } }.sorted()
            let lo = vals[vals.count / 50], hi = vals[vals.count * 99 / 100]
            return (s, makeImage(s, rows: rows, lo: lo, hi: hi), (lo, hi))
        }.value
        guard let (s, img, r) = result else { image = nil; return }
        spec = s; image = img; range = r
    }
}

/// Render rows 0..<rows (low frequency at the bottom) through the magma colormap.
private func makeImage(_ s: Spectrogram, rows: Int, lo: Float, hi: Float) -> CGImage? {
    let w = s.nFrames, h = rows
    var px = [UInt8](repeating: 255, count: w * h * 4)
    for f in 0..<h {
        for t in 0..<w {
            let v = Double((s.value(freqIndex: f, frame: t) - lo) / max(hi - lo, 1e-6))
            let (r, g, b) = Colormap.rgb(v)
            let i = ((h - 1 - f) * w + t) * 4
            px[i] = UInt8(r * 255); px[i + 1] = UInt8(g * 255); px[i + 2] = UInt8(b * 255)
        }
    }
    guard let provider = CGDataProvider(data: Data(px) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                   space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
}
