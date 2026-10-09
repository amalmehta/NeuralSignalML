import SignalCore
import SwiftUI

struct SignalsView: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    @AppStorage(SettingsKey.showRaw) var showRaw = true
    @AppStorage(SettingsKey.showFiltered) var showFiltered = true
    @AppStorage(SettingsKey.filterLow) var low = 8.0
    @AppStorage(SettingsKey.filterHigh) var high = 30.0
    @AppStorage(SettingsKey.showEOG) var showEOG = true
    @AppStorage(SettingsKey.rateSmoothingMs) var smoothing = 50.0
    @State private var gain = 1.0

    var body: some View {
        VStack(spacing: 6) {
            controls
            if rec.isSpikes {
                SpikeSignals(rec: rec, showRaw: showRaw, showRates: showFiltered)
            } else {
                EEGTraces(rec: rec, showRaw: showRaw, showFiltered: showFiltered, showEOG: showEOG, gain: gain)
            }
        }
    }

    var controls: some View {
        HStack(spacing: 14) {
            Toggle(rec.isSpikes ? "Spike raster" : "Raw", isOn: $showRaw)
            Toggle(rec.isSpikes ? "Smoothed rates" : "Filtered", isOn: $showFiltered)
            if rec.isSpikes {
                Stepper(value: $smoothing, in: 10...200, step: 10) { Text("Smoothing σ \(Int(smoothing)) ms").monospacedDigit() }
                    .onChange(of: smoothing) { model.refilter() }
            } else {
                HStack(spacing: 4) {
                    Text("Band")
                    TextField("", value: $low, format: .number).frame(width: 40)
                    Text("–")
                    TextField("", value: $high, format: .number).frame(width: 40)
                    Text("Hz")
                }
                .onSubmit { model.refilter() }
                .onChange(of: low) { model.refilter() }
                .onChange(of: high) { model.refilter() }
                Menu("Presets") {
                    Button("Mu + beta, 8–30 Hz (CSP)") { low = 8; high = 30 }
                    Button("Wide, 4–40 Hz (EEGNet)") { low = 4; high = 40 }
                    Button("Mu, 8–13 Hz") { low = 8; high = 13 }
                    Button("Beta, 13–30 Hz") { low = 13; high = 30 }
                    Button("Broadband, 0.5–45 Hz") { low = 0.5; high = 45 }
                }
                .frame(width: 90)
                Toggle("EOG", isOn: $showEOG)
                Slider(value: $gain, in: 0.25...4) { Text("Gain") }.frame(width: 150)
            }
            Spacer()
            if model.filtered.isEmpty { ProgressView().controlSize(.small) }
        }
        .toggleStyle(.checkbox)
        .font(.callout)
    }
}

// MARK: - EEG

struct EEGTraces: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    let showRaw: Bool, showFiltered: Bool, showEOG: Bool
    let gain: Double
    @AppStorage(SettingsKey.windowSeconds) var windowSeconds = 10.0

    var body: some View {
        let nCh = showEOG ? rec.meta.n_channels : 22
        let fs = rec.meta.fs
        let t0 = model.viewStart, t1 = model.viewStart + windowSeconds
        let i0 = max(0, Int(t0 * fs)), i1 = min(rec.meta.n_samples, Int(t1 * fs))
        let decoded = model.decodeResult?.kind == "eeg" ? model.decodeResult : nil
        GeometryReader { geo in
            let labelW: CGFloat = 54
            let plotW = geo.size.width - labelW
            let rowH = geo.size.height / CGFloat(nCh)
            let x = { (t: Double) in labelW + CGFloat((t - t0) / (t1 - t0)) * plotW }
            ZStack(alignment: .topLeading) {
                Canvas { ctx, size in
                    // Trial markers: cue line and the shaded epoch the decoders use (cue + 0.5 to 2.5 s).
                    for e in rec.meta.events where e.time_s > t0 - 3 && e.time_s < t1 {
                        let band = CGRect(x: x(e.time_s + 0.5), y: 0, width: x(e.time_s + 2.5) - x(e.time_s + 0.5), height: size.height)
                        ctx.fill(Path(band), with: .color(.green.opacity(0.06)))
                        var line = Path(); line.move(to: CGPoint(x: x(e.time_s), y: 0)); line.addLine(to: CGPoint(x: x(e.time_s), y: size.height))
                        ctx.stroke(line, with: .color(e.artifact == true ? .red.opacity(0.6) : .secondary.opacity(0.6)), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    }
                    for r in rec.meta.runs where r.start_s > t0 && r.start_s < t1 {
                        var line = Path(); line.move(to: CGPoint(x: x(r.start_s), y: 0)); line.addLine(to: CGPoint(x: x(r.start_s), y: size.height))
                        ctx.stroke(line, with: .color(.blue.opacity(0.5)), lineWidth: 2)
                    }
                    guard i1 > i0 + 1 else { return }
                    let buckets = Int(plotW)
                    for c in 0..<nCh {
                        let mid = rowH * (CGFloat(c) + 0.5)
                        if showRaw {
                            draw(ctx, rec.channels[c][i0..<i1], mid: mid, rowH: rowH, buckets: buckets, labelW: labelW, plotW: plotW,
                                 color: .secondary.opacity(showFiltered ? 0.45 : 0.9))
                        }
                        if showFiltered, c < model.filtered.count {
                            draw(ctx, model.filtered[c][i0..<i1], mid: mid, rowH: rowH, buckets: buckets, labelW: labelW, plotW: plotW,
                                 color: c >= 22 ? .purple : .accentColor)
                        }
                    }
                }
                ForEach(0..<nCh, id: \.self) { c in
                    Text(rec.meta.channels[c]).font(.system(size: max(8, min(11, rowH * 0.6)))).foregroundStyle(c >= 22 ? .purple : .primary)
                        .frame(width: labelW - 6, alignment: .trailing)
                        .position(x: (labelW - 6) / 2, y: rowH * (CGFloat(c) + 0.5))
                }
                ForEach(Array(rec.meta.events.enumerated()).filter { $0.element.time_s > t0 - 1 && $0.element.time_s < t1 }, id: \.offset) { i, e in
                    eventLabel(e, decoded: decoded?.trials?.first { abs($0.time_s - e.time_s) < 0.01 }, classes: decoded?.classes)
                        .position(x: x(e.time_s) + 52, y: 10)
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder
    func eventLabel(_ e: RecordingEvent, decoded: EEGTrialResult?, classes: [String]?) -> some View {
        HStack(spacing: 3) {
            if let d = decoded, let cls = classes {
                Image(systemName: d.pred == d.true ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(d.pred == d.true ? .green : .red)
                Text(d.pred == d.true ? e.label : "\(e.label) (said \(cls[d.pred]))")
            } else {
                Text(e.label)
            }
        }
        .font(.caption2.weight(.medium))
        .padding(.horizontal, 5).padding(.vertical, 2)
        .background(.regularMaterial, in: Capsule())
        .fixedSize()
    }

    func draw(_ ctx: GraphicsContext, _ x: ArraySlice<Float>, mid: CGFloat, rowH: CGFloat, buckets: Int,
              labelW: CGFloat, plotW: CGFloat, color: Color) {
        let scale = Decimate.robustScale(x)
        let mean = x.reduce(0, +) / Float(x.count)
        let k = rowH * 0.18 * CGFloat(gain) / CGFloat(scale)
        let env = Decimate.minMax(x, buckets: buckets)
        var p = Path()
        for (i, (lo, hi)) in env.enumerated() {
            let px = labelW + CGFloat(i) / CGFloat(max(env.count - 1, 1)) * plotW
            let y1 = mid - CGFloat(hi - mean) * k, y2 = mid - CGFloat(lo - mean) * k
            if i == 0 { p.move(to: CGPoint(x: px, y: y1)) } else { p.addLine(to: CGPoint(x: px, y: y1)) }
            if y2 != y1 { p.addLine(to: CGPoint(x: px, y: y2)) }
        }
        ctx.stroke(p, with: .color(color), lineWidth: 0.8)
    }
}

// MARK: - Spikes

struct SpikeSignals: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    let showRaw: Bool, showRates: Bool
    @AppStorage(SettingsKey.windowSeconds) var windowSeconds = 10.0

    var body: some View {
        let t0 = model.viewStart, t1 = model.viewStart + windowSeconds
        VStack(spacing: 6) {
            if showRaw {
                panel("Spikes, \(rec.spikeTimes.count) units (each row one neuron)") { ctx, size in
                    markers(ctx, size, t0, t1)
                    let rowH = size.height / CGFloat(rec.spikeTimes.count)
                    for (u, times) in rec.spikeTimes.enumerated() {
                        let lo = lowerBound(times, Float(t0)), hi = lowerBound(times, Float(t1))
                        guard hi > lo else { continue }
                        var p = Path()
                        for t in times[lo..<hi] {
                            let x = CGFloat((Double(t) - t0) / (t1 - t0)) * size.width
                            p.move(to: CGPoint(x: x, y: CGFloat(u) * rowH)); p.addLine(to: CGPoint(x: x, y: CGFloat(u + 1) * rowH))
                        }
                        ctx.stroke(p, with: .color(.primary.opacity(0.8)), lineWidth: 0.7)
                    }
                }
            }
            if showRates, let r = model.rates {
                panel("Smoothed firing rates (Gaussian kernel), brighter = faster") { ctx, size in
                    let b0 = max(0, Int(t0 / r.binS)), b1 = min(r.nBins, Int(t1 / r.binS))
                    guard b1 > b0 else { return }
                    let rowH = size.height / CGFloat(r.rates.count)
                    let cols = min(b1 - b0, Int(size.width))
                    let colW = size.width / CGFloat(cols)
                    for (u, rate) in r.rates.enumerated() {
                        let peak = max(rate[b0..<b1].max() ?? 1, 5)
                        for c in 0..<cols {
                            let b = b0 + c * (b1 - b0) / cols
                            let v = Double(rate[b] / peak)
                            guard v > 0.03 else { continue }
                            ctx.fill(Path(CGRect(x: CGFloat(c) * colW, y: CGFloat(u) * rowH, width: colW + 0.5, height: rowH + 0.5)),
                                     with: .color(Colormap.magma(v)))
                        }
                    }
                    markers(ctx, size, t0, t1)
                }
            }
            panel("Hand velocity (mm/s): x blue, y orange" + (model.decodeResult?.kind == "spikes" ? "; decoded dashed" : "")) { ctx, size in
                markers(ctx, size, t0, t1)
                let fs = rec.meta.fs
                let i0 = max(0, Int(t0 * fs)), i1 = min(rec.meta.n_samples, Int(t1 * fs))
                guard i1 > i0 + 1 else { return }
                let vmax: CGFloat = 1200
                let y = { (v: Double) in size.height / 2 - CGFloat(v) / vmax * size.height / 2 }
                for (c, color) in [(0, Color.blue), (1, Color.orange)] {
                    var p = Path()
                    for i in i0..<i1 {
                        let pt = CGPoint(x: CGFloat(Double(i - i0) / Double(i1 - i0)) * size.width, y: y(Double(rec.channels[c][i])))
                        if i == i0 { p.move(to: pt) } else { p.addLine(to: pt) }
                    }
                    ctx.stroke(p, with: .color(color), lineWidth: 1.2)
                    for tr in model.decodeResult?.spikeTrials ?? [] where tr.t_s.last! > t0 && tr.t_s.first! < t1 {
                        var q = Path()
                        for (k, t) in tr.t_s.enumerated() {
                            let pt = CGPoint(x: CGFloat((t - t0) / (t1 - t0)) * size.width, y: y(tr.pred[k][c]))
                            if k == 0 { q.move(to: pt) } else { q.addLine(to: pt) }
                        }
                        ctx.stroke(q, with: .color(color.opacity(tr.held_out ? 1 : 0.35)), style: StrokeStyle(lineWidth: 1.6, dash: [4, 2]))
                    }
                }
            }
            .frame(maxHeight: 180)
        }
    }

    func markers(_ ctx: GraphicsContext, _ size: CGSize, _ t0: Double, _ t1: Double) {
        for e in rec.meta.events where e.time_s > t0 && e.time_s < t1 {
            let x = CGFloat((e.time_s - t0) / (t1 - t0)) * size.width
            var p = Path(); p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: size.height))
            ctx.stroke(p, with: .color(e.split == "val" ? .green : .secondary), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        }
    }

    func panel(_ title: String, draw: @escaping (GraphicsContext, CGSize) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Canvas { ctx, size in draw(ctx, size) }
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    func lowerBound(_ a: [Float], _ v: Float) -> Int {
        var lo = 0, hi = a.count
        while lo < hi { let m = (lo + hi) / 2; if a[m] < v { lo = m + 1 } else { hi = m } }
        return lo
    }
}

enum Colormap {
    /// Approximation of matplotlib's magma, v in 0...1.
    static let stops: [(Double, Double, Double)] = [
        (0.001, 0.000, 0.014), (0.173, 0.067, 0.373), (0.445, 0.123, 0.506),
        (0.716, 0.215, 0.475), (0.945, 0.377, 0.365), (0.994, 0.624, 0.427), (0.987, 0.991, 0.750),
    ]

    static func rgb(_ v: Double) -> (Double, Double, Double) {
        let x = min(max(v, 0), 1) * Double(stops.count - 1)
        let i = min(Int(x), stops.count - 2), f = x - Double(i)
        let a = stops[i], b = stops[i + 1]
        return (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f)
    }

    static func magma(_ v: Double) -> Color {
        let (r, g, b) = rgb(v)
        return Color(red: r, green: g, blue: b)
    }
}
