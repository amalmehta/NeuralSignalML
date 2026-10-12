import SwiftUI

/// Run a decoder online and watch it as a user would: a cursor driven by decoded spikes,
/// or motor-imagery probabilities updating every 250 ms, with measured latency.
struct LiveView: View {
    @EnvironmentObject var model: AppModel
    @StateObject private var live = LiveSession()
    let rec: LoadedRecording
    @State private var chosen: DecoderInfo.ID?
    @State private var speed = 1.0

    var body: some View {
        let options = model.modelsForLoaded()
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Picker("Decoder", selection: $chosen) {
                    ForEach(options) { m in Text(m.displayName).tag(Optional(m.id)) }
                }
                .frame(maxWidth: 460)
                Picker("Speed", selection: $speed) {
                    ForEach([1.0, 2.0, 5.0, 10.0], id: \.self) { Text("\(Int($0))×").tag($0) }
                }
                .frame(width: 110)
                .onChange(of: speed) { live.speed = speed }
                if live.running {
                    Button { live.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                } else {
                    Button {
                        guard let m = options.first(where: { $0.id == chosen }) else { return }
                        live.speed = speed
                        live.start(bridge: model.bridge, path: rec.info.path, model: m.path)
                    } label: { Label("Start", systemImage: "play.fill") }
                    .disabled(chosen == nil)
                }
                Spacer()
            }
            if let e = live.error { Banner(text: e, warning: true) }
            latencyBar
            if rec.isSpikes { SpikeLive(live: live) } else { EEGLive(live: live, rec: rec) }
        }
        .padding(8)
        .onAppear { if chosen == nil { chosen = options.first?.id } }
        .onDisappear { live.stop() }
    }

    var latencyBar: some View {
        let budget = rec.isSpikes ? 50.0 : 250.0
        let med = LiveSession.percentile(live.computeMs, 0.5), p95 = LiveSession.percentile(live.computeMs, 0.95)
        return HStack(spacing: 22) {
            stat("Decoder compute, median", String(format: "%.2f ms", med))
            stat("95th percentile", String(format: "%.2f ms", p95), warn: p95 > budget)
            stat("Pipe to app, median", String(format: "%.1f ms", LiveSession.percentile(live.transportMs, 0.5)))
            stat("Update budget", String(format: "%.0f ms", budget))
            stat("Updates", "\(live.computeMs.count)")
            Spacer()
            Text(String(format: "t = %.1f s", live.clock)).monospacedDigit().foregroundStyle(.secondary)
        }
    }

    func stat(_ title: String, _ value: String, warn: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(value).font(.system(.title3, design: .rounded).weight(.semibold)).monospacedDigit()
                .foregroundStyle(warn ? .orange : .primary)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

/// Cursor driven by decoded velocity (dashed) next to the real hand (solid), within the current trial.
struct SpikeLive: View {
    @ObservedObject var live: LiveSession

    var body: some View {
        let trial = live.shown.last?.trial
        let samples = live.shown.filter { $0.trial == trial }
        let dt = 0.05
        func path(_ key: (LiveSample) -> [Double]?) -> [CGPoint] {
            var p = CGPoint.zero, out = [CGPoint.zero]
            for s in samples { if let v = key(s) { p.x += v[0] * dt; p.y += v[1] * dt; out.append(p) } }
            return out
        }
        let truth = path { $0.true }, decoded = path { $0.pred }
        return HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(trial.map { "Trial \($0)" + ((samples.last?.held_out ?? false) ? " (held out)" : " (training trial)") } ?? "Press Start")
                    .font(.callout)
                Canvas { ctx, size in
                    let scale = min(size.width, size.height) / 320
                    let c = CGPoint(x: size.width / 2, y: size.height / 2)
                    let map = { (p: CGPoint) in CGPoint(x: c.x + p.x * scale, y: c.y - p.y * scale) }
                    ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 100 * scale, y: c.y - 100 * scale, width: 200 * scale, height: 200 * scale)),
                               with: .color(.secondary.opacity(0.25)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    for (pts, color, dash) in [(truth, Color.primary, [CGFloat]()), (decoded, Color.orange, [5, 3])] {
                        var p = Path()
                        for (i, q) in pts.enumerated() { i == 0 ? p.move(to: map(q)) : p.addLine(to: map(q)) }
                        ctx.stroke(p, with: .color(color), style: StrokeStyle(lineWidth: 2, dash: dash))
                        if let last = pts.last {
                            ctx.fill(Path(ellipseIn: CGRect(x: map(last).x - 6, y: map(last).y - 6, width: 12, height: 12)), with: .color(color))
                        }
                    }
                }
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                HStack(spacing: 14) {
                    Label("real hand", systemImage: "circle.fill").foregroundStyle(.primary)
                    Label("decoded cursor", systemImage: "circle.fill").foregroundStyle(.orange)
                    Text("dashed circle: 100 mm").foregroundStyle(.secondary)
                }
                .font(.caption)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Velocity, last 3 s (mm/s): x top, y bottom").font(.callout)
                Canvas { ctx, size in
                    let recent = live.shown.suffix(60)
                    guard let t1 = recent.last?.t else { return }
                    let t0 = t1 - 3
                    for comp in 0..<2 {
                        let top = CGFloat(comp) * size.height / 2, h = size.height / 2
                        let y = { (v: Double) in top + h / 2 - CGFloat(v) / 1200 * h / 2 }
                        for (key, color) in [({ (s: LiveSample) in s.true }, Color.primary), ({ (s: LiveSample) in s.pred }, Color.orange)] {
                            var p = Path(); var first = true
                            for s in recent where s.t >= t0 {
                                guard let v = key(s) else { continue }
                                let pt = CGPoint(x: CGFloat((s.t - t0) / 3) * size.width, y: y(v[comp]))
                                if first { p.move(to: pt); first = false } else { p.addLine(to: pt) }
                            }
                            ctx.stroke(p, with: .color(color), lineWidth: 1.5)
                        }
                    }
                }
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }
}

/// Motor-imagery probabilities, updated every 250 ms from a sliding 2 s window.
struct EEGLive: View {
    @ObservedObject var live: LiveSession
    let rec: LoadedRecording

    var body: some View {
        let classes = rec.meta.classes ?? ["left hand", "right hand", "feet", "tongue"]
        let colors: [Color] = [.blue, .orange, .green, .purple]
        let current = live.shown.last?.probs ?? Array(repeating: 0.25, count: 4)
        let cue = rec.meta.events.last { $0.time_s <= live.clock && $0.time_s > live.clock - 4.5 }
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .bottom, spacing: 18) {
                ForEach(0..<4, id: \.self) { i in
                    VStack {
                        Text(String(format: "%.0f%%", current[i] * 100)).monospacedDigit()
                        RoundedRectangle(cornerRadius: 4).fill(colors[i].opacity(0.8))
                            .frame(width: 60, height: max(2, 160 * current[i]))
                        Text(classes[i]).font(.caption)
                    }
                    .frame(height: 210, alignment: .bottom)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Cue on screen").font(.caption).foregroundStyle(.secondary)
                    Text(cue.map { "imagine \($0.label)" } ?? "rest / between trials").font(.title2.weight(.semibold))
                    Text("The decoder sees only the last 2 s of EEG, filtered causally, and never the cue.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.leading, 20)
            }
            Text("Probabilities over the last 20 s (dashed lines: cues)").font(.callout)
            Canvas { ctx, size in
                let t1 = live.clock, t0 = t1 - 20
                let x = { (t: Double) in CGFloat((t - t0) / 20) * size.width }
                for e in rec.meta.events where e.time_s > t0 && e.time_s < t1 {
                    var p = Path(); p.move(to: CGPoint(x: x(e.time_s), y: 0)); p.addLine(to: CGPoint(x: x(e.time_s), y: size.height))
                    ctx.stroke(p, with: .color(colors[e.label_index].opacity(0.7)), style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                }
                let recent = live.shown.filter { $0.t > t0 }
                for c in 0..<4 {
                    var p = Path(); var first = true
                    for s in recent {
                        guard let pr = s.probs else { continue }
                        let pt = CGPoint(x: x(s.t), y: size.height * (1 - CGFloat(pr[c])))
                        if first { p.move(to: pt); first = false } else { p.addLine(to: pt) }
                    }
                    ctx.stroke(p, with: .color(colors[c]), lineWidth: 1.6)
                }
            }
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}
