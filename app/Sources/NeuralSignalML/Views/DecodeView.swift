import Charts
import SwiftUI

/// Pick a trained decoder, run it on the loaded recording, and see how it did.
struct DecodeView: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    let jump: (Double) -> Void
    @State private var chosen: DecoderInfo.ID?

    var body: some View {
        let options = model.modelsForLoaded()
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Picker("Decoder", selection: $chosen) {
                    ForEach(options) { m in Text(m.displayName).tag(Optional(m.id)) }
                }
                .frame(maxWidth: 520)
                Button {
                    if let m = options.first(where: { $0.id == chosen }) { Task { await model.decode(with: m) } }
                } label: {
                    Label("Run decoder", systemImage: "play.fill")
                }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(chosen == nil || model.decoding)
                if model.decoding { ProgressView().controlSize(.small) }
                Spacer()
            }
            if options.isEmpty {
                ContentUnavailableView("No trained decoders for this kind of recording",
                                       systemImage: "cpu",
                                       description: Text("Train some with `neuralsignal train --all`, then reload (⇧⌘R)."))
            } else if let r = model.decodeResult, r.recording == rec.meta.id || r.kind == "spikes" && rec.isSpikes {
                if r.kind == "eeg" { EEGResultView(result: r, jump: jump) } else { SpikeResultView(result: r, jump: jump) }
            } else {
                ContentUnavailableView("Choose a decoder and press Run", systemImage: "brain",
                                       description: Text(rec.isSpikes
                                            ? "The decoder predicts hand velocity from the spikes, trial by trial."
                                            : "The decoder classifies each trial's imagined movement from the 2 s of EEG after the cue."))
            }
            Spacer(minLength: 0)
        }
        .padding(8)
        .onAppear { if chosen == nil { chosen = options.first?.id } }
        .onChange(of: options.map(\.id)) { if chosen == nil { chosen = options.first?.id } }
    }
}

struct Banner: View {
    let text: String
    let warning: Bool
    var body: some View {
        Label(text, systemImage: warning ? "exclamationmark.triangle.fill" : "checkmark.seal")
            .font(.callout)
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background((warning ? Color.orange : Color.green).opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
    }
}

struct EEGResultView: View {
    let result: DecodeResult
    let jump: (Double) -> Void

    var body: some View {
        let classes = result.classes ?? []
        let trials = result.trials ?? []
        VStack(alignment: .leading, spacing: 12) {
            Banner(text: result.note ?? "", warning: result.seen_in_training == true)
            HStack(alignment: .top, spacing: 28) {
                VStack(alignment: .leading, spacing: 10) {
                    big("Accuracy", String(format: "%.1f%%", (result.accuracy ?? 0) * 100))
                    big("Cohen's kappa", String(format: "%.2f", result.kappa ?? 0))
                    big("Chance (p < .05)", String(format: "%.1f%%", (result.chance_95 ?? 0.25) * 100))
                    Text("\(trials.count) trials, 4 classes").font(.caption).foregroundStyle(.secondary)
                }
                confusion(classes: classes, trials: trials)
                Table(trials) {
                    TableColumn("Time") { t in Text(String(format: "%.1f s", t.time_s)).monospacedDigit() }.width(60)
                    TableColumn("Imagined") { t in Text(classes[t.true]) }
                    TableColumn("Decoded") { t in
                        Label(classes[t.pred], systemImage: t.pred == t.true ? "checkmark" : "xmark")
                            .foregroundStyle(t.pred == t.true ? .green : .red)
                    }
                    TableColumn("Confidence") { t in
                        ProgressView(value: t.probs[t.pred]).frame(width: 70)
                    }
                    TableColumn("") { t in Button("Show") { jump(t.time_s) }.buttonStyle(.link) }.width(40)
                }
                .frame(minWidth: 380)
            }
        }
    }

    func big(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(value).font(.system(size: 26, weight: .semibold, design: .rounded)).monospacedDigit()
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
    }

    func confusion(classes: [String], trials: [EEGTrialResult]) -> some View {
        let n = classes.count
        var m = Array(repeating: Array(repeating: 0, count: n), count: n)
        for t in trials { m[t.true][t.pred] += 1 }
        return VStack(alignment: .leading, spacing: 2) {
            Text("Confusion (rows: imagined, columns: decoded)").font(.caption).foregroundStyle(.secondary)
            Grid(horizontalSpacing: 2, verticalSpacing: 2) {
                GridRow {
                    Text("")
                    ForEach(0..<n, id: \.self) { j in Text(classes[j]).font(.caption2).frame(width: 58) }
                }
                ForEach(0..<n, id: \.self) { i in
                    GridRow {
                        Text(classes[i]).font(.caption2).frame(width: 64, alignment: .trailing)
                        ForEach(0..<n, id: \.self) { j in
                            let total = max(1, m[i].reduce(0, +))
                            let frac = Double(m[i][j]) / Double(total)
                            Text("\(m[i][j])").font(.caption.monospacedDigit())
                                .frame(width: 58, height: 34)
                                .background((i == j ? Color.green : Color.red).opacity(0.08 + 0.7 * frac), in: RoundedRectangle(cornerRadius: 3))
                        }
                    }
                }
            }
        }
    }
}

struct SpikeResultView: View {
    let result: DecodeResult
    let jump: (Double) -> Void
    @State private var selected: Int?

    var body: some View {
        let trials = result.spikeTrials ?? []
        VStack(alignment: .leading, spacing: 12) {
            Banner(text: result.note ?? "", warning: false)
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(String(format: "%.3f", result.r2_held_out ?? 0)).font(.system(size: 26, weight: .semibold, design: .rounded))
                        Text("velocity R² on \(result.n_held_out ?? 0) held-out trials").font(.caption).foregroundStyle(.secondary)
                    }
                    List(trials, id: \.trial, selection: $selected) { t in
                        HStack {
                            Text("Trial \(t.trial)")
                            Spacer()
                            Text(t.held_out ? "held out" : "training").font(.caption)
                                .foregroundStyle(t.held_out ? .green : .secondary)
                        }
                        .tag(t.trial)
                    }
                    .frame(width: 220)
                }
                if let t = trials.first(where: { $0.trial == selected }) ?? trials.first(where: \.held_out) {
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Trial \(t.trial): true (solid) vs decoded (dashed), 0 s = movement onset").font(.callout)
                            Spacer()
                            Button("Show in signals") { jump(t.t_s.first! + 0.5) }
                        }
                        chart(t)
                        if !t.held_out {
                            Text("This trial was in the training set, so the fit here is optimistic.").font(.caption).foregroundStyle(.orange)
                        }
                    }
                }
            }
        }
    }

    func chart(_ t: SpikeTrialResult) -> some View {
        let onset = t.t_s.first! + (t.t_s.last! - t.t_s.first!) * 750 / 1200
        struct Pt: Identifiable { let id = UUID(); let t: Double; let v: Double; let series: String; let kind: String }
        var pts: [Pt] = []
        for (k, time) in t.t_s.enumerated() {
            for (c, axis) in ["x", "y"].enumerated() {
                pts.append(Pt(t: time - onset, v: t.true[k][c], series: "v\(axis)", kind: "true"))
                pts.append(Pt(t: time - onset, v: t.pred[k][c], series: "v\(axis)", kind: "decoded"))
            }
        }
        return Chart(pts) { p in
            LineMark(x: .value("s", p.t), y: .value("mm/s", p.v), series: .value("s", p.series + p.kind))
                .foregroundStyle(by: .value("axis", p.series))
                .lineStyle(p.kind == "true" ? StrokeStyle(lineWidth: 2) : StrokeStyle(lineWidth: 2, dash: [5, 3]))
        }
        .chartXAxisLabel("seconds from movement onset")
        .chartYAxisLabel("hand velocity (mm/s)")
        .frame(minHeight: 260)
    }
}
