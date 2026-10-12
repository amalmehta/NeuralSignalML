import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @Binding var showFeedback: Bool

    var body: some View {
        NavigationSplitView {
            Sidebar(showFeedback: $showFeedback)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        } detail: {
            if let rec = model.loaded {
                RecordingView(rec: rec)
            } else {
                Placeholder()
            }
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                SettingsLink { Label("Settings", systemImage: "gearshape") }
                    .help("Settings (⌘,)")
            }
        }
        .sheet(isPresented: $showFeedback) { FeedbackView() }
    }
}

struct Sidebar: View {
    @EnvironmentObject var model: AppModel
    @Binding var showFeedback: Bool

    var groups: [(String, [RecordingInfo])] {
        let all = model.allRecordings
        let names = all.map(\.group).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        return names.map { g in (g, all.filter { $0.group == g }) }
    }

    var body: some View {
        VStack(spacing: 0) {
            List(selection: Binding(get: { model.selection }, set: { id in Task { await model.select(id) } })) {
                ForEach(groups, id: \.0) { group, recs in
                    Section(group) {
                        ForEach(recs) { r in
                            Label {
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(r.kind == "eeg" ? "Subject \(Int(r.id.dropFirst().prefix(2)) ?? 0), \(r.id.hasSuffix("T") ? "day 1" : "day 2")" : r.title)
                                    Text(r.kind == "eeg" ? r.id : "MC_Maze Medium").font(.caption).foregroundStyle(.secondary)
                                }
                            } icon: {
                                Image(systemName: r.kind == "eeg" ? "waveform.path" : "circle.grid.cross")
                            }
                            .tag(r.id)
                        }
                    }
                }
            }
            Divider()
            HStack {
                Text("\(model.models.count) trained decoders").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button { showFeedback = true } label: {
                    Label("Feedback", systemImage: "bubble.left").font(.caption)
                }
                .buttonStyle(.borderless)
                .help("Send feedback about the app")
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
        }
    }
}

struct Placeholder: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        VStack(spacing: 12) {
            if let status = model.status {
                ProgressView()
                Text(status).foregroundStyle(.secondary)
            } else if let err = model.error {
                Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                Text(err).multilineTextAlignment(.center).frame(maxWidth: 520).textSelection(.enabled)
                HStack {
                    Button("Try Again") { Task { await model.refresh() } }
                    SettingsLink { Text("Open Settings") }
                }
            } else {
                Image(systemName: "brain.head.profile").font(.system(size: 48)).foregroundStyle(.secondary)
                Text("Pick a recording on the left, or open one with ⌘O").foregroundStyle(.secondary)
            }
        }
        .padding()
    }
}

/// The detail pane: header, shared time navigator, and the three tabs.
struct RecordingView: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    @AppStorage(SettingsKey.windowSeconds) var windowSeconds = 10.0
    @State private var tab = "signals"

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            TabView(selection: $tab) {
                SignalsView(rec: rec).tabItem { Text("Signals") }.tag("signals")
                SpectrogramView(rec: rec).tabItem { Text("Spectrogram") }.tag("spectrogram")
                DecodeView(rec: rec, jump: { t in model.viewStart = max(0, t - windowSeconds * 0.3); tab = "signals" })
                    .tabItem { Text("Decode") }.tag("decode")
                LiveView(rec: rec).tabItem { Text("Live") }.tag("live")
            }
            .padding(8)
            if tab != "decode" && tab != "live" { TimeNavigator(rec: rec) }
        }
        .navigationTitle(rec.isSpikes ? "MC_Maze Medium" : rec.meta.id)
        .navigationSubtitle(rec.info.title)
    }

    var header: some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 2) {
                Text(rec.info.title).font(.headline)
                Text(rec.info.group).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            stat("Duration", String(format: "%.0f s", rec.duration))
            if rec.isSpikes {
                stat("Units", "\(rec.meta.n_units ?? 0)")
                stat("Trials", "\(rec.meta.events.count)")
            } else {
                stat("Channels", "22 EEG + 3 EOG")
                stat("Rate", String(format: "%.0f Hz", rec.meta.fs))
                stat("Trials", "\(rec.meta.events.count)")
            }
            if let err = model.error {
                Label(err, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).lineLimit(2)
                    .frame(maxWidth: 280)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 1) {
            Text(value).font(.system(.body, design: .rounded)).monospacedDigit()
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct TimeNavigator: View {
    @EnvironmentObject var model: AppModel
    let rec: LoadedRecording
    @AppStorage(SettingsKey.windowSeconds) var windowSeconds = 10.0

    var body: some View {
        HStack(spacing: 10) {
            Button { jumpEvent(-1) } label: { Image(systemName: "backward.end") }.help("Previous trial")
            Button { model.viewStart = max(0, model.viewStart - windowSeconds) } label: { Image(systemName: "chevron.left") }
            Slider(value: Binding(get: { model.viewStart }, set: { model.viewStart = $0 }),
                   in: 0...max(1, rec.duration - windowSeconds))
            Button { model.viewStart = min(rec.duration - windowSeconds, model.viewStart + windowSeconds) } label: { Image(systemName: "chevron.right") }
            Button { jumpEvent(1) } label: { Image(systemName: "forward.end") }.help("Next trial")
            Text(String(format: "%.1f–%.1f s", model.viewStart, model.viewStart + windowSeconds))
                .monospacedDigit().frame(width: 110)
            Picker("", selection: $windowSeconds) {
                ForEach([2.0, 5.0, 10.0, 20.0, 60.0], id: \.self) { Text("\(Int($0)) s").tag($0) }
            }
            .frame(width: 80)
            .help("Seconds shown")
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.bar)
    }

    func jumpEvent(_ dir: Int) {
        let anchor = model.viewStart + windowSeconds * 0.3
        let times = rec.meta.events.map(\.time_s)
        let next = dir > 0 ? times.first { $0 > anchor + 0.01 } : times.last { $0 < anchor - 0.01 }
        if let t = next { model.viewStart = max(0, t - windowSeconds * 0.3) }
    }
}
