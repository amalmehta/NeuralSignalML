import SwiftUI

/// A small feedback form. Notes are appended to a local JSON-lines file; nothing is sent anywhere.
struct FeedbackView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var kind = "Idea"
    @State private var text = ""
    @State private var saved = false

    static var fileURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Neural Signal ML/feedback.jsonl")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Feedback").font(.title3.weight(.semibold))
            Picker("", selection: $kind) {
                ForEach(["Idea", "Bug", "Question"], id: \.self) { Text($0) }
            }
            .pickerStyle(.segmented).labelsHidden()
            TextEditor(text: $text)
                .font(.body)
                .frame(width: 420, height: 140)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.separator))
            Text("Saved on this Mac only, with the recording you have open.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Show File") { NSWorkspace.shared.activateFileViewerSelecting([Self.fileURL]) }
                    .disabled(!FileManager.default.fileExists(atPath: Self.fileURL.path))
                Spacer()
                if saved { Label("Saved", systemImage: "checkmark").foregroundStyle(.green) }
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save") { save() }.keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
    }

    func save() {
        let entry: [String: String] = [
            "date": ISO8601DateFormatter().string(from: Date()),
            "kind": kind,
            "text": text,
            "recording": model.loaded?.meta.id ?? "",
            "app_version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev",
        ]
        let url = Self.fileURL
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard var line = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
        line.append(0x0A)
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(line); try? h.close()
        } else {
            try? line.write(to: url)
        }
        saved = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { dismiss() }
    }
}
