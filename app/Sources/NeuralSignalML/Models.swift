import Foundation

/// A recording the Python side found in ./data (from `neuralsignal app recordings`).
struct RecordingInfo: Codable, Identifiable, Hashable {
    let id: String
    let kind: String          // "eeg" | "spikes"
    let title: String
    let group: String
    let path: String
}

struct RecordingEvent: Codable, Hashable {
    let time_s: Double
    let label: String
    let label_index: Int
    var artifact: Bool?
    var split: String?
}

struct RunInfo: Codable, Hashable {
    let index: Int
    let start_s: Double
    let calibration: Bool
}

/// meta.json written by `neuralsignal app export`.
struct RecordingMeta: Codable {
    let kind: String
    let id: String
    let fs: Double
    let n_channels: Int
    let n_samples: Int
    let channels: [String]
    let units: String
    let channel_types: [String]
    let events: [RecordingEvent]
    let runs: [RunInfo]
    var classes: [String]?
    var n_units: Int?
    var spike_counts: [Int]?
    var duration_s: Double?
}

/// A loaded recording held in memory: raw channels plus (for spikes) spike times per unit.
struct LoadedRecording {
    let info: RecordingInfo
    let meta: RecordingMeta
    let channels: [[Float]]         // raw continuous signals (EEG/EOG, or hand velocity)
    let spikeTimes: [[Float]]       // per unit, seconds (spikes recordings only)

    var duration: Double {
        meta.duration_s ?? Double(meta.n_samples) / meta.fs
    }
    var isSpikes: Bool { meta.kind == "spikes" }
}

struct DecoderInfo: Codable, Identifiable, Hashable {
    let path: String
    let method: String
    let kind: String
    let name: String
    let protocol_: String
    let target: String
    let trained_on: [String]?
    let test_accuracy: Double?
    let test_r2: Double?
    let test_recording: String?
    var id: String { path }

    enum CodingKeys: String, CodingKey {
        case path, method, kind, name, target, trained_on, test_accuracy, test_r2, test_recording
        case protocol_ = "protocol"
    }

    var protocolTitle: String {
        switch protocol_ {
        case "cross_session": return "held-out session"
        case "cross_subject": return "held-out subject"
        case "nlb_split": return "NLB train split"
        default: return protocol_
        }
    }

    var displayName: String {
        let method = name.components(separatedBy: " - ").last ?? name
        switch protocol_ {
        case "cross_session": return "\(method), trained on subject \(Int(target.dropFirst()) ?? 0) day 1"
        case "cross_subject": return "\(method), trained on everyone except subject \(Int(target.dropFirst()) ?? 0)"
        default: return "\(method), trained on the NLB train trials"
        }
    }
}

struct EEGTrialResult: Codable, Hashable {
    let time_s: Double
    let `true`: Int
    let pred: Int
    let probs: [Double]
}

struct SpikeTrialResult: Codable, Hashable {
    let trial: Int
    let held_out: Bool
    let t_s: [Double]
    let `true`: [[Double]]
    let pred: [[Double]]
}

struct DecodeResult: Codable {
    let kind: String
    let model: String
    let `protocol`: String
    let recording: String
    var note: String?
    // EEG
    var seen_in_training: Bool?
    var classes: [String]?
    var trials: [EEGTrialResult]?
    var accuracy: Double?
    var kappa: Double?
    var chance_95: Double?
    // spikes
    var spikeTrials: [SpikeTrialResult]?
    var r2_held_out: Double?
    var n_held_out: Int?

    enum CodingKeys: String, CodingKey {
        case kind, model, `protocol`, recording, note, seen_in_training, classes, accuracy, kappa, chance_95,
             r2_held_out, n_held_out, trials
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = try c.decode(String.self, forKey: .kind)
        model = try c.decode(String.self, forKey: .model)
        `protocol` = try c.decode(String.self, forKey: .protocol)
        recording = try c.decode(String.self, forKey: .recording)
        note = try c.decodeIfPresent(String.self, forKey: .note)
        seen_in_training = try c.decodeIfPresent(Bool.self, forKey: .seen_in_training)
        classes = try c.decodeIfPresent([String].self, forKey: .classes)
        accuracy = try c.decodeIfPresent(Double.self, forKey: .accuracy)
        kappa = try c.decodeIfPresent(Double.self, forKey: .kappa)
        chance_95 = try c.decodeIfPresent(Double.self, forKey: .chance_95)
        r2_held_out = try c.decodeIfPresent(Double.self, forKey: .r2_held_out)
        n_held_out = try c.decodeIfPresent(Int.self, forKey: .n_held_out)
        if kind == "eeg" {
            trials = try c.decodeIfPresent([EEGTrialResult].self, forKey: .trials)
        } else {
            spikeTrials = try c.decodeIfPresent([SpikeTrialResult].self, forKey: .trials)
        }
    }

    func encode(to encoder: Encoder) throws {}
}

extension EEGTrialResult: Identifiable {
    var id: Double { time_s }
}
