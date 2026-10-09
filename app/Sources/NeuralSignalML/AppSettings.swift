import SwiftUI

/// Every preference in one place; the Settings window (⌘,) edits these.
enum SettingsKey {
    static let projectFolder = "projectFolder"
    static let pythonPath = "pythonPath"
    static let windowSeconds = "windowSeconds"
    static let filterLow = "filterLow"
    static let filterHigh = "filterHigh"
    static let notchOn = "notchOn"
    static let notchHz = "notchHz"
    static let zeroPhase = "zeroPhase"
    static let showRaw = "showRaw"
    static let showFiltered = "showFiltered"
    static let spectrogramWindow = "spectrogramWindow"
    static let spectrogramMaxHz = "spectrogramMaxHz"
    static let showEOG = "showEOG"
    static let rateSmoothingMs = "rateSmoothingMs"
}

enum Defaults {
    /// The project folder is baked into Info.plist by build-app.sh; fall back to the folder
    /// two levels above the .app (app/build/Neural Signal ML.app -> project).
    static var projectFolder: String {
        if let p = Bundle.main.object(forInfoDictionaryKey: "NSMLProjectFolder") as? String,
           FileManager.default.fileExists(atPath: p) { return p }
        return Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().path
    }

    static func register() {
        UserDefaults.standard.register(defaults: [
            SettingsKey.projectFolder: projectFolder,
            SettingsKey.pythonPath: "",
            SettingsKey.windowSeconds: 10.0,
            SettingsKey.filterLow: 8.0,
            SettingsKey.filterHigh: 30.0,
            SettingsKey.notchOn: false,
            SettingsKey.notchHz: 50.0,
            SettingsKey.zeroPhase: false,
            SettingsKey.showRaw: true,
            SettingsKey.showFiltered: true,
            SettingsKey.spectrogramWindow: 256,
            SettingsKey.spectrogramMaxHz: 45.0,
            SettingsKey.showEOG: true,
            SettingsKey.rateSmoothingMs: 50.0,
        ])
    }
}
