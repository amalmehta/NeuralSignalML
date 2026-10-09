import XCTest
@testable import SignalCore

final class SignalCoreTests: XCTestCase {
    func sine(_ f: Double, fs: Double, seconds: Double = 4) -> [Float] {
        (0..<Int(fs * seconds)).map { Float(sin(2 * Double.pi * f * Double($0) / fs)) }
    }

    func rms(_ x: ArraySlice<Float>) -> Float { sqrt(x.map { $0 * $0 }.reduce(0, +) / Float(x.count)) }

    func testBandpassResponseIsButterworthLike() {
        let spec = FilterSpec(low: 8, high: 30)
        let sos = spec.sections(fs: 250)
        let g = { (f: Double) in sos.reduce(1.0) { $0 * $1.gain(at: f, fs: 250) } }
        XCTAssertEqual(g(15), 1, accuracy: 0.05, "passband should be flat")
        XCTAssertEqual(g(30), 1 / sqrt(2), accuracy: 0.05, "-3 dB at the cutoff")
        XCTAssertLessThan(g(60), 0.05, "4th order: about -30 dB at twice the cutoff")
        XCTAssertLessThan(g(2), 0.01, "4th order: (2/8)^4 = -48 dB")
    }

    func testFilteringPassesInBandAndRemovesOutOfBand() {
        let fs = 250.0
        let inBand = SignalFilter.apply(sine(12, fs: fs), fs: fs, spec: FilterSpec(low: 8, high: 30))
        let outBand = SignalFilter.apply(sine(80, fs: fs), fs: fs, spec: FilterSpec(low: 8, high: 30))
        XCTAssertEqual(rms(inBand[500...]), Float(1 / sqrt(2.0)), accuracy: 0.05)
        XCTAssertLessThan(rms(outBand[500...]), 0.02)
    }

    func testZeroPhaseHasNoLag() {
        let fs = 250.0
        let x = sine(10, fs: fs)
        let y = SignalFilter.apply(x, fs: fs, spec: FilterSpec(low: 4, high: 40, zeroPhase: true))
        let mid = 300..<700
        let corr = zip(x[mid], y[mid]).map { $0 * $1 }.reduce(0, +) / (rms(x[mid]) * rms(y[mid]) * Float(mid.count))
        XCTAssertGreaterThan(corr, 0.98)
    }

    func testNotchRemovesMains() {
        let fs = 250.0
        let y = SignalFilter.apply(sine(50, fs: fs), fs: fs, spec: FilterSpec(low: 1, high: 100, notch: 50))
        XCTAssertLessThan(rms(y[500...]), 0.05)
    }

    func testSpectrogramPeakAtSignalFrequency() throws {
        let fs = 250.0
        let s = try XCTUnwrap(Spectrogram.compute(sine(20, fs: fs), fs: fs, window: 256, hop: 32))
        let frame = s.nFrames / 2
        let peak = (0..<s.nFreqs).max { s.value(freqIndex: $0, frame: frame) < s.value(freqIndex: $1, frame: frame) }!
        XCTAssertEqual(Double(peak) * s.freqResolution, 20, accuracy: s.freqResolution)
    }

    func testMinMaxDecimationKeepsExtremes() {
        var x = [Float](repeating: 0, count: 10_000)
        x[1234] = 5; x[8000] = -3
        let mm = Decimate.minMax(x[...], buckets: 100)
        XCTAssertEqual(mm.count, 100)
        XCTAssertEqual(mm.map(\.1).max(), 5)
        XCTAssertEqual(mm.map(\.0).min(), -3)
    }
}
