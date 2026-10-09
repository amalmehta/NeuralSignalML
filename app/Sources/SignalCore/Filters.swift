import Accelerate
import Foundation

/// One second-order IIR section: y = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2).
public struct Biquad: Equatable, Sendable {
    public var b0, b1, b2, a1, a2: Double

    /// Magnitude response at `freq` Hz for sample rate `fs`.
    public func gain(at freq: Double, fs: Double) -> Double {
        let w = 2 * Double.pi * freq / fs
        let (c1, s1, c2, s2) = (cos(w), sin(w), cos(2 * w), sin(2 * w))
        let nr = b0 + b1 * c1 + b2 * c2, ni = -(b1 * s1 + b2 * s2)
        let dr = 1 + a1 * c1 + a2 * c2, di = -(a1 * s1 + a2 * s2)
        return sqrt((nr * nr + ni * ni) / (dr * dr + di * di))
    }

    // RBJ audio-EQ-cookbook designs (bilinear transform with frequency pre-warping).
    static func lowpass(_ f: Double, fs: Double, q: Double) -> Biquad {
        let w = 2 * Double.pi * f / fs, alpha = sin(w) / (2 * q), c = cos(w), a0 = 1 + alpha
        return Biquad(b0: (1 - c) / 2 / a0, b1: (1 - c) / a0, b2: (1 - c) / 2 / a0,
                      a1: -2 * c / a0, a2: (1 - alpha) / a0)
    }

    static func highpass(_ f: Double, fs: Double, q: Double) -> Biquad {
        let w = 2 * Double.pi * f / fs, alpha = sin(w) / (2 * q), c = cos(w), a0 = 1 + alpha
        return Biquad(b0: (1 + c) / 2 / a0, b1: -(1 + c) / a0, b2: (1 + c) / 2 / a0,
                      a1: -2 * c / a0, a2: (1 - alpha) / a0)
    }

    static func notch(_ f: Double, fs: Double, q: Double) -> Biquad {
        let w = 2 * Double.pi * f / fs, alpha = sin(w) / (2 * q), c = cos(w), a0 = 1 + alpha
        return Biquad(b0: 1 / a0, b1: -2 * c / a0, b2: 1 / a0, a1: -2 * c / a0, a2: (1 - alpha) / a0)
    }
}

/// Settings for the viewer's filter. Mirrors the Python pipeline's causal Butterworth band-pass,
/// built here as a 4th-order Butterworth high-pass cascaded with a 4th-order Butterworth low-pass.
public struct FilterSpec: Equatable, Sendable {
    public var low: Double
    public var high: Double
    public var notch: Double?      // e.g. 50 Hz mains; nil = off
    public var zeroPhase: Bool     // forward-backward (display only) vs causal (what decoders see)

    public init(low: Double, high: Double, notch: Double? = nil, zeroPhase: Bool = false) {
        self.low = low; self.high = high; self.notch = notch; self.zeroPhase = zeroPhase
    }

    /// Q factors of the two biquads that make up a 4th-order Butterworth.
    static let butterworth4Q = [1 / (2 * cos(Double.pi / 8)), 1 / (2 * cos(3 * Double.pi / 8))]

    public func sections(fs: Double) -> [Biquad] {
        let nyq = fs / 2
        var s: [Biquad] = []
        if low > 0 && low < nyq { s += Self.butterworth4Q.map { Biquad.highpass(low, fs: fs, q: $0) } }
        if high > 0 && high < nyq * 0.98 { s += Self.butterworth4Q.map { Biquad.lowpass(high, fs: fs, q: $0) } }
        if let n = notch, n < nyq { s.append(Biquad.notch(n, fs: fs, q: 30)) }
        return s
    }
}

public enum SignalFilter {
    /// Apply biquad sections in series using vDSP's optimised cascade.
    public static func cascade(_ x: [Float], _ sections: [Biquad]) -> [Float] {
        guard !sections.isEmpty, !x.isEmpty else { return x }
        let coeffs = sections.flatMap { [$0.b0, $0.b1, $0.b2, $0.a1, $0.a2] }
        guard let setup = vDSP.Biquad<Float>(coefficients: coeffs, channelCount: 1,
                                            sectionCount: vDSP_Length(sections.count), ofType: Float.self)
        else { return x }
        var b = setup
        return b.apply(input: x)
    }

    public static func apply(_ x: [Float], fs: Double, spec: FilterSpec) -> [Float] {
        let sos = spec.sections(fs: fs)
        // Remove the mean first so the high-pass doesn't start with a huge step transient.
        let mean = vDSP.mean(x)
        let centred = vDSP.add(-mean, x)
        let y = cascade(centred, sos)
        guard spec.zeroPhase else { return y }
        return Array(cascade(Array(y.reversed()), sos).reversed())
    }
}
