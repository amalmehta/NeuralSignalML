import Accelerate
import Foundation

/// Short-time Fourier transform power, in dB, computed with vDSP's real FFT.
public struct Spectrogram: Sendable {
    public let power: [Float]       // row-major (frequencies x frames), dB
    public let nFreqs: Int
    public let nFrames: Int
    public let freqResolution: Double
    public let frameStep: Double    // seconds between frames
    public let startTime: Double    // seconds, centre of the first frame

    public func value(freqIndex f: Int, frame t: Int) -> Float { power[f * nFrames + t] }

    /// - Parameters:
    ///   - window: samples per frame (rounded up to a power of two for the FFT)
    ///   - hop: samples between frames
    public static func compute(_ x: [Float], fs: Double, window: Int = 256, hop: Int = 16,
                               startTime: Double = 0) -> Spectrogram? {
        let n = 1 << Int(ceil(log2(Double(max(window, 8)))))
        guard x.count >= n, let fft = vDSP.FFT(log2n: vDSP_Length(log2(Double(n))), radix: .radix2,
                                                ofType: DSPSplitComplex.self) else { return nil }
        let hann = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: n, isHalfWindow: false)
        let frames = (x.count - n) / hop + 1
        let half = n / 2
        var out = [Float](repeating: 0, count: half * frames)
        var re = [Float](repeating: 0, count: half), im = [Float](repeating: 0, count: half)
        var mags = [Float](repeating: 0, count: half)
        let norm = 1 / (fs * Double(vDSP.sumOfSquares(hann)))
        for f in 0..<frames {
            let seg = Array(x[(f * hop)..<(f * hop + n)])
            let m = vDSP.mean(seg)
            let w = vDSP.multiply(vDSP.add(-m, seg), hann)
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    w.withUnsafeBufferPointer { wp in
                        wp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                        }
                    }
                    fft.forward(input: split, output: &split)
                    vDSP.squareMagnitudes(split, result: &mags)
                }
            }
            for k in 0..<half {
                // vDSP's packed real FFT is scaled by 2; divide out to get a true periodogram.
                let p = Double(mags[k]) / 4 * norm * (k == 0 ? 1 : 2)
                out[k * frames + f] = Float(10 * log10(p + 1e-12))
            }
        }
        return Spectrogram(power: out, nFreqs: half, nFrames: frames, freqResolution: fs / Double(n),
                           frameStep: Double(hop) / fs, startTime: startTime + Double(n) / 2 / fs)
    }
}

public enum Decimate {
    /// Min/max envelope with `buckets` pairs - draws a faithful trace with one line per pixel column.
    public static func minMax(_ x: ArraySlice<Float>, buckets: Int) -> [(Float, Float)] {
        guard !x.isEmpty, buckets > 0 else { return [] }
        let n = x.count, start = x.startIndex
        if n <= buckets * 2 { return x.map { ($0, $0) } }
        return (0..<buckets).map { b in
            let lo = start + b * n / buckets, hi = start + max((b + 1) * n / buckets, b * n / buckets + 1)
            let s = x[lo..<hi]
            return (s.min()!, s.max()!)
        }
    }

    /// Robust amplitude scale (median absolute value / 0.6745), ignores spikes and artefacts.
    public static func robustScale(_ x: ArraySlice<Float>) -> Float {
        guard !x.isEmpty else { return 1 }
        let stride = max(1, x.count / 4000)
        var s = Swift.stride(from: x.startIndex, to: x.endIndex, by: stride).map { abs(x[$0]) }
        s.sort()
        return max(s[s.count / 2] / 0.6745, 1e-6)
    }
}
