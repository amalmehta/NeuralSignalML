"""Spike sorting, end to end, on a synthetic recording where the right answer is known.

MC_Maze arrives already sorted, so this teaching demo shows what happened upstream:

  1. Simulate 10 s of broadband extracellular voltage at 30 kHz: 3 neurons with distinct
     action-potential shapes firing as Poisson processes (with a 2 ms refractory period),
     plus pink-ish background noise and slow LFP.
  2. Band-pass 300-3000 Hz: removes the LFP, keeps the spike energy.
  3. Detect threshold crossings at -4.5 x the noise level, where noise is estimated
     robustly as median(|x|)/0.6745 (spikes would inflate a plain std).
  4. Cut a 1.6 ms waveform around each crossing, aligned on its trough.
  5. Reduce each waveform to 3 principal components.
  6. Cluster with a Gaussian mixture; pick the number of units by BIC.
  7. Score against ground truth: detection recall/precision and per-unit sorting accuracy.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

FS = 30_000


def _template(fs, width_ms, trough, peak, t_peak_ms):
    t = np.arange(int(fs * 0.0016)) / fs * 1000 - 0.5   # -0.5 .. 1.1 ms
    trough_shape = -trough * np.exp(-(t / (width_ms / 2.5)) ** 2)
    peak_shape = peak * np.exp(-((t - t_peak_ms) / (width_ms / 1.5)) ** 2)
    return trough_shape + peak_shape


TEMPLATES = [  # (width ms, trough uV, after-hyperpolarisation uV, AHP time ms)
    (0.30, 140, 35, 0.35),   # narrow and large, "fast-spiking interneuron"-like
    (0.65, 95, 20, 0.80),    # broad with a slow, shallow after-hyperpolarisation, pyramidal-like
    (0.40, 70, 65, 0.40),    # small trough with a large, fast repolarisation
]


def simulate(duration_s=10.0, rates=(12.0, 8.0, 15.0), noise_uv=11.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    white = rng.standard_normal(n)
    b, a = signal.butter(1, 2000, fs=FS)
    noise = signal.lfilter(b, a, white)
    noise *= noise_uv / noise.std()
    lfp = 60 * np.sin(2 * np.pi * 8 * np.arange(n) / FS + rng.uniform(0, 6))
    x = noise + lfp
    truth = []
    for unit, ((w, tr, pk, tp), rate) in enumerate(zip(TEMPLATES, rates)):
        tmpl = _template(FS, w, tr, pk, tp)
        isi = rng.exponential(1 / rate, int(rate * duration_s * 2)) + 0.002
        times = np.cumsum(isi)
        times = times[times < duration_s - 0.01]
        for t in times:
            i = int(t * FS)
            x[i:i + len(tmpl)] += tmpl * rng.normal(1, 0.05)
            truth.append((i + int(0.5e-3 * FS), unit))  # sample of the trough
    truth.sort()
    return x, np.array(truth)


def sort_spikes(x, fs=FS, threshold=4.5, max_units=6, seed=0):
    sos = signal.butter(3, [300, 3000], btype="bandpass", fs=fs, output="sos")
    f = signal.sosfiltfilt(sos, x)
    sigma = np.median(np.abs(f)) / 0.6745
    thr = -threshold * sigma
    cross = np.flatnonzero((f[1:] < thr) & (f[:-1] >= thr)) + 1
    pre, post = int(0.5e-3 * fs), int(1.1e-3 * fs)
    troughs, waves, last = [], [], -10 ** 9
    for c in cross:
        seg = f[c:c + int(0.5e-3 * fs)]
        tr = c + int(np.argmin(seg))
        if tr - last < int(1e-3 * fs) or tr - pre < 0 or tr + post > len(f):
            continue  # 1 ms dead time, edge spikes
        troughs.append(tr)
        waves.append(f[tr - pre:tr + post])
        last = tr
    waves = np.array(waves)
    feats = PCA(3, random_state=seed).fit_transform(waves)
    fits = [GaussianMixture(k, n_init=3, random_state=seed).fit(feats) for k in range(1, max_units + 1)]
    bics = [g.bic(feats) for g in fits]
    best = fits[int(np.argmin(bics))]
    return dict(filtered=f, sigma=sigma, threshold=thr, troughs=np.array(troughs), waves=waves,
                features=feats, labels=best.predict(feats), bics=bics, n_units=best.n_components)


def score(truth, troughs, labels, tol=int(0.4e-3 * FS)):
    """Match detections to true spikes within +/-0.4 ms, then map clusters to units by majority."""
    t_true = truth[:, 0]
    j = np.clip(np.searchsorted(t_true, troughs), 1, len(t_true) - 1)
    nearest = np.where(np.abs(t_true[j - 1] - troughs) < np.abs(t_true[j] - troughs), j - 1, j)
    hit = np.abs(t_true[nearest] - troughs) <= tol
    true_unit = truth[nearest, 1]
    mapping = {int(c): int(np.bincount(true_unit[hit & (labels == c)], minlength=3).argmax())
               for c in np.unique(labels) if (hit & (labels == c)).any()}
    pred_unit = np.array([mapping.get(int(c), -1) for c in labels])
    per_unit = {f"unit {u}": float(np.mean(pred_unit[hit & (true_unit == u)] == u)) for u in range(3)}
    return dict(recall=float(len(np.unique(nearest[hit])) / len(t_true)), precision=float(hit.mean()),
                sorting_accuracy=float(np.mean(pred_unit[hit] == true_unit[hit])), per_unit=per_unit,
                n_true=int(len(t_true)), n_detected=int(len(troughs)), cluster_to_unit=mapping)


def main():
    from neuralsignal.paths import FIGURES_DIR, RESULTS_DIR
    x, truth = simulate()
    out = sort_spikes(x)
    sc = score(truth, out["troughs"], out["labels"])
    sc["n_units_found"] = int(out["n_units"])
    # Repeat on fresh simulations: how stable is the pipeline (and BIC's unit count)?
    reps = []
    for seed in range(1, 5):
        xs, ts = simulate(seed=seed)
        o = sort_spikes(xs)
        r = score(ts, o["troughs"], o["labels"])
        reps.append(dict(seed=seed, n_units_found=int(o["n_units"]), recall=r["recall"],
                         sorting_accuracy=r["sorting_accuracy"]))
    sc["other_seeds"] = reps
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runs" / "spike-sorting-demo.json").write_text(json.dumps(sc, indent=2))
    print(json.dumps(sc, indent=2))
    from neuralsignal.report import plot_spikesort
    plot_spikesort(x, out, truth, sc, FIGURES_DIR / "spike-sorting-demo.png")


if __name__ == "__main__":
    main()
