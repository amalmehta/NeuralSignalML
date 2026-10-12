"""Aggregate every run into results/Results.md and draw the figures in docs/figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neuralsignal.evaluation import metrics as M  # noqa: E402
from neuralsignal.paths import FIGURES_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402

COLORS = {"csp_lda": "#4C78A8", "eegnet": "#F58518", "kalman": "#4C78A8", "gru": "#F58518"}
LABELS = {"csp_lda": "CSP + LDA", "eegnet": "EEGNet", "kalman": "Kalman filter", "gru": "GRU"}
PROTO = {"cross_session": "Held-out session (train day 1, test day 2)",
         "cross_subject": "Held-out subject (train on the other 8)",
         "nlb_split": "NLB split (188 train / 62 test trials)",
         "blocked_cv": "5-fold blocked CV (all 250 trials)"}
# Approximate 2D layout of the 2a montage (10-20 positions on a 3.5 cm grid)
LAYOUT = {"Fz": (0, 3), "FC3": (-2, 2), "FC1": (-1, 2), "FCz": (0, 2), "FC2": (1, 2), "FC4": (2, 2),
          "C5": (-3, 1), "C3": (-2, 1), "C1": (-1, 1), "Cz": (0, 1), "C2": (1, 1), "C4": (2, 1),
          "C6": (3, 1), "CP3": (-2, 0), "CP1": (-1, 0), "CPz": (0, 0), "CP2": (1, 0), "CP4": (2, 0),
          "P1": (-1, -1), "Pz": (0, -1), "P2": (1, -1), "POz": (0, -2)}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10})


def _runs() -> dict:
    out = {}
    for p in sorted((RESULTS_DIR / "runs").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        if "method" in m:  # the single-method experiments; Phase 2 runs are reported by report_phase2
            out[m["method"]] = m
    return out


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote docs/figures/{name}")


# ------------------------------------------------------------------ EEG figures

def plot_eeg_subjects(runs):
    methods = [m for m in ("csp_lda", "eegnet") if m in runs]
    protos = [p for p in ("cross_session", "cross_subject") if p in runs[methods[0]]["protocols"]]
    fig, axes = plt.subplots(1, len(protos), figsize=(5.2 * len(protos), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, proto in zip(axes, protos):
        subs = list(runs[methods[0]]["protocols"][proto]["per_subject_accuracy"])
        x = np.arange(len(subs) + 1)
        w = 0.8 / len(methods)
        for k, m in enumerate(methods):
            r = runs[m]["protocols"].get(proto)
            if not r:
                continue
            acc = [r["per_subject_accuracy"][s] for s in subs] + [r["mean_accuracy"]]
            ax.bar(x + (k - (len(methods) - 1) / 2) * w, np.array(acc) * 100, w, color=COLORS[m],
                   label=f"{LABELS[m]} ({r['mean_accuracy']*100:.1f}%)")
        ch = runs[methods[0]]["protocols"][proto]["chance_95"] * 100
        ax.axhline(25, color="grey", lw=0.8, ls=":")
        ax.axhline(ch, color="grey", lw=0.8, ls="--")
        ax.text(len(subs) + 0.45, ch + 1, "chance (p<.05)", ha="right", fontsize=7, color="grey")
        ax.set_xticks(x, [s.replace("A0", "S") for s in subs] + ["mean"])
        ax.set_title(PROTO[proto])
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].set_ylabel("4-class accuracy (%)")
    fig.suptitle("Motor imagery decoding, BCI Competition IV 2a (9 subjects, 288 test trials each)", y=1.02)
    _save(fig, "eeg-accuracy-by-subject.png")


def plot_confusions(runs):
    from neuralsignal.data.bci2a import CLASS_NAMES
    methods = [m for m in ("csp_lda", "eegnet") if m in runs]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.2))
    for ax, m in zip(np.atleast_1d(axes), methods):
        d = np.load(RESULTS_DIR / "runs" / runs[m]["id"] / "predictions.npz", allow_pickle=True)
        C = np.zeros((4, 4))
        for key in d.files:
            if key.startswith("cross_session/") and key.endswith("/proba") and "/seed0/" in key:
                subj = key.split("/")[1]
                y, yh = d[f"cross_session/{subj}/y"], d[key].argmax(1)
                np.add.at(C, (y, yh), 1)
        C = C / C.sum(1, keepdims=True)
        ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{C[i, j]*100:.0f}", ha="center", va="center",
                        color="white" if C[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(4), CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_title(f"{LABELS[m]}, held-out session")
    np.atleast_1d(axes)[0].set_ylabel("true")
    fig.suptitle("Confusion matrices pooled over 9 subjects (% of each true class)", y=1.02)
    _save(fig, "eeg-confusion.png")


def _scalp(ax, values, names, title):
    from scipy.interpolate import griddata
    pts = np.array([LAYOUT[n] for n in names], float)
    gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 120), np.linspace(-2.5, 3.5, 120))
    gz = griddata(pts, values, (gx, gy), method="cubic")
    head = (gx ** 2 / 3.6 ** 2 + (gy - 0.5) ** 2 / 3.2 ** 2) <= 1
    gz[~head] = np.nan
    v = np.nanmax(np.abs(values))
    ax.contourf(gx, gy, gz, 20, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c="k")
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(3.6 * np.cos(t), 0.5 + 3.2 * np.sin(t), "k", lw=0.8)
    ax.plot([-0.4, 0, 0.4], [3.65, 4.0, 3.65], "k", lw=0.8)  # nose
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def plot_csp_patterns(subject="A03"):
    import joblib
    from neuralsignal.data.bci2a import CLASS_NAMES, EEG_CHANNELS
    path = MODELS_DIR / "eeg-csp-lda" / "cross_session" / subject / "model.joblib"
    if not path.exists():
        return
    model = joblib.load(path)
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for ax, A, name in zip(axes, model.csp.patterns_, CLASS_NAMES):
        _scalp(ax, A[:, -1], EEG_CHANNELS, f"{name} vs rest")
    fig.suptitle(f"CSP spatial patterns, subject {int(subject[1:])}: where each class's most "
                 "distinctive power change sits on the scalp", y=1.04)
    _save(fig, "csp-patterns.png")


def plot_erd(subject=3):
    """The physiology CSP exploits: mu/beta power drops over the motor cortex opposite the imagined hand."""
    from scipy import signal as sg
    from neuralsignal.data import bci2a
    from neuralsignal.preprocess.eeg import epoch_session
    sess = bci2a.load(subject, "T")
    ep = epoch_session(sess, band=(1, 45), tmin=0.5, tmax=3.5, reject_uv=None)
    f, P = sg.welch(ep.X, fs=ep.fs, nperseg=256, axis=-1)
    ch = {"C3": bci2a.EEG_CHANNELS.index("C3"), "C4": bci2a.EEG_CHANNELS.index("C4")}
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for ax, (name, c) in zip(axes, ch.items()):
        for cls, col in ((0, "#4C78A8"), (1, "#F58518")):
            keep = (f >= 4) & (f <= 35)
            ax.plot(f[keep], P[ep.y == cls, c].mean(0)[keep], color=col, lw=1.6,
                    label=f"imagine {bci2a.CLASS_NAMES[cls]}")
        ax.axvspan(8, 30, color="grey", alpha=0.1)
        ax.set_xlim(4, 35)
        ax.set_title(f"{name} ({'left' if name == 'C3' else 'right'} motor cortex)")
        ax.set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("power (uV^2/Hz)")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Event-related desynchronisation, subject {subject}: imagining one hand lowers "
                 "8-30 Hz power over the opposite hemisphere", y=1.03)
    _save(fig, "eeg-erd-spectra.png")


def plot_preprocessing(subject=1):
    from neuralsignal.data import bci2a
    from neuralsignal.preprocess.eeg import bandpass, fit_eog_regression
    sess = bci2a.load(subject, "T")
    B = fit_eog_regression(sess)
    run = [r for r in sess.runs if not r.is_calibration][0]
    fs = sess.fs
    t0 = run.trial_starts[3]
    sl = slice(t0, t0 + int(7.5 * fs))
    c = bci2a.EEG_CHANNELS.index("Fz")
    raw = np.nan_to_num(run.eeg)
    clean = raw - B @ np.nan_to_num(run.eog)
    t = np.arange(sl.stop - sl.start) / fs
    fig, axes = plt.subplots(3, 1, figsize=(8, 4.6), sharex=True)
    axes[0].plot(t, bandpass(raw, fs, 0.5, 45)[c, sl], color="#999", lw=0.7, label="raw (0.5-45 Hz)")
    axes[0].plot(t, bandpass(clean, fs, 0.5, 45)[c, sl], color="#4C78A8", lw=0.7, label="after EOG regression")
    axes[0].set_title("Fz, one trial: eye-artefact removal")
    axes[1].plot(t, bandpass(clean, fs, 8, 30)[c, sl], color="#F58518", lw=0.7)
    axes[1].set_title("band-passed 8-30 Hz (mu + beta) for CSP")
    from scipy import signal as sg
    f, tt, S = sg.spectrogram(clean[bci2a.EEG_CHANNELS.index("C3"), sl], fs=fs, nperseg=125, noverlap=110)
    axes[2].pcolormesh(tt, f, 10 * np.log10(S + 1e-6), shading="auto", cmap="magma")
    axes[2].set_ylim(1, 40)
    axes[2].set_ylabel("Hz")
    axes[2].set_title("C3 spectrogram")
    for ax in axes:
        ax.axvline(2.0, color="k", lw=0.8, ls="--")
        ax.axvspan(2.5, 4.5, color="green", alpha=0.08)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[2].set_xlabel("seconds from trial start (cue at 2 s, shaded = epoch used)")
    fig.tight_layout()
    _save(fig, "eeg-preprocessing.png")


# ------------------------------------------------------------------ spike figures

def plot_spike_r2(runs):
    methods = [m for m in ("kalman", "gru") if m in runs]
    protos = ["nlb_split", "blocked_cv"]
    fig, ax = plt.subplots(figsize=(5.6, 3))
    x = np.arange(len(protos))
    w = 0.38
    for k, m in enumerate(methods):
        rs = [runs[m]["protocols"][p] for p in protos]
        vals = [r["mean_r2"] for r in rs]
        lo = [r["mean_r2"] - r["ci95_trials"][0] for r in rs]
        hi = [r["ci95_trials"][1] - r["mean_r2"] for r in rs]
        ax.bar(x + (k - 0.5) * w, vals, w, yerr=[lo, hi], capsize=3, color=COLORS[m], label=LABELS[m])
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, 0.02, f"{v:.2f}", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(x, [PROTO[p] for p in protos], fontsize=8)
    ax.set_ylabel("hand-velocity R²")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Hand-velocity decoding, MC_Maze Medium (error bars: 95% CI over trials)")
    _save(fig, "spikes-r2.png")


def plot_trajectories(runs):
    methods = [m for m in ("kalman", "gru") if m in runs]
    d = {m: np.load(RESULTS_DIR / "runs" / runs[m]["id"] / "predictions.npz", allow_pickle=True) for m in methods}
    first = d[methods[0]]
    Y, mask, split = first["Y"], first["mask"], first["split"]
    val = np.flatnonzero(split == "val")[:24]
    cfg = json.loads((RESULTS_DIR / "runs" / runs[methods[0]]["id"] / "metrics.json").read_text())
    import yaml
    pp = yaml.safe_load((RESULTS_DIR / "runs" / cfg["id"] / "config.resolved.yaml").read_text())["preprocess"]
    dt = pp["bin_ms"] / 1000
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4 * (len(methods) + 1), 3.8), sharex=True, sharey=True)
    cmap = plt.get_cmap("hsv")

    def traj(V):
        return np.cumsum(V[mask] * dt, axis=0)
    for i, tr in enumerate(val):
        c = cmap(i / len(val))
        axes[0].plot(*traj(Y[tr]).T, color=c, lw=1)
        for ax, m in zip(axes[1:], methods):
            ax.plot(*traj(d[m]["nlb_split/seed0"][tr]).T, color=c, lw=1)
    axes[0].set_title("true reaches (held-out trials)")
    for ax, m in zip(axes[1:], methods):
        ax.set_title(f"{LABELS[m]} (R² {runs[m]['protocols']['nlb_split']['mean_r2']:.2f})")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("mm")
    axes[0].set_ylabel("mm")
    fig.suptitle("Reach paths rebuilt by integrating decoded velocity, from 250 ms before movement onset "
                 "to 450 ms after", y=1.0)
    _save(fig, "spikes-trajectories.png")


def plot_velocity_traces(runs):
    methods = [m for m in ("kalman", "gru") if m in runs]
    d = {m: np.load(RESULTS_DIR / "runs" / runs[m]["id"] / "predictions.npz", allow_pickle=True) for m in methods}
    Y, split = d[methods[0]]["Y"], d[methods[0]]["split"]
    val = np.flatnonzero(split == "val")[:4]
    fig, axes = plt.subplots(2, 4, figsize=(10, 3.8), sharex=True)
    for j, tr in enumerate(val):
        for i, comp in enumerate("xy"):
            ax = axes[i, j]
            ax.plot(Y[tr, :, i], "k", lw=1.4, label="true")
            for m in methods:
                ax.plot(d[m]["nlb_split/seed0"][tr, :, i], color=COLORS[m], lw=1, label=LABELS[m])
            if j == 0:
                ax.set_ylabel(f"v{comp} (mm/s)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("50 ms bins")
    fig.suptitle("Decoded vs true hand velocity on four held-out trials", y=1.0)
    fig.tight_layout()
    _save(fig, "spikes-velocity-traces.png")


def plot_spikesort(x, out, truth, sc, path):
    fs = 30_000
    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :])
    n = int(0.25 * fs)
    t = np.arange(n) / fs * 1000
    ax.plot(t, x[:n], color="#bbb", lw=0.5, label="broadband")
    ax.plot(t, out["filtered"][:n], color="k", lw=0.6, label="300-3000 Hz")
    ax.axhline(out["threshold"], color="r", lw=0.8, ls="--", label="-4.5 sigma threshold")
    cols = plt.get_cmap("tab10")
    for tr, lab in zip(out["troughs"], out["labels"]):
        if tr < n:
            ax.plot(tr / fs * 1000, out["filtered"][tr] - 15, "v", color=cols(lab), ms=5)
    ax.set_xlabel("ms")
    ax.set_ylabel("uV")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    ax.set_title("1. filter and detect threshold crossings (markers coloured by assigned unit)")
    ax2 = fig.add_subplot(gs[1, 0])
    tw = (np.arange(out["waves"].shape[1]) / fs - 0.5e-3) * 1000
    for k in np.unique(out["labels"]):
        w = out["waves"][out["labels"] == k]
        ax2.plot(tw, w[:40].T, color=cols(k), lw=0.3, alpha=0.5)
        ax2.plot(tw, w.mean(0), color=cols(k), lw=2)
    ax2.set_title("2. aligned waveforms")
    ax2.set_xlabel("ms")
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(out["features"][:, 0], out["features"][:, 1], c=[cols(k) for k in out["labels"]], s=3)
    ax3.set_title("3. PCA features + Gaussian mixture")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(range(1, len(out["bics"]) + 1), out["bics"], "o-", color="k")
    ax4.axvline(out["n_units"], color="r", ls="--", lw=0.8)
    ax4.set_title(f"4. BIC picks {out['n_units']} units (truth: 3)")
    ax4.set_xlabel("clusters")
    fig.suptitle(f"Spike sorting on synthetic data: detection recall {sc['recall']*100:.0f}%, "
                 f"precision {sc['precision']*100:.0f}%, sorting accuracy {sc['sorting_accuracy']*100:.0f}%")
    fig.tight_layout()
    _save(fig, Path(path).name)


# ------------------------------------------------------------------ tables

def results_markdown(runs) -> str:
    L = ["# Results", "",
         "Generated by `neuralsignal report` from `results/runs/*/metrics.json`. Don't edit by hand.", ""]
    eeg = [m for m in ("csp_lda", "eegnet") if m in runs]
    if eeg:
        L += ["## Motor imagery EEG: BCI Competition IV 2a", "",
              "4-class accuracy on each subject's 288-trial evaluation session (chance 25%; a random "
              f"classifier beats {M.chance_upper(288, 4)*100:.1f}% only 5% of the time).", ""]
        for proto in ("cross_session", "cross_subject"):
            if proto not in runs[eeg[0]]["protocols"]:
                continue
            subs = list(runs[eeg[0]]["protocols"][proto]["per_subject_accuracy"])
            L += [f"### {PROTO[proto]}", "", "| Method | " + " | ".join(s.replace("A0", "S") for s in subs)
                  + " | Mean +/- SD | Kappa |", "|---" * (len(subs) + 3) + "|"]
            for m in eeg:
                r = runs[m]["protocols"].get(proto)
                if r:
                    L.append(f"| {LABELS[m]} | " + " | ".join(f"{r['per_subject_accuracy'][s]*100:.1f}" for s in subs)
                             + f" | **{r['mean_accuracy']*100:.1f} +/- {r['std_accuracy']*100:.1f}** | {r['mean_kappa']:.2f} |")
            if len(eeg) == 2 and all(proto in runs[m]["protocols"] for m in eeg):
                a = [runs["csp_lda"]["protocols"][proto]["per_subject_accuracy"][s] for s in subs]
                b = [runs["eegnet"]["protocols"][proto]["per_subject_accuracy"][s] for s in subs]
                wins = sum(y > x for x, y in zip(a, b))
                L += ["", f"EEGNet beats CSP + LDA on {wins}/9 subjects; mean difference "
                      f"{(np.mean(b) - np.mean(a))*100:+.1f} points; Wilcoxon signed-rank p = {M.paired_wilcoxon(a, b):.3f}."]
            L.append("")
    sp = [m for m in ("kalman", "gru") if m in runs]
    if sp:
        L += ["## Hand velocity from motor-cortex spikes: NLB MC_Maze Medium", "",
              "Variance-weighted R² of 2D hand velocity in the window 250 ms before to 450 ms after "
              "movement onset. CI = 95% bootstrap over test trials; seeds = spread over training seeds.", "",
              "| Method | Protocol | R² | 95% CI (trials) | SD over seeds | Test trials |", "|---|---|---|---|---|---|"]
        for m in sp:
            for proto in ("nlb_split", "blocked_cv"):
                r = runs[m]["protocols"][proto]
                L.append(f"| {LABELS[m]} | {PROTO[proto]} | **{r['mean_r2']:.3f}** | {r['ci95_trials'][0]:.3f} - "
                         f"{r['ci95_trials'][1]:.3f} | {r['std_r2_over_seeds']:.3f} | {r['n_test_trials']} |")
        if len(sp) == 2:
            L.append("")
            for proto in ("nlb_split", "blocked_cv"):
                d = {m: np.load(RESULTS_DIR / "runs" / runs[m]["id"] / "predictions.npz", allow_pickle=True) for m in sp}
                k = d["kalman"]
                te = np.flatnonzero(k["split"] == "val") if proto == "nlb_split" else np.arange(len(k["Y"]))
                b = M.paired_bootstrap_r2_diff(k["Y"][te], k[f"{proto}/seed0"][te], d["gru"][f"{proto}/seed0"][te], k["mask"])
                L.append(f"- {PROTO[proto]}: GRU - Kalman = **{b['diff']:+.3f}** R\u00b2 (paired bootstrap 95% CI "
                         f"{b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f}; share of resamples where GRU is not better: {b['p_not_better']:.3f}).")
        L.append("")
    ss = RESULTS_DIR / "runs" / "spike-sorting-demo.json"
    if ss.exists():
        s = json.loads(ss.read_text())
        L += ["## Spike sorting demo (synthetic ground truth)", "",
              f"{s['n_true']} true spikes from 3 units; {s['n_detected']} detections; recall "
              f"{s['recall']*100:.1f}%, precision {s['precision']*100:.1f}%, sorting accuracy "
              f"{s['sorting_accuracy']*100:.1f}%, units found by BIC: {s['n_units_found']}.", ""]
    from neuralsignal import report_phase2
    report_phase2.build(L)
    L += ["## Run details", "", "| Run | Config hash | Minutes |", "|---|---|---|"]
    every = [json.loads(q.read_text()) for q in sorted((RESULTS_DIR / "runs").glob("*/metrics.json"))]
    L += [f"| {r['name']} | `{r['config_hash']}` | {r['minutes']} |" for r in every if "config_hash" in r]
    return "\n".join(L) + "\n"


def main():
    runs = _runs()
    print("Building report...")
    if "csp_lda" in runs or "eegnet" in runs:
        plot_eeg_subjects(runs)
        plot_confusions(runs)
        plot_csp_patterns()
        plot_erd()
        plot_preprocessing()
    if "kalman" in runs:
        plot_spike_r2(runs)
        plot_trajectories(runs)
        plot_velocity_traces(runs)
    (RESULTS_DIR / "Results.md").write_text(results_markdown(runs))
    print("  wrote results/Results.md")
