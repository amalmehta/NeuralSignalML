"""Figures and tables for the Phase 2 experiments (cross-day, calibration-free EEG, NLB/NDT,
closed loop, latency). Called from `neuralsignal report`; each part is skipped if its run
hasn't happened yet."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuralsignal.paths import RESULTS_DIR
from neuralsignal.report import _save, plt

C = {"kalman": "#4C78A8", "gru": "#F58518"}
LBL = {"kalman": "Kalman filter", "gru": "GRU"}


def _load(run_id):
    p = RESULTS_DIR / "runs" / run_id / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ cross-day

CD_METHODS = [("within_day", "Recalibrated\n(same day)"), ("no_adaptation", "Yesterday's\ndecoder as is"),
              ("renormalize", "Re-normalise\n(no labels)"), ("latent_align", "Latent alignment\n(no labels)"),
              ("fewshot", "Refit with\n20 labelled trials")]


def crossday(L):
    m = _load("spikes-crossday")
    if not m:
        return
    df = pd.DataFrame(m["rows"])
    df = df[(df.k.isna()) | (df.k == 20) | (df.method == "within_day")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, (gap_name, gaps) in zip(axes, [("1 day apart", [1]), ("7-8 days apart", [7, 8])]):
        x = np.arange(len(CD_METHODS))
        for j, dec in enumerate(["kalman", "gru"]):
            vals, errs = [], []
            for meth, _ in CD_METHODS:
                if meth == "within_day":
                    tg = df[(df.method == "fewshot") & df.gap_days.isin(gaps)].target.unique()
                    sel = df[(df.method == meth) & (df.decoder == dec) & df.target.isin(tg)]
                else:
                    sel = df[(df.method == meth) & (df.decoder == dec) & df.gap_days.isin(gaps) & ((df.k == 20) | df.k.isna())]
                vals.append(sel.r2.mean())
                errs.append(sel.groupby(["source", "target"]).r2.mean().std() if len(sel) > 1 else 0)
            ax.bar(x + (j - 0.5) * 0.38, np.clip(vals, -0.5, None), 0.38, color=C[dec], label=LBL[dec])
            for xi, v in zip(x + (j - 0.5) * 0.38, vals):
                ax.text(xi, max(v, -0.5) + (0.02 if v >= 0 else -0.07), f"{v:.2f}", ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [n for _, n in CD_METHODS], fontsize=7.5)
        ax.set_title(f"Decoding a recording day {gap_name}")
        ax.set_ylim(-0.55, 0.9)
    axes[0].set_ylabel("hand-velocity R² (held-out trials)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Monkey J, three days (Sep 28, Sep 29, Oct 6 2009): what each fix recovers when the decoder is a day or a week old", y=1.03)
    _save(fig, "crossday-methods.png")

    full = pd.DataFrame(m["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, dec in zip(axes, ["kalman", "gru"]):
        for meth, style, name in [("fewshot", "-o", "reuse yesterday's decoder, refit the drifting part"),
                                  ("scratch", "--s", "train from scratch on the same trials")]:
            s = full[(full.method == meth) & (full.decoder == dec)].groupby("k").r2.mean()
            ax.plot(s.index, s.values, style, color=C[dec], label=name, alpha=1 if meth == "fewshot" else 0.6)
        ref = full[(full.method == "within_day") & (full.decoder == dec)].r2.mean()
        ren = full[(full.method == "renormalize") & (full.decoder == dec)].r2.mean()
        ax.axhline(ref, color="k", ls=":", lw=1)
        ax.text(41, ref, " full day of\n calibration", va="center", fontsize=7)
        ax.axhline(ren, color="grey", ls="-.", lw=1)
        ax.text(41, ren, " re-normalise,\n 0 labels", va="center", fontsize=7, color="grey")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 40], ["5", "10", "20", "40"])
        ax.set_xlabel("labelled trials from the new day")
        ax.set_title(LBL[dec])
        ax.set_ylim(-0.3, 0.9)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_ylabel("R² (mean over 6 day pairs)")
    fig.suptitle("How much new-day calibration is needed?", y=1.02)
    _save(fig, "crossday-fewshot.png")

    L += ["## Decoders across days: MC_Maze Small, Medium, Large (same monkey, 3 days)", "",
          "Electrode-level spike counts (units summed per electrode). Mean R² over every ordered pair of days "
          "(6 pairs) and seeds; tested on the target day's NLB val trials, which no method sees.", "",
          "| Method | Labels from new day | Kalman | GRU |", "|---|---|---|---|"]
    names = {"within_day": ("Recalibrated on the new day (reference)", "all train trials"),
             "latent_within_day": ("Latent decoder, same day (ceiling for alignment)", "all train trials"),
             "no_adaptation": ("Yesterday's decoder as is", "0"), "renormalize": ("Re-normalise channels", "0"),
             "latent_align": ("Latent alignment (Degenhart et al. 2020)", "0")}
    for meth, (name, lab) in names.items():
        v = {d: full[(full.method == meth) & (full.decoder == d)].r2.mean() for d in ("kalman", "gru")}
        L.append(f"| {name} | {lab} | {v['kalman']:.3f} | {v['gru']:.3f} |")
    for k in sorted(full.k.dropna().unique()):
        for meth, name in (("fewshot", "Refit drifting part"), ("scratch", "Train from scratch")):
            v = {d: full[(full.method == meth) & (full.decoder == d) & (full.k == k)].r2.mean() for d in ("kalman", "gru")}
            L.append(f"| {name} | {int(k)} trials | {v['kalman']:.3f} | " + (f"{v['gru']:.3f} |" if not np.isnan(v['gru']) else "n/a |"))
    L.append("")


# ------------------------------------------------------------------ EEG calibration

EA = {"csp_lda": ("CSP + LDA", "#4C78A8"), "ea_csp_lda": ("Euclidean-aligned CSP + LDA", "#72B7B2"),
      "ts_lda": ("Tangent space + LDA", "#B279A2"), "ra_ts_lda": ("Riemannian re-centred tangent space + LDA", "#E45756"),
      "ea_eegnet": ("Euclidean-aligned EEGNet", "#F58518")}


def eeg_calibration(L):
    m = _load("eeg-calibration")
    if not m:
        return
    df = pd.DataFrame(m["rows"])
    zero = df[(df.k_per_class == 0) & (df.variant == "pooled")].groupby(["method", "subject"]).accuracy.mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    meths = [k for k in EA if k in zero.index]
    ax.bar(range(len(meths)), [zero.loc[k].mean() * 100 for k in meths], color=[EA[k][1] for k in meths])
    for i, k in enumerate(meths):
        ax.text(i, zero.loc[k].mean() * 100 + 1, f"{zero.loc[k].mean()*100:.1f}%", ha="center", fontsize=8)
        ax.scatter(np.full(9, i) + np.linspace(-0.2, 0.2, 9), zero.loc[k].values * 100, s=6, color="k", zorder=3)
    ax.axhline(m["chance_95"] * 100, color="grey", ls="--", lw=0.8)
    ax.set_xticks(range(len(meths)), [EA[k][0].replace(" + ", "\n+ ").replace("-aligned ", "-aligned\n") for k in meths], fontsize=7)
    ax.set_ylabel("accuracy on the new person (%)")
    ax.set_title("Zero calibration: trained on 8 other people")
    ax.set_ylim(0, 100)
    ax = axes[1]
    for k in [k for k in ("csp_lda", "ea_csp_lda", "ra_ts_lda") if k in df.method.unique()]:
        for variant, style in (("pooled", "-o"), ("own_only", "--s")):
            s = df[(df.method == k) & (df.variant == variant)].groupby("k_per_class").accuracy.mean() * 100
            if variant == "own_only":
                s = s[s.index > 0]
            ax.plot(s.index * 4, s.values, style, color=EA[k][1], alpha=1 if variant == "pooled" else 0.5,
                    label=f"{EA[k][0]}, " + ("others + new person" if variant == "pooled" else "new person only"))
        full = df[(df.method == k) & (df.variant == "full_calibration")].accuracy.mean() * 100
        ax.axhline(full, color=EA[k][1], ls=":", lw=1)
    ax.set_xlabel("labelled trials from the new person (4 classes)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Calibration curve (dotted: a full 288-trial session)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_ylim(20, 85)
    fig.suptitle("BCI Competition IV 2a: decoding someone new, with little or none of their own labelled data", y=1.03)
    _save(fig, "eeg-calibration.png")

    L += ["## EEG without calibration: BCI IV 2a", "",
          "Test: each person's day-2 session. Alignment uses only their *unlabelled* day-1 EEG. Mean over 9 people "
          f"(chance p<.05: {m['chance_95']*100:.1f}%).", "",
          "| Method | Zero calibration | 20 labelled trials | 80 labelled trials | 160 labelled trials | Full session, own data only |",
          "|---|---|---|---|---|---|"]
    for k in [k for k in EA if k in df.method.unique()]:
        def acc(kpc, variant="pooled"):
            s = df[(df.method == k) & (df.k_per_class == kpc) & (df.variant == variant)].accuracy
            return f"{s.mean()*100:.1f}%" if len(s) else "-"
        L.append(f"| {EA[k][0]} | **{acc(0)}** | {acc(5)} | {acc(20)} | {acc(40)} | {acc(-1, 'full_calibration')} |")
    L.append("")


# ------------------------------------------------------------------ NLB / NDT

def nlb(L):
    m = _load("spikes-nlb")
    if not m:
        return
    df = pd.DataFrame(m["rows"])
    agg = df.groupby("model").agg(co_bps=("co_bps", "mean"), co_sd=("co_bps", "std"), vel=("vel_r2", "mean"), n=("seed", "count"))
    pub = m.get("published_test_split", {}).get("MC_Maze_Medium_co_bps", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    order = [k for k in ("smoothing", "ndt") if k in agg.index]
    names = {"smoothing": "Smoothing", "ndt": "NDT"}
    axes[0].bar(range(len(order)), agg.loc[order].co_bps, yerr=agg.loc[order].co_sd.fillna(0), capsize=3,
                color=["#999", "#F58518"][:len(order)])
    axes[0].set_xticks(range(len(order)), [names[k] for k in order])
    axes[0].set_ylabel("co-smoothing bits/spike")
    axes[0].set_title("This project (NLB val split)")
    axes[1].bar(range(len(pub)), list(pub.values()), color="#ccc")
    axes[1].set_xticks(range(len(pub)), list(pub.keys()), fontsize=8)
    axes[1].set_title("Published (NLB'21, hidden test split)")
    for ax in axes:
        ax.set_ylim(0, max(0.35, float(agg.co_bps.max()) + 0.05))
    fig.suptitle("MC_Maze Medium co-smoothing: predicting held-out neurons from held-in ones", y=1.03)
    _save(fig, "nlb-cobps.png")
    L += ["## Neural Latents Benchmark protocol: MC_Maze Medium", "",
          "co-bps on the NLB 'val' split (5 ms bins, -250 to +450 ms around movement onset); vel R² = ridge from "
          "inferred rates to hand velocity (acausal: uses the whole window).", "",
          "| Model | co-bps (val) | vel R² (val) | Seeds | Published co-bps (test split) |", "|---|---|---|---|---|"]
    for k in order:
        r = agg.loc[k]
        L.append(f"| {names[k]} | **{r.co_bps:.3f}**" + (f" ± {r.co_sd:.3f}" if r.n > 1 else "")
                 + f" | {r.vel:.3f} | {int(r.n)} | {pub.get(names[k], '-')} |")
    L.append("")


# ------------------------------------------------------------------ closed loop

def closedloop(L):
    m = _load("spikes-closedloop")
    if not m:
        return
    df = pd.DataFrame(m["rows"])
    agg = df.groupby("decoder").agg(off=("offline_r2", "mean"), succ=("success_rate", "mean"),
                                    t=("median_time_s", "mean"), eff=("median_path_efficiency", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, dec in zip(axes[:2], ["kalman", "gru"]):
        for tr in m["paths"].get(f"{dec}/seed0", []):
            p = np.array(tr["path"])
            ax.plot(p[:, 0], p[:, 1], color=C[dec] if tr["success"] else "#d62728", lw=1)
            a = np.radians(tr["angle"])
            ax.add_patch(plt.Circle((100 * np.cos(a), 100 * np.sin(a)), m["simulation"]["target_radius_mm"],
                                    fill=False, color="grey", lw=0.8))
        ax.set_aspect("equal")
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_title(f"{LBL[dec]}: {agg.loc[dec].succ*100:.0f}% of reaches succeed")
        ax.set_xlabel("mm")
    ax = axes[2]
    for dec in agg.index:
        r = df[df.decoder == dec]
        ax.scatter(r.offline_r2, r.success_rate * 100, color=C[dec], label=LBL[dec], s=30)
    ax.set_xlabel("offline R² (simulated open-loop data)")
    ax.set_ylabel("closed-loop success (%)")
    ax.set_title("Offline accuracy vs closed-loop use")
    ax.legend(frameon=False)
    fig.suptitle("Simulated closed-loop cursor control (user steers using the decoded cursor; red = timed out)", y=1.03)
    _save(fig, "closedloop.png")
    L += ["## Simulated closed-loop cursor control", "",
          "Neurons simulated from Poisson GLMs fitted to MC_Maze Medium; simulated user with 100 ms visual delay "
          "steers toward 8 targets 100 mm away (15 mm radius, 250 ms hold, 3 s limit). Mean over 3 seeds x 40 reaches.", "",
          "| Decoder | Offline R² | Success | Median time to target | Path efficiency |", "|---|---|---|---|---|"]
    for dec, r in agg.iterrows():
        L.append(f"| {LBL[dec]} | {r.off:.3f} | **{r.succ*100:.0f}%** | {r.t:.2f} s | {r.eff:.2f} |")
    L.append("")


def latency(L):
    m = _load("latency")
    if not m:
        return
    L += ["## Online latency (each decoder run bin by bin, single thread)", "",
          f"Measured on {m['processor'] or 'this machine'} under a load average of {m['load_average']:.0f}.", "",
          "| Decoder | Median | 95th percentile | Worst | Budget per update |", "|---|---|---|---|---|"]
    for r in m["rows"]:
        L.append(f"| {r['decoder']} | {r['compute_ms_median']:.2f} ms | {r['compute_ms_p95']:.2f} ms | "
                 f"{r['compute_ms_max']:.0f} ms | {r['budget_ms']} ms |")
    L.append("")


def build(L):
    for part in (crossday, eeg_calibration, nlb, closedloop, latency):
        part(L)
