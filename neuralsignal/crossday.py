"""Decoders that survive across days: train on one recording day, decode another.

The problem. An implanted array records a slightly different mix of neurons each day
(micromotion, scarring, cells going quiet), so a decoder fitted on Monday degrades by
Tuesday. Clinical systems handle this by recalibrating every session - minutes of the
user's time, every day. This experiment measures how bad the drift is and how much each
fix recovers, on three MC_Maze recordings from the same monkey (Sep 28, Sep 29, Oct 6 2009).

Shared input space. Sorted units differ between days, but the electrodes don't, so every
decoder here reads spike counts summed per electrode (192 channels).

Methods, for every ordered (source day -> target day) pair; all tested on the target
day's NLB 'val' trials, which no method ever sees labels *or* spikes from:

  within_day      reference: fully recalibrated - trained on the target day's own train trials
  latent_within_day  the ceiling for latent alignment: factor space + decoder fitted on the same day
  no_adaptation   yesterday's decoder, used as is
  renormalize     per-channel mean/std from the target day's unlabelled train-trial spikes
  latent_align    factor analysis on each day, then rotate the target day's factors onto the
                  source day's using the channels whose loadings agree (Degenhart et al. 2020).
                  Unsupervised: uses target spikes, never target movements.
  fewshot_k       k labelled target trials: refit only the part that drifts (Kalman: the
                  observation model H,Q; GRU: the input layer) - vs. training from scratch on
                  the same k trials (scratch_k).
"""
from __future__ import annotations

import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import FactorAnalysis

from neuralsignal import config as cfgmod
from neuralsignal.data import mc_maze
from neuralsignal.evaluation import metrics as M
from neuralsignal.models.kalman import KalmanDecoder
from neuralsignal.models.torch_fit import GRURegressor
from neuralsignal.paths import RESULTS_DIR
from neuralsignal.preprocess.spikes import segment_trials


# ---------------------------------------------------------------- latent alignment

class LatentSpace:
    """Factor analysis on one day's z-scored sqrt counts (that day's own statistics)."""

    def __init__(self, X: np.ndarray, channels: np.ndarray, d: int, seed: int = 0):
        S = np.sqrt(X[..., channels]).reshape(-1, len(channels))
        self.channels, self.mu, self.sd = channels, S.mean(0), S.std(0) + 1e-3
        self.fa = FactorAnalysis(d, random_state=seed).fit((S - self.mu) / self.sd)
        self.L = self.fa.components_.T  # (channels, d) loadings
        self.R = np.eye(d)              # rotation into the reference day's latent coordinates

    def transform(self, X: np.ndarray) -> np.ndarray:
        S = (np.sqrt(X[..., self.channels]).reshape(-1, len(self.channels)) - self.mu) / self.sd
        return (self.fa.transform(S) @ self.R).reshape(*X.shape[:-1], -1).astype(np.float32)


def align(target: LatentSpace, source: LatentSpace, keep: float = 0.6, iters: int = 3) -> np.ndarray:
    """Find the rotation R with target.L @ R ~= source.L on the most stable channels.

    Start with every channel, then repeatedly drop the channels whose aligned loadings
    disagree most - those are the electrodes whose neurons changed - and refit.
    """
    stable = np.arange(len(source.channels))
    for _ in range(iters):
        R, _ = orthogonal_procrustes(target.L[stable], source.L[stable])
        err = np.linalg.norm(target.L @ R - source.L, axis=1)
        stable = np.argsort(err)[: int(keep * len(err))]
    target.R = R
    return stable


# ---------------------------------------------------------------- experiment

def _fit(kind, cfg, X, Y, P, mask, seed, transform="sqrt"):
    if kind == "kalman":
        return KalmanDecoder(bin_s=cfg["preprocess"]["bin_ms"] / 1000, **cfg.get("kalman", {})).fit(X, Y, P)
    m = GRURegressor(seed=seed, input_transform=transform, **cfg.get("gru", {}))
    return m.fit(X, Y, score_mask=mask)


def run(config_path: str, log=print) -> dict:
    import copy
    import torch
    cfg = cfgmod.load_config(config_path)
    out = RESULTS_DIR / "runs" / cfg["id"]
    cfgmod.write_run_header(cfg, out)
    if "threads" in cfg:
        torch.set_num_threads(int(cfg["threads"]))
    log(f"== {cfg['name']}  (config {cfg['_source']}, hash {cfg['_hash']})")
    t_start = time.time()
    pp, days, d = cfg["preprocess"], cfg["days"], cfg["latent_dim"]
    seg = {}
    for day in days:
        seg[day] = segment_trials(mc_maze.load_file(mc_maze.SESSIONS[day]), electrodes=mc_maze.N_ELECTRODES, **pp)
        log(f"  {day} ({mc_maze.SESSION_DATES[day]}): {seg[day].X.shape[0]} trials, "
            f"{int((seg[day].X.sum((0, 1)) > 0).sum())} active electrodes")
    mask = seg[days[0]].score_mask
    tr = {k: np.flatnonzero(v.split == "train") for k, v in seg.items()}
    te = {k: np.flatnonzero(v.split == "val") for k, v in seg.items()}
    rows, preds = [], {}

    bin_s = pp["bin_ms"] / 1000

    def active(day):  # channels a decoder trained on this day would read
        return np.flatnonzero(seg[day].X.mean((0, 1)) / bin_s >= cfg.get("min_rate_hz", 0.5))

    def score(method, kind, src, tgt, seed, Yhat, k=None, extra=None):
        y = seg[tgt].Y[te[tgt]]
        r = dict(method=method, decoder=kind, source=src, target=tgt, seed=seed, k=k,
                 gap_days=abs((np.datetime64(mc_maze.SESSION_DATES[tgt]) - np.datetime64(mc_maze.SESSION_DATES[src])).astype(int)),
                 r2=M.velocity_r2(y, Yhat, mask), ci95=M.bootstrap_r2(y, Yhat, mask, n_boot=500))
        r.update(extra or {})
        rows.append(r)
        preds[f"{kind}/{method}/{src}->{tgt}/seed{seed}" + (f"/k{k}" if k else "")] = Yhat
        log(f"  {kind:6s} {method:14s} {src:>6s}->{tgt:6s} seed {seed}" + (f" k={k:<3d}" if k else "       ")
            + f" R2 {r['r2']:.3f}")

    for kind in cfg["decoders"]:
        seeds = cfg["seeds"] if kind == "gru" else [0]
        for seed in seeds:
            cfgmod.seed_everything(seed)
            # within-day references (fully calibrated on the target day)
            for day in days:
                s, ch = seg[day], active(day)
                m = _fit(kind, cfg, s.X[tr[day]][..., ch], s.Y[tr[day]], s.P[tr[day]], mask, seed)
                score("within_day", kind, day, day, seed, m.predict(s.X[te[day]][..., ch]))
                # ceiling for latent alignment: latent space + decoder fitted on this same day
                sp = LatentSpace(s.X[tr[day]], ch, d, seed)
                m = _fit(kind, cfg, sp.transform(s.X[tr[day]]), s.Y[tr[day]], s.P[tr[day]], mask, seed, transform="none")
                score("latent_within_day", kind, day, day, seed, m.predict(sp.transform(s.X[te[day]])))
            for src in days:
                s, ch = seg[src], active(src)
                base = _fit(kind, cfg, s.X[..., ch], s.Y, s.P, mask, seed)  # all of the source day's trials
                for tgt in days:
                    if tgt == src:
                        continue
                    t = seg[tgt]
                    Xte, Xun = t.X[te[tgt]][..., ch], t.X[tr[tgt]][..., ch]  # test / unlabelled target spikes
                    score("no_adaptation", kind, src, tgt, seed, base.predict(Xte))
                    score("renormalize", kind, src, tgt, seed, copy.deepcopy(base).renormalize(Xun).predict(Xte))
                    # latent space on channels active on both days (an unsupervised check of target spikes)
                    both = np.intersect1d(ch, active(tgt))
                    src_space = LatentSpace(s.X, both, d, seed)
                    lat = _fit(kind, cfg, src_space.transform(s.X), s.Y, s.P, mask, seed, transform="none")
                    tgt_space = LatentSpace(t.X[tr[tgt]], both, d, seed)
                    stable = align(tgt_space, src_space, keep=cfg.get("stable_fraction", 0.6))
                    score("latent_align", kind, src, tgt, seed, lat.predict(tgt_space.transform(t.X[te[tgt]])),
                          extra=dict(stable_channels=int(len(stable))))
                    rng = np.random.default_rng(seed)
                    for k in cfg["fewshot_k"]:
                        if k > len(tr[tgt]):
                            continue
                        pick = rng.choice(tr[tgt], k, replace=False)
                        m = copy.deepcopy(base)
                        if kind == "kalman":
                            m.refit_observation(t.X[pick][..., ch], t.Y[pick], t.P[pick])
                        else:
                            m.finetune_input(t.X[pick][..., ch], t.Y[pick], mask, epochs=cfg.get("finetune_epochs", 150))
                        score("fewshot", kind, src, tgt, seed, m.predict(Xte), k=k)
                        # from scratch on the same k trials (GRU needs >= 10 to carve out a validation split)
                        scratch = _fit(kind, cfg, t.X[pick][..., ch], t.Y[pick], t.P[pick], mask, seed) \
                            if kind == "kalman" or k >= 10 else None
                        if scratch is not None:
                            score("scratch", kind, src, tgt, seed, scratch.predict(Xte), k=k)
    np.savez_compressed(out / "predictions.npz", **preds)
    summary = dict(name=cfg["name"], id=cfg["id"], experiment="crossday", config_hash=cfg["_hash"],
                   minutes=round((time.time() - t_start) / 60, 1), days=mc_maze.SESSION_DATES, rows=rows)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"== done in {summary['minutes']} min -> {out}")
    return summary
