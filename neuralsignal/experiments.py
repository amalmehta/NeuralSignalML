"""Run a config end to end: preprocess, train under each evaluation protocol, score, save.

EEG protocols (BCI 2a)
  cross_session   train on a subject's day-1 session (T), test on their day-2 session (E).
                  The standard competition protocol; tests robustness to day-to-day drift.
  cross_subject   leave one subject out: train on the other 8 subjects (T+E), test on the
                  held-out subject's E session. No calibration data from the test user at all.

Spike protocols (MC_Maze Medium - a single session, so "held-out session/subject" isn't
possible with this dataset; we hold out trials instead)
  nlb_split       train on the NLB 'train' trials, test on the NLB 'val' trials.
  blocked_cv      5 folds of *contiguous* trials in time, so neighbouring trials (which are
                  correlated) can't straddle train and test.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from neuralsignal import config as cfgmod
from neuralsignal.data import bci2a, mc_maze
from neuralsignal.evaluation import metrics as M
from neuralsignal.models import registry
from neuralsignal.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR
from neuralsignal.preprocess.eeg import Epochs, epoch_session
from neuralsignal.preprocess.spikes import segment_trials


# ---------------------------------------------------------------- EEG

def get_epochs(subject: int, session: str, pp: dict) -> Epochs:
    """Preprocess one session, cached on disk by a hash of the preprocessing settings."""
    key = hashlib.sha1(json.dumps(pp, sort_keys=True).encode()).hexdigest()[:10]
    cache = DATA_DIR / "cache" / f"A{subject:02d}{session}_{key}.npz"
    if cache.exists():
        d = np.load(cache)
        return Epochs(X=d["X"], y=d["y"], fs=float(d["fs"]), bad=d["bad"], session=str(d["session"]))
    ep = epoch_session(bci2a.load(subject, session), **pp)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=ep.X, y=ep.y, fs=ep.fs, bad=ep.bad, session=ep.session)
    return ep


def build_eeg_model(cfg: dict, fs: float, seed: int, log, protocol: str = ""):
    from neuralsignal.models.csp_lda import CSPLDA
    from neuralsignal.models.torch_fit import EEGNetClassifier
    m = cfg["model"]
    if cfg["method"] == "csp_lda":
        return CSPLDA(**m)
    if cfg["method"] == "eegnet":
        training = {**cfg.get("training", {}), **cfg.get("training_overrides", {}).get(protocol, {})}
        return EEGNetClassifier(fs=fs, seed=seed, log=log, **m, **training)
    raise ValueError(cfg["method"])


def run_eeg(cfg: dict, out: Path, log=print) -> dict:
    pp, subjects = cfg["preprocess"], cfg["subjects"]
    log("Preprocessing sessions...")
    ep = {(s, x): get_epochs(s, x, pp) for s in subjects for x in "TE"}
    results, preds = {}, {}
    for protocol in cfg["protocols"]:
        seeds = cfg.get("seeds", {}).get(protocol, [0]) if isinstance(cfg.get("seeds"), dict) else cfg.get("seeds", [0])
        rows = []
        for s in subjects:
            train_keys = [(s, "T")] if protocol == "cross_session" else \
                [(o, x) for o in subjects if o != s for x in "TE"]
            test = ep[(s, "E")]
            Xtr = np.concatenate([ep[k].X[~ep[k].bad] for k in train_keys])
            ytr = np.concatenate([ep[k].y[~ep[k].bad] for k in train_keys])
            for seed in seeds:
                cfgmod.seed_everything(seed)
                t0 = time.time()
                model = build_eeg_model(cfg, test.fs, seed, log if cfg.get("verbose") else None, protocol).fit(Xtr, ytr)
                proba = model.predict_proba(test.X)
                yhat = proba.argmax(1)
                row = dict(subject=s, seed=seed, n_train=int(len(ytr)), n_test=int(len(test.y)),
                           accuracy=M.accuracy(test.y, yhat), kappa=M.kappa(test.y, yhat),
                           seconds=round(time.time() - t0, 1))
                if hasattr(model, "info"):
                    row["epochs_run"] = model.info["epochs_run"]
                rows.append(row)
                preds[f"{protocol}/A{s:02d}/seed{seed}/proba"] = proba
                preds[f"{protocol}/A{s:02d}/y"] = test.y
                log(f"  {protocol:14s} A{s:02d} seed {seed}: acc {row['accuracy']:.3f}  kappa {row['kappa']:.3f}  ({row['seconds']}s)")
                if seed == seeds[0]:
                    registry.save(model, MODELS_DIR / cfg["id"] / protocol / f"A{s:02d}", dict(
                        method=cfg["method"], kind="eeg", name=cfg["name"], config_id=cfg["id"],
                        protocol=protocol, target=f"A{s:02d}", preprocess=pp,
                        trained_on=[f"A{o:02d}{x}" for o, x in train_keys],
                        n_channels=int(test.X.shape[1]), n_times=int(test.X.shape[2]),
                        test_accuracy=row["accuracy"], test_recording=f"A{s:02d}E"))
        acc = _per_subject(rows, "accuracy")
        kap = _per_subject(rows, "kappa")
        results[protocol] = dict(rows=rows, per_subject_accuracy=acc, per_subject_kappa=kap,
                                 mean_accuracy=float(np.mean(list(acc.values()))),
                                 std_accuracy=float(np.std(list(acc.values()))),
                                 mean_kappa=float(np.mean(list(kap.values()))),
                                 chance_95=M.chance_upper(288, 4))
        log(f"  => {protocol}: mean accuracy {results[protocol]['mean_accuracy']:.3f} "
            f"+/- {results[protocol]['std_accuracy']:.3f}")
    np.savez_compressed(out / "predictions.npz", **preds)
    return results


def _per_subject(rows, key):
    by = {}
    for r in rows:
        by.setdefault(f"A{r['subject']:02d}", []).append(r[key])
    return {k: float(np.mean(v)) for k, v in by.items()}


# ---------------------------------------------------------------- spikes

def build_spike_model(cfg: dict, seed: int, log):
    from neuralsignal.models.kalman import KalmanDecoder
    from neuralsignal.models.torch_fit import GRURegressor
    if cfg["method"] == "kalman":
        return KalmanDecoder(bin_s=cfg["preprocess"]["bin_ms"] / 1000, **cfg.get("model", {}))
    if cfg["method"] == "gru":
        return GRURegressor(seed=seed, log=log, **cfg.get("model", {}), **cfg.get("training", {}))
    raise ValueError(cfg["method"])


def _fit_spikes(model, seg, idx):
    if hasattr(model, "kw"):  # GRU
        return model.fit(seg.X[idx], seg.Y[idx], score_mask=seg.score_mask)
    return model.fit(seg.X[idx], seg.Y[idx], seg.P[idx])


def run_spikes(cfg: dict, out: Path, log=print) -> dict:
    pp = cfg["preprocess"]
    log("Loading MC_Maze and binning spikes...")
    sess = mc_maze.load()
    seg = segment_trials(sess, **pp)
    mask = seg.score_mask
    seeds = cfg.get("seeds", [0])
    results, preds = {}, {"Y": seg.Y, "mask": mask, "split": seg.split.astype(str), "trial_index": seg.trial_index}
    for protocol in cfg["protocols"]:
        if protocol == "nlb_split":
            folds = [(np.flatnonzero(seg.split == "train"), np.flatnonzero(seg.split == "val"))]
        elif protocol == "blocked_cv":
            order = np.argsort(seg.onset_time)
            blocks = np.array_split(order, cfg.get("n_folds", 5))
            folds = [(np.setdiff1d(order, b), b) for b in blocks]
        else:
            raise ValueError(protocol)
        rows = []
        for seed in seeds:
            cfgmod.seed_everything(seed)
            Yhat = np.full_like(seg.Y, np.nan)
            for k, (tr, te) in enumerate(folds):
                t0 = time.time()
                model = _fit_spikes(build_spike_model(cfg, seed, log if cfg.get("verbose") else None), seg, tr)
                Yhat[te] = model.predict(seg.X[te])
                r2 = M.velocity_r2(seg.Y[te], Yhat[te], mask)
                rows.append(dict(fold=k, seed=seed, n_train=int(len(tr)), n_test=int(len(te)), r2=r2,
                                 seconds=round(time.time() - t0, 1),
                                 **({"epochs_run": model.info["epochs_run"]} if hasattr(model, "info") else {})))
                log(f"  {protocol:10s} fold {k} seed {seed}: R2 {r2:.3f}  ({rows[-1]['seconds']}s)")
                if protocol == "nlb_split" and seed == seeds[0]:
                    registry.save(model, MODELS_DIR / cfg["id"] / protocol / "MC_Maze_Medium", dict(
                        method=cfg["method"], kind="spikes", name=cfg["name"], config_id=cfg["id"],
                        protocol=protocol, target="MC_Maze_Medium", preprocess=pp,
                        trained_on_trials=seg.trial_index[tr].tolist(), n_units=int(seg.X.shape[-1]),
                        test_r2=r2))
            preds[f"{protocol}/seed{seed}"] = Yhat
        tested = np.concatenate([te for _, te in folds])
        pooled = [M.velocity_r2(seg.Y[tested], preds[f"{protocol}/seed{s}"][tested], mask) for s in seeds]
        ci = M.bootstrap_r2(seg.Y[tested], preds[f"{protocol}/seed{seeds[0]}"][tested], mask)
        results[protocol] = dict(rows=rows, r2_per_seed=pooled, mean_r2=float(np.mean(pooled)),
                                 std_r2_over_seeds=float(np.std(pooled)), ci95_trials=ci,
                                 n_test_trials=int(len(tested)))
        log(f"  => {protocol}: R2 {np.mean(pooled):.3f}  (95% CI over trials {ci[0]:.3f}-{ci[1]:.3f})")
    np.savez_compressed(out / "predictions.npz", **preds)
    return results


# ---------------------------------------------------------------- entry point

def run(config_path: str, log=print) -> dict:
    cfg = cfgmod.load_config(config_path)
    out = RESULTS_DIR / "runs" / cfg["id"]
    cfgmod.write_run_header(cfg, out)
    log(f"== {cfg['name']}  (config {cfg['_source']}, hash {cfg['_hash']})")
    if "threads" in cfg:  # small models train fastest on few threads, especially on a busy machine
        import torch
        torch.set_num_threads(int(cfg["threads"]))
    t0 = time.time()
    res = run_eeg(cfg, out, log) if cfg["experiment"] == "bci2a" else run_spikes(cfg, out, log)
    summary = dict(name=cfg["name"], id=cfg["id"], method=cfg["method"], experiment=cfg["experiment"],
                   config_hash=cfg["_hash"], minutes=round((time.time() - t0) / 60, 1), protocols=res)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    log(f"== done in {summary['minutes']} min -> {out}")
    return summary
