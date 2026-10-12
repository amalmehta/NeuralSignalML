"""EEG that works without calibration: decode a new person with little or none of their own labelled data.

Protocol (extends the held-out-subject test). For each target person:
  * test set: their day-2 session (all 288 trials), never used for anything else;
  * their day-1 session is "the first minutes of using the BCI": its EEG may be used
    *unlabelled* for alignment, and in the calibration curve k labelled trials per class
    are drawn from it;
  * training data: the other 8 people, both days, each session aligned to itself.

Methods:
  csp_lda      no alignment (the original held-out-subject baseline)
  ea_csp_lda   Euclidean alignment per session, then CSP + LDA
  ts_lda       tangent space + LDA, no re-centring (reference point = training grand mean)
  ra_ts_lda    Riemannian re-centring per session, then tangent space + LDA
  ea_eegnet    Euclidean alignment, then EEGNet (expensive; optional)

Calibration curve: k labelled trials per class from the target's day 1 (k = 0 is
zero-calibration), either pooled with the other people ("pooled") or used alone
("own_only"), compared with full calibration (all 288 day-1 trials, own data only).
"""
from __future__ import annotations

import json
import time

import numpy as np

from neuralsignal import config as cfgmod
from neuralsignal.evaluation import metrics as M
from neuralsignal.experiments import get_epochs
from neuralsignal.models import riemann as R
from neuralsignal.models.csp_lda import CSPLDA
from neuralsignal.paths import RESULTS_DIR


class Pipeline:
    """One method: how to align a person's session, and how to train/predict on aligned data."""

    def __init__(self, name: str, cfg: dict, seed: int = 0, fs: float = 250.0):
        self.name, self.cfg, self.seed, self.fs = name, cfg, seed, fs

    def align(self, X, ref):
        """Transform one session's trials X given unlabelled trials `ref` from the same session."""
        if self.name in ("ea_csp_lda", "ea_eegnet"):
            return R.euclidean_align(X, ref)
        if self.name == "ra_ts_lda":
            return R.recenter(R.covariances(X), R.covariances(ref))
        if self.name == "ts_lda":
            return R.covariances(X)
        return X

    def model(self):
        if self.name in ("csp_lda", "ea_csp_lda"):
            return CSPLDA(n_pairs=3)
        if self.name in ("ts_lda", "ra_ts_lda"):
            return R.TangentSpaceLDA()
        from neuralsignal.models.torch_fit import EEGNetClassifier
        return EEGNetClassifier(fs=self.fs, seed=self.seed, **self.cfg["eegnet"]["model"], **self.cfg["eegnet"]["training"])

    def fit(self, X, y):
        if self.name == "ts_lda":  # no re-centring: everyone shares the training grand mean
            self.W = R.invsqrtm(R.riemann_mean(X))
            X = self.W @ X @ self.W
        self.m = self.model().fit(X, y)
        return self

    def predict_proba(self, X):
        if self.name == "ts_lda":
            X = self.W @ X @ self.W
        return self.m.predict_proba(X)


def run(config_path: str, log=print) -> dict:
    cfg = cfgmod.load_config(config_path)
    out = RESULTS_DIR / "runs" / cfg["id"]
    cfgmod.write_run_header(cfg, out)
    if "threads" in cfg:
        import torch
        torch.set_num_threads(int(cfg["threads"]))
    log(f"== {cfg['name']}  (config {cfg['_source']}, hash {cfg['_hash']})")
    t0 = time.time()
    subjects = cfg["subjects"]
    rows = []

    for method in cfg["methods"]:
        pp = cfg["eegnet"]["preprocess"] if method == "ea_eegnet" else cfg["preprocess"]
        ep = {(s, x): get_epochs(s, x, pp) for s in subjects for x in "TE"}
        ks = cfg["calibration_k"] if method in cfg["curve_methods"] else [0]
        draws = cfg["draws"] if method in cfg["curve_methods"] else [0]
        for s in subjects:
            pipe = Pipeline(method, cfg, fs=ep[(s, "E")].fs)
            # each session aligned with its own unlabelled trials
            aligned = {k: pipe.align(e.X, e.X) for k, e in ep.items() if k[0] != s}
            tgt_T, tgt_E = ep[(s, "T")], ep[(s, "E")]
            # the new person: aligned with their unlabelled day-1 EEG; day 2 is the untouched test set
            test = pipe.align(tgt_E.X, tgt_T.X)
            day1 = pipe.align(tgt_T.X, tgt_T.X)
            Xo = np.concatenate([aligned[k][~ep[k].bad] for k in aligned])
            yo = np.concatenate([ep[k].y[~ep[k].bad] for k in aligned])
            for k in ks:
                for draw in draws if k > 0 else [0]:
                    rng = np.random.default_rng(draw)
                    good = np.flatnonzero(~tgt_T.bad)
                    pick = np.concatenate([rng.choice(good[tgt_T.y[good] == c], k, replace=False) for c in range(4)]) \
                        if k > 0 else np.array([], int)
                    variants = [("pooled", np.concatenate([Xo, day1[pick]]), np.concatenate([yo, tgt_T.y[pick]]))]
                    if k >= cfg.get("own_only_min_k", 5) and method in cfg["curve_methods"]:
                        variants.append(("own_only", day1[pick], tgt_T.y[pick]))
                    for variant, Xtr, ytr in variants:
                        p = Pipeline(method, cfg, seed=draw, fs=tgt_E.fs).fit(Xtr, ytr)
                        yhat = p.predict_proba(test).argmax(1)
                        rows.append(dict(method=method, subject=f"A{s:02d}", k_per_class=int(k), draw=int(draw),
                                         variant=variant, n_train=int(len(ytr)),
                                         accuracy=M.accuracy(tgt_E.y, yhat), kappa=M.kappa(tgt_E.y, yhat)))
                        log(f"  {method:10s} A{s:02d} k={k:<3d} draw {draw} {variant:8s} acc {rows[-1]['accuracy']:.3f}")
            if method in cfg["curve_methods"]:  # full calibration reference: all of day 1, own data only
                p = Pipeline(method, cfg, fs=tgt_E.fs).fit(day1[~tgt_T.bad], tgt_T.y[~tgt_T.bad])
                yhat = p.predict_proba(test).argmax(1)
                rows.append(dict(method=method, subject=f"A{s:02d}", k_per_class=-1, draw=0, variant="full_calibration",
                                 n_train=int((~tgt_T.bad).sum()), accuracy=M.accuracy(tgt_E.y, yhat),
                                 kappa=M.kappa(tgt_E.y, yhat)))
                log(f"  {method:10s} A{s:02d} full calibration acc {rows[-1]['accuracy']:.3f}")

    summary = dict(name=cfg["name"], id=cfg["id"], experiment="eeg_calibration", config_hash=cfg["_hash"],
                   minutes=round((time.time() - t0) / 60, 1), chance_95=M.chance_upper(288, 4), rows=rows)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    log(f"== done in {summary['minutes']} min -> {out}")
    return summary
