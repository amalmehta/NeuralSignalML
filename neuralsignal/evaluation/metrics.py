"""Metrics and the statistics used to compare decoders honestly."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, r2_score


def accuracy(y, yhat) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


def kappa(y, yhat) -> float:
    """Cohen's kappa - the official BCI Competition IV 2a metric. 0 = chance, 1 = perfect."""
    return float(cohen_kappa_score(y, yhat))


def chance_upper(n_trials: int, n_classes: int, alpha: float = 0.05) -> float:
    """Accuracy a random classifier exceeds with probability alpha (Combrisson & Jerbi 2015).
    With 288 trials and 4 classes this is ~29.5%, not 25%."""
    return float(stats.binom.ppf(1 - alpha, n_trials, 1 / n_classes) / n_trials)


def velocity_r2(Y, Yhat, mask=None) -> float:
    """Variance-weighted R^2 over x/y velocity on the scored bins (the NLB convention)."""
    if mask is not None:
        Y, Yhat = Y[:, mask], Yhat[:, mask]
    return float(r2_score(Y.reshape(-1, 2), Yhat.reshape(-1, 2), multioutput="variance_weighted"))


def bootstrap_r2(Y, Yhat, mask=None, n_boot=1000, seed=0) -> tuple[float, float]:
    """95% CI by resampling *trials* (bins within a trial are not independent)."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    vals = [velocity_r2(Y[i], Yhat[i], mask) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_r2_diff(Y, Ya, Yb, mask=None, n_boot=2000, seed=0) -> dict:
    """R^2(b) - R^2(a) on the *same* test trials, with a 95% CI from resampling trials.
    Pairing removes the trial-to-trial difficulty that both models share."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    d = [velocity_r2(Y[i], Yb[i], mask) - velocity_r2(Y[i], Ya[i], mask)
         for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return dict(diff=velocity_r2(Y, Yb, mask) - velocity_r2(Y, Ya, mask),
                ci95=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                p_not_better=float(np.mean(np.array(d) <= 0)))


def paired_wilcoxon(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired per-subject scores."""
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)
