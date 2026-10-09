"""Common Spatial Patterns + Linear Discriminant Analysis - the classic motor-imagery baseline.

Why it works: imagining a left-hand movement desynchronises the mu/beta rhythm
(8-30 Hz) over the *right* motor cortex (and vice versa). The signal is a change in
band *power* at particular scalp locations. CSP finds spatial filters w (weightings
of the 22 electrodes) that maximise variance for one class while minimising it for
the other:

    w* = argmax  (w' C_a w) / (w' C_b w)

which is the generalised eigenproblem C_a w = lambda C_b w (here solved as
C_a w = lambda (C_a + C_b) w so lambda is in [0, 1]). Eigenvectors with lambda near 1
or 0 are the most discriminative. The log of each filtered signal's variance is then
an almost-Gaussian feature, which is exactly what LDA assumes.

For 4 classes we use one-vs-rest CSP: n_pairs filters from each end of the spectrum
for each class, giving 4 * 2 * n_pairs log-variance features, then one shrinkage LDA.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _cov(X: np.ndarray) -> np.ndarray:
    """Average trace-normalised covariance of trials X (n, ch, t)."""
    C = np.einsum("nct,ndt->ncd", X, X)
    C /= np.trace(C, axis1=1, axis2=2)[:, None, None]
    return C.mean(0)


class CSP:
    def __init__(self, n_pairs: int = 3, reg: float = 1e-6):
        self.n_pairs, self.reg = n_pairs, reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        X = X - X.mean(-1, keepdims=True)
        self.classes_ = np.unique(y)
        filters, self.patterns_ = [], []
        for c in self.classes_:
            Ca, Cb = _cov(X[y == c]), _cov(X[y != c])
            n = Ca.shape[0]
            evals, evecs = eigh(Ca, Ca + Cb + self.reg * np.eye(n))
            order = np.argsort(evals)
            pick = np.r_[order[: self.n_pairs], order[-self.n_pairs:]]
            W = evecs[:, pick]
            filters.append(W)
            # Patterns (A = C W (W' C W)^-1) are what you'd plot on a scalp map;
            # filters are not interpretable on their own (Haufe et al. 2014).
            C = Ca + Cb
            self.patterns_.append(C @ W @ np.linalg.pinv(W.T @ C @ W))
        self.filters_ = np.concatenate(filters, axis=1)  # (ch, n_features)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X - X.mean(-1, keepdims=True)
        Z = np.einsum("cf,nct->nft", self.filters_, X)
        var = Z.var(-1)
        return np.log(var / var.sum(1, keepdims=True))


class CSPLDA:
    """The full baseline pipeline: CSP log-variance features -> shrinkage LDA."""

    def __init__(self, n_pairs: int = 3, shrinkage: str | float = "auto"):
        self.csp = CSP(n_pairs=n_pairs)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)

    def fit(self, X, y):
        self.lda.fit(self.csp.fit(X, y).transform(X), y)
        return self

    def predict_proba(self, X):
        return self.lda.predict_proba(self.csp.transform(X))

    def predict(self, X):
        return self.lda.predict(self.csp.transform(X))
