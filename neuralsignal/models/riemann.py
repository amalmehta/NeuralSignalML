"""Covariance-matrix geometry for EEG: alignment across people, and a tangent-space classifier.

Why covariances. Motor imagery changes band power at particular electrodes and the
correlations between them - all of which lives in each trial's spatial covariance matrix
C = X X' / T (22 x 22, symmetric positive definite). CSP looks at these through a few
filters; here we use the whole matrix.

Why people differ. Every head has a different skull, electrode fit and baseline rhythm,
so each person's covariances sit in a different region of the space. A decoder trained
on others sees the new person's data as shifted. Alignment removes the shift *without
labels*: estimate where the person's data is centred (their mean covariance R, from any
unlabelled EEG) and move it to the identity.

  Euclidean alignment (He & Wu 2019):   X  ->  R^(-1/2) X          with R the arithmetic mean
  Riemannian re-centring (Zanini 2018): C  ->  M^(-1/2) C M^(-1/2)  with M the Riemannian mean

Tangent space. Covariance matrices live on a curved manifold, where straight-line
averages are wrong (the arithmetic mean of SPD matrices inflates determinant). Taking
the matrix logarithm at the re-centred origin "unrolls" it locally into a flat vector
space, where ordinary LDA works well (Barachant et al. 2012).
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def sqrtm(C):
    return _eig_fn(C, np.sqrt)


def invsqrtm(C):
    return _eig_fn(C, lambda w: 1 / np.sqrt(w))


def logm(C):
    return _eig_fn(C, np.log)


def expm(C):
    return _eig_fn(C, np.exp)


def covariances(X: np.ndarray, shrink: float = 1e-3) -> np.ndarray:
    """Per-trial spatial covariance (trials, ch, ch), lightly regularised to stay positive definite."""
    X = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    return (1 - shrink) * C + shrink * tr * np.eye(C.shape[1])


def riemann_mean(C: np.ndarray, iters: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Fréchet mean under the affine-invariant metric: the point minimising the summed squared
    geodesic distances. Fixed-point iteration: map to the tangent space at the current
    estimate, average there, map back."""
    M = C.mean(0)
    for _ in range(iters):
        Mh, Mih = sqrtm(M), invsqrtm(M)
        T = logm(Mih @ C @ Mih).mean(0)
        M = Mh @ expm(T) @ Mh
        if np.linalg.norm(T) < tol:
            break
    return M


def euclidean_align(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Whiten trials X by the mean covariance of `reference` trials (same person, unlabelled)."""
    W = invsqrtm(covariances(reference).mean(0))
    return np.einsum("cd,ndt->nct", W, X).astype(np.float32)


def recenter(C: np.ndarray, reference_C: np.ndarray) -> np.ndarray:
    W = invsqrtm(riemann_mean(reference_C))
    return W @ C @ W


def tangent_vectors(C: np.ndarray) -> np.ndarray:
    """Upper triangle of log(C) at the identity, off-diagonals scaled by sqrt(2) to keep the norm."""
    L = logm(C)
    iu = np.triu_indices(C.shape[1])
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return L[:, iu[0], iu[1]] * coef


class TangentSpaceLDA:
    """Re-centred covariances -> tangent vectors -> shrinkage LDA.

    fit/predict take *already re-centred* covariance matrices; re-centring needs to know
    which person each trial came from, so the caller does it per person."""

    def __init__(self):
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, C, y):
        self.lda.fit(tangent_vectors(C), y)
        return self

    def predict_proba(self, C):
        return self.lda.predict_proba(tangent_vectors(C))

    def predict(self, C):
        return self.lda.predict(tangent_vectors(C))
