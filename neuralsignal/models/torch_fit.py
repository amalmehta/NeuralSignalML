"""sklearn-style wrappers around the torch models, sharing one training loop.

Training loop: Adam, mini-batches, a validation split carved out of the *training*
data only (stratified for classification), early stopping on validation loss, and the
best-epoch weights restored at the end. The test data never touches any of this.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from neuralsignal.models.eegnet import EEGNet
from neuralsignal.models.rnn import GRUDecoder


def _split(n: int, val_frac: float, rng: np.random.Generator, y: np.ndarray | None = None):
    if y is None:
        idx = rng.permutation(n)
        k = max(1, int(round(n * val_frac)))
        return idx[k:], idx[:k]
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = max(1, int(round(len(idx) * val_frac)))
        va.extend(idx[:k])
        tr.extend(idx[k:])
    return np.array(tr), np.array(va)


def train_loop(model: nn.Module, loss_fn, Xtr, Ytr, Xva, Yva, *, lr=1e-3, weight_decay=0.0,
               batch_size=64, max_epochs=300, patience=50, seed=0, log=None) -> dict:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr, Ytr, Xva, Yva = map(torch.as_tensor, (Xtr, Ytr, Xva, Yva))
    best, best_state, wait, history = np.inf, None, 0, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), Yva).item()
        history.append((tot / len(Xtr), vl))
        if vl < best - 1e-5:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
        if log and epoch % 25 == 0:
            log(f"    epoch {epoch:3d}  train {history[-1][0]:.4f}  val {vl:.4f}")
    model.load_state_dict(best_state)
    return {"epochs_run": len(history), "best_val_loss": best, "history": history}


class EEGNetClassifier:
    def __init__(self, fs: float, dropout=0.5, F1=8, D=2, F2=16, lr=1e-3, weight_decay=1e-3,
                 batch_size=64, max_epochs=300, patience=60, val_frac=0.2, seed=0, log=None):
        self.kw = dict(F1=F1, D=D, F2=F2, dropout=dropout)
        self.fs, self.seed, self.val_frac, self.log = fs, seed, val_frac, log
        self.train_kw = dict(lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                             max_epochs=max_epochs, patience=patience, seed=seed)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.scale = float(X.std())  # one global scale, from training data only
        self.n_classes = int(y.max()) + 1
        torch.manual_seed(self.seed)
        self.model = EEGNet(X.shape[1], X.shape[2], self.n_classes, fs=self.fs, **self.kw)
        tr, va = _split(len(X), self.val_frac, rng, y)
        Xs = (X / self.scale).astype(np.float32)
        self.info = train_loop(self.model, nn.CrossEntropyLoss(), Xs[tr], y[tr].astype(np.int64),
                               Xs[va], y[va].astype(np.int64), log=self.log, **self.train_kw)
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(torch.as_tensor((X / self.scale).astype(np.float32))), -1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(-1)

    def state(self):
        return {"state_dict": self.model.state_dict(), "scale": self.scale, "n_classes": self.n_classes,
                "kw": self.kw, "fs": self.fs, "shape": None}


class GRURegressor:
    def __init__(self, hidden=128, layers=1, dropout=0.3, input_dropout=0.2, lr=2e-3,
                 weight_decay=1e-4, batch_size=16, max_epochs=400, patience=60, val_frac=0.2,
                 seed=0, log=None):
        self.kw = dict(hidden=hidden, layers=layers, dropout=dropout, input_dropout=input_dropout)
        self.seed, self.val_frac, self.log = seed, val_frac, log
        self.train_kw = dict(lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                             max_epochs=max_epochs, patience=patience, seed=seed)

    def _x(self, X):
        return ((np.sqrt(X) - self.mu) / self.sd).astype(np.float32)

    def fit(self, X, Y, score_mask=None):
        rng = np.random.default_rng(self.seed)
        S = np.sqrt(X).reshape(-1, X.shape[-1])
        self.mu, self.sd = S.mean(0), S.std(0) + 1e-3
        self.y_sd = float(Y.std())
        mask = torch.as_tensor(np.ones(X.shape[1], bool) if score_mask is None else score_mask)

        def loss_fn(pred, target):  # MSE on the scored window, in units of velocity std
            return ((pred - target)[:, mask] ** 2).mean()

        torch.manual_seed(self.seed)
        self.model = GRUDecoder(X.shape[-1], **self.kw)
        tr, va = _split(len(X), self.val_frac, rng)
        Xs, Ys = self._x(X), (Y / self.y_sd).astype(np.float32)
        self.info = train_loop(self.model, loss_fn, Xs[tr], Ys[tr], Xs[va], Ys[va], log=self.log,
                               **self.train_kw)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(self._x(X))).numpy() * self.y_sd
