"""Bin-by-bin (online) versions of the decoders: what a real-time BCI actually runs.

Offline decoders get a whole trial at once. Online, a new bin of spikes (or EEG samples)
arrives every few tens of milliseconds and the decoder must update immediately, using
only the past. Both spike decoders are already causal, so streaming them gives exactly
the offline predictions - the tests check that - but now each update can be timed.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
from scipy import signal

from neuralsignal.models.kalman import KalmanDecoder
from neuralsignal.models.torch_fit import GRURegressor


class StreamingKalman:
    """The Kalman gain K_t depends only on the model (A, W, H, Q, P0), never on the spikes, so
    the whole sequence of gains can be computed before decoding starts. It converges within
    about a second to a steady state, which is reused from then on. Each online update is
    then a few matrix-vector products (microseconds) instead of solving a units x units
    system, with outputs identical to the offline filter."""

    def __init__(self, model: KalmanDecoder, max_steps: int = 400, tol: float = 1e-9):
        self.m = m = model
        P, self.gains = m.P0.copy(), []
        for _ in range(max_steps):
            P = m.A @ P @ m.A.T + m.W
            K = np.linalg.solve(m.H @ P @ m.H.T + m.Q, m.H @ P).T
            P = P - K @ m.H @ P
            if self.gains and np.abs(K - self.gains[-1]).max() < tol:
                break
            self.gains.append(K)
        self.reset()

    def reset(self):
        self.x, self.t = self.m.x0.copy(), 0

    def step(self, counts: np.ndarray) -> np.ndarray:
        m = self.m
        z = (counts.astype(np.float64) - m.z_mean) / m.z_std
        K = self.gains[min(self.t, len(self.gains) - 1)]
        x = m.A @ self.x
        self.x = x + K @ (z - m.H @ x)
        self.t += 1
        return self.x[:2].astype(np.float32)


class StreamingGRU:
    def __init__(self, model: GRURegressor):
        self.m = model
        self.m.model.eval()
        self.reset()

    def reset(self):
        self.h = None

    @torch.no_grad()
    def step(self, counts: np.ndarray) -> np.ndarray:
        net = self.m.model
        x = torch.as_tensor(self.m._x(counts[None, None, :]))
        out, self.h = net.gru(torch.relu(net.inp(x)), self.h)
        return (net.readout(out)[0, 0].numpy() * self.m.y_sd).astype(np.float32)


def streaming_decoder(model):
    return StreamingKalman(model) if isinstance(model, KalmanDecoder) else StreamingGRU(model)


class StreamingEEG:
    """Continuous motor-imagery classification: causal band-pass with persistent filter state,
    EOG regression, a sliding window, and a classification every `step_s` seconds."""

    def __init__(self, model, preprocess: dict, fs: float, eog_weights: np.ndarray | None,
                 n_eeg: int = 22, step_s: float = 0.25):
        self.model, self.fs, self.B, self.n_eeg = model, fs, eog_weights, n_eeg
        self.sos = signal.butter(4, preprocess["band"], btype="bandpass", fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], n_eeg, 2))
        self.win = int(round((preprocess["tmax"] - preprocess["tmin"]) * fs))
        self.buf = deque(maxlen=self.win)
        self.step_n = int(round(step_s * fs))
        self.resample = preprocess.get("resample_fs")
        self.since = 0

    def push(self, chunk: np.ndarray):
        """chunk: (25, n) raw EEG+EOG samples. Returns (class probabilities, compute ms) when a
        classification is due, else None."""
        t0 = time.perf_counter()
        eeg = np.nan_to_num(chunk[: self.n_eeg])
        if self.B is not None:
            eeg = eeg - self.B @ np.nan_to_num(chunk[self.n_eeg:self.n_eeg + 3])
        y, self.zi = signal.sosfilt(self.sos, eeg, axis=-1, zi=self.zi)
        self.buf.extend(y.T)
        self.since += chunk.shape[1]
        if len(self.buf) < self.win or self.since < self.step_n:
            return None
        self.since = 0
        X = np.asarray(self.buf).T[None].astype(np.float32)
        if self.resample and self.resample != self.fs:
            X = signal.resample_poly(X, int(self.resample), int(self.fs), axis=-1).astype(np.float32)
        p = self.model.predict_proba(X)[0]
        return p, (time.perf_counter() - t0) * 1000
