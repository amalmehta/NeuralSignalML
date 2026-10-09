"""Kalman filter velocity decoder (Wu et al. 2003; Gilja et al. 2012) - the classic spike baseline.

Two linear-Gaussian models, both fit by least squares on training data:

  movement model     x_t = A x_{t-1} + w,   w ~ N(0, W)     (how velocity evolves)
  observation model  z_t = H x_t     + q,   q ~ N(0, Q)     (how neurons encode velocity)

x is the hand's kinematic state - velocity, position and acceleration, the formulation
of Wu et al. (2003) - plus a constant 1 (so H can absorb each neuron's baseline rate);
z is the vector of binned spike counts. Including position and acceleration matters:
motor-cortex neurons are tuned to more than velocity, and on MC_Maze a velocity-only
state loses roughly half the decodable variance (see docs/WRITEUP.md). Decoding alternates predict (push the last
estimate through A) and update (correct it with the new spikes, weighted by the Kalman
gain K - large when the neurons are reliable relative to the movement prior). The
result is a smoothed, causal estimate that blends "where the hand was going" with
"what the neurons say now".
"""
from __future__ import annotations

import numpy as np


class KalmanDecoder:
    def __init__(self, ridge: float = 1e-3, state=("velocity", "position", "acceleration"),
                 bin_s: float = 0.05):
        self.ridge, self.state, self.bin_s = ridge, tuple(state), bin_s

    def _state(self, Y, P):
        parts = {"velocity": Y, "position": P,
                 "acceleration": None if Y is None else np.gradient(Y, self.bin_s, axis=1)}
        return np.concatenate([parts[k] for k in self.state], -1)

    @staticmethod
    def _aug(Y):  # append the constant state
        return np.concatenate([Y, np.ones((*Y.shape[:-1], 1), Y.dtype)], -1)

    def fit(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray | None = None) -> "KalmanDecoder":
        """X: (trials, bins, units) counts; Y: (trials, bins, 2) velocity; P: position.
        Velocity is always the first two state dimensions, which is what predict() returns."""
        assert self.state[0] == "velocity"
        Xs = self._aug(self._state(Y.astype(np.float64), None if P is None else P.astype(np.float64)))
        Z = X.astype(np.float64)
        self.z_mean = Z.reshape(-1, Z.shape[-1]).mean(0)
        self.z_std = Z.reshape(-1, Z.shape[-1]).std(0) + 1e-3
        Z = (Z - self.z_mean) / self.z_std
        x_prev = Xs[:, :-1].reshape(-1, Xs.shape[-1])
        x_next = Xs[:, 1:].reshape(-1, Xs.shape[-1])
        x_all, z_all = Xs.reshape(-1, Xs.shape[-1]), Z.reshape(-1, Z.shape[-1])
        n = Xs.shape[-1]
        reg = self.ridge * np.eye(n)
        self.A = np.linalg.solve(x_prev.T @ x_prev + reg, x_prev.T @ x_next).T
        self.A[-1] = 0
        self.A[-1, -1] = 1  # the constant stays constant
        r = x_next - x_prev @ self.A.T
        self.W = np.cov(r.T)
        self.W[-1] = self.W[:, -1] = 0
        self.H = np.linalg.solve(x_all.T @ x_all + reg, x_all.T @ z_all).T
        q = z_all - x_all @ self.H.T
        self.Q = np.cov(q.T) + 1e-6 * np.eye(Z.shape[-1])
        self.x0 = Xs[:, 0].mean(0)
        self.P0 = np.cov(Xs[:, 0].T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the filter independently on each trial. Returns (trials, bins, 2)."""
        Z = (X.astype(np.float64) - self.z_mean) / self.z_std
        A, W, H, Q = self.A, self.W, self.H, self.Q
        out = np.zeros((*X.shape[:2], 2))
        for i in range(X.shape[0]):
            x, P = self.x0.copy(), self.P0.copy()
            for t in range(X.shape[1]):
                x, P = A @ x, A @ P @ A.T + W  # predict
                # update, covariance form. (The information form needs P^-1, which doesn't
                # exist for the constant state or a velocity known to start at zero.)
                S = H @ P @ H.T + Q
                K = np.linalg.solve(S, H @ P).T  # = P H' S^-1, using symmetry of P and S
                x = x + K @ (Z[i, t] - H @ x)
                P = P - K @ H @ P
                out[i, t] = x[:2]
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("A", "W", "H", "Q", "x0", "P0", "z_mean", "z_std")}
        d["state"] = np.array(self.state)
        return d

    @classmethod
    def from_state(cls, d) -> "KalmanDecoder":
        m = cls(state=[str(v) for v in d["state"]]) if "state" in d else cls()
        for k in ("A", "W", "H", "Q", "x0", "P0", "z_mean", "z_std"):
            setattr(m, k, np.asarray(d[k]))
        return m
