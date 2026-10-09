"""EEG cleaning: band-pass filtering, EOG artifact regression, epoching, trial rejection.

Order of operations (per session):
  1. Fit EOG regression weights on the eye-movement calibration runs.
  2. Subtract the EOG contribution from every run.
  3. Band-pass each run separately (causal Butterworth, so it could run online).
  4. Cut epochs around each cue and optionally downsample.
  5. Flag bad trials (expert marks / peak-to-peak amplitude) - only ever dropped from
     *training* data; test trials are always all kept so accuracy isn't flattered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from neuralsignal.data.bci2a import CUE_OFFSET_S, Session


@dataclass
class Epochs:
    X: np.ndarray        # (n_trials, n_channels, n_times)
    y: np.ndarray        # (n_trials,)
    fs: float
    bad: np.ndarray      # (n_trials,) bool - candidates for rejection in training
    session: str         # e.g. "A01T"


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4,
             causal: bool = True) -> np.ndarray:
    """Butterworth band-pass along the last axis. Second-order sections for stability."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x, axis=-1) if causal else signal.sosfiltfilt(sos, x, axis=-1)


def fit_eog_regression(session: Session) -> np.ndarray:
    """Least-squares weights B so that EEG_clean = EEG - B @ EOG  (Schlogl et al. 2007).

    The 2a sessions start with runs recorded specifically for this: eyes open, eyes
    closed and deliberate eye movements. If those runs are missing (A04T), fall back
    to all of the session's data. Signals are high-passed at 1 Hz first so slow drifts
    don't dominate the fit.
    """
    calib = [r for r in session.runs if r.is_calibration] or session.runs
    eeg = np.concatenate([bandpass(r.eeg, session.fs, 1, 40) for r in calib], axis=1)
    eog = np.concatenate([bandpass(r.eog, session.fs, 1, 40) for r in calib], axis=1)
    ok = np.isfinite(eeg).all(0) & np.isfinite(eog).all(0)
    eeg, eog = eeg[:, ok], eog[:, ok]
    # Solve eog.T @ B.T = eeg.T  ->  B is (n_eeg, n_eog)
    B, *_ = np.linalg.lstsq(eog.T, eeg.T, rcond=None)
    return B.T


def epoch_session(session: Session, *, band=(8.0, 30.0), tmin=0.5, tmax=2.5,
                  eog_regression=True, resample_fs: float | None = None,
                  reject_uv: float | None = 100.0, causal=True) -> Epochs:
    """Clean a session and return one epoch per trial. tmin/tmax are seconds after the cue."""
    fs = session.fs
    B = fit_eog_regression(session) if eog_regression else None
    X, y, bad = [], [], []
    start, stop = int(round((CUE_OFFSET_S + tmin) * fs)), int(round((CUE_OFFSET_S + tmax) * fs))
    for run in session.runs:
        if run.is_calibration:
            continue
        eeg = np.nan_to_num(run.eeg)
        if B is not None:
            eeg = eeg - B @ np.nan_to_num(run.eog)
        eeg = bandpass(eeg, fs, *band, causal=causal)
        for t0, label, art in zip(run.trial_starts, run.labels, run.artifacts):
            seg = eeg[:, t0 + start:t0 + stop]
            X.append(seg)
            y.append(label)
            ptp = np.ptp(seg, axis=1).max()
            bad.append(bool(art) or (reject_uv is not None and ptp > reject_uv))
    X = np.stack(X).astype(np.float32)
    out_fs = fs
    if resample_fs and resample_fs != fs:
        up, down = _ratio(resample_fs, fs)
        X = signal.resample_poly(X, up, down, axis=-1).astype(np.float32)
        out_fs = fs * up / down
    return Epochs(X=X, y=np.array(y), fs=out_fs, bad=np.array(bad), session=session.name)


def _ratio(target: float, fs: float) -> tuple[int, int]:
    from fractions import Fraction
    f = Fraction(target / fs).limit_denominator(100)
    return f.numerator, f.denominator
