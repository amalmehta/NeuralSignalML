"""Turn spike times into decoder inputs: bin counts, apply a neural-to-movement lag,
and cut one segment per trial around movement onset.

Motor cortex fires ~100 ms *before* the hand moves, so the counts used to predict the
velocity at time t come from the bin ending at t - lag. Everything here is causal: no
decoder input ever uses spikes from after the moment it is predicting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuralsignal.data.mc_maze import SpikeSession


@dataclass
class TrialSegments:
    X: np.ndarray        # (n_trials, n_bins, n_units) spike counts
    Y: np.ndarray        # (n_trials, n_bins, 2) hand velocity (mm/s), bin-averaged
    P: np.ndarray        # (n_trials, n_bins, 2) hand position (mm) at the end of each bin
    score_mask: np.ndarray  # (n_bins,) bool - bins that count toward R^2
    split: np.ndarray    # (n_trials,) 'train' / 'val'
    trial_index: np.ndarray
    onset_time: np.ndarray  # (n_trials,) seconds
    bin_s: float


def bin_counts(spike_times: list[np.ndarray], edges: np.ndarray) -> np.ndarray:
    """(n_bins, n_units) spike counts for arbitrary bin edges."""
    return np.stack([np.histogram(st, edges)[0] for st in spike_times], axis=1).astype(np.float32)


def segment_trials(sess: SpikeSession, *, bin_ms=20, lag_ms=100, pre_ms=750, post_ms=450,
                   score_pre_ms=250) -> TrialSegments:
    """One fixed-length segment per trial: [onset - pre_ms, onset + post_ms].

    The first (pre_ms - score_pre_ms) are warm-up for the decoder's state; R^2 is
    computed on [onset - score_pre_ms, onset + post_ms], the window NLB uses.
    """
    bin_s, lag_s = bin_ms / 1000, lag_ms / 1000
    n_bins = int(round((pre_ms + post_ms) / bin_ms))
    n_warm = int(round((pre_ms - score_pre_ms) / bin_ms))
    X, Y, P, keep = [], [], [], []
    for i, tr in sess.trials.iterrows():
        onset = tr.move_onset_time
        if not np.isfinite(onset):
            continue
        edges = onset - pre_ms / 1000 + bin_s * np.arange(n_bins + 1)
        X.append(bin_counts(sess.spike_times, edges - lag_s))
        idx = np.searchsorted(sess.behavior_t, edges)
        vel = np.stack([np.nanmean(sess.hand_vel[a:b], axis=0) for a, b in zip(idx[:-1], idx[1:])])
        Y.append(vel.astype(np.float32))
        P.append(sess.hand_pos[np.maximum(idx[1:] - 1, 0)].astype(np.float32))
        keep.append(i)
    keep = np.array(keep)
    mask = np.zeros(n_bins, bool)
    mask[n_warm:] = True
    return TrialSegments(X=np.stack(X), Y=np.stack(Y), P=np.stack(P), score_mask=mask,
                         split=sess.trials.split.values[keep], trial_index=keep,
                         onset_time=sess.trials.move_onset_time.values[keep], bin_s=bin_s)

