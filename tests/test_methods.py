"""Fast tests on synthetic data: each method must recover a signal we planted."""
import numpy as np
import pytest

from neuralsignal.evaluation import metrics as M
from neuralsignal.models.csp_lda import CSPLDA
from neuralsignal.models.kalman import KalmanDecoder
from neuralsignal.preprocess.eeg import bandpass


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    fs = 250
    t = np.arange(0, 8, 1 / fs)
    x = np.stack([np.sin(2 * np.pi * 12 * t), np.sin(2 * np.pi * 60 * t)])
    y = bandpass(x, fs, 8, 30)[:, 500:]
    assert y[0].std() == pytest.approx(1 / np.sqrt(2), rel=0.1)
    assert y[1].std() < 0.05


def test_csp_lda_finds_class_specific_power():
    """Classes differ only in the *variance* of particular spatial mixtures - CSP's assumption."""
    rng = np.random.default_rng(0)
    n, ch, T = 240, 8, 200
    mix = rng.standard_normal((ch, ch))
    y = np.repeat(np.arange(4), n // 4)
    src = rng.standard_normal((n, ch, T))
    for c in range(4):
        src[y == c, c] *= 3.0  # class c boosts source c
    X = np.einsum("cd,ndt->nct", mix, src)
    idx = rng.permutation(n)
    tr, te = idx[:160], idx[160:]
    acc = M.accuracy(y[te], CSPLDA(n_pairs=2).fit(X[tr], y[tr]).predict(X[te]))
    assert acc > 0.9


def test_kalman_recovers_linear_tuning():
    rng = np.random.default_rng(1)
    trials, bins, units = 120, 40, 30
    t = np.linspace(0, 2 * np.pi, bins)
    ang = rng.uniform(0, 2 * np.pi, trials)
    V = np.stack([np.cos(ang)[:, None] * np.sin(t / 2), np.sin(ang)[:, None] * np.sin(t / 2)], -1) * 300
    P = np.cumsum(V, axis=1) * 0.05
    H = rng.standard_normal((units, 2)) * 0.01
    rate = np.clip(2 + V @ H.T, 0, None)
    X = rng.poisson(rate).astype(np.float32)
    m = KalmanDecoder(bin_s=0.05).fit(X[:90], V[:90], P[:90])
    assert M.velocity_r2(V[90:], m.predict(X[90:])) > 0.6


def test_chance_level_and_kappa():
    assert 0.28 < M.chance_upper(288, 4) < 0.31
    assert M.kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_eegnet_and_gru_shapes_and_registry_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from neuralsignal.models import registry
    from neuralsignal.models.torch_fit import EEGNetClassifier, GRURegressor
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 22, 250)).astype(np.float32)
    y = np.repeat(np.arange(4), 10)
    clf = EEGNetClassifier(fs=125, max_epochs=2, patience=2).fit(X, y)
    assert clf.predict_proba(X).shape == (40, 4)
    registry.save(clf, tmp_path / "e", dict(method="eegnet", kind="eeg", n_channels=22, n_times=250))
    clf2, meta = registry.load(tmp_path / "e")
    np.testing.assert_allclose(clf.predict_proba(X), clf2.predict_proba(X), atol=1e-5)

    S = rng.poisson(1.0, (20, 24, 30)).astype(np.float32)
    V = rng.standard_normal((20, 24, 2)).astype(np.float32)
    gru = GRURegressor(hidden=16, max_epochs=2, patience=2).fit(S, V)
    assert gru.predict(S).shape == V.shape
    registry.save(gru, tmp_path / "g", dict(method="gru", kind="spikes", n_units=30))
    gru2, _ = registry.load(tmp_path / "g")
    np.testing.assert_allclose(gru.predict(S), gru2.predict(S), atol=1e-4)


def test_spike_sorting_demo_recovers_units():
    from neuralsignal.spikesort.demo import score, simulate, sort_spikes
    x, truth = simulate(seed=0)
    out = sort_spikes(x)
    s = score(truth, out["troughs"], out["labels"])
    assert s["recall"] > 0.9 and s["precision"] > 0.95 and s["sorting_accuracy"] > 0.9
