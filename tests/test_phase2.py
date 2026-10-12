"""Phase 2 methods on synthetic data: alignment, Riemannian geometry, cross-day adaptation."""
import numpy as np

from neuralsignal.models import riemann as R


def _spd(rng, n=6):
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def test_euclidean_alignment_whitens_mean_covariance():
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((6, 6))
    X = np.einsum("cd,ndt->nct", mix, rng.standard_normal((50, 6, 400)))
    Z = R.euclidean_align(X, X)
    np.testing.assert_allclose(R.covariances(Z).mean(0), np.eye(6), atol=0.02)


def test_riemann_mean_of_identical_matrices_is_that_matrix():
    rng = np.random.default_rng(1)
    C = _spd(rng)
    np.testing.assert_allclose(R.riemann_mean(np.stack([C] * 5)), C, rtol=1e-6)


def test_riemann_mean_is_affine_invariant():
    """mean(W C W') == W mean(C) W' - the property that makes re-centring meaningful."""
    rng = np.random.default_rng(2)
    Cs = np.stack([_spd(rng) for _ in range(8)])
    W = rng.standard_normal((6, 6)) + 3 * np.eye(6)
    lhs = R.riemann_mean(W @ Cs @ W.T)
    rhs = W @ R.riemann_mean(Cs) @ W.T
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5)


def test_recentering_removes_a_per_person_shift():
    """Two 'people' whose data differ by a mixing matrix land in the same place after re-centring."""
    rng = np.random.default_rng(3)
    base = np.stack([_spd(rng) for _ in range(20)])
    W = rng.standard_normal((6, 6)) + 3 * np.eye(6)
    a = R.recenter(base, base)
    b = R.recenter(W @ base @ W.T, W @ base @ W.T)
    # re-centred sets agree up to a rotation; tangent-vector norms are rotation invariant
    na = np.linalg.norm(R.tangent_vectors(a), axis=1)
    nb = np.linalg.norm(R.tangent_vectors(b), axis=1)
    np.testing.assert_allclose(na, nb, rtol=1e-4)


def test_latent_alignment_recovers_a_rotation():
    """Same latent dynamics, channels permuted in meaning by a rotation: align() should undo it."""
    from neuralsignal.crossday import LatentSpace, align
    rng = np.random.default_rng(4)
    z = rng.standard_normal((40, 30, 3))
    L = rng.standard_normal((25, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    rate = lambda Lm: np.exp(0.5 * z @ Lm.T + 1.0)
    Xa = rng.poisson(rate(L)).astype(float)
    Xb = rng.poisson(rate(L)).astype(float)  # same day statistics, independent noise
    ch = np.arange(25)
    A, B = LatentSpace(Xa, ch, 3), LatentSpace(Xb, ch, 3)
    align(B, A, keep=1.0)
    za, zb = A.transform(Xa).reshape(-1, 3), B.transform(Xa).reshape(-1, 3)
    corr = np.mean([abs(np.corrcoef(za[:, i], zb[:, i])[0, 1]) for i in range(3)])
    assert corr > 0.9


def test_kalman_observation_refit_keeps_dynamics():
    from neuralsignal.models.kalman import KalmanDecoder
    rng = np.random.default_rng(5)
    V = np.cumsum(rng.standard_normal((30, 20, 2)), 1)
    P = np.cumsum(V, 1) * 0.05
    X = rng.poisson(2, (30, 20, 10)).astype(float)
    m = KalmanDecoder().fit(X, V, P)
    A = m.A.copy()
    m.refit_observation(X[:5] * 2, V[:5], P[:5])
    np.testing.assert_array_equal(m.A, A)
    assert m.H.shape == (10, A.shape[0])


def test_streaming_matches_offline_decoding():
    """A causal decoder run bin-by-bin must give exactly its offline predictions."""
    from neuralsignal.models.kalman import KalmanDecoder
    from neuralsignal.models.streaming import streaming_decoder
    from neuralsignal.models.torch_fit import GRURegressor
    rng = np.random.default_rng(6)
    V = np.cumsum(rng.standard_normal((12, 15, 2)), 1)
    P = np.cumsum(V, 1) * 0.05
    X = rng.poisson(2, (12, 15, 8)).astype(np.float32)
    for m in (KalmanDecoder().fit(X, V, P), GRURegressor(hidden=8, max_epochs=2, patience=2).fit(X, V)):
        s = streaming_decoder(m)
        online = np.stack([s.step(x) for x in X[0]])
        np.testing.assert_allclose(online, m.predict(X[:1])[0], rtol=1e-4, atol=1e-3)


def test_closed_loop_simulator_reaches_with_a_perfect_decoder():
    """Sanity check: if the 'decoder' returns the intended velocity, the simulated user succeeds."""
    from neuralsignal import closedloop

    class Oracle:  # decodes intended velocity exactly from a 2-unit "population" that encodes it
        pass

    class Enc:
        def sample(self, V, rng):
            return np.asarray(V, dtype=np.float32)

    import neuralsignal.closedloop as cl
    orig = cl.streaming_decoder
    cl.streaming_decoder = lambda d: type("S", (), {"step": staticmethod(lambda c: c), "reset": staticmethod(lambda: None)})()
    try:
        sim = dict(bin_ms=50, n_targets=4, reps=1, target_dist_mm=100, target_radius_mm=15, hold_ms=250,
                   timeout_s=3.0, feedback_delay_ms=100, gain=6.0, max_speed=600, motor_noise=0, warmup_bins=2)
        trials = closedloop.run_closed_loop(Oracle(), Enc(), sim, np.random.default_rng(0), "oracle")
    finally:
        cl.streaming_decoder = orig
    assert all(t["success"] for t in trials)
    assert all(t["time_s"] < 1.5 for t in trials)


def test_bits_per_spike_reference_points():
    """co-bps is 0 for each neuron's mean rate, positive for the true rates, negative for bad ones."""
    from neuralsignal.nlb import bits_per_spike
    rng = np.random.default_rng(7)
    true = np.exp(rng.normal(-3, 1, (50, 40, 6)))
    spikes = rng.poisson(true)
    mean = np.broadcast_to(spikes.reshape(-1, 6).mean(0), true.shape)
    assert abs(bits_per_spike(mean, spikes)) < 1e-9
    assert bits_per_spike(true, spikes) > 0.1
    assert bits_per_spike(true * 5, spikes) < 0


def test_ndt_starts_at_mean_rates():
    """The readout bias is initialised at each neuron's log mean rate, so an untrained NDT is ~the null model."""
    import torch
    from neuralsignal.nlb import NDT, NLBData, bits_per_spike, train_ndt
    rng = np.random.default_rng(8)
    d = NLBData(heldin=rng.poisson(0.05, (24, 20, 10)).astype(float), heldout=rng.poisson(0.05, (24, 20, 4)).astype(float),
                vel=np.zeros((24, 20, 2), np.float32), split=np.array(["train"] * 20 + ["val"] * 4))
    cfg = dict(model=dict(d_model=16, layers=1, heads=2, dropout=0.0), lr=1e-9, weight_decay=0.0, epochs=1,
               batch_size=8, mask_ratio=0.25, eval_every=1)
    rates, _ = train_ndt(d, cfg)
    assert abs(rates[..., 10:].mean() - d.heldout[:20].mean()) < 0.02
