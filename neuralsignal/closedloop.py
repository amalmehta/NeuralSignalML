"""Simulated closed-loop cursor control: does a better offline decoder make a better BCI?

Offline, a decoder predicts movements the arm actually made. In a real BCI there is no arm:
the user watches the cursor the decoder produces and *corrects* it, so decoder errors feed
back into the neural activity. Offline accuracy is known to be a weak predictor of how well
a decoder works in this loop (Koyama et al. 2010; Cunningham et al. 2011).

The simulator (after Cunningham et al. 2011):
  * Neurons: each MC_Maze unit's tuning to intended velocity, fitted on real data with a
    Poisson GLM:  rate = exp(b + w . v + c |v|)   (50 ms bins, spikes leading by 100 ms).
  * User: sees the cursor with a visual-feedback delay, aims at the target with a
    proportional controller (speed capped at what the monkey actually reached), plus
    motor noise.
  * Decoder: trained the usual way on open-loop data from the same simulated neurons
    (real reach kinematics -> simulated spikes), then put in the loop.
  * Task: centre-out to 8 targets 100 mm away; success = cursor inside a 15 mm target for
    250 ms within 3 s.

Limits: real neurons have dynamics and adaptation the GLM lacks, and real users learn the
decoder. The simulator ranks decoders under a stated, simple model of the user.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.linear_model import PoissonRegressor

from neuralsignal import config as cfgmod
from neuralsignal.data import mc_maze
from neuralsignal.evaluation import metrics as M
from neuralsignal.models.kalman import KalmanDecoder
from neuralsignal.models.streaming import streaming_decoder
from neuralsignal.models.torch_fit import GRURegressor
from neuralsignal.paths import RESULTS_DIR
from neuralsignal.preprocess.spikes import segment_trials


class EncodingModel:
    """Per-unit Poisson GLM from velocity to spike counts in one bin."""

    def fit(self, X, Y, alpha=1e-3):
        V = Y.reshape(-1, 2)
        F = self._features(V)
        self.models = [PoissonRegressor(alpha=alpha, max_iter=300).fit(F, X.reshape(-1, X.shape[-1])[:, j])
                       for j in range(X.shape[-1])]
        self.coef = np.stack([m.coef_ for m in self.models])
        self.intercept = np.array([m.intercept_ for m in self.models])
        return self

    @staticmethod
    def _features(V):
        V = np.asarray(V) / 300.0  # mm/s -> roughly unit scale
        return np.concatenate([V, np.linalg.norm(V, axis=-1, keepdims=True)], -1)

    def rates(self, V):
        return np.exp(np.clip(self._features(V) @ self.coef.T + self.intercept, -10, 4))

    def sample(self, V, rng):
        return rng.poisson(self.rates(V)).astype(np.float32)


def simulate_open_loop(enc, Y, rng):
    """Spikes for real reach kinematics Y (trials, bins, 2)."""
    return enc.sample(Y.reshape(-1, 2), rng).reshape(*Y.shape[:2], -1)


def run_closed_loop(decoder, enc, cfg, rng, kind):
    dt = cfg["bin_ms"] / 1000
    delay = int(round(cfg["feedback_delay_ms"] / cfg["bin_ms"]))
    n_steps = int(round(cfg["timeout_s"] / dt))
    hold = int(round(cfg["hold_ms"] / cfg["bin_ms"]))
    angles = np.linspace(0, 2 * np.pi, cfg["n_targets"], endpoint=False)
    out = []
    stream = streaming_decoder(decoder)
    for rep in range(cfg["reps"]):
        for ang in angles:
            target = cfg["target_dist_mm"] * np.array([np.cos(ang), np.sin(ang)])
            stream.reset()
            pos = np.zeros(2)
            hist = [pos.copy()] * (delay + 1)
            path, inside, t_success = [pos.copy()], 0, None
            # warm-up: the user holds still at the centre while the decoder settles (as in the offline segments)
            for _ in range(cfg["warmup_bins"]):
                stream.step(enc.sample(np.zeros((1, 2)), rng)[0])
            for t in range(n_steps):
                seen = hist[-1 - delay]                     # delayed visual feedback
                err = target - seen
                v_int = cfg["gain"] * err
                sp = np.linalg.norm(v_int)
                if sp > cfg["max_speed"]:
                    v_int *= cfg["max_speed"] / sp
                v_int = v_int + rng.normal(0, cfg["motor_noise"], 2)
                v_dec = stream.step(enc.sample(v_int[None], rng)[0])
                pos = pos + v_dec * dt
                hist.append(pos.copy())
                path.append(pos.copy())
                inside = inside + 1 if np.linalg.norm(pos - target) < cfg["target_radius_mm"] else 0
                if inside >= hold:
                    t_success = (t + 1) * dt
                    break
            path = np.array(path)
            length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            out.append(dict(decoder=kind, target_angle=float(np.degrees(ang)), rep=rep, success=t_success is not None,
                            time_s=t_success, path_efficiency=float(cfg["target_dist_mm"] / max(length, 1e-6))
                            if t_success else None, path=path[::2].round(1).tolist()))
    return out


def run(config_path: str, log=print) -> dict:
    import torch
    cfg = cfgmod.load_config(config_path)
    out_dir = RESULTS_DIR / "runs" / cfg["id"]
    cfgmod.write_run_header(cfg, out_dir)
    torch.set_num_threads(int(cfg.get("threads", 1)))
    log(f"== {cfg['name']}  (config {cfg['_source']}, hash {cfg['_hash']})")
    t0 = time.time()
    pp, sim = cfg["preprocess"], cfg["simulation"]
    seg = segment_trials(mc_maze.load(), **pp)
    tr, te = seg.split == "train", seg.split == "val"
    enc = EncodingModel().fit(seg.X[tr], seg.Y[tr])
    log(f"  encoding model fitted on {tr.sum()} real trials, {seg.X.shape[-1]} units")
    rows, paths = [], {}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        cfgmod.seed_everything(seed)
        Xs = simulate_open_loop(enc, seg.Y, rng)  # same reaches, simulated neurons
        P = np.cumsum(seg.Y, axis=1) * (pp["bin_ms"] / 1000)
        for kind in cfg["decoders"]:
            if kind == "kalman":
                dec = KalmanDecoder(bin_s=pp["bin_ms"] / 1000, **cfg.get("kalman", {})).fit(Xs[tr], seg.Y[tr], P[tr])
            else:
                dec = GRURegressor(seed=seed, **cfg.get("gru", {})).fit(Xs[tr], seg.Y[tr], score_mask=seg.score_mask)
            offline = M.velocity_r2(seg.Y[te], dec.predict(Xs[te]), seg.score_mask)
            trials = run_closed_loop(dec, enc, sim, rng, kind)
            succ = [t for t in trials if t["success"]]
            row = dict(decoder=kind, seed=seed, offline_r2=offline,
                       success_rate=len(succ) / len(trials),
                       median_time_s=float(np.median([t["time_s"] for t in succ])) if succ else None,
                       median_path_efficiency=float(np.median([t["path_efficiency"] for t in succ])) if succ else None,
                       n_trials=len(trials))
            rows.append(row)
            paths[f"{kind}/seed{seed}"] = [dict(angle=t["target_angle"], success=t["success"], path=t["path"])
                                           for t in trials if t["rep"] == 0]
            log(f"  {kind:6s} seed {seed}: offline R2 {offline:.3f} | closed loop: success {row['success_rate']*100:.0f}%, "
                f"median time {row['median_time_s']}, path efficiency {row['median_path_efficiency']}")
    summary = dict(name=cfg["name"], id=cfg["id"], experiment="closedloop", config_hash=cfg["_hash"],
                   minutes=round((time.time() - t0) / 60, 1), rows=rows, paths=paths, simulation=sim)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"== done in {summary['minutes']} min -> {out_dir}")
    return summary
