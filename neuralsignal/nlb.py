"""Neural Latents Benchmark protocol on MC_Maze, so results compare with published work.

The NLB task is *co-smoothing*: each trial's neurons are split into "held-in" (given to
the model) and "held-out" (hidden). A model of the population's latent dynamics should
predict the held-out neurons' firing rates from the held-in ones alone. Score:

  co-bps = (LL(model rates) - LL(each neuron's mean rate)) / (n held-out spikes * ln 2)

the Poisson log-likelihood improvement per spike, in bits. Hand velocity is then decoded
from the inferred rates with ridge regression (vel R²), as the benchmark does.

Data: 5 ms bins, window [-250, +450] ms around movement onset (140 bins), NLB 'train'
trials to fit, 'val' trials to score. (Published numbers use NLB's hidden test split on
EvalAI; we can only score 'val' locally, so we also re-run the Smoothing baseline here as
an anchor between the two.)

Models
  smoothing   Gaussian-smooth the held-in spikes, Poisson-GLM to each held-out neuron
              (the official NLB baseline)
  ndt         Neural Data Transformer (Ye & Pandarinath 2021): a transformer encoder over
              time bins, trained BERT-style - random bins of the input are masked and the
              model must reconstruct every neuron's spikes there - which forces it to learn
              population dynamics rather than copy its input.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor, Ridge
from torch import nn

from neuralsignal import config as cfgmod
from neuralsignal.data import mc_maze
from neuralsignal.evaluation import metrics as M
from neuralsignal.paths import RESULTS_DIR
from neuralsignal.preprocess.spikes import bin_counts


@dataclass
class NLBData:
    heldin: np.ndarray    # (trials, 140, n_in)
    heldout: np.ndarray   # (trials, 140, n_out)
    vel: np.ndarray       # (trials, 140, 2)
    split: np.ndarray     # 'train' / 'val'


def load(day: str = "medium", bin_ms: int = 5, pre_ms: int = 250, post_ms: int = 450) -> NLBData:
    from pynwb import NWBHDF5IO
    path = mc_maze.SESSIONS[day]
    sess = mc_maze.load_file(path)
    with NWBHDF5IO(str(path), "r") as io:
        heldout_flag = np.asarray(io.read().units.heldout.data[:], bool)
    n = int((pre_ms + post_ms) / bin_ms)
    X, V = [], []
    for _, tr in sess.trials.iterrows():
        edges = tr.move_onset_time - pre_ms / 1000 + bin_ms / 1000 * np.arange(n + 1)
        X.append(bin_counts(sess.spike_times, edges))
        idx = np.searchsorted(sess.behavior_t, edges)
        V.append(np.stack([sess.hand_vel[a:b].mean(0) for a, b in zip(idx[:-1], idx[1:])]))
    X = np.stack(X)
    return NLBData(heldin=X[..., ~heldout_flag], heldout=X[..., heldout_flag],
                   vel=np.stack(V).astype(np.float32), split=sess.trials.split.values.astype(str))


def bits_per_spike(rates: np.ndarray, spikes: np.ndarray) -> float:
    """co-bps: Poisson log-likelihood gain over each neuron's mean rate, per spike, in bits."""
    rates = np.clip(rates.reshape(-1, rates.shape[-1]), 1e-9, None)
    s = spikes.reshape(-1, spikes.shape[-1])
    null = np.clip(s.mean(0, keepdims=True), 1e-9, None)
    ll = lambda r: (s * np.log(r) - r).sum()  # log(s!) cancels in the difference
    return float((ll(rates) - ll(np.broadcast_to(null, s.shape))) / (s.sum() * math.log(2)))


def velocity_r2(rates_tr, vel_tr, rates_te, vel_te, alphas=(1e-2, 1e-1, 1, 10, 100, 1e3)) -> float:
    """Ridge from (log) rates to velocity, alpha by 5-fold CV on the training trials only."""
    Ftr, Fte = np.log(rates_tr + 1e-6).reshape(-1, rates_tr.shape[-1]), np.log(rates_te + 1e-6).reshape(-1, rates_te.shape[-1])
    ytr = vel_tr.reshape(-1, 2)
    folds = np.array_split(np.arange(len(rates_tr)), 5)
    nb = rates_tr.shape[1]

    def cv(a):
        sc = []
        for f in folds:
            m = np.ones(len(rates_tr), bool)
            m[f] = False
            rows = np.repeat(m, nb)
            sc.append(M.velocity_r2(vel_tr[f], Ridge(alpha=a).fit(Ftr[rows], ytr[rows]).predict(Ftr[~rows]).reshape(len(f), nb, 2)))
        return np.mean(sc)
    best = max(alphas, key=cv)
    pred = Ridge(alpha=best).fit(Ftr, ytr).predict(Fte).reshape(vel_te.shape)
    return M.velocity_r2(vel_te, pred)


# ---------------------------------------------------------------- smoothing baseline

def smoothing_baseline(d: NLBData, kern_sd_ms=50, bin_ms=5, alpha=0.01):
    """Official NLB baseline: smoothed held-in rates -> Poisson GLM per held-out neuron."""
    sm = gaussian_filter1d(d.heldin.astype(float), kern_sd_ms / bin_ms, axis=1)
    tr, te = d.split == "train", d.split == "val"
    F = lambda m: np.log(sm[m].reshape(-1, sm.shape[-1]) + 1e-4)
    out = np.zeros(d.heldout.shape)
    for j in range(d.heldout.shape[-1]):
        g = PoissonRegressor(alpha=alpha, max_iter=500).fit(F(tr), d.heldout[tr][..., j].ravel())
        out[..., j] = g.predict(np.log(sm.reshape(-1, sm.shape[-1]) + 1e-4)).reshape(sm.shape[:2])
    rates = np.concatenate([sm / 1.0 + 1e-6, out], -1)  # held-in: smoothed counts per bin
    return rates


# ---------------------------------------------------------------- NDT

class NDT(nn.Module):
    """Transformer encoder over time bins -> log firing rates for every neuron."""

    def __init__(self, n_in, n_out, n_bins, d_model=128, layers=2, heads=4, dropout=0.3, context=None):
        super().__init__()
        self.embed = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_bins, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Linear(d_model, n_in + n_out)
        self.context = context  # optional limit on how far apart (in bins) tokens may attend
        if context is not None:
            i = torch.arange(n_bins)
            self.register_buffer("attn_mask", (i[None] - i[:, None]).abs() > context)
        else:
            self.attn_mask = None

    def forward(self, x):  # x: (batch, bins, n_in) spike counts (log1p-scaled)
        h = self.drop(self.embed(x)) + self.pos
        h = self.encoder(h, mask=self.attn_mask)
        return self.readout(self.drop(h))  # log rates


def train_ndt(d: NLBData, cfg: dict, seed: int = 0, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(d.split == "train")
    va_inner = rng.choice(tr, max(1, int(0.15 * len(tr))), replace=False)  # early-stop split from train
    fit = np.setdiff1d(tr, va_inner)
    Xin = torch.as_tensor(np.log1p(d.heldin), dtype=torch.float32)
    Yall = torch.as_tensor(np.concatenate([d.heldin, d.heldout], -1), dtype=torch.float32)
    n_in, n_out, nb = d.heldin.shape[-1], d.heldout.shape[-1], d.heldin.shape[1]
    m = cfg["model"]
    model = NDT(n_in, n_out, nb, **m)
    # Start every neuron at its mean training rate. With the default init the readout predicts ~1 spike
    # per 5 ms bin (about 50x too high) and most of a CPU-sized training budget goes on unlearning that.
    with torch.no_grad():
        mean_rate = Yall[fit].reshape(-1, n_in + n_out).mean(0).clamp_min(1e-4)
        model.readout.bias.copy_(torch.log(mean_rate))
        model.readout.weight.mul_(0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * math.ceil(len(fit) / cfg["batch_size"]))
    pois = nn.PoissonNLLLoss(log_input=True, full=False, reduction="none")
    best, best_state, hist = np.inf, None, []
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(fit)
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            x = Xin[b].clone()
            # BERT-style masking: hide random time bins of the input; score reconstruction there
            mask = torch.rand(x.shape[:2]) < cfg["mask_ratio"]
            zero = mask & (torch.rand(x.shape[:2]) < 0.8)
            x[zero] = 0
            loss = pois(model(x), Yall[b])
            # held-in neurons: scored only where their input was masked (BERT-style, so the model can't
            # just copy its input); held-out neurons are never in the input, so score them everywhere.
            # (Scoring held-out neurons only at masked bins left them untrained for unmasked input: co-bps ~0.)
            loss = torch.cat([loss[..., :n_in][mask].reshape(-1), loss[..., n_in:].reshape(-1)]).mean() \
                if cfg.get("loss_on_masked_only", True) else loss.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 200)
            opt.step()
            sched.step()
        if ep % cfg.get("eval_every", 10) == 0 or ep == cfg["epochs"] - 1:
            model.eval()
            with torch.no_grad():
                # validation: predict held-out neurons from unmasked held-in input (the real task)
                lr_ = model(Xin[va_inner])
                vl = pois(lr_[..., n_in:], Yall[va_inner][..., n_in:]).mean().item()
            hist.append((ep, vl))
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            if log and ep % 100 == 0:
                log(f"    epoch {ep:4d}  val heldout NLL {vl:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rates = torch.exp(model(Xin)).numpy()
    return rates, dict(best_val_nll=best, history=hist, n_params=sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- experiment

def run(config_path: str, log=print) -> dict:
    cfg = cfgmod.load_config(config_path)
    out = RESULTS_DIR / "runs" / cfg["id"]
    cfgmod.write_run_header(cfg, out)
    if "threads" in cfg:
        torch.set_num_threads(int(cfg["threads"]))
    log(f"== {cfg['name']}  (config {cfg['_source']}, hash {cfg['_hash']})")
    t0 = time.time()
    rows, preds = [], {}
    for day in cfg["days"]:
        d = load(day)
        tr, te = d.split == "train", d.split == "val"
        n_in = d.heldin.shape[-1]
        log(f"  {day}: {tr.sum()} train / {te.sum()} val trials, {n_in} held-in + {d.heldout.shape[-1]} held-out neurons")
        runs = [("smoothing", 0, smoothing_baseline(d), {})]
        for seed in cfg["seeds"]:
            cfgmod.seed_everything(seed)
            r, info = train_ndt(d, cfg["ndt"], seed, log if cfg.get("verbose") else None)
            runs.append(("ndt", seed, r, info))
        for model, seed, rates, info in runs:
            row = dict(day=day, model=model, seed=seed,
                       co_bps=bits_per_spike(rates[te][..., n_in:], d.heldout[te]),
                       vel_r2=velocity_r2(rates[tr], d.vel[tr], rates[te], d.vel[te]),
                       **{k: v for k, v in info.items() if k != "history"})
            rows.append(row)
            preds[f"{day}/{model}/seed{seed}"] = rates[te].astype(np.float32)
            log(f"  {day:6s} {model:9s} seed {seed}: co-bps {row['co_bps']:.3f}  vel R2 {row['vel_r2']:.3f}")
    np.savez_compressed(out / "predictions.npz", **preds)
    summary = dict(name=cfg["name"], id=cfg["id"], experiment="nlb", config_hash=cfg["_hash"],
                   minutes=round((time.time() - t0) / 60, 1), rows=rows,
                   published_test_split=cfg.get("published_test_split", {}))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"== done in {summary['minutes']} min -> {out}")
    return summary
