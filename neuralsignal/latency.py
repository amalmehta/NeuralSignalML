"""Measure online decoding latency: every decoder run bin-by-bin, exactly as the Mac app's Live tab does.

Budgets: a spike decoder must finish each 50 ms bin before the next arrives; the EEG
classifier updates every 250 ms. Numbers are wall-clock compute per update on this machine
(single thread), so they include whatever else the machine is doing.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
from argparse import Namespace

from neuralsignal.bridge import cmd_stream
from neuralsignal.data import bci2a, mc_maze
from neuralsignal.paths import MODELS_DIR, RESULTS_DIR

CASES = [
    ("Kalman filter", "spikes-kalman/nlb_split/MC_Maze_Medium", str(mc_maze.DEFAULT_FILE), 50),
    ("GRU", "spikes-gru/nlb_split/MC_Maze_Medium", str(mc_maze.DEFAULT_FILE), 50),
    ("CSP + LDA", "eeg-csp-lda/cross_session/A01", str(bci2a.path_for(1, "E")), 250),
    ("EEGNet", "eeg-eegnet/cross_session/A01", str(bci2a.path_for(1, "E")), 250),
]


def main():
    rows = []
    for name, model, path, budget in CASES:
        with contextlib.redirect_stdout(io.StringIO()):
            r = cmd_stream(Namespace(path=path, model=str(MODELS_DIR / model)))
        r.update(decoder=name, budget_ms=budget, within_budget_p95=r["compute_ms_p95"] < budget)
        rows.append(r)
        print(f"  {name:14s} median {r['compute_ms_median']:7.3f} ms   p95 {r['compute_ms_p95']:7.3f} ms   "
              f"max {r['compute_ms_max']:7.1f} ms   budget {budget} ms")
    out = RESULTS_DIR / "runs" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    try:
        import subprocess
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    (out / "metrics.json").write_text(json.dumps(dict(rows=rows, machine=platform.platform(),
                                                      processor=cpu or platform.processor(), load_average=load), indent=2))
    print(f"  (machine load average during the run: {load:.0f})")
