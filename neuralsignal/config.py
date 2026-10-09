"""Reproducible configs: YAML in, a fully-resolved copy plus environment info out."""
from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_source"] = str(path)
    cfg["_hash"] = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment() -> dict:
    pkgs = ["numpy", "scipy", "scikit-learn", "torch", "pynwb"]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: metadata.version(p) for p in pkgs},
    }


def write_run_header(cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "environment.json").write_text(json.dumps(environment(), indent=2))
