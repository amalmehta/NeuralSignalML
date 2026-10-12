"""Save and load trained decoders with the metadata the app needs to run them honestly.

Each model is a folder:  results/models/<config>/<protocol>/<target>/
    meta.json   - method, signal kind, preprocessing, which recordings it was trained on
    model.*     - the weights (joblib for CSP+LDA, npz for Kalman, pt for torch models)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from neuralsignal.models.kalman import KalmanDecoder
from neuralsignal.models.torch_fit import EEGNetClassifier, GRURegressor
from neuralsignal.models.eegnet import EEGNet
from neuralsignal.models.rnn import GRUDecoder


def save(model, folder: Path, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    method = meta["method"]
    if method == "csp_lda":
        joblib.dump(model, folder / "model.joblib")
    elif method == "kalman":
        np.savez(folder / "model.npz", **model.state_dict())
    elif method == "eegnet":
        torch.save({"state_dict": model.model.state_dict(), "scale": model.scale,
                    "n_classes": model.n_classes, "kw": model.kw, "fs": model.fs,
                    "n_channels": meta["n_channels"], "n_times": meta["n_times"]}, folder / "model.pt")
    elif method == "gru":
        torch.save({"state_dict": model.model.state_dict(), "mu": model.mu, "sd": model.sd,
                    "y_sd": model.y_sd, "kw": model.kw, "n_units": meta["n_units"],
                    "input_transform": getattr(model, "input_transform", "sqrt")}, folder / "model.pt")
    else:
        raise ValueError(method)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def load(folder: Path):
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    method = meta["method"]
    if method == "csp_lda":
        return joblib.load(folder / "model.joblib"), meta
    if method == "kalman":
        return KalmanDecoder.from_state(np.load(folder / "model.npz")), meta
    ck = torch.load(folder / "model.pt", weights_only=False)
    if method == "eegnet":
        m = EEGNetClassifier(fs=ck["fs"])
        m.model = EEGNet(ck["n_channels"], ck["n_times"], ck["n_classes"], fs=ck["fs"], **ck["kw"])
        m.model.load_state_dict(ck["state_dict"])
        m.scale, m.n_classes = ck["scale"], ck["n_classes"]
        return m, meta
    if method == "gru":
        m = GRURegressor()
        m.model = GRUDecoder(ck["n_units"], **ck["kw"])
        m.model.load_state_dict(ck["state_dict"])
        m.mu, m.sd, m.y_sd = ck["mu"], ck["sd"], ck["y_sd"]
        m.input_transform = ck.get("input_transform", "sqrt")
        return m, meta
    raise ValueError(method)


def list_models(root: Path) -> list[dict]:
    out = []
    for meta_path in sorted(Path(root).glob("**/meta.json")):
        meta = json.loads(meta_path.read_text())
        meta["path"] = str(meta_path.parent)
        out.append(meta)
    return out
