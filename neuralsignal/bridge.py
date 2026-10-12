"""JSON helper the Mac app calls as a subprocess: `python -m neuralsignal app <command> ...`.

  recordings                         list recordings found in ./data
  export  --path FILE --out DIR      write DIR/meta.json + DIR/signals.f32 (+ spikes.f32)
  models                             list trained decoders
  decode  --path FILE --model DIR    run a decoder on a recording, print JSON results

Every command prints exactly one JSON object on stdout. Errors are {"error": "..."} with
exit code 1. The app does its own filtering and spectrograms on the exported raw
signals, so it only needs Python to read the file formats and to run models.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from neuralsignal.data import bci2a, mc_maze
from neuralsignal.paths import DATA_DIR, MODELS_DIR


def _kind(path: Path) -> str:
    if path.suffix == ".mat":
        return "eeg"
    if path.suffix == ".nwb":
        return "spikes"
    raise ValueError(f"Don't know how to open {path.name} - expected a BCI 2a .mat or an NLB .nwb file")


def cmd_recordings(_) -> dict:
    recs = []
    for name in bci2a.available():
        day = "day 1 (training session)" if name.endswith("T") else "day 2 (evaluation session)"
        recs.append(dict(id=name, kind="eeg", title=f"EEG subject {int(name[1:3])}, {day}",
                         group="BCI Competition IV 2a", path=str(bci2a.path_for(int(name[1:3]), name[3]))))
    if mc_maze.DEFAULT_FILE.exists():
        recs.append(dict(id="MC_Maze_Medium", kind="spikes", title="Monkey J reaching, 152 units",
                         group="NLB MC_Maze Medium", path=str(mc_maze.DEFAULT_FILE)))
    return {"recordings": recs}


def _eeg_timeline(sess: bci2a.Session):
    """Concatenate runs into one timeline (with run boundaries) for viewing."""
    sigs, events, runs, offset = [], [], [], 0
    for k, r in enumerate(sess.runs):
        x = np.nan_to_num(np.concatenate([r.eeg, r.eog]))
        sigs.append(x)
        runs.append(dict(index=k, start_s=offset / sess.fs, calibration=r.is_calibration))
        for t0, y, art in zip(r.trial_starts, r.labels, r.artifacts):
            events.append(dict(time_s=(offset + t0) / sess.fs + bci2a.CUE_OFFSET_S,
                               label=bci2a.CLASS_NAMES[y], label_index=int(y), artifact=bool(art)))
        offset += x.shape[1]
    return np.concatenate(sigs, axis=1).astype(np.float32), events, runs


def cmd_export(a) -> dict:
    path, out = Path(a.path), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if _kind(path) == "eeg":
        sess = bci2a.load_file(path)
        X, events, runs = _eeg_timeline(sess)
        X.tofile(out / "signals.f32")
        meta = dict(kind="eeg", id=sess.name, fs=sess.fs, n_channels=X.shape[0], n_samples=X.shape[1],
                    channels=bci2a.EEG_CHANNELS + bci2a.EOG_CHANNELS, units="uV",
                    channel_types=["eeg"] * 22 + ["eog"] * 3, events=events, runs=runs,
                    classes=bci2a.CLASS_NAMES)
    else:
        sess = mc_maze.load_file(path)
        fs = 100.0  # behaviour downsampled 1 kHz -> 100 Hz for display
        step = int(round(1000 / fs))
        vel = sess.hand_vel[::step].T.astype(np.float32)
        vel.tofile(out / "signals.f32")
        counts = np.array([len(s) for s in sess.spike_times], dtype=np.int32)
        np.concatenate(sess.spike_times).astype(np.float32).tofile(out / "spikes.f32")
        events = [dict(time_s=float(r.move_onset_time), label=f"trial {i} ({r.split})", label_index=0,
                       split=r.split, start_s=float(r.start_time), stop_s=float(r.stop_time))
                  for i, r in sess.trials.iterrows() if np.isfinite(r.move_onset_time)]
        meta = dict(kind="spikes", id="MC_Maze_Medium", fs=fs, n_channels=2, n_samples=int(vel.shape[1]),
                    channels=["hand velocity x", "hand velocity y"], units="mm/s",
                    channel_types=["behavior", "behavior"], n_units=sess.n_units,
                    spike_counts=counts.tolist(), events=events, runs=[], duration_s=sess.duration)
    (out / "meta.json").write_text(json.dumps(meta))
    return {"ok": True, "out": str(out), "kind": meta["kind"], "id": meta["id"]}


def cmd_models(_) -> dict:
    from neuralsignal.models import registry
    models = registry.list_models(MODELS_DIR) if MODELS_DIR.exists() else []
    keep = ("path", "method", "kind", "name", "protocol", "target", "trained_on", "test_accuracy",
            "test_r2", "test_recording")
    return {"models": [{k: m.get(k) for k in keep} for m in models]}


def cmd_decode(a) -> dict:
    from neuralsignal.evaluation import metrics as M
    from neuralsignal.models import registry
    path = Path(a.path)
    model, meta = registry.load(Path(a.model))
    kind = _kind(path)
    if kind != meta["kind"]:
        raise ValueError(f"This is a {meta['kind']} decoder but the recording is {kind}.")
    if kind == "eeg":
        from neuralsignal.preprocess.eeg import epoch_session
        sess = bci2a.load_file(path)
        ep = epoch_session(sess, **meta["preprocess"])
        if ep.X.shape[1:] != (meta["n_channels"], meta["n_times"]):
            raise ValueError("Epoch shape doesn't match what the model was trained on.")
        proba = model.predict_proba(ep.X)
        yhat = proba.argmax(1)
        _, events, _ = _eeg_timeline(sess)
        seen = sess.name in meta.get("trained_on", [])
        trials = [dict(time_s=e["time_s"], true=int(y), pred=int(p), probs=[round(float(v), 4) for v in pr])
                  for e, y, p, pr in zip(events, ep.y, yhat, proba)]
        return dict(kind="eeg", model=meta["name"], protocol=meta["protocol"], recording=sess.name,
                    seen_in_training=seen, classes=bci2a.CLASS_NAMES, trials=trials,
                    accuracy=M.accuracy(ep.y, yhat), kappa=M.kappa(ep.y, yhat),
                    chance_95=M.chance_upper(len(ep.y), 4),
                    note=("Warning: this model was trained on this recording, so accuracy here is "
                          "optimistic." if seen else "Held-out recording: this model never saw it."))
    from neuralsignal.preprocess.spikes import segment_trials
    sess = mc_maze.load_file(path)
    seg = segment_trials(sess, **meta["preprocess"])
    Yhat = model.predict(seg.X)
    seen = np.isin(seg.trial_index, meta.get("trained_on_trials", []))
    centers = (np.arange(seg.X.shape[1]) + 0.5) * seg.bin_s - meta["preprocess"]["pre_ms"] / 1000
    trials = [dict(trial=int(ti), held_out=bool(not s), t_s=[round(float(o + c), 4) for c in centers],
                   true=np.round(Y, 2).tolist(), pred=np.round(P, 2).tolist())
              for ti, s, o, Y, P in zip(seg.trial_index, seen, seg.onset_time, seg.Y, Yhat)]
    r2 = M.velocity_r2(seg.Y[~seen], Yhat[~seen], seg.score_mask) if (~seen).any() else None
    return dict(kind="spikes", model=meta["name"], protocol=meta["protocol"], recording="MC_Maze_Medium",
                trials=trials, r2_held_out=r2, n_held_out=int((~seen).sum()), n_trials=len(trials),
                note="R² is computed on held-out trials only; training trials are shown greyed out.")


def cmd_stream(a) -> dict:
    """Run a decoder online, one update at a time, printing one JSON line per update
    (then a final summary object). The app plays the lines back in real time."""
    import time
    from neuralsignal.models import registry
    from neuralsignal.models.streaming import StreamingEEG, streaming_decoder
    import torch
    torch.set_num_threads(1)  # one update at a time: thread start-up costs more than it saves
    model, meta = registry.load(Path(a.model))
    path = Path(a.path)
    if _kind(path) != meta["kind"]:
        raise ValueError(f"This is a {meta['kind']} decoder but the recording is {_kind(path)}.")
    lat = []

    def emit(obj):
        obj["sent"] = time.time()
        print(json.dumps(obj), flush=True)

    if meta["kind"] == "spikes":
        from neuralsignal.preprocess.spikes import segment_trials
        sess = mc_maze.load_file(path)
        seg = segment_trials(sess, **meta["preprocess"])
        seen = set(meta.get("trained_on_trials", []))
        pre = meta["preprocess"]["pre_ms"] / 1000
        dec = streaming_decoder(model)
        for i in np.argsort(seg.onset_time):
            dec.reset()  # each trial starts from the decoder's resting state, as in training
            for b in range(seg.X.shape[1]):
                t0 = time.perf_counter()
                v = dec.step(seg.X[i, b])
                ms = (time.perf_counter() - t0) * 1000
                lat.append(ms)
                emit(dict(t=round(float(seg.onset_time[i] - pre + (b + 1) * seg.bin_s), 4), trial=int(seg.trial_index[i]),
                          held_out=int(seg.trial_index[i]) not in seen, true=np.round(seg.Y[i, b], 1).tolist(),
                          pred=np.round(v, 1).tolist(), compute_ms=round(ms, 3)))
    else:
        from neuralsignal.preprocess.eeg import fit_eog_regression
        sess = bci2a.load_file(path)
        pp = meta["preprocess"]
        B = fit_eog_regression(sess) if pp.get("eog_regression") else None  # unlabelled calibration runs
        X, events, _ = _eeg_timeline(sess)
        stream = StreamingEEG(model, pp, sess.fs, B)
        chunk = int(sess.fs * 0.1)
        for start in range(0, X.shape[1] - chunk, chunk):
            r = stream.push(X[:, start:start + chunk])
            if r is not None:
                probs, ms = r
                lat.append(ms)
                emit(dict(t=round((start + chunk) / sess.fs, 3), probs=np.round(probs, 4).tolist(), compute_ms=round(ms, 3)))
    lat = np.array(lat)
    return dict(done=True, updates=int(len(lat)), compute_ms_median=float(np.median(lat)),
                compute_ms_p95=float(np.percentile(lat, 95)), compute_ms_max=float(lat.max()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neuralsignal app")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("recordings")
    e = sub.add_parser("export")
    e.add_argument("--path", required=True)
    e.add_argument("--out", required=True)
    sub.add_parser("models")
    st = sub.add_parser("stream")
    st.add_argument("--path", required=True)
    st.add_argument("--model", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--path", required=True)
    d.add_argument("--model", required=True)
    try:
        a = p.parse_args(argv)
        res = dict(recordings=cmd_recordings, export=cmd_export, models=cmd_models, decode=cmd_decode,
                   stream=cmd_stream)[a.cmd](a)
        print(json.dumps(res))
        return 0
    except SystemExit as ex:
        return int(ex.code or 0)
    except Exception as ex:  # surface the message in the app, details on stderr
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(ex)}))
        return 1
