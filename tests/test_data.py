"""Tests against the real downloaded data. Skipped if `neuralsignal download` hasn't been run."""
import json

import numpy as np
import pytest

from neuralsignal.data import bci2a, mc_maze

needs_2a = pytest.mark.skipif(not bci2a.path_for(1, "T").exists(), reason="BCI 2a not downloaded")
needs_maze = pytest.mark.skipif(not mc_maze.DEFAULT_FILE.exists(), reason="MC_Maze not downloaded")


@needs_2a
def test_bci2a_session_structure_and_epochs():
    from neuralsignal.preprocess.eeg import epoch_session
    s = bci2a.load(1, "T")
    assert s.fs == 250 and s.n_trials == 288
    ep = epoch_session(s)
    assert ep.X.shape == (288, 22, 500)
    assert np.bincount(ep.y).tolist() == [72, 72, 72, 72]


@needs_2a
def test_eog_regression_reduces_eye_artifact_correlation():
    from neuralsignal.preprocess.eeg import bandpass, fit_eog_regression
    s = bci2a.load(1, "T")
    B = fit_eog_regression(s)
    r = s.runs[0]
    eeg, eog = bandpass(r.eeg, 250, 1, 40), bandpass(r.eog, 250, 1, 40)
    fz_before = abs(np.corrcoef(eeg[0], eog[1])[0, 1])
    fz_after = abs(np.corrcoef(eeg[0] - (B @ eog)[0], eog[1])[0, 1])
    assert fz_after < fz_before * 0.5


@needs_maze
def test_mc_maze_segments():
    from neuralsignal.preprocess.spikes import segment_trials
    s = mc_maze.load()
    assert s.n_units == 152
    seg = segment_trials(s, bin_ms=50)
    assert seg.X.shape == (250, 24, 152) and seg.Y.shape == (250, 24, 2)
    assert (seg.split == "val").sum() == 62
    assert not np.isnan(seg.Y).any()


@needs_2a
def test_bridge_recordings_and_export(tmp_path, capsys):
    from neuralsignal.bridge import main
    assert main(["recordings"]) == 0
    recs = json.loads(capsys.readouterr().out)["recordings"]
    assert any(r["id"] == "A01T" for r in recs)
    assert main(["export", "--path", str(bci2a.path_for(1, "T")), "--out", str(tmp_path)]) == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    data = np.fromfile(tmp_path / "signals.f32", dtype=np.float32)
    assert data.size == meta["n_channels"] * meta["n_samples"]
    assert len(meta["events"]) == 288


def test_bridge_reports_errors_as_json(capsys):
    from neuralsignal.bridge import main
    assert main(["export", "--path", "nope.txt", "--out", "/tmp/x"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
