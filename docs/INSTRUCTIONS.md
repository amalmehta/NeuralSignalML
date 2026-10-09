# Instructions

How to set up, run and use Neural Signal ML on a Mac.

## 1. Set up

You need macOS 14 or later, Xcode (or the Xcode command-line tools) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

PyTorch is pinned to 2.2.2 because that's the last version with wheels for Intel Macs. It works on Apple Silicon too.

## 2. Download the data (about 830 MB)

```bash
.venv/bin/neuralsignal download
```

| Dataset | What | Where it goes |
|---|---|---|
| BCI Competition IV 2a | 4-class motor-imagery EEG, 9 subjects × 2 days | `data/bci2a/A01T.mat` … `A09E.mat` |
| NLB MC_Maze Medium | 152 motor-cortex units + hand velocity, 250 reaches | `data/mc_maze_medium/*.nwb` |

`data/` is git-ignored.

## 3. Train and evaluate

```bash
.venv/bin/neuralsignal train --all
```

That runs every config in `configs/`. Or run one at a time:

```bash
.venv/bin/neuralsignal train "configs/EEG CSP LDA.yaml"
```

| Config | Time on an 8-core Intel Mac |
|---|---|
| `EEG CSP LDA.yaml` | ~8 min (mostly the first-time preprocessing, cached in `data/cache/`) |
| `EEG EEGNet.yaml` | ~1–3 h (CPU) |
| `Spikes Kalman.yaml` | < 1 min |
| `Spikes GRU.yaml` | ~10–30 min |

Each run writes to `results/runs/<id>/`:
- `config.resolved.yaml`: the exact settings, with a hash
- `environment.json`: Python and package versions
- `metrics.json`: every fold, subject and seed
- `predictions.npz`: test-set predictions

Trained models go to `results/models/<id>/<protocol>/<target>/`.

Then build the summary tables and figures:

```bash
.venv/bin/neuralsignal report
.venv/bin/neuralsignal spikesort
```

The first writes `results/Results.md` and `docs/figures/*.png`. The second is the synthetic spike-sorting demo.

### Changing an experiment

Copy a YAML file in `configs/`, change its `id` (the output folder) and whatever settings you like, and pass it to `train`. All preprocessing and model choices live in the config. Nothing is hard-coded in the scripts.

## 4. Run the Mac app

```bash
app/build-app.sh --open
```

This builds `app/build/Neural Signal ML.app`. Drag it to Applications if you like. It remembers where the project folder is, and you can change that in **Settings (⌘,) → General**.

**First launch:** macOS asks whether *Neural Signal ML* may access files in your Desktop (or Documents) folder if the project lives there. Click **Allow**. Until you do, the app shows "Looking for recordings…" because macOS blocks its file access.

### Using it

- **Sidebar.** Lists every recording found in `data/`. Open other `.mat` or `.nwb` files with **⌘O**.
- **Signals tab.**
  - EEG: raw traces (grey) and filtered traces (blue). Set the band in the toolbar or pick a preset. Dashed lines mark cues and the shaded box is the 2 s the decoders read. EOG channels are purple.
  - Spikes: a spike raster, smoothed firing rates and hand velocity. Dashed lines mark movement onsets; green ones are NLB test ("val") trials.
- **Spectrogram tab.** Time-frequency power for one channel. Try C3 or C4 during left- vs right-hand trials.
- **Decode tab.** Pick a trained decoder and press **Run decoder** (⌘↩).
  - EEG: accuracy, kappa, a confusion matrix and a trial list. **Show** jumps to a trial; ✓/✗ badges then appear on the signal view.
  - Spikes: held-out R² and true vs decoded velocity for each trial.
  - The app warns you if the decoder was trained on the recording you're decoding.
- **Settings (⌘,).** Project folder and Python, filter defaults, mains notch, causal vs zero-phase display filtering, spike-rate smoothing, spectrogram window.
- **Feedback.** The small button at the bottom of the sidebar (or **Help → Send Feedback…**) saves a note to `~/Library/Application Support/Neural Signal ML/feedback.jsonl`. Nothing is sent anywhere.

## 5. Tests

```bash
.venv/bin/python -m pytest
```

Tests on real data skip themselves if the data isn't downloaded. The app's DSP has its own tests:

```bash
cd app && swift test
```
