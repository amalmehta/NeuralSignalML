# File structure

```
Neural Signal ML/
├── README.md                    what it is, headline figures, links
├── pyproject.toml               Python package + pinned dependencies
├── configs/                     one YAML per experiment - every setting lives here
│   ├── EEG CSP LDA.yaml
│   ├── EEG EEGNet.yaml
│   ├── Spikes Kalman.yaml
│   ├── Spikes GRU.yaml
│   ├── Spikes Cross Day.yaml        decoders across recording days
│   ├── EEG Without Calibration.yaml alignment + calibration curve for a new person
│   ├── Spikes NDT Benchmark.yaml    NLB co-smoothing: Smoothing vs Neural Data Transformer
│   └── Spikes Closed Loop.yaml      simulated closed-loop cursor control
├── neuralsignal/                the Python pipeline
│   ├── cli.py                   `neuralsignal download | train | report | spikesort | app`
│   ├── paths.py                 where data, results and figures live
│   ├── config.py                load configs, seed everything, record the environment
│   ├── experiments.py           evaluation protocols: train, score, save models
│   ├── report.py                Results.md tables + all figures
│   ├── bridge.py                JSON helper the Mac app calls (incl. `stream` for live decoding)
│   ├── crossday.py              train on one recording day, decode another; adaptation methods
│   ├── eeg_calibration.py       decode a new person with little or no labelled data
│   ├── nlb.py                   Neural Latents Benchmark protocol, Smoothing baseline, NDT
│   ├── closedloop.py            closed-loop cursor simulator (fitted neurons + simulated user)
│   ├── latency.py               time every decoder running online
│   ├── report_phase2.py         figures + tables for the experiments above
│   ├── data/
│   │   ├── download.py          fetch BCI 2a (BNCI) and MC_Maze (DANDI)
│   │   ├── bci2a.py             read 2a .mat sessions into runs/trials
│   │   └── mc_maze.py           read the NWB file: spikes, hand kinematics, trials
│   ├── preprocess/
│   │   ├── eeg.py               band-pass, EOG regression, epoching, trial rejection
│   │   └── spikes.py            binning, neural lag, per-trial segments
│   ├── models/
│   │   ├── csp_lda.py           Common Spatial Patterns + shrinkage LDA
│   │   ├── eegnet.py            EEGNet CNN
│   │   ├── kalman.py            Kalman filter velocity decoder
│   │   ├── rnn.py               GRU velocity decoder
│   │   ├── riemann.py           covariance geometry: alignment, Riemannian mean, tangent space
│   │   ├── streaming.py         bin-by-bin (online) versions of every decoder
│   │   ├── torch_fit.py         shared training loop + sklearn-style wrappers
│   │   └── registry.py          save/load models with their metadata
│   ├── evaluation/metrics.py    accuracy, kappa, chance level, R², bootstrap, Wilcoxon
│   └── spikesort/demo.py        synthetic spike-sorting walkthrough
├── tests/                       pytest: synthetic method tests + real-data checks
├── app/                         the Mac app (Swift package)
│   ├── Package.swift
│   ├── build-app.sh             builds "app/build/Neural Signal ML.app"
│   ├── make_icon.py             draws the app icon
│   ├── Resources/               AppIcon.icns
│   ├── Sources/SignalCore/      filters, spectrogram, decimation (no UI; unit tested)
│   ├── Sources/NeuralSignalML/  SwiftUI app: model, Python bridge, views (Signals, Spectrogram, Live, Decode), settings, feedback
│   └── Tests/SignalCoreTests/
├── docs/
│   ├── INSTRUCTIONS.md          setup, run, use
│   ├── SYSTEM-DESIGN.md         architecture, flows, decisions, limits
│   ├── FILE-STRUCTURE.md        this file
│   ├── WRITEUP.md               the methods explained, and what worked
│   └── figures/                 PNGs made by `neuralsignal report` / `spikesort`
├── results/
│   ├── Results.md               generated tables
│   ├── runs/<id>/               config, environment, metrics, predictions per experiment
│   ├── models/<id>/<protocol>/<target>/   trained decoders (git-ignored)
│   └── logs/                    training logs (git-ignored)
└── data/                        downloaded datasets + preprocessing cache (git-ignored)
```
