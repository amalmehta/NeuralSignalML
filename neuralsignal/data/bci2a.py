"""BCI Competition IV dataset 2a: 4-class motor imagery EEG (BNCI Horizon .mat version).

9 subjects x 2 sessions on different days (T = training session, E = evaluation session).
Each session: up to 3 short EOG calibration runs (eyes open / closed / eye movements),
then 6 runs x 48 trials = 288 motor-imagery trials. 22 EEG + 3 EOG channels at 250 Hz.

Trial timeline (seconds from trial start): 0 fixation cross, 2 cue appears
(left hand / right hand / feet / tongue), 2-6 imagery, then a short break.
The .mat files give `trial` = trial start sample, so the cue is at +2 s.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from neuralsignal.paths import DATA_DIR

EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]
CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]
CUE_OFFSET_S = 2.0


@dataclass
class Run:
    """One continuous run. Filtering is done per run so we never filter across a gap."""
    eeg: np.ndarray            # (22, n_samples) microvolts
    eog: np.ndarray            # (3, n_samples) microvolts
    trial_starts: np.ndarray   # sample index of each trial start (0-based)
    labels: np.ndarray         # 0..3, or empty for calibration runs
    artifacts: np.ndarray      # expert-marked artifact flag per trial

    @property
    def is_calibration(self) -> bool:
        return len(self.trial_starts) == 0


@dataclass
class Session:
    subject: int
    session: str               # "T" or "E"
    fs: float
    runs: list[Run] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"A{self.subject:02d}{self.session}"

    @property
    def n_trials(self) -> int:
        return sum(len(r.labels) for r in self.runs)


def path_for(subject: int, session: str) -> Path:
    return DATA_DIR / "bci2a" / f"A{subject:02d}{session}.mat"


def load_file(path: Path) -> Session:
    path = Path(path)
    name = path.stem  # e.g. A01T
    subject, session = int(name[1:3]), name[3]
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
    runs, fs = [], None
    for r in np.atleast_1d(mat):
        fs = float(r.fs)
        X = np.asarray(r.X, dtype=np.float64).T  # (25, n)
        trial = np.atleast_1d(r.trial).astype(int) - 1  # MATLAB is 1-based
        y = np.atleast_1d(r.y).astype(int) - 1
        art = np.atleast_1d(r.artifacts).astype(bool)
        runs.append(Run(eeg=X[:22], eog=X[22:25], trial_starts=trial, labels=y, artifacts=art))
    return Session(subject=subject, session=session, fs=fs, runs=runs)


def load(subject: int, session: str) -> Session:
    return load_file(path_for(subject, session))


def available() -> list[str]:
    folder = DATA_DIR / "bci2a"
    return sorted(p.stem for p in folder.glob("A0[1-9][TE].mat")) if folder.exists() else []
