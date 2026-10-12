"""NLB MC_Maze Medium: motor-cortex spiking while a monkey (Jenkins) reaches through a maze.

One session, 250 successful trials (188 'train' + 62 'val' per the Neural Latents
Benchmark split), 152 sorted units from Utah arrays in M1 and PMd, hand position and
velocity sampled at 1 kHz. Spike sorting has already been done by the data providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neuralsignal.paths import DATA_DIR

DEFAULT_FILE = DATA_DIR / "mc_maze_medium" / "sub-Jenkins_ses-medium_desc-train_behavior+ecephys.nwb"

# The same monkey on three different days, recorded on the same two 96-channel Utah arrays.
SESSIONS = {name: DATA_DIR / f"mc_maze_{name}" / f"sub-Jenkins_ses-{name}_desc-train_behavior+ecephys.nwb"
            for name in ("small", "medium", "large")}
SESSION_DATES = {"small": "2009-09-28", "medium": "2009-09-29", "large": "2009-10-06"}
N_ELECTRODES = 192


@dataclass
class SpikeSession:
    name: str
    spike_times: list[np.ndarray]   # one array of times (s) per unit
    behavior_t: np.ndarray          # (n,) seconds, 1 kHz
    hand_vel: np.ndarray            # (n, 2)
    hand_pos: np.ndarray            # (n, 2)
    trials: pd.DataFrame            # start/stop/move_onset_time/split/...
    unit_electrode: np.ndarray | None = None  # electrode (0..191) each unit was sorted from

    @property
    def n_units(self) -> int:
        return len(self.spike_times)

    @property
    def duration(self) -> float:
        return float(self.behavior_t[-1])


def load_file(path: Path = DEFAULT_FILE) -> SpikeSession:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        units = f.units.to_dataframe()
        spikes = [np.asarray(s, dtype=np.float64) for s in units.spike_times]
        electrode = np.asarray(f.units.electrodes.data[:], dtype=int)
        beh = f.processing["behavior"]
        vel = np.asarray(beh["hand_vel"].data[:], dtype=np.float64)
        pos = np.asarray(beh["hand_pos"].data[:], dtype=np.float64)
        t = np.asarray(beh["hand_vel"].timestamps[:], dtype=np.float64)
        trials = f.trials.to_dataframe()[
            ["start_time", "stop_time", "go_cue_time", "move_onset_time", "split", "trial_type"]
        ].reset_index(drop=True)
    trials = trials[trials.split.isin(["train", "val"])].reset_index(drop=True)
    return SpikeSession(name=Path(path).stem.split("_desc")[0], spike_times=spikes,
                        behavior_t=t, hand_vel=vel, hand_pos=pos, trials=trials, unit_electrode=electrode)


def load() -> SpikeSession:
    return load_file(DEFAULT_FILE)


def available() -> list[str]:
    return [DEFAULT_FILE.stem] if DEFAULT_FILE.exists() else []
