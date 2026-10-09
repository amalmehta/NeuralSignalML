"""GRU velocity decoder: a recurrent net that reads spike counts bin by bin (causally).

Where the Kalman filter assumes a fixed linear tuning (z = Hx) and linear dynamics,
the GRU learns its own nonlinear state update from data. Its hidden state can carry
information like "the reach has started" or "this is a curved reach" that a linear
velocity state can't. The cost is far more parameters, so with only ~190 training
trials it needs dropout, weight decay and early stopping to avoid memorising them.

Inputs are sqrt-transformed counts (stabilises Poisson variance), z-scored per unit.
"""
from __future__ import annotations

import torch
from torch import nn


class GRUDecoder(nn.Module):
    def __init__(self, n_units: int, hidden: int = 128, layers: int = 1, dropout: float = 0.3,
                 input_dropout: float = 0.2):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
        self.inp = nn.Linear(n_units, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out_drop = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, bins, units)
        h, _ = self.gru(torch.relu(self.inp(self.in_drop(x))))
        return self.readout(self.out_drop(h))
