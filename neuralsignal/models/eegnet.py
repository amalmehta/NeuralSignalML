"""EEGNet (Lawhern et al. 2018): a compact CNN that learns CSP-like processing end to end.

Block 1 mirrors the classic pipeline:
  * temporal conv (F1 kernels, ~0.5 s long)   -> learns band-pass filters (like 8-30 Hz)
  * depthwise conv across all electrodes       -> learns D spatial filters per band (like CSP)
  * square-ish nonlinearity (ELU) + avg pool   -> a smoothed power/envelope estimate
Block 2 (separable conv) mixes the band x spatial feature maps over time, and a
single linear layer classifies. ~3k parameters, so it can train on a few hundred
trials without memorising them - the main reason it works on small EEG datasets.
"""
from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int, fs: float = 125.0,
                 F1: int = 8, D: int = 2, F2: int = 16, dropout: float = 0.5):
        super().__init__()
        k1 = int(fs // 2)  # half-second temporal kernel -> frequency resolution ~2 Hz
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),  # separable = depthwise + pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_flat = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_times))).numel()
        self.classify = nn.Linear(n_flat, n_classes)
        # max-norm constraints from the paper are approximated by weight decay in training

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channels, times)
        z = self.block2(self.block1(x.unsqueeze(1)))
        return self.classify(z.flatten(1))
