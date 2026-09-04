"""
FR: Fourier-reparameterized training of a coordinate MLP.

Shi et al., "Improved Implicit Neural Representation with Fourier
Reparameterized Training", CVPR 2024.
Reference implementation: https://github.com/LabShuHangGU/FR-INR
"""

import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F


class FourierReparamLayer(nn.Module):
    """
    Fourier Reparameterized Linear Layer.
    
    Decomposes the weight matrix as W = λ · B where B is a fixed set of
    Fourier bases and λ are learned coefficients.
    """
    def __init__(self, in_features, out_features, high_freq_num, low_freq_num,
                 phi_num, alpha):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num = high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha = alpha

        # 1. Fixed Fourier bases.
        self.bases = self._init_bases()

        # 2. Learned coefficients.
        self.lamb = self._init_lamb()

        self.bias = nn.Parameter(torch.zeros(out_features))

    def _init_bases(self):
        phi_set = np.array([2 * math.pi * i / self.phi_num
                            for i in range(self.phi_num)])
        high_freq = np.array([i + 1 for i in range(self.high_freq_num)])
        low_freq = np.array([(i + 1) / self.low_freq_num
                             for i in range(self.low_freq_num)])

        if len(low_freq) != 0:
            T_max = 2 * math.pi / low_freq[0]
        else:
            T_max = 2 * math.pi / min(high_freq)

        points = np.linspace(-T_max / 2, T_max / 2, self.in_features)

        total_bases = (self.high_freq_num + self.low_freq_num) * self.phi_num
        bases = torch.zeros(total_bases, self.in_features)

        idx = 0
        for freq in low_freq:
            for phi in phi_set:
                bases[idx, :] = torch.tensor(
                    [math.cos(freq * x + phi) for x in points])
                idx += 1

        for freq in high_freq:
            for phi in phi_set:
                bases[idx, :] = torch.tensor(
                    [math.cos(freq * x + phi) for x in points])
                idx += 1

        bases = self.alpha * bases
        return nn.Parameter(bases.float(), requires_grad=False)

    def _init_lamb(self):
        total_bases = (self.high_freq_num + self.low_freq_num) * self.phi_num
        lamb = torch.zeros(self.out_features, total_bases)

        with torch.no_grad():
            for i in range(total_bases):
                dominator = torch.norm(self.bases[i, :], p=2)
                limit = np.sqrt(6 / total_bases) / dominator
                nn.init.uniform_(lamb[:, i], -limit, limit)

        return nn.Parameter(lamb, requires_grad=True)

    def forward(self, x):
        weight = torch.matmul(self.lamb, self.bases)
        return F.linear(x, weight, self.bias)


class FR(nn.Module):
    """
    Fourier Reparameterized Implicit Neural Representation.
    """
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3,
                 out_features=3, high_freq_num=128, low_freq_num=128,
                 phi_num=32, alpha=0.05):
        super().__init__()

        layers = []

        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(hidden_layers):
            layers.append(
                FourierReparamLayer(
                    hidden_features, hidden_features,
                    high_freq_num, low_freq_num,
                    phi_num, alpha
                )
            )
            layers.append(nn.ReLU(inplace=True))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            nn.init.uniform_(final_linear.weight,
                             -np.sqrt(6 / hidden_features),
                             np.sqrt(6 / hidden_features))
            nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)
        return self.net(coords), coords