"""
MFN: multiplicative filter network (Fourier / Gabor filter variants).

Fathony et al., "Multiplicative Filter Networks", ICLR 2021.
Reference implementation: https://github.com/boschresearch/multiplicative-filter-networks
"""

import torch
from torch import nn
import numpy as np

class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Frequency (mu) and Scale (gamma) initialization
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        # Calculate squared Euclidean distance between x and mu
        # D = ||x - mu||^2
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])

class MFN(nn.Module):
    """
    Multiplicative Filter Network (GaborNet variant).
    """
    def __init__(
        self,
        in_features=2,
        hidden_features=256,
        hidden_layers=3,
        out_features=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__()

        self.out_features = out_features
        self.output_act = output_act

        # 1. Initialize Gabor Filters (from GaborNet)
        # We need n_layers + 1 filters
        self.net = nn.ModuleList(
            [
                GaborLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(hidden_layers + 1),
                    alpha / (hidden_layers + 1),
                    beta,
                )
                for _ in range(hidden_layers + 1)
            ]
        )

        # 2. Initialize Linear Layers (from MFNBase)
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features, bias) for _ in range(hidden_layers)]
        )
        
        self.output_linear = nn.Linear(hidden_features, out_features)

        # 3. Initialize Linear Weights (MFNBase logic)
        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_features),
                np.sqrt(weight_scale / hidden_features),
            )

    def forward(self, coords):
        # Enable gradients for coordinates if needed
        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)

        # Recursive MFN application
        # out = g_0(x)
        out = self.net[0](coords)
        
        # out = g_i(x) * W_{i-1}(out)
        for i in range(1, len(self.net)):
            out = self.net[i](coords) * self.linear[i - 1](out)
            
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out, coords