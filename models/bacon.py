"""
BACON: band-limited coordinate network.

Lindell et al., "BACON: Band-limited Coordinate Networks for Multiscale Scene
Representation", CVPR 2022.
Reference implementation: https://github.com/computational-imaging/bacon
"""

import torch
import torch.nn as nn
import numpy as np


class BACONLayer(nn.Module):
    def __init__(self, in_features, hidden_features, is_first=False):
        super().__init__()

        self.is_first = is_first

        # Linear transformation.
        self.linear = nn.Linear(hidden_features, hidden_features, bias=True)

        # Frequencies are frozen.
        self.freq = nn.Parameter(
            torch.zeros(in_features, hidden_features), requires_grad=False)

        # Phase shift for sine.
        self.phase = nn.Parameter(
            torch.zeros(hidden_features), requires_grad=False)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            num_input = self.linear.in_features
            w_std = np.sqrt(1 / num_input)
            self.linear.weight.uniform_(-w_std, w_std)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-w_std, w_std)

    def forward(self, h, coords):
        """
        Args:
            h: hidden features [batch, hidden_features].
            coords: input coordinates [batch, in_features].
        Returns:
            output: [batch, hidden_features].
        """
        # Linear transform (no sine — nonlinearity comes from the
        # Hadamard product with the frequency encoding).
        lin_out = self.linear(h)

        # Frequency modulation: sin(omega · x + phase).
        freq_encoding = torch.sin(coords @ self.freq + self.phase)

        # Hadamard product.
        return lin_out * freq_encoding


class BACON(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=8,
                 out_features=3, frequency_limit=128.0, quantize_freq=True):
        """
        Band-limited Coordinate Networks (BACON).

        Args:
            in_features: Input coordinate dimension.
            hidden_features: Width of the network.
            hidden_layers: Number of multiscale layers.
            out_features: Output dimension.
            frequency_limit: Maximum frequency (Nyquist limit).
            quantize_freq: Quantize frequencies to 2π intervals
                           (makes network periodic over [-0.5, 0.5]).
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.frequency_limit = frequency_limit

        # 1. Learned constant input (z0).
        self.z0 = nn.Parameter(torch.zeros(1, hidden_features))
        nn.init.uniform_(self.z0, -1 / hidden_features, 1 / hidden_features)

        # 2. Main layers.
        self.net = nn.ModuleList([
            BACONLayer(in_features, hidden_features, is_first=(i == 0))
            for i in range(hidden_layers)
        ])

        # 3. Multiscale readouts.
        self.readouts = nn.ModuleList([
            nn.Linear(hidden_features, out_features)
            for _ in range(hidden_layers)
        ])

        for readout in self.readouts:
            nn.init.uniform_(readout.weight,
                             -np.sqrt(1 / hidden_features),
                             np.sqrt(1 / hidden_features))
            if readout.bias is not None:
                nn.init.zeros_(readout.bias)

        # 4. Set frequency schedule.
        self._set_frequencies(in_features, hidden_features,
                              hidden_layers, frequency_limit, quantize_freq)

    def _set_frequencies(self, in_features, hidden_features, layers,
                         limit, quantize):
        """
        Linear frequency schedule: each layer covers an equal band
        of [0, limit].
        """
        for i, layer in enumerate(self.net):
            min_freq = limit * (i / layers)
            max_freq = limit * ((i + 1) / layers)

            freqs = torch.FloatTensor(
                in_features, hidden_features).uniform_(min_freq, max_freq)

            if quantize:
                freqs = torch.round(freqs / (2 * np.pi)) * (2 * np.pi)

            layer.freq.data = freqs

    def forward(self, coords):
        """
        Args:
            coords: [batch, in_features] in [-1, 1].

        Returns:
            outputs: list of [batch, out_features] (one per scale).
            coords: input coordinates (gradients enabled if needed).

        Note: BACON expects coords in [-0.5, 0.5]. The division by 2 is done here so the rest of the codebase can use [-1, 1] consistently. 
        """
        # Scale from [-1, 1] to [-0.5, 0.5] (not in-place!).
        coords = coords / 2.0

        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)

        batch_size = coords.shape[0]

        h = self.z0.expand(batch_size, -1)

        outputs = []
        for layer, readout in zip(self.net, self.readouts):
            h = layer(h, coords)
            outputs.append(readout(h))

        return outputs, coords

    def forward_single_scale(self, coords, scale_idx=-1):
        """Get output at a specific scale."""
        outputs, _ = self.forward(coords)
        return outputs[scale_idx]