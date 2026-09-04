"""
Fourier feature network: random Fourier feature encoding followed by a ReLU MLP.

Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in
Low Dimensional Domains", NeurIPS 2020.
Reference implementation: https://github.com/tancik/fourier-feature-networks
"""

import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        
        # "B" Matrix from the paper: Sampled from N(0, scale^2)
        # Shape: [mapping_size, in_features]
        self.B = nn.Parameter(torch.randn(mapping_size, in_features) * scale, requires_grad=True)
        
        self.mapping_size = mapping_size
        self.output_dim = mapping_size * 2 # sin and cos for each freq

    def forward(self, x):
        # 1. Projection: x @ B.T
        # x: [batch, in_features]
        # B: [mapping_size, in_features]
        # proj: [batch, mapping_size]
        x_proj = (2.0 * np.pi * x) @ self.B.t()
        
        # 2. Fourier Features: [sin(proj), cos(proj)]
        # Concatenate along the last dimension
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFeatures(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3, mapping_size=256, scale=10.0):
        """
        Args:
            mapping_size (int): Number of Fourier features (often equal to hidden_features/2).
            scale (float): The standard deviation of the Gaussian distribution. 
                           Controls the frequency bandwidth.
        """
        super().__init__()

        # 1. Fourier Feature Mapping
        self.mapping = FourierFeatureMapping(in_features, mapping_size, scale)
        
        # 2. Standard MLP
        # The input to the MLP is the fourier features (size * 2)
        self.net = []
        
        # First layer: mapped_features -> hidden
        self.net.append(nn.Linear(mapping_size * 2, hidden_features))
        self.net.append(nn.ReLU())
        
        # Hidden self.net
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())
            
        # Output layer (Linear, no activation)
        self.net.append(nn.Linear(hidden_features, out_features))
        
        self.mlp = nn.Sequential(*self.net)
        
        # self.init_weights()

    def init_weights(self):
        # Standard He/Xavier initialization for the MLP parts
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords):
        if not coords.requires_grad:
            coords = coords.clone().requires_grad_(True)
            
        # 1. Map coordinates to Fourier features
        h = self.mapping(coords)
        
        # 2. Pass through MLP
        output = self.mlp(h)
        
        return output, coords