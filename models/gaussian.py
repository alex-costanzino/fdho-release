"""
Gaussian-activated coordinate MLP.

Ramasinghe and Lucey, "Beyond Periodicity: Towards a Unifying Framework for
Activations in Coordinate-MLPs", ECCV 2022.
"""

import torch
import torch.nn as nn

class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, sigma = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sigma = sigma

    def forward(self, x):
        z = self.linear(x) / self.sigma
        return torch.exp(-0.5 * (z**2))
    
    def forward_with_intermediate(self, x):
        linear_out = self.linear(x)
        return torch.exp(-0.5 * ((linear_out / self.sigma) **2)), linear_out

class Gaussian(nn.Module):
    def __init__(self, in_features = 2, hidden_features = 256, hidden_layers = 3, out_features = 3, sigma = 0.1):
        """
        Args:
            sigma (float): The sigma parameter (sigma).
        """
        super().__init__()
        
        layers = []
        
        layers.append(GaussianLayer(in_features, hidden_features, sigma = sigma))
        
        for _ in range(hidden_layers):
            layers.append(GaussianLayer(hidden_features, hidden_features, sigma = sigma))
            
        final_linear = nn.Linear(hidden_features, out_features)
        
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords