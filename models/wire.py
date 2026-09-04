"""
WIRE: complex Gabor wavelet implicit neural representation.

Saragadam et al., "WIRE: Wavelet Implicit Neural Representations", CVPR 2023.
Reference implementation: https://github.com/vishwa91/wire
"""

import numpy as np
import torch
from torch import nn

class GaborLayer(nn.Module):
    '''
    Implicit representation with complex Gabor nonlinearity.
    
    Args:
        in_features: Input features;
        out_features: Output features;
        bias: if True, enable bias for the linear operation;
        is_first: Legacy SIREN parameter;
        omega_0: Legacy SIREN parameter;
        omega0: Frequency of Gabor sinusoid term;
        sigma0: Scaling of Gabor Gaussian term;
        trainable: If True, omega and sigma are trainable parameters.
    '''
    
    def __init__(self, in_features, out_features, bias = True, is_first = False, omega0 = 10.0, sigma0 = 40.0, trainable = False):
        
        super().__init__()
        
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features, out_features, bias = bias, dtype = dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class WIRE(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0 = 20.0, hidden_omega_0 = 20.0, scale = 30.0):
        super().__init__()
        
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2.
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat

        self.net = []
        self.net.append(GaborLayer(in_features, hidden_features, 
                                    omega0 = first_omega_0, sigma0 = scale,
                                    is_first = True, trainable = False))

        for i in range(hidden_layers):
            self.net.append(GaborLayer(hidden_features, hidden_features,
                                       omega0 = hidden_omega_0, sigma0 = scale))

        final_linear = nn.Linear(hidden_features, out_features, dtype = dtype)
                    
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)

        """
        self._init_weights()
        """

    """
    def _init_weights(self):
            with torch.no_grad():
                for layer in self.net:
                    if isinstance(layer, GaborLayer):
                        # WIRE typically uses a tighter initialization than He/Kaiming to ensure the Gaussian envelope doesn't kill the signal early.
                        # Standard practice is U(-1/n, 1/n) or similar small bounds.
                        limit = np.sqrt(6 / layer.in_features) # or 1 / in_features
                        nn.init.uniform_(layer.linear.weight, -limit, limit)
                        if layer.linear.bias is not None:
                            nn.init.zeros_(layer.linear.bias)
    """

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output.real, coords