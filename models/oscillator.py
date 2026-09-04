"""
The proposed forced damped harmonic oscillator (FDHO) representation.

The activation is the steady-state amplitude response of a sine-forced damped
harmonic oscillator, with the physical parameters (forcing frequency, natural
frequency, damping ratio and phase) optionally learned per layer.

Costanzino et al., "Spectral Gating via Damped Oscillations for Adaptive
Implicit Neural Representations", ECCV 2026.
"""

import torch
from torch import nn
import numpy as np

class OscillatorLayer(nn.Module):
    """
    Oscillator activation layer.

    Uses the steady-state amplitude response of a sine-forced damped oscillator:
    activation(z) = sin(omega * z + phi) / sqrt((omega_n^2 - omega^2)^2 + (2*xi*omega_n*omega)^2)
    
    where:
    - omega: forcing frequency;
    - omega_n: natural frequency:  
    - xi: damping ratio;
    - phi: phase offset.
    """
    
    def __init__(self, in_features, out_features, bias = True, is_first = False, 
                 omega = 5.0, omega_n = 7.0, xi = 0.5, phi = 0.0, learn_params = False):
        
        super().__init__()
        
        self.is_first = is_first
        self.learn_params = learn_params
        
        # Oscillator parameters - earnable or fixed per layer
        if learn_params:
            self.omega = nn.Parameter(torch.tensor(omega))
            self.omega_n = nn.Parameter(torch.tensor(omega_n))

            # Normalize to sigmoid range [0.05, 0.95] and apply Inverse Sigmoid (Logit function) logit(p) = ln(p / (1-p)).
            sigmoid_target = (xi - 0.05) / 0.95        
            xi_init = np.log(sigmoid_target / (1.0 - sigmoid_target))

            self.xi = nn.Parameter(torch.tensor(float(xi_init)))

            # We use the initial values of omega, omega_n, xi to calculate the initial physical phase lag.
            real_part = omega_n**2 - omega**2
            imag_part = 2 * xi * omega_n * omega

            phi_init = -np.arctan2(imag_part, real_part)

            self.phi = nn.Parameter(torch.tensor(float(phi_init)))

        else:
            self.register_buffer('omega', torch.tensor(omega))
            self.register_buffer('omega_n', torch.tensor(omega_n))
            self.register_buffer('xi', torch.tensor(xi))
            self.register_buffer('phi', torch.tensor(phi))
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize weights following SIREN's scheme.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_n, np.sqrt(6 / self.in_features) / self.omega_n)
    
    def get_params(self):
        """
        Get current parameter values (constrained if learnable).
        """
        if self.learn_params:
            # Apply constraints to raw parameters.
            omega = torch.nn.functional.softplus(self.omega) + 0.1
            omega_n = torch.nn.functional.softplus(self.omega_n) + 0.1
            xi = 0.05 + 0.95 * torch.sigmoid(self.xi)  # [0.05, 1.0] - relaxed range
            phi = self.phi
        else:
            omega = self.omega
            omega_n = self.omega_n
            xi = self.xi
            phi = self.phi
        
        return omega, omega_n, xi, phi

    @staticmethod
    @torch.jit.script
    def oscillator_activation(z: torch.Tensor, omega: torch.Tensor, 
                              omega_n: torch.Tensor, xi: torch.Tensor, 
                              phi: torch.Tensor) -> torch.Tensor:

        real_part = (omega_n**2 - omega**2)**2
        imag_part = (2 * xi * omega_n * omega)**2
        amp_denom = torch.sqrt(real_part + imag_part)
        
        return (torch.sin(omega * z + phi) / (amp_denom + 1e-8)) * omega_n**2
  
    def forward(self, input):
        z = self.linear(input)
        
        # Get current parameters (with constraints if learnable).
        omega, omega_n, xi, phi = self.get_params()
        
        return self.oscillator_activation(z, omega, omega_n, xi, phi)
    
    def forward_with_intermediate(self, input):
        """
        For visualization of activation distributions.
        """
        z = self.linear(input)
        
        omega, omega_n, xi, phi = self.get_params()
        amp_denom = torch.sqrt((omega_n**2 - omega**2)**2 + (2 * xi * omega_n * omega)**2)
        
        intermediate = omega * z + phi
        return (torch.sin(omega * z + phi) / (amp_denom + 1e-8)) * omega_n**2, intermediate

class Oscillator(nn.Module):
    """
    SIREN-style network with damped oscillator activations.
    
    Args:
        in_features: Number of input features;
        hidden_features: Number of hidden layer features;
        hidden_layers: Number of hidden layers;
        out_features: Number of output features;
        outermost_linear: If True, final layer is linear (no activation);
        omega: Forcing frequency for oscillator activation (initial value if learnable);
        omega_n: Natural frequency for oscillator activation (initial value if learnable);
        xi: Damping ratio (initial value if learnable);
        phi: Phase offset for oscillator activation (initial value if learnable);
        learn_params: If True, omega, omega_n, xi, phi are learnable per layer.
    """
    
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear = False,
                 omega = 5.0, omega_n = 7.0, xi = 0.5, phi = 0.0, 
                 learn_params = False):
        
        super().__init__()
        
        self.net = []
        self.net.append(OscillatorLayer(in_features, hidden_features,
                                        is_first = True,
                                        omega = omega, omega_n = omega_n, xi = xi, phi = phi,
                                        learn_params = learn_params))
        
        for i in range(hidden_layers):
            self.net.append(OscillatorLayer(hidden_features, hidden_features,
                                            is_first = False,
                                            omega = omega, omega_n = omega_n, xi = xi, phi = phi,
                                            learn_params = learn_params))
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_n, np.sqrt(6 / hidden_features) / omega_n)
            
            self.net.append(final_linear)
        else:
            self.net.append(OscillatorLayer(hidden_features, out_features,
                                            is_first = False,
                                            omega = omega, omega_n = omega_n, xi = xi, phi = phi,
                                            learn_params = learn_params))
        
        self.net = nn.Sequential(*self.net)
    
        self.params_history = {f'layer_{idx}' : {'omega': [], 'omega_n': [], 'xi': [], 'phi': []} for idx, _ in enumerate(self.net[:-1])}
    
    def forward(self, coords):
    
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        # if self.training:
        #     segments = max(1, len(self.net) // 2) # Split into two segments to trade approx. 20% of inference time to gain approx. 60%-80% of memory during training.
        #     output = torch.utils.checkpoint.checkpoint_sequential(self.net, segments, coords)
        # else:
        #     output = self.net(coords)
        output = self.net(coords)
    
        return output, coords
    
    def update_params_history(self):
        for idx, layer in enumerate(self.net[:-1]):
            w, w_n, xi, phi = layer.get_params()
            
            self.params_history[f'layer_{idx}']['omega'].append(w.item())
            self.params_history[f'layer_{idx}']['omega_n'].append(w_n.item())
            self.params_history[f'layer_{idx}']['xi'].append(xi.item())
            self.params_history[f'layer_{idx}']['phi'].append(phi.item())