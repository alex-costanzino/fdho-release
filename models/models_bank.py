import torch

from models.fr import FR
from models.fourier_features import FourierFeatures
from models.mfn import MFN
from models.finer import Finer
from models.bacon import BACON
from models.wire import WIRE
from models.gaussian import Gaussian
from models.siren import Siren
from models.oscillator import Oscillator

# Every model the bank can build. Set `model_names = ALL_MODELS` (or edit the list below) to
# reproduce the full comparison reported in the paper.
ALL_MODELS = ['oscillator', 'siren', 'gaussian', 'wire', 'bacon', 'finer', 'mfn', 'fourier_features', 'fr']

# Models the task scripts sweep over. Defaults to the proposed method alone so a plain run is quick.
model_names = ['oscillator']


def get_model(cfg, signal):

    scheduler = None
    # optim_extra_args = {"betas": (0.9, signal.adam_adjusted_beta2)} if cfg.train.adjust_beta2 else {}

    if cfg.net.model_name == 'oscillator':

        inr = Oscillator(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.oscillator.hidden_features, hidden_layers = cfg.net.oscillator.hidden_layers, outermost_linear = True,
            omega = cfg.net.oscillator.omega, omega_n = cfg.net.oscillator.omega_n, xi = cfg.net.oscillator.xi, phi = cfg.net.oscillator.phi, 
            learn_params = cfg.net.oscillator.learn_params
        )

        physics_params = []
        linear_params = []

        for name, param in inr.named_parameters():
            if 'omega' in name or 'xi' in name or 'phi' in name:
                physics_params.append(param)
            else:
                linear_params.append(param)

        optim = torch.optim.Adam([
            {'params': linear_params, 'lr': cfg.net.oscillator.lr_linear},
            {'params': physics_params, 'lr': cfg.net.oscillator.lr_physics}
        ], eps = 1e-10)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode = 'min', factor = 0.1, patience = 500, threshold = 1e-4, min_lr = 1e-6)

    elif cfg.net.model_name == 'siren':
        inr = Siren(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.siren.hidden_features, hidden_layers = cfg.net.siren.hidden_layers, outermost_linear = True,
            first_omega_0 = cfg.net.siren.omega_0, hidden_omega_0 = cfg.net.siren.omega_0
        )

        optim = torch.optim.Adam(lr = cfg.net.siren.lr, params=inr.parameters())

    elif cfg.net.model_name == 'gaussian':
        inr = Gaussian(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.gaussian.hidden_features, hidden_layers = cfg.net.gaussian.hidden_layers,
            sigma = cfg.net.gaussian.sigma
        )
    
        optim = torch.optim.Adam(lr = cfg.net.gaussian.lr, params=inr.parameters())

    elif cfg.net.model_name == 'wire':
        inr = WIRE(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.wire.hidden_features, hidden_layers = cfg.net.wire.hidden_layers,
            first_omega_0 = cfg.net.wire.omega_0, hidden_omega_0 = cfg.net.wire.omega_0, scale = cfg.net.wire.scale
        )

        optim = torch.optim.Adam(lr = cfg.net.wire.lr, params=inr.parameters())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 0.1 * min(x / cfg.train.steps, 1))

        
    elif cfg.net.model_name == 'bacon':
        inr = BACON(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.bacon.hidden_features, hidden_layers = cfg.net.bacon.hidden_layers, 
            frequency_limit = cfg.net.bacon.frequency_limit, quantize_freq = cfg.net.bacon.quantize_freq
        )
    
        optim = torch.optim.Adam(lr = cfg.net.bacon.lr, params=inr.parameters())

    elif cfg.net.model_name == 'finer':
        inr = Finer(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.finer.hidden_features, hidden_layers = cfg.net.finer.hidden_layers,
            first_omega_0 = cfg.net.finer.first_omega_0, hidden_omega_0 = cfg.net.finer.hidden_omega_0, first_bias_scale = cfg.net.finer.first_bias_scale, scale_req_grad = cfg.net.finer.scale_req_grad
        )

        optim = torch.optim.Adam(lr = cfg.net.finer.lr, params=inr.parameters())

    elif cfg.net.model_name == 'mfn':
        inr = MFN(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.mfn.hidden_features, hidden_layers = cfg.net.mfn.hidden_layers, 
            weight_scale = cfg.net.mfn.weight_scale, alpha = cfg.net.mfn.alpha
        )

        optim = torch.optim.Adam(lr = cfg.net.mfn.lr, params=inr.parameters())

    elif cfg.net.model_name == 'fourier_features':
        inr = FourierFeatures(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.fourier_features.hidden_features, hidden_layers = cfg.net.fourier_features.hidden_layers, 
            scale = cfg.net.fourier_features.scale
        )

        optim = torch.optim.Adam(lr = cfg.net.fourier_features.lr, params=inr.parameters())

    elif cfg.net.model_name == 'fr':
        inr = FR(
            in_features = cfg.net.in_features, out_features = signal.ch, hidden_features = cfg.net.fr.hidden_features, hidden_layers = cfg.net.fr.hidden_layers, 
            high_freq_num = cfg.net.fr.high_freq_num
        )

        optim = torch.optim.Adam(lr = cfg.net.fr.lr, params=inr.parameters())

        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = 3000, gamma = 0.1)

    else:
        raise ValueError(f"Unknown model name '{cfg.net.model_name}'. Available: {ALL_MODELS}.")

    return inr, optim, scheduler