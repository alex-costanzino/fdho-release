"""Grid construction, PSNR and seeding helpers shared by the task scripts."""

import torch
import random
import numpy as np

def get_mgrid(h, w, dim=2):
    """
    sidelen: tuple or list of length `dim`, e.g. (H, W)
    dim: dimensionality
    """
    axes = [torch.linspace(-1, 1, steps = s) for s in (h, w)]
    mgrid = torch.stack(torch.meshgrid(*axes, indexing = "ij"), dim=-1)
    return mgrid.reshape(-1, dim)

def compute_psnr(img_pred, img_gt):
    """
    Computes PSNR for models trained on data in range [-1, 1].
    It normalizes both inputs to [0, 1] before computing MSE.
    """
    # 1. Normalize from [-1, 1] to [0, 1]
    # Formula: (x + 1) / 2
    img_pred = (img_pred + 1.0) / 2.0
    img_gt = (img_gt + 1.0) / 2.0

    # 2. Clamp Prediction to [0, 1]
    # Essential for INRs as they often slightly overshoot (e.g. -1.05 or 1.05) which would become -0.025 or 1.025 after normalization.
    img_pred = torch.clamp(img_pred, 0.0, 1.0)
    img_gt = torch.clamp(img_gt, 0.0, 1.0)
        
    # 3. Compute MSE (Mean Squared Error)
    mse = torch.mean((img_pred - img_gt) ** 2)
        
    return -10.0 * torch.log10(mse)


# Seeds each task script averages its metrics over.
seeds = [98, 68, 41, 15, 39, 54, 82, 27, 11, 51]

def set_seed(seed: int = 42):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
