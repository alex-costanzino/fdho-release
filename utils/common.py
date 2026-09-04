"""
Helpers shared by the task scripts: output paths, model accounting, image I/O
and the spectral fidelity metric.
"""

import os

import matplotlib.pyplot as plt
import numpy
import torch


# ------------------------------------------------------------------ #
#  Output paths                                                       #
# ------------------------------------------------------------------ #

def path_stem(path: str) -> str:
    """Filename of `path` without directories or extension."""
    return os.path.splitext(os.path.basename(path))[0]


def make_output_dir(stem: str, task: str, create: bool = True) -> str:
    """Return the output directory `<stem>_<task>`, creating it unless `create` is False."""
    out = f"{stem}_{task}"
    if create:
        os.makedirs(out, exist_ok=True)
    return out


# ------------------------------------------------------------------ #
#  Model accounting                                                   #
# ------------------------------------------------------------------ #

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_memory_mb(model: torch.nn.Module) -> float:
    """Total memory occupied by model parameters + buffers (MB)."""
    mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    mem += sum(b.nelement() * b.element_size() for b in model.buffers())
    return mem / (1024 ** 2)


# ------------------------------------------------------------------ #
#  Image I/O                                                          #
# ------------------------------------------------------------------ #

def save_image_flat(tensor, h, w, ch, path):
    """Save a [-1, 1] flat (H*W, C) image tensor as a .png file."""
    img = (tensor.cpu().view(h, w, ch).detach().numpy() + 1.0) / 2.0
    img = numpy.clip(img, 0.0, 1.0)
    if ch == 1:
        plt.imsave(path, img.squeeze(-1), cmap='gray')
    else:
        plt.imsave(path, img)


def save_image_bchw(tensor, path):
    """Save a [-1, 1] BCHW tensor as a .png file."""
    img = (tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() + 1.0) / 2.0
    img = numpy.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        plt.imsave(path, img.squeeze(-1), cmap='gray')
    else:
        plt.imsave(path, img)


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #

def spectral_fidelity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Tuning-free spectral fidelity metric between two signals/images/volumes.
    Expects channel-first tensors of shape (C, *spatial_dims).
    """
    x = x.float()
    y = y.float()

    if x.shape[0] <= 4:  # channel-first -> move channels last
        x = x.permute(*range(1, x.dim()), 0)
        y = y.permute(*range(1, y.dim()), 0)

    spatial_axes = tuple(range(x.ndim - 1))

    X = torch.fft.fftn(x, dim=spatial_axes)
    Y = torch.fft.fftn(y, dim=spatial_axes)

    amp_X, amp_Y = torch.abs(X), torch.abs(Y)

    amp_err = torch.linalg.norm(amp_X - amp_Y, dim=spatial_axes)
    amp_err = amp_err / torch.linalg.norm(amp_X, dim=spatial_axes)

    phase_diff = torch.exp(1j * torch.angle(X)) - torch.exp(1j * torch.angle(Y))
    phase_err = torch.linalg.norm(amp_X * torch.abs(phase_diff), dim=spatial_axes)
    phase_err = phase_err / torch.linalg.norm(X, dim=spatial_axes)

    sf = torch.sqrt(amp_err ** 2 + phase_err ** 2) / (2 ** 0.5)
    return sf.mean()
