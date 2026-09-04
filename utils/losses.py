"""
Differential operators and the SDF fitting objective.

Adapted from the official SIREN implementation:
Sitzmann et al., "Implicit Neural Representations with Periodic Activation
Functions", NeurIPS 2020. https://github.com/vsitzmann/siren (MIT License).
"""

import torch
import torch.nn.functional as F


def gradient(y, x, grad_outputs=None):
    """Gradient of y with respect to x, keeping the graph for higher-order terms."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def sdf_loss(model_output, gt):
    """
    Eikonal-regularized SDF fitting loss.

    Combines an on-surface data term, an off-surface repulsion term, a normal
    alignment term and a unit-gradient (Eikonal) constraint.
    """
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    grad = gradient(pred_sdf, coords)

    # Wherever gt_sdf is not -1, the sample lies on the surface and acts as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None], torch.zeros_like(grad[..., :1]))
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
            'inter': inter_constraint.mean() * 3e3,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1}
