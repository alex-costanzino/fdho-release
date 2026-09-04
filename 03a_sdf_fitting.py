import argparse
import json
import os
import time

import torch
import numpy
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from loaders.utils import DotDict
from utils.utils import compute_psnr
from utils.common import count_parameters, model_memory_mb, make_output_dir, path_stem
from utils.losses import sdf_loss
from models.models_bank import get_model, model_names
from loaders.sdf_fitting_loader import SDFFitting


# ------------------------------------------------------------------ #
#  Core SDF fitting routine                                           #
# ------------------------------------------------------------------ #

def sdf_fitting(sdf, cfg, out_dir: str):
    model_dir = os.path.join(out_dir, cfg.net.model_name)
    os.makedirs(model_dir, exist_ok=True)

    print_every = 100
    save_every = 10000

    inr, optim, scheduler = get_model(cfg, sdf)
    inr = inr.cuda()

    # ---- Resource metrics ---- #
    n_parameters = count_parameters(inr)
    model_memory = model_memory_mb(inr)

    if cfg.net.model_name == 'oscillator':
        params_history = {
            f'layer_{idx}': {'omega': [], 'omega_n': [], 'xi': [], 'phi': []}
            for idx, _ in enumerate(inr.net[:-1])
        }

    metric_history = {'loss': [], 'psnr': []}

    # ---- Gradient accumulation & mixed precision ---- #
    accum_steps = getattr(cfg.train, 'accum_steps', 1)
    use_amp = getattr(cfg.train, 'use_amp', False)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    if accum_steps > 1 or use_amp:
        mode_parts = []
        if use_amp:
            mode_parts.append("AMP")
        if accum_steps > 1:
            mode_parts.append(f"{accum_steps}× grad accum")
        eff_samples = cfg.data.on_surface_samples * accum_steps
        print(f"  Training with {' + '.join(mode_parts)} "
              f"(effective on-surface samples/step: {eff_samples:,})")

    # ---- Measure allocated memory before training ---- #
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ---- Training ---- #

    peak_psnr = torch.tensor(0.0)
    best_psnr_state = None

    # sdf_loss returns already-weighted terms:
    #   sdf=3e3, inter=3e3, normal_constraint=1e2, grad_constraint=5e1
    # Dividing by their sum (6150) normalises magnitude, preserving
    # relative weighting.  This follows the SIREN formulation.
    weight_sum = 3e3 + 3e3 + 1e2 + 5e1

    train_start = time.perf_counter()

    for step in tqdm(range(cfg.train.steps), desc=f"SDF {cfg.net.model_name}..."):

        optim.zero_grad()

        for micro in range(accum_steps):
            # SDFFitting.__getitem__ ignores idx and randomly samples
            # fresh on/off-surface points each call.
            sample = sdf[0]
            model_input = sample['coords'].unsqueeze(0).cuda().requires_grad_(True)

            ground_truth = {
                'sdf': sample['sdf'].unsqueeze(0).cuda(),
                'normals': sample['normals'].unsqueeze(0).cuda(),
            }

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                model_output, coords = inr(model_input)

            # Loss in fp32 — eikonal gradient computation is
            # precision-sensitive and should not run in fp16.
            if cfg.net.model_name != 'bacon':
                output_dict = {'model_in': coords, 'model_out': model_output}
                loss_dict = sdf_loss(output_dict, ground_truth)
                loss = sum(loss_dict.values()) / weight_sum / accum_steps
            else:
                loss = 0
                for output in model_output:
                    output_dict = {'model_in': model_input, 'model_out': output}
                    loss_dict = sdf_loss(output_dict, ground_truth)
                    loss += sum(loss_dict.values()) / weight_sum / accum_steps

            scaler.scale(loss).backward()

            # Free computation graph between micro-batches, but keep
            # the last one for metrics/logging.
            if micro < accum_steps - 1:
                del model_output, coords, loss, loss_dict, output_dict
                del model_input, ground_truth, sample

        scaler.step(optim)
        scaler.update()

        # loss currently = last micro-batch's loss / accum_steps.
        # Reconstruct the full-step loss for scheduler & logging.
        step_loss = loss.item() * accum_steps

        if scheduler is not None:
            try:
                scheduler.step(metrics=step_loss)
            except TypeError:
                scheduler.step()

        if cfg.log:
            with torch.no_grad():
                out_for_metrics = model_output[-1] if cfg.net.model_name == 'bacon' else model_output

                on_mask = (ground_truth['sdf'] != -1)
                psnr = compute_psnr(out_for_metrics[on_mask], ground_truth['sdf'][on_mask])

                if psnr > peak_psnr:
                    peak_psnr = psnr
                    best_psnr_state = {k: v.cpu().clone() for k, v in inr.state_dict().items()}

                metric_history['loss'].append(step_loss)
                metric_history['psnr'].append(psnr.item())

                if step % print_every == 0:
                    pred_on = out_for_metrics[on_mask]
                    pred_off = out_for_metrics[~on_mask]
                    print(
                        f"Step {step} | Loss: {step_loss:.5f} | PSNR: {psnr.item():.3f} dB"
                    )
                    print(
                        f"  On-surface:  mean={pred_on.mean().item():.4f}, "
                        f"std={pred_on.std().item():.4f}"
                    )
                    print(
                        f"  Off-surface: mean={pred_off.mean().item():.4f}, "
                        f"std={pred_off.std().item():.4f}"
                    )

                # Periodic checkpoint during training.
                if step % save_every == 0 and step > 0:
                    ckpt_path = os.path.join(model_dir, f"checkpoint_step_{step}.pt")
                    torch.save(
                        {k: v.cpu().clone() for k, v in inr.state_dict().items()},
                        ckpt_path,
                    )
                    print(f"  Saved checkpoint to {ckpt_path}.")

                if cfg.net.model_name == 'oscillator':
                    for idx, layer in enumerate(inr.net[:-1]):
                        w, w_n, xi, phi = layer.get_params()
                        params_history[f'layer_{idx}']['omega'].append(w.item())
                        params_history[f'layer_{idx}']['omega_n'].append(w_n.item())
                        params_history[f'layer_{idx}']['xi'].append(xi.item())
                        params_history[f'layer_{idx}']['phi'].append(phi.item())

    torch.cuda.synchronize()
    train_end = time.perf_counter()
    training_time = train_end - train_start
    allocated_training_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---- Final outputs ---- #

    if cfg.net.model_name == 'bacon' and not cfg.log:
        model_output = model_output[-1]

    final_output = model_output[-1] if cfg.net.model_name == 'bacon' else model_output

    on_mask = (ground_truth['sdf'] != -1)
    final_psnr = compute_psnr(final_output[on_mask], ground_truth['sdf'][on_mask])

    # ---- Inference time & memory ---- #

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inr.eval()
    with torch.no_grad():
        test_sample = sdf[0]
        test_input = test_sample['coords'].unsqueeze(0).cuda()

        inf_start = time.perf_counter()
        inf_output, _ = inr(test_input)
        torch.cuda.synchronize()
        inf_end = time.perf_counter()

    inference_time = inf_end - inf_start
    allocated_inference_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---- Save model checkpoints ---- #

    final_state = {k: v.cpu().clone() for k, v in inr.state_dict().items()}
    torch.save(final_state, os.path.join(model_dir, "final_model.pt"))
    print(f"Training finished. Final model saved to {model_dir}/final_model.pt.")

    if best_psnr_state is not None:
        torch.save(best_psnr_state, os.path.join(model_dir, "best_psnr_model.pt"))
    else:
        torch.save(final_state, os.path.join(model_dir, "best_psnr_model.pt"))

    # ================================================================ #
    #  Visualisations                                                   #
    # ================================================================ #

    # ---- Training Dynamic ---- #

    plt.figure(figsize=(20, 6))
    plt.title(f"Training Dynamic {cfg.net.model_name} (PSNR: {final_psnr:.2f})")
    plt.plot(range(cfg.train.steps), metric_history['psnr'], label='PSNR (dB)', color='gray')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_dynamic.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Physics Dashboard ---- #

    if cfg.net.model_name == 'oscillator':

        n_layers = len(inr.net[:-1])
        fig, axes = plt.subplots(n_layers, 3, figsize=(18, 5 * n_layers), constrained_layout=True)
        if n_layers == 1:
            axes = axes[numpy.newaxis, :]

        for idx in range(n_layers):

            # Spectral Gating.
            axes[idx, 0].set_title(f"Spectral Gating at Layer {idx}", fontsize=12)
            axes[idx, 0].set_xlabel("Steps")
            axes[idx, 0].set_ylabel("Frequency (rad/s)")
            axes[idx, 0].plot(params_history[f'layer_{idx}']['omega'],
                              label=r'Forcing Frequency ($\omega$)', linestyle='--',
                              color='tab:blue', alpha=0.7)
            axes[idx, 0].plot(params_history[f'layer_{idx}']['omega_n'],
                              label=r'Natural Frequency ($\omega_n$)', linewidth=2.5,
                              color='tab:orange')
            axes[idx, 0].legend(loc='upper right')
            axes[idx, 0].grid(True, alpha=0.3)

            # Selectivity.
            axes[idx, 1].set_title("Selectivity", fontsize=12)
            axes[idx, 1].set_xlabel("Steps")
            axes[idx, 1].set_ylabel(r"Damping Ratio $\xi$")
            axes[idx, 1].axhline(0.707, color='k', linestyle=':', alpha=0.5, label='Butterworth')
            axes[idx, 1].axhline(0.0, color='r', linestyle='--', alpha=0.3, label='Unstable')
            axes[idx, 1].plot(params_history[f'layer_{idx}']['xi'],
                              color='tab:green', linewidth=2)
            axes[idx, 1].legend(loc='upper right')
            axes[idx, 1].grid(True, alpha=0.3)

            # Phase Response.
            axes[idx, 2].set_title("Phase Response", fontsize=12)
            axes[idx, 2].set_xlabel("Steps")
            axes[idx, 2].set_ylabel(r"Phase (rad)")
            axes[idx, 2].axhline(0, color='gray', linestyle='--', alpha=0.5,
                                 label='Passband ($0$)')
            axes[idx, 2].axhline(-numpy.pi / 2, color='red', linestyle=':', linewidth=2,
                                 label=r'Resonance ($-\pi/2$)')
            axes[idx, 2].axhline(-numpy.pi, color='gray', linestyle='--', alpha=0.5,
                                 label=r'Stopband ($-\pi$)')
            axes[idx, 2].plot(params_history[f'layer_{idx}']['phi'],
                              color='tab:purple', linewidth=2, label=r'Phase $\phi$')
            axes[idx, 2].legend(loc='upper right')
            axes[idx, 2].grid(True, alpha=0.3)

        plt.savefig(os.path.join(model_dir, "physics_dashboard.png"), dpi=300)
        plt.close()

    # ================================================================ #
    #  Return metrics                                                   #
    # ================================================================ #

    return {
        'final_psnr': final_psnr.item(),
        'peak_psnr': peak_psnr.item(),
        'inference_time': inference_time,
        'training_time': training_time,
        'n_parameters': n_parameters,
        'model_memory_mb': model_memory,
        'allocated_inference_memory_mb': allocated_inference_memory,
        'allocated_training_memory_mb': allocated_training_memory,
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fit a signed distance function to an oriented point cloud with an INR.")
    parser.add_argument("--config", default="configs/config_sdf_fitting.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotDict(**conf)

    sdf = SDFFitting(
        path=cfg.data.sdf_path,
        on_surface_samples=cfg.data.on_surface_samples,
    )

    out_dir = make_output_dir(path_stem(cfg.data.sdf_path), "sdf_fitting")

    # Save config.
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(conf, f, default_flow_style=False)

    logs = {}

    for model_name in model_names:

        print(f'Running {model_name}.')
        cfg.net.model_name = model_name

        logs[model_name] = {
            'finals_psnr': [],
            'peaks_psnr': [],
            'inference_times': [], 'training_times': [],
            'n_parameters': None,
            'model_memory_mb': None,
            'allocated_inference_memory_mb': [],
            'allocated_training_memory_mb': [],
        }

        # Single seed — SDF training is heavy.

        metrics = sdf_fitting(sdf, cfg, out_dir)

        logs[model_name]['finals_psnr'].append(metrics['final_psnr'])
        logs[model_name]['peaks_psnr'].append(metrics['peak_psnr'])
        logs[model_name]['inference_times'].append(metrics['inference_time'])
        logs[model_name]['training_times'].append(metrics['training_time'])
        logs[model_name]['allocated_inference_memory_mb'].append(metrics['allocated_inference_memory_mb'])
        logs[model_name]['allocated_training_memory_mb'].append(metrics['allocated_training_memory_mb'])

        logs[model_name]['n_parameters'] = metrics['n_parameters']
        logs[model_name]['model_memory_mb'] = metrics['model_memory_mb']

        # Aggregate statistics (still compute mean/std for consistency,
        # even with a single seed).
        for key_base in ['finals_psnr', 'peaks_psnr',
                         'inference_times', 'training_times',
                         'allocated_inference_memory_mb', 'allocated_training_memory_mb']:
            vals = logs[model_name][key_base]
            logs[model_name][f'{key_base}_mean'] = float(numpy.mean(vals))
            logs[model_name][f'{key_base}_std'] = float(numpy.std(vals))

        print(
            f"  {model_name}: "
            f"final_psnr={logs[model_name]['finals_psnr_mean']:.2f}, "
            f"peak_psnr={logs[model_name]['peaks_psnr_mean']:.2f}, "
            f"params={logs[model_name]['n_parameters']}, "
            f"train_time={logs[model_name]['training_times_mean']:.2f}s, "
            f"infer_time={logs[model_name]['inference_times_mean']:.4f}s"
        )

        # Save logs incrementally.
        with open(os.path.join(out_dir, "logs.json"), "w") as f:
            json.dump(logs, f, indent=4)