import argparse
import json
import os
import time

import torch
import numpy
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from pytorch_msssim import ssim as compute_ssim

from loaders.utils import DotDict
from utils.utils import compute_psnr, set_seed, seeds
from utils.common import count_parameters, model_memory_mb, make_output_dir, path_stem, save_image_flat, save_image_bchw, spectral_fidelity
from models.models_bank import get_model, model_names
from loaders.ct_scanning_loader import CTScanning
from torch.utils.data import DataLoader


# ------------------------------------------------------------------ #
#  Core CT scanning routine                                           #
# ------------------------------------------------------------------ #

def ct_scanning(scan, cfg, out_dir: str):
    model_dir = os.path.join(out_dir, cfg.net.model_name)
    os.makedirs(model_dir, exist_ok=True)

    dataloader = DataLoader(scan, batch_size=1, pin_memory=True, num_workers=0)

    inr, optim, scheduler = get_model(cfg, scan)
    inr = inr.cuda()

    # ---- Resource metrics ---- #
    n_parameters = count_parameters(inr)
    model_memory = model_memory_mb(inr)

    if cfg.net.model_name == 'oscillator':
        params_history = {
            f'layer_{idx}': {'omega': [], 'omega_n': [], 'xi': [], 'phi': []}
            for idx, _ in enumerate(inr.net[:-1])
        }

    metric_history = {'loss': [], 'psnr': [], 'ssim': []}

    model_input, sinogram_gt, ground_truth_img = next(iter(dataloader))
    model_input = model_input.cuda()
    sinogram_gt = sinogram_gt.cuda()
    ground_truth_img = ground_truth_img.cuda()  # (1, C, H, W)

    # ---- Measure allocated memory before training ---- #
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ---- Training ---- #

    peak_psnr = torch.tensor(0.0)
    peak_ssim = torch.tensor(0.0)
    best_ssim_state = None
    obj_loss = None

    train_start = time.perf_counter()

    for step in tqdm(range(cfg.train.steps), desc=f"CT {cfg.net.model_name}..."):

        model_output, _ = inr(model_input)

        if cfg.net.model_name != 'bacon':
            img_pred = model_output.permute(0, 2, 1).view(1, scan.ch, scan.h, scan.w)
            sinogram_pred = scan.radon(img_pred, scan.angles.cuda())
            loss = ((sinogram_pred - sinogram_gt) ** 2).mean()
        else:
            loss = 0
            for output in model_output:
                img_pred = output.permute(0, 2, 1).view(1, scan.ch, scan.h, scan.w)
                sinogram_pred = scan.radon(img_pred, scan.angles.cuda())
                loss += ((sinogram_pred - sinogram_gt) ** 2).mean()

        if cfg.log:
            with torch.no_grad():
                if cfg.net.model_name == 'bacon':
                    model_output = model_output[-1]

                obj_loss = ((img_pred - ground_truth_img) ** 2).mean()
                psnr = compute_psnr(img_pred, ground_truth_img)
                ssim_val = compute_ssim(
                    (img_pred + 1.0) / 2.0,
                    (ground_truth_img + 1.0) / 2.0,
                )

                if psnr > peak_psnr:
                    peak_psnr = psnr
                if ssim_val > peak_ssim:
                    peak_ssim = ssim_val
                    best_ssim_state = {k: v.cpu().clone() for k, v in inr.state_dict().items()}

                metric_history['loss'].append(loss.item())
                metric_history['psnr'].append(psnr.item())
                metric_history['ssim'].append(ssim_val.item())

                if cfg.net.model_name == 'oscillator':
                    for idx, layer in enumerate(inr.net[:-1]):
                        w, w_n, xi, phi = layer.get_params()
                        params_history[f'layer_{idx}']['omega'].append(w.item())
                        params_history[f'layer_{idx}']['omega_n'].append(w_n.item())
                        params_history[f'layer_{idx}']['xi'].append(xi.item())
                        params_history[f'layer_{idx}']['phi'].append(phi.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if scheduler is not None:
            try:
                scheduler.step(metrics=obj_loss)
            except TypeError:
                scheduler.step()

    torch.cuda.synchronize()
    train_end = time.perf_counter()
    training_time = train_end - train_start
    allocated_training_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---- Final outputs ---- #

    if cfg.net.model_name == 'bacon' and not cfg.log:
        model_output = model_output[-1]

    final_output = model_output  # (1, H*W, C)
    # Also get the BCHW reconstruction for metrics.
    img_pred = final_output.permute(0, 2, 1).view(1, scan.ch, scan.h, scan.w)

    final_psnr = compute_psnr(img_pred, ground_truth_img)
    final_ssim = compute_ssim(
        (img_pred + 1.0) / 2.0,
        (ground_truth_img + 1.0) / 2.0,
    )
    final_sf = spectral_fidelity(
        img_pred.squeeze(0),
        ground_truth_img.squeeze(0),
    )

    # ---- Inference time & memory ---- #

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inr.eval()
    with torch.no_grad():
        inf_start = time.perf_counter()
        inf_output, _ = inr(model_input)
        torch.cuda.synchronize()
        inf_end = time.perf_counter()

    inference_time = inf_end - inf_start
    allocated_inference_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---- Save final model checkpoint ---- #

    final_state = {k: v.cpu().clone() for k, v in inr.state_dict().items()}
    torch.save(final_state, os.path.join(model_dir, "final_model.pt"))

    # ---- Best-SSIM model output ---- #

    if best_ssim_state is not None:
        inr.load_state_dict({k: v.cuda() for k, v in best_ssim_state.items()})
        torch.save(best_ssim_state, os.path.join(model_dir, "best_ssim_model.pt"))
    else:
        torch.save(final_state, os.path.join(model_dir, "best_ssim_model.pt"))

    inr.eval()
    with torch.no_grad():
        best_ssim_output, _ = inr(model_input)
    best_ssim_output = best_ssim_output[-1] if cfg.net.model_name == 'bacon' else best_ssim_output

    # ---- Save output images ---- #

    save_image_bchw(ground_truth_img, os.path.join(model_dir, "ground_truth.png"))
    save_image_flat(final_output, scan.h, scan.w, scan.ch,
                     os.path.join(model_dir, "output_final.png"))
    save_image_flat(best_ssim_output, scan.h, scan.w, scan.ch,
                     os.path.join(model_dir, "output_best_ssim.png"))

    # ================================================================ #
    #  Visualisations                                                   #
    # ================================================================ #

    # ---- Frequency Analysis ---- #

    gt_reshaped = ground_truth_img.reshape(scan.h, scan.w, scan.ch)
    fitted_reshaped = final_output.reshape(scan.h, scan.w, scan.ch)

    gt_gray = gt_reshaped.mean(dim=-1) if scan.ch == 3 else gt_reshaped.squeeze(-1)
    fitted_gray = fitted_reshaped.mean(dim=-1) if scan.ch == 3 else fitted_reshaped.squeeze(-1)

    fft_gt = torch.fft.fftshift(torch.fft.fft2(gt_gray))
    fft_fitted = torch.fft.fftshift(torch.fft.fft2(fitted_gray))

    mag_gt = torch.abs(fft_gt)
    mag_fitted = torch.abs(fft_fitted)

    norm_gt = ((mag_gt / mag_gt.max()) - 0.5) * 2.0
    norm_fitted = ((mag_fitted / mag_gt.max()) - 0.5) * 2.0

    spectrum_psnr = compute_psnr(norm_fitted, norm_gt)

    vis_gt = torch.log(mag_gt + 1e-8).cpu().detach().numpy()
    vis_fitted = torch.log(mag_fitted + 1e-8).cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    v_min, v_max = vis_gt.min(), vis_gt.max()

    im1 = axes[0].imshow(vis_gt, cmap='inferno', vmin=v_min, vmax=v_max)
    axes[0].set_title("Ground-truth Spectrum (Log)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(vis_fitted, cmap='inferno', vmin=v_min, vmax=v_max)
    axes[1].set_title(f"Fitted Spectrum (s-PSNR: {spectrum_psnr:.3f} dB)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "spectrum_ct_scanning.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

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

    # ---- Side-by-side output ---- #

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow((ground_truth_img.permute(0, 2, 3, 1).cpu().view(scan.h, scan.w, scan.ch).detach().numpy() + 1.0) / 2.0,
                   cmap='gray')
    axes[0].set_title("Ground-truth Scan")

    axes[1].imshow((final_output.cpu().view(scan.h, scan.w, scan.ch).detach().numpy() + 1.0) / 2.0,
                   cmap='gray')
    axes[1].set_title(f"Fitted Scan (PSNR: {final_psnr:.3f} dB)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "ct_reconstruction.png"),
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
        'final_ssim': final_ssim.item(),
        'peak_ssim': peak_ssim.item(),
        'spectrum_psnr': spectrum_psnr.item(),
        'spectral_fidelity': final_sf.item(),
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

    parser = argparse.ArgumentParser(description="Reconstruct a CT scan from sinogram measurements with an INR.")
    parser.add_argument("--config", default="configs/config_ct_scanning.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotDict(**conf)

    scan = CTScanning(cfg.data.scan_path, cfg.data.size, cfg.data.angles)
    out_dir = make_output_dir(path_stem(cfg.data.scan_path), "ct_scanning")

    # Save config.
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(conf, f, default_flow_style=False)

    logs = {}

    for model_name in model_names:

        print(f'Running CT Scanning with {model_name}...')
        cfg.net.model_name = model_name

        logs[model_name] = {
            'finals_psnr': [], 'peaks_psnr': [],
            'finals_ssim': [], 'peaks_ssim': [],
            'spectrum_psnrs': [],
            'spectral_fidelities': [],
            'inference_times': [], 'training_times': [],
            'n_parameters': None,
            'model_memory_mb': None,
            'allocated_inference_memory_mb': [],
            'allocated_training_memory_mb': [],
        }

        for seed in seeds:
            print(f'  Running seed {seed}.')
            set_seed(seed)

            metrics = ct_scanning(scan, cfg, out_dir)

            logs[model_name]['finals_psnr'].append(metrics['final_psnr'])
            logs[model_name]['peaks_psnr'].append(metrics['peak_psnr'])
            logs[model_name]['finals_ssim'].append(metrics['final_ssim'])
            logs[model_name]['peaks_ssim'].append(metrics['peak_ssim'])
            logs[model_name]['spectrum_psnrs'].append(metrics['spectrum_psnr'])
            logs[model_name]['spectral_fidelities'].append(metrics['spectral_fidelity'])
            logs[model_name]['inference_times'].append(metrics['inference_time'])
            logs[model_name]['training_times'].append(metrics['training_time'])
            logs[model_name]['allocated_inference_memory_mb'].append(metrics['allocated_inference_memory_mb'])
            logs[model_name]['allocated_training_memory_mb'].append(metrics['allocated_training_memory_mb'])

            # These are constant across seeds.
            logs[model_name]['n_parameters'] = metrics['n_parameters']
            logs[model_name]['model_memory_mb'] = metrics['model_memory_mb']

        # Aggregate statistics.
        for key_base in ['finals_psnr', 'peaks_psnr', 'finals_ssim', 'peaks_ssim',
                         'spectrum_psnrs', 'spectral_fidelities',
                         'inference_times', 'training_times',
                         'allocated_inference_memory_mb', 'allocated_training_memory_mb']:
            vals = logs[model_name][key_base]
            logs[model_name][f'{key_base}_mean'] = float(numpy.mean(vals))
            logs[model_name][f'{key_base}_std'] = float(numpy.std(vals))

        print(
            f"  {model_name}: "
            f"final_psnr={logs[model_name]['finals_psnr_mean']:.2f}±{logs[model_name]['finals_psnr_std']:.2f}, "
            f"peak_psnr={logs[model_name]['peaks_psnr_mean']:.2f}±{logs[model_name]['peaks_psnr_std']:.2f}, "
            f"final_ssim={logs[model_name]['finals_ssim_mean']:.4f}±{logs[model_name]['finals_ssim_std']:.4f}, "
            f"sf={logs[model_name]['spectral_fidelities_mean']:.4f}±{logs[model_name]['spectral_fidelities_std']:.4f}, "
            f"params={logs[model_name]['n_parameters']}, "
            f"train_time={logs[model_name]['training_times_mean']:.2f}s, "
            f"infer_time={logs[model_name]['inference_times_mean']:.4f}s"
        )

        # Save logs incrementally.
        with open(os.path.join(out_dir, "logs.json"), "w") as f:
            json.dump(logs, f, indent=4)