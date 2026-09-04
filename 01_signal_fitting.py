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
from utils.utils import compute_psnr, set_seed, seeds
from utils.common import count_parameters, model_memory_mb, make_output_dir, spectral_fidelity
from models.models_bank import get_model, model_names
from loaders.signal_fitting_loader import SignalFitting, AudioFitting
from torch.utils.data import DataLoader


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_output_dir(cfg) -> str:
    """Create and return an output directory named after the data source."""
    if cfg.data.type == 'audio':
        stem = os.path.splitext(os.path.basename(cfg.data.audio_path))[0]
    elif cfg.data.signal == 'chirp_signal':
        stem = f"chirp_signal_{cfg.data.chirp_freq_high}hz_{cfg.data.num_samples}n"
    elif cfg.data.signal == 'square_wave':
        stem = f"square_wave_{cfg.data.square_wave_freq}hz_{cfg.data.num_samples}n"
    else:
        stem = f"{cfg.data.signal}_{cfg.data.num_samples}n"

    return make_output_dir(stem, "signal_fitting")


# ------------------------------------------------------------------ #
#  Core signal fitting routine                                        #
# ------------------------------------------------------------------ #

def signal_fitting(signal, cfg, out_dir: str):
    model_dir = os.path.join(out_dir, cfg.net.model_name)
    os.makedirs(model_dir, exist_ok=True)

    dataloader = DataLoader(signal, batch_size=1, pin_memory=True, num_workers=0)

    inr, optim, scheduler = get_model(cfg, signal)
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

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    # ---- Measure allocated memory before training ---- #
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ---- Training ---- #

    peak_psnr = torch.tensor(0.0)
    best_psnr_state = None

    train_start = time.perf_counter()

    for step in tqdm(range(cfg.train.steps), desc=f"Fitting {cfg.net.model_name}..."):

        model_output, coords = inr(model_input)

        if cfg.net.model_name != 'bacon':
            loss = ((model_output - ground_truth) ** 2).mean()
        else:
            loss = sum(((out - ground_truth) ** 2).mean() for out in model_output)

        if cfg.log:
            with torch.no_grad():
                out_for_metrics = model_output[-1] if cfg.net.model_name == 'bacon' else model_output

                psnr = compute_psnr(out_for_metrics, ground_truth)

                if psnr > peak_psnr and not torch.isinf(psnr):
                    peak_psnr = psnr
                    best_psnr_state = {k: v.cpu().clone() for k, v in inr.state_dict().items()}

                metric_history['loss'].append(loss.item())
                metric_history['psnr'].append(psnr.item())

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
                scheduler.step(metrics=loss)
            except TypeError:
                scheduler.step()

    torch.cuda.synchronize()
    train_end = time.perf_counter()
    training_time = train_end - train_start
    allocated_training_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ---- Final outputs ---- #

    if cfg.net.model_name == 'bacon' and not cfg.log:
        model_output = model_output[-1]

    final_output = model_output[-1] if cfg.net.model_name == 'bacon' else model_output

    final_psnr = compute_psnr(final_output, ground_truth)
    final_sf = spectral_fidelity(
        final_output.reshape(1, -1),
        ground_truth.reshape(1, -1),
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

    # ---- Best-PSNR model output ---- #

    if best_psnr_state is not None:
        inr.load_state_dict({k: v.cuda() for k, v in best_psnr_state.items()})
        torch.save(best_psnr_state, os.path.join(model_dir, "best_psnr_model.pt"))
    else:
        torch.save(final_state, os.path.join(model_dir, "best_psnr_model.pt"))

    inr.eval()
    with torch.no_grad():
        best_psnr_output, _ = inr(model_input)
    best_psnr_output = best_psnr_output[-1] if cfg.net.model_name == 'bacon' else best_psnr_output

    # ---- Save fitted signals ---- #

    torch.save({
        'coords': signal.coords.cpu(),
        'ground_truth': ground_truth.cpu(),
        'final_output': final_output.cpu(),
        'best_psnr_output': best_psnr_output.cpu(),
    }, os.path.join(model_dir, "fitted_signals.pt"))

    # ================================================================ #
    #  Visualisations                                                   #
    # ================================================================ #

    # ---- Frequency Analysis ---- #

    fft_gt = torch.fft.rfft(ground_truth.reshape(signal.num_samples))
    mag_gt = torch.abs(fft_gt).view(-1).cpu().detach().numpy()
    freq_gt = torch.fft.rfftfreq(signal.num_samples, d=1.0 / signal.fs).view(-1).cpu().detach().numpy()

    fft_fitted = torch.fft.rfft(final_output.reshape(signal.num_samples))
    mag_fitted = torch.abs(fft_fitted).view(-1).cpu().detach().numpy()
    freq_fitted = torch.fft.rfftfreq(signal.num_samples, d=1.0 / signal.fs).view(-1).cpu().detach().numpy()

    norm_gt_t = ((torch.from_numpy(mag_gt) / mag_gt.max()) - 0.5) * 2.0
    norm_fitted_t = ((torch.from_numpy(mag_fitted) / mag_gt.max()) - 0.5) * 2.0
    spectrum_psnr = compute_psnr(norm_fitted_t, norm_gt_t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(freq_gt, mag_gt, color='black', linewidth=1)
    axes[0].set_title("Ground-truth Spectrum")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq_fitted, mag_fitted, color='red', linewidth=1)
    axes[1].set_title(f"Fitted Spectrum (s-PSNR: {spectrum_psnr:.3f} dB)")
    axes[1].grid(True, alpha=0.3)

    y_max = max(mag_gt[1:].max(), mag_fitted[1:].max())
    for ax in axes:
        ax.set_ylim(0, y_max + y_max * 0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "spectrum_fitting.png"),
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

    # ---- Signal output ---- #

    t = signal.coords.view(-1).cpu().numpy()
    y_gt = ground_truth.view(-1).cpu().detach().numpy()
    y_fitted = final_output.view(-1).cpu().detach().numpy()

    plt.figure(figsize=(20, 6))
    plt.title(f"Signal Fitting {cfg.net.model_name} (PSNR: {final_psnr:.2f})")
    plt.plot(t, y_gt, label='Ground Truth Signal', color='gray', alpha=0.6)
    plt.plot(t, y_fitted, label='Fitted Signal', color='red', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "signal_fitting.png"),
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

    parser = argparse.ArgumentParser(description="Fit a 1D signal (audio, chirp or square wave) with an INR.")
    parser.add_argument("--config", default="configs/config_signal_fitting.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotDict(**conf)

    def chirp_signal(coords, freq_low=cfg.data.chirp_freq_low, freq_high=cfg.data.chirp_freq_high):
        t = (coords + 1) / 2
        phase = 2 * numpy.pi * (freq_low * t + ((freq_high - freq_low) / 2) * t ** 2)
        return torch.sin(phase)

    def square_wave(coords, freq=cfg.data.square_wave_freq):
        return torch.sign(torch.sin(freq * numpy.pi * coords))

    def target_signal(t):
        return torch.cos(2 * numpy.pi * 100 * t) + torch.cos(2 * numpy.pi * 500 * t) + torch.cos(2 * numpy.pi * 1000 * t) + torch.cos(2 * numpy.pi * 2500 * t)

    if cfg.data.type == 'signal':
        if cfg.data.signal == 'chirp_signal':
            fcn = chirp_signal
        elif cfg.data.signal == 'square_wave':
            fcn = square_wave
        elif cfg.data.signal == 'target_signal':
            fcn = target_signal

        signal = SignalFitting(function=fcn, num_samples=cfg.data.num_samples)

    elif cfg.data.type == 'audio':
        signal = AudioFitting(file_path=cfg.data.audio_path)

    out_dir = _make_output_dir(cfg)

    # Save config.
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(conf, f, default_flow_style=False)

    logs = {}

    for model_name in model_names:

        print(f'Running {model_name}.')
        cfg.net.model_name = model_name

        logs[model_name] = {
            'finals_psnr': [], 'peaks_psnr': [],
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

            metrics = signal_fitting(signal, cfg, out_dir)

            logs[model_name]['finals_psnr'].append(metrics['final_psnr'])
            logs[model_name]['peaks_psnr'].append(metrics['peak_psnr'])
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
        for key_base in ['finals_psnr', 'peaks_psnr',
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
            f"sf={logs[model_name]['spectral_fidelities_mean']:.4f}±{logs[model_name]['spectral_fidelities_std']:.4f}, "
            f"params={logs[model_name]['n_parameters']}, "
            f"train_time={logs[model_name]['training_times_mean']:.2f}s, "
            f"infer_time={logs[model_name]['inference_times_mean']:.4f}s"
        )

        # Save logs incrementally.
        with open(os.path.join(out_dir, "logs.json"), "w") as f:
            json.dump(logs, f, indent=4)