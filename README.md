# Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations

**Spotlight Oral at ECCV 2026**

[Alex Costanzino](https://alex-costanzino.github.io/)<sup>1</sup>,
Pierluigi Zama Ramirez<sup>2</sup>,
Giuseppe Lisanti<sup>1</sup>,
Luigi Di Stefano<sup>1</sup>

<sup>1</sup>CVLab, University of Bologna &nbsp;&nbsp; <sup>2</sup>Ca' Foscari, University of Venice

[Project page](https://alex-costanzino.github.io/fdho/) · [arXiv](https://arxiv.org/abs/2606.23129)

Official implementation of the forced damped harmonic oscillator (FDHO) activation for implicit
neural representations. The activation is the steady-state amplitude response of a sine-forced damped
oscillator,

```
activation(z) = sin(omega * z + phi) / sqrt((omega_n^2 - omega^2)^2 + (2 * xi * omega_n * omega)^2)
```

where the forcing frequency `omega`, natural frequency `omega_n`, damping ratio `xi` and phase `phi`
are learned per layer alongside the linear weights. The implementation lives in
[`models/oscillator.py`](models/oscillator.py).

The repository reproduces the comparison against nine INR baselines across seven tasks: 1D signal
fitting, image fitting, denoising, inpainting, super-resolution, SDF fitting, CT reconstruction and
Poisson gradient-domain reconstruction.

## Installation

Requires **Python 3.9+ and a CUDA GPU** — the task scripts move models and data to the GPU
unconditionally.

```bash
conda create -n fdho python=3.10 -y
conda activate fdho

# Install PyTorch matched to your CUDA version first: https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install -r requirements.txt
```

## Data

Images (`data/images/`), audio (`data/audio/`) and the CT phantom (`data/scan/`) ship with the
repository. The meshes for the SDF experiments do not — they are downloaded from the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) and converted to the
oriented point clouds the loader expects:

```bash
bash scripts/download_sdf_data.sh              # armadillo, dragon, lucy, thai_statue
python scripts/prepare_sdf_data.py             # .ply -> six-column .xyz (x y z nx ny nz)
```

Lucy and the Thai statue have ~14M and ~5M vertices, which produce large `.xyz` files that are slow to
load. Pass `--max-points 2000000` to `prepare_sdf_data.py` to subsample them.

Please cite the Stanford 3D Scanning Repository if you use these meshes.

## Running

Every task is a standalone script that takes a YAML config:

```bash
python 01_signal_fitting.py            --config configs/config_signal_fitting.yaml
python 02a_image_fitting.py            --config configs/config_image_fitting.yaml
python 02b_image_denoising.py          --config configs/config_image_fitting.yaml
python 02c_image_inpainting.py         --config configs/config_image_fitting.yaml
python 02d_image_super_resolution.py   --config configs/config_image_fitting.yaml
python 03a_sdf_fitting.py              --config configs/config_sdf_fitting.yaml
python 03b_sdf_meshing.py              --config configs/config_sdf_fitting.yaml
python 04_ct_scanning.py               --config configs/config_ct_scanning.yaml
python 05a_poisson_solver.py           --config configs/config_poisson_solver.yaml
```

`--config` defaults to the path shown, so the scripts also run with no arguments.
`03b_sdf_meshing.py` extracts and evaluates a mesh from the checkpoint written by `03a_sdf_fitting.py`,
so run `03a` first.

Results are written to `<input_stem>_<task>/<model_name>/` next to the repository root: the resolved
config, per-model reconstructions, training curves, and a `logs.json` with PSNR, SSIM, spectral
fidelity, parameter count, memory and timings averaged over the seeds in
[`utils/utils.py`](utils/utils.py) (`seeds = [7, 66, 69]`).

### Reproducing the baseline comparison

By default the scripts run the proposed method alone, so a plain invocation is quick. To sweep every
model in the paper's tables, edit [`models/models_bank.py`](models/models_bank.py):

```python
model_names = ALL_MODELS   # oscillator, siren, gaussian, wire, bacon, finer, mfn, fourier_features, fr
```

Per-model hyperparameters live under `net:` in each config file.

## Repository layout

```
0*.py                  one script per task
configs/               one YAML per task, with a hyperparameter block per model
models/                the proposed oscillator plus the baseline implementations
models/models_bank.py  builds a model, its optimizer and its scheduler from a config
loaders/               dataset classes for signals, images, CT scans and point clouds
utils/                 shared metrics, losses and helpers
scripts/               SDF data download and preparation
```

## Baselines

The baseline implementations are re-implementations of published methods, each attributed in its file
header:

| Model | Paper |
| --- | --- |
| `siren` | Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions*, NeurIPS 2020 |
| `wire` | Saragadam et al., *WIRE: Wavelet Implicit Neural Representations*, CVPR 2023 |
| `bacon` | Lindell et al., *BACON: Band-limited Coordinate Networks*, CVPR 2022 |
| `finer` | Liu et al., *FINER: Flexible Spectral-bias Tuning*, CVPR 2024 |
| `mfn` | Fathony et al., *Multiplicative Filter Networks*, ICLR 2021 |
| `fourier_features` | Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions*, NeurIPS 2020 |
| `gaussian` | Ramasinghe and Lucey, *Beyond Periodicity*, ECCV 2022 |
| `fr` | Shi et al., *Improved INR with Fourier Reparameterized Training*, CVPR 2024 |

The SDF objective in [`utils/losses.py`](utils/losses.py) is adapted from the official SIREN
implementation (MIT License).

## Citation

```bibtex
@inproceedings{costanzino2026fdho,
  author    = {Costanzino, Alex and Zama Ramirez, Pierluigi and Lisanti, Giuseppe and Di Stefano, Luigi},
  title     = {Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## License

Released under the [MIT License](LICENSE). The datasets retain their original licenses.
