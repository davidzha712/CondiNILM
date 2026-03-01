# CondiNILMformer

**Condition-modulated Transformer for Multi-Device Non-Intrusive Load Monitoring**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5.svg)](https://lightning.ai/)

<p align="center">
  <img src="assets/intro.png" alt="CondiNILMformer Architecture" width="800"/>
</p>

CondiNILMformer is a research framework for Non-Intrusive Load Monitoring (NILM) built on PyTorch Lightning. Given an aggregate household power signal, the model simultaneously disaggregates per-appliance power consumption for up to five devices (fridge, kettle, dishwasher, washing machine, microwave) in a single forward pass. The framework includes 15 baseline architectures for systematic comparison, Optuna-based hyperparameter optimization, and comprehensive evaluation pipelines across three public NILM datasets.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Monitoring Training](#monitoring-training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)
- [Baseline Models](#baseline-models)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **FiLM Conditioning**: Feature-wise Linear Modulation (FiLM) layers that inject frequency-domain and electrical-domain context into the transformer encoder, enabling the model to adapt its representation to different appliance operating characteristics.

- **Multi-Device Joint Training with PCGrad**: Simultaneous disaggregation of multiple appliances using gradient surgery (PCGrad) to resolve gradient conflicts between device-specific loss functions. Includes soft gradient balancing with configurable maximum ratio and randomized projection order.

- **AdaptiveDeviceLoss**: Per-device loss function that automatically derives weighting parameters from appliance electrical characteristics (duty cycle, ON/OFF power ratios, peak power). Supports asymmetric ON/OFF weighting with focal-style gate classification loss.

- **Seq2SubSeq Centre Supervision**: Only the centre region of the output window is supervised (configurable `output_ratio`), reducing boundary artifacts from sliding-window segmentation and improving temporal consistency.

- **Soft Gating for Energy/State Separation**: Learnable per-device gate with sigmoid sharpness, bias, and floor parameters. Uses soft gating during training for gradient flow and hard thresholding at inference for clean ON/OFF separation.

- **Hyperparameter Optimization via Optuna**: Integrated Optuna search over loss parameters, learning rates, architecture hyperparameters, and sampling configurations with configurable search spaces.

- **15 Baseline Architectures**: Comprehensive set of NILM-specific and general time-series models for fair benchmarking, all sharing the same training pipeline, loss functions, and evaluation protocol.

- **Anti-Collapse Stabilization**: Automatic detection and recovery from training collapse in multi-task learning, preventing dominant devices from suppressing sparse ones.

- **Dual-Path Inference**: Transformer path for complex multi-state devices (fridge, dishwasher, washing machine) and a lightweight CNN bypass path for sparse high-power devices (kettle, microwave) with learnable blend weights.

---

## Installation

### Prerequisites

- Python 3.10 or later (tested with 3.12 on Windows 11)
- CUDA-capable GPU recommended (see [Hardware Requirements](#hardware-requirements))

### Method 1: Conda (Recommended)

Conda environment files are provided with all dependencies pre-configured:

```bash
# Windows
conda env create -f environment_win.yaml
conda activate condinilm

# macOS
conda env create -f environment_mac.yaml
conda activate condinilm
```

### Method 2: pip

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install CondiNILM as an editable package:

```bash
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.nilmformer.model import NILMFormer; print('NILMFormer import OK')"
```

---

## Dataset Preparation

The framework supports three public NILM datasets. All data files should be placed under the `data/` directory (gitignored; users must download the data themselves).

### UK-DALE

**Source**: [UK Domestic Appliance-Level Electricity Dataset](https://jack-kelly.com/data/)

- 5 houses, 6-second sampling interval
- Appliances: Fridge, Kettle, Dishwasher, Washing Machine, Microwave
- Default split: House 1 (train/val), House 2 (test)

**Expected directory structure**:

```
data/
  UKDALE/
    house_1/
      channel_1.dat
      channel_2.dat
      ...
      labels.dat
    house_2/
      ...
```

### REFIT

**Source**: [REFIT Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements)

- 20 houses, 8-second sampling interval (6 high-quality houses used: 2, 5, 6, 7, 9, 15)
- Appliances: Fridge, Kettle, Dishwasher, Washing Machine
- Default split: Houses 2, 5, 6, 9 (train/val), Houses 7, 15 (test)

**Expected directory structure**:

```
data/
  REFIT/
    RAW_DATA_CLEAN/
      CLEAN_House1.csv
      CLEAN_House2.csv
      ...
```

### REDD

**Source**: [Reference Energy Disaggregation Dataset](http://redd.csail.mit.edu/)

- 6 houses, 1-second sampling interval
- Appliances: Fridge, Dishwasher, Microwave (Washing Machine available in REDD_STRESS config)
- Default split: Houses 1, 2 (train/val), House 3 (test)

**Expected directory structure**:

```
data/
  REDD/
    house_1/
      channel_1.dat
      channel_2.dat
      ...
      labels.dat
    house_2/
      ...
```

---

## Quick Start

### Smoke Test

Verify your setup with a quick 1-epoch test using reduced data:

```bash
python scripts/run_single_device_quick.py
```

This runs a fast sanity check on UKDALE with 10% of the data and minimal epochs.

### Single-Device Training

Train NILMFormer on a single appliance:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance Kettle \
    --name_model NILMFormer \
    --epochs 25
```

### Multi-Device Joint Training

Use `multi` to train on all target appliances simultaneously:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance multi \
    --name_model NILMFormer \
    --epochs 25
```

Or specify a subset of devices as a comma-separated list:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance Kettle,Fridge,Microwave \
    --name_model NILMFormer \
    --epochs 25
```

### Hyperparameter Optimization

Run Optuna-based hyperparameter search:

```bash
python scripts/run_optuna_search.py \
    --dataset UKDALE \
    --appliance multi \
    --name_model NILMFormer \
    --n_trials 50
```

### Baseline Comparison

Run structured baseline comparison experiments for paper tables:

```bash
# Table 1: UKDALE single-device per-model best
python scripts/run_baseline_comparison.py --phase 1

# Table 2: UKDALE multi-device per-model best
python scripts/run_baseline_comparison.py --phase 2

# Table 3: UKDALE multi-device controlled (same loss)
python scripts/run_baseline_comparison.py --phase 3

# Table 4: Ablation study
python scripts/run_baseline_comparison.py --phase 4

# Table 5: REFIT cross-dataset
python scripts/run_baseline_comparison.py --phase 5

# Run all phases sequentially
python scripts/run_baseline_comparison.py --phase all
```

### View All CLI Arguments

```bash
python scripts/run_experiment.py -h
```

---

## Monitoring Training

### TensorBoard

Training metrics are logged to `log/tensorboard/`. Launch the viewer with:

```bash
tensorboard --logdir=log/tensorboard/
# Open http://localhost:6006 in your browser
```

Key metrics logged per device:
- `val/F1_{device}` -- validation F1 score
- `val/MAE_{device}` -- validation mean absolute error
- `val/NDE_{device}` -- normalised disaggregation error
- `train/loss` -- total training loss

### Interactive Validation Viewer

For detailed per-epoch validation analysis with interactive plots:

```bash
streamlit run scripts/streamlit_val_viewer.py
```

### Resuming Training

To resume training from an existing checkpoint:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --appliance multi \
    --name_model NILMFormer \
    --epochs 50 \
    --resume
```

The `--resume` flag loads the latest checkpoint for the same experiment configuration (dataset, appliance, model) and continues training.

---

## Project Structure

```
CondiNILM/
├── src/
│   ├── nilmformer/                     # NILMFormer model implementation
│   │   ├── model.py                    # NILMFormer, SparseDeviceCNN, SimpleDeviceHead
│   │   ├── config.py                   # NILMFormerConfig dataclass
│   │   └── layers/
│   │       ├── embedding.py            # DilatedBlock, ResUnit convolution embedding
│   │       └── transformer.py          # EncoderLayer with multi-head attention + FiLM
│   ├── baselines/
│   │   ├── nilm/                       # NILM-specific baseline architectures
│   │   │   ├── bilstm.py              # BiLSTM
│   │   │   ├── bigru.py               # BiGRU
│   │   │   ├── cnn1d.py               # CNN1D
│   │   │   ├── unetnilm.py            # UNet-NILM
│   │   │   ├── fcn.py                 # FCN
│   │   │   ├── dresnets.py            # DResNet, DAResNet
│   │   │   ├── bert4nilm.py           # BERT4NILM
│   │   │   ├── diffnilm.py            # DiffNILM
│   │   │   ├── stnilm.py             # ST-NILM (with Mixture of Experts)
│   │   │   ├── tsilnet.py            # TSILNet (TCN + LSTM)
│   │   │   └── energformer.py         # Energformer
│   │   └── tser/                       # Time-series regression baselines
│   │       ├── convnet.py             # ConvNet
│   │       ├── resnet.py             # ResNet
│   │       └── inceptiontime.py       # InceptionTime
│   └── helpers/
│       ├── experiment.py               # Model factory and utility re-exports
│       ├── trainer.py                  # PyTorch Lightning trainer, AdaptiveDeviceLoss
│       ├── training.py                 # Training loop orchestration
│       ├── evaluation.py               # Evaluation and metric computation
│       ├── callbacks.py                # Lightning callbacks (validation, logging)
│       ├── loss.py                     # Loss function definitions
│       ├── inference.py                # Sliding window inference and stitching
│       ├── postprocess.py              # Post-processing (threshold, gate suppression)
│       ├── preprocessing.py            # Data loading, windowing, train/valid splitting
│       ├── dataset.py                  # PyTorch Dataset classes, NILMscaler
│       ├── dataset_params.py           # Per-dataset device parameter loading
│       ├── device_config.py            # Device type classification and gate configs
│       ├── metrics.py                  # F1, precision, recall, MAE, SAE metrics
│       ├── loss_tuning.py              # Loss parameter tuning utilities
│       ├── gradient_conflict.py        # Gradient conflict detection (PCGrad)
│       └── utils.py                    # General utilities
├── configs/
│   ├── expes.yaml                      # Experiment defaults (training, loss, gate, scheduler)
│   ├── dataset_params.yaml             # Per-dataset device parameters and postprocessing
│   ├── datasets.yaml                   # House-level train/valid/test splits
│   ├── models.yaml                     # Model architecture hyperparameters
│   ├── hpo_search_spaces.yaml          # Optuna search space definitions
│   └── hpo_sparse_devices.yaml         # HPO config for sparse devices
├── scripts/
│   ├── archive/                        # Development and paper-generation scripts
│   │   ├── tensorboard/               # TensorBoard log extraction utilities
│   │   ├── paper/                     # Paper figure and table generation
│   │   └── profiling/                 # VRAM estimation and data visualization
│   ├── run_experiment.py               # Main training entry point
│   ├── run_optuna_search.py            # Hyperparameter optimization
│   ├── run_baseline_comparison.py      # Baseline model comparison (5 phases)
│   ├── run_ukdale_all.py               # Batch runner for UK-DALE experiments
│   ├── run_redd_all.py                 # Batch runner for REDD experiments
│   ├── run_ukdale_hires.py             # High-resolution (6s) UK-DALE experiments
│   ├── run_single_device_quick.py      # Quick smoke test (10% data)
│   ├── collect_results.py              # Result aggregation and table generation
│   ├── check_hpo_progress.py           # Monitor Optuna HPO progress
│   ├── streamlit_val_viewer.py         # Interactive validation viewer
│   ├── generate_ukdale_labels.py       # UK-DALE label generation from metadata
│   ├── analyze_appliance_stats.py      # Appliance electrical statistics
│   └── analyze_redd_devices.py         # REDD device availability analysis
├── assets/                             # Architecture diagrams and figures
├── data/                               # Dataset root (gitignored, user-provided)
├── pyproject.toml                      # PEP 621 project metadata and packaging
├── requirements.txt                    # Pip dependency list
├── environment_win.yaml                # Conda environment (Windows)
├── environment_mac.yaml                # Conda environment (macOS)
└── LICENSE                             # Apache License 2.0
```

---

## Configuration

CondiNILM uses a layered configuration system where later sources override earlier ones:

```
expes.yaml (base defaults)
  -> models.yaml (architecture-specific overrides)
    -> datasets.yaml (house-level splits)
      -> dataset_params.yaml (per-dataset device parameters -- overrides expes.yaml)
        -> HPO overrides (highest priority, filtered by dataset)
          -> Runtime derivation (trainer.py derives loss alphas from data statistics)
```

### expes.yaml

Core experiment settings including:

| Category | Parameters |
|----------|-----------|
| **Training** | `batch_size`, `epochs`, `scheduler_type`, `n_warmup_epochs`, `p_es` (early stopping patience) |
| **Data** | `sampling_rate`, `window_size`, `train_num_crops`, `train_crop_event_bias` |
| **Loss** | `loss_type`, `output_ratio`, `loss_lambda_on_recall`, `loss_alpha_on`, `loss_alpha_off` |
| **Gate** | `gate_soft_scale`, `gate_floor`, `gate_cls_weight`, `loss_gate_focal_gamma` |
| **PCGrad** | `use_gradient_conflict_resolution`, `pcgrad_every_n_steps`, `gradient_conflict_balance_method` |

### dataset_params.yaml

Per-dataset, per-device parameters including:

- ON/OFF power thresholds and minimum event durations
- Device type classification (`sparse_high_power`, `frequent_switching`, `long_cycle`, etc.)
- Loss function overrides (per-device alpha, recall, energy weights)
- Postprocessing settings (power thresholds, minimum ON steps)

Entries in this file take precedence over `expes.yaml` for dataset-specific tuning.

### datasets.yaml

House-level data splits for each dataset and appliance:

- `ind_house_train_val`: Houses used for training (80%) and validation (20%, temporal split)
- `ind_house_test`: Houses reserved for final evaluation (never seen during training)
- `overlap`: Window overlap ratio for sliding-window segmentation

Includes standard and extended configurations: `UKDALE`, `UKDALE_EXT`, `UKDALE_BI`, `REFIT`, `REDD`, and `REDD_STRESS`.

### models.yaml

Architecture-specific hyperparameters for NILMFormer and all 15 baselines, including layer counts, hidden dimensions, dropout rates, and learning rates.

### hpo_search_spaces.yaml

Optuna search space definitions with parameter types (`categorical`, `int`, `loguniform`, `uniform`), ranges, and step sizes for automated hyperparameter optimization.

---

## Hardware Requirements

The default configuration (`batch_size=2048`, 4 crops, 5-device multi-task training) is optimized for high-VRAM GPUs. Mixed precision (`bf16-mixed` on Ampere+ GPUs, `16-mixed` otherwise) is automatically enabled when CUDA is available.

| GPU | VRAM | Status | Notes |
|-----|------|--------|-------|
| RTX 4080 | 16 GB | Limited | Reduce `batch_size` to 512--1024 |
| RTX 4090 | 24 GB | Good | May need `batch_size=1024` for 5-device training |
| RTX 5090 | 32 GB | Optimal | Default config fits comfortably (~17 GB peak) |
| A100 | 40/80 GB | Excellent | Full config with room for larger windows |

**To reduce memory usage**, override `batch_size` via CLI:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance multi \
    --name_model NILMFormer \
    --batch_size 512 \
    --epochs 25
```

Additional tips:
- Reduce `train_num_crops` in `expes.yaml` (default 4) to lower effective batch size
- Use `--appliance Kettle` (single device) instead of `multi` for smaller memory footprint
- CPU training is supported but significantly slower (`device: cpu` in `expes.yaml`)

---

## Baseline Models

CondiNILM includes 15 baseline architectures spanning NILM-specific models and general time-series regression models. All baselines share the same training pipeline, data preprocessing, loss functions, and evaluation protocol for fair comparison.

### NILM-Specific Baselines

| Model | Type | Reference |
|-------|------|-----------|
| **BiGRU** | Recurrent | Bidirectional GRU for sequence-to-sequence regression |
| **BiLSTM** | Recurrent | Bidirectional LSTM for sequence-to-sequence regression |
| **CNN1D** | Convolutional | 1D convolutional network with pooling |
| **FCN** | Convolutional | Fully convolutional network |
| **UNet-NILM** | Encoder-Decoder | U-Net architecture adapted for NILM |
| **DResNet** | Residual | Dilated residual network |
| **DAResNet** | Residual | Dilated attention residual network |
| **BERT4NILM** | Transformer | BERT-style masked transformer for NILM |
| **DiffNILM** | Diffusion | Diffusion-based generative model for NILM |
| **ST-NILM** | Transformer + MoE | Spatial-temporal NILM with Mixture of Experts |
| **TSILNet** | Hybrid | TCN + LSTM integration network |
| **Energformer** | Transformer | Energy-focused transformer architecture |

### Time-Series Regression Baselines

| Model | Type | Reference |
|-------|------|-----------|
| **ConvNet** | Convolutional | General time-series convolutional network |
| **ResNet** | Residual | Residual network for time-series regression |
| **InceptionTime** | Inception | Inception-based time-series model |

### Running a Baseline

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance Kettle \
    --name_model CNN1D \
    --epochs 25
```

---

## Results

Experiment results are saved under `result/` (configurable via `result_path` in `expes.yaml`), including:

- **val_report.jsonl** -- Per-epoch validation metrics with per-device breakdowns
- **val_compare.html** -- Interactive visualization of validation curves
- **FINAL_EVAL_JSON** -- Test set evaluation results with postprocessing applied
- **experiment_config.json** -- Full configuration snapshot for reproducibility
- **TensorBoard logs** -- Training curves under `log/tensorboard/`

### UK-DALE Multi-Device Performance

Test set F1 scores using CondiNILMformer (V8.1) with multi-device joint training:

| Device | Test F1 |
|--------|---------|
| Fridge | 0.837 |
| Dishwasher | 0.796 |
| Kettle | 0.758 |
| Washing Machine | 0.342 |
| Microwave | 0.240 |

Results on REFIT and REDD datasets are available through the baseline comparison script (`run_baseline_comparison.py --phase 5`).

### Parsing Results

Use the provided script to aggregate and rank experiment results:

```bash
python scripts/collect_results.py
```

---

## Troubleshooting

### CUDA Out of Memory

If you encounter `RuntimeError: CUDA out of memory`:

1. Reduce batch size: add `--batch_size 512` (or 256) to your command
2. Reduce crops in `configs/expes.yaml`: set `train_num_crops: 2` (default 4)
3. Use single-device training instead of `multi`: `--appliance Kettle`
4. Ensure no other processes are using GPU memory: `nvidia-smi`

### Dataset Not Found

If you see `FileNotFoundError` for data paths:

1. Verify data is downloaded to `data/` directory (see [Dataset Preparation](#dataset-preparation))
2. Check that directory names match exactly (case-sensitive): `UKDALE`, `REFIT`, `REDD`
3. For UK-DALE, ensure `labels.dat` files exist in each house directory

### Training Collapse (F1 = 0)

Multi-device training may collapse for sparse devices (microwave, washing machine):

1. Check that `AdaptiveDeviceLoss` is enabled (default in `expes.yaml`)
2. Ensure `use_gradient_conflict_resolution: true` for PCGrad
3. Verify `gate_floor` is not too low for sparse devices (minimum 0.01 recommended)
4. Try increasing `loss_alpha_on` for the collapsing device in `configs/dataset_params.yaml`

### PyTorch / CUDA Version Mismatch

To check your CUDA version:

```bash
nvidia-smi  # Shows driver CUDA version
python -c "import torch; print(torch.version.cuda)"  # Shows PyTorch CUDA version
```

The PyTorch CUDA version must be compatible with (not necessarily identical to) your driver CUDA version. Install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Citation

If you use CondiNILMformer in your research, please cite:

```bibtex
@misc{li2026condinilmformer,
  title   = {CondiNILMformer: Condition-modulated Transformer for Multi-Device
             Non-Intrusive Load Monitoring},
  author  = {Li, Siyi},
  year    = {2026},
  note    = {Software available at https://github.com/davidzha712/CondiNILM}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for full details.
