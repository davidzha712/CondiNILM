# CondiNILM

## Overview

CondiNILM is a research framework for Non-Intrusive Load Monitoring (NILM) built on PyTorch Lightning. It centers on NILMFormer, a transformer-based architecture designed for multi-device joint energy disaggregation. The framework includes device-type-aware adaptive loss functions, soft gate mechanisms for sparse appliance handling, and anti-collapse training stabilization for robust multi-task learning.

Given an aggregate household power signal, the model simultaneously disaggregates per-appliance power consumption for up to five devices (fridge, kettle, dishwasher, washing machine, microwave) in a single forward pass.

## Key Features

- **NILMFormer architecture**: Transformer encoder with dilated convolution embedding, FiLM (Feature-wise Linear Modulation) conditioning, instance normalization, and device-type-grouped prediction heads.
- **Dual-path inference**: Transformer path for complex multi-state devices (fridge, dishwasher, washing machine) and a lightweight CNN bypass path for sparse high-power devices (kettle, microwave) with learnable blend weights.
- **AdaptiveDeviceLoss**: Per-device loss function that automatically derives parameters from appliance electrical characteristics (duty cycle, ON/OFF power ratios), with asymmetric ON/OFF weighting and focal-style gate classification loss.
- **Soft gate mechanism**: Learnable per-device gate with scale, bias, and floor parameters. Uses soft gating during training and hard thresholding at inference for clean ON/OFF separation.
- **Anti-collapse stabilization**: Automatic detection and recovery from training collapse in multi-task learning, preventing dominant devices from suppressing sparse ones.
- **Multi-dataset support**: UKDALE (5 houses, 6s sampling), REFIT (20 houses, 8s sampling), and REDD (6 houses, 1s sampling) with configurable house-level train/validation/test splits.
- **Hyperparameter optimization**: Optuna integration with configurable search spaces for loss parameters, learning rates, and architecture hyperparameters.
- **15 baseline architectures**: BiGRU, BiLSTM, CNN1D, UNet-NILM, FCN, DResNet, DAResNet, BERT4NILM, DiffNILM, STNILM, TSILNet, Energformer, ConvNet, ResNet, and InceptionTime for systematic comparison.

## Installation

### Prerequisites

- Python 3.10 or later (tested with 3.12 on Windows)
- CUDA-capable GPU recommended

### Setup with pip

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install PyTorch according to your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)), then install the remaining dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Alternative: Conda

Conda environment files are provided for reference:

- `environment_win.yaml` (Windows)
- `environment_mac.yaml` (macOS)

## Quick Start

### Single-device training

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance Kettle \
    --name_model NILMFormer \
    --n_epochs 25
```

### Multi-device joint training

Use `multi` to train on all five target appliances simultaneously:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance multi \
    --name_model NILMFormer \
    --n_epochs 25
```

Or specify a subset of devices as a comma-separated list:

```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --sampling_rate 1min \
    --window_size 128 \
    --appliance Kettle,Fridge,Microwave \
    --name_model NILMFormer \
    --n_epochs 25
```

### Hyperparameter optimization

```bash
python scripts/run_optuna_search.py \
    --dataset UKDALE \
    --appliance multi \
    --name_model NILMFormer \
    --n_trials 50
```

### Run all arguments

```bash
python scripts/run_experiment.py -h
```

## Project Structure

```
CondiNILM/
├── src/
│   ├── nilmformer/                 # NILMFormer model
│   │   ├── model.py                # NILMFormer, SparseDeviceCNN, SimpleDeviceHead
│   │   ├── config.py               # NILMFormerConfig dataclass
│   │   └── layers/
│   │       ├── embedding.py        # DilatedBlock, ResUnit convolution embedding
│   │       └── transformer.py      # EncoderLayer with multi-head attention
│   ├── baselines/
│   │   ├── nilm/                   # NILM-specific baselines
│   │   │   ├── bilstm.py           # BiLSTM
│   │   │   ├── bigru.py            # BiGRU
│   │   │   ├── cnn1d.py            # CNN1D
│   │   │   ├── unetnilm.py         # UNet-NILM
│   │   │   ├── fcn.py              # FCN
│   │   │   ├── dresnets.py         # DResNet, DAResNet
│   │   │   ├── bert4nilm.py        # BERT4NILM
│   │   │   ├── diffnilm.py         # DiffNILM
│   │   │   ├── stnilm.py           # STNILM (with MoE)
│   │   │   ├── tsilnet.py          # TSILNet (TCN + LSTM)
│   │   │   └── energformer.py      # Energformer
│   │   └── tser/                   # Time-series regression baselines
│   │       ├── convnet.py          # ConvNet
│   │       ├── resnet.py           # ResNet
│   │       └── inceptiontime.py    # InceptionTime
│   └── helpers/
│       ├── experiment.py           # Model factory, utility re-exports
│       ├── trainer.py              # PyTorch Lightning trainer, AdaptiveDeviceLoss
│       ├── training.py             # Training loop orchestration
│       ├── evaluation.py           # Evaluation and metric computation
│       ├── callbacks.py            # Lightning callbacks (validation, logging)
│       ├── loss.py                 # Loss function definitions
│       ├── inference.py            # Sliding window inference and stitching
│       ├── postprocess.py          # Post-processing (threshold, gate suppression)
│       ├── preprocessing.py        # Data loading, windowing, train/valid splitting
│       ├── dataset.py              # PyTorch Dataset classes
│       ├── dataset_params.py       # Per-dataset device parameter loading
│       ├── device_config.py        # Device type classification and gate configs
│       ├── metrics.py              # F1, precision, recall, MAE, SAE metrics
│       ├── loss_tuning.py          # Loss parameter tuning utilities
│       ├── gradient_conflict.py    # Gradient conflict detection (PCGrad)
│       └── utils.py                # General utilities
├── configs/
│   ├── expes.yaml                  # Experiment settings (training, loss, gate, scheduler)
│   ├── dataset_params.yaml         # Per-dataset device parameters and postprocessing
│   ├── datasets.yaml               # House-level train/valid/test splits
│   ├── models.yaml                 # Model architecture hyperparameters
│   ├── hpo_search_spaces.yaml      # Optuna search space definitions
│   └── hpo_sparse_devices.yaml     # HPO config for sparse devices
├── scripts/
│   ├── run_experiment.py           # Main entry point for single experiments
│   ├── run_optuna_search.py        # Optuna hyperparameter search
│   ├── run_ukdale_all.py           # Run all UKDALE experiments
│   ├── run_redd_all.py             # Run all REDD experiments
│   ├── run_all_experiments.sh      # Batch runner for all dataset/model combinations
│   ├── streamlit_val_viewer.py     # Interactive validation visualization
│   ├── viz_preprocessing.py        # Preprocessing visualization
│   ├── analyze_appliance_stats.py  # Appliance statistics analysis
│   └── find_best_results.py        # Parse and rank experiment results
├── assets/                         # Architecture diagrams and figures
├── data/                           # Dataset root (UKDALE/, REFIT/, REDD/)
├── pyproject.toml                  # PEP 621 project metadata and packaging
├── requirements.txt
├── environment_win.yaml
├── environment_mac.yaml
└── LICENSE
```

## Configuration

### expes.yaml

Core experiment settings: model selection, training hyperparameters (batch size, epochs, learning rate scheduler), loss function parameters (output ratio, gate weights, focal gamma), and data augmentation (cropping, event-biased sampling).

### dataset_params.yaml

Per-dataset, per-device parameters including ON/OFF power thresholds, minimum event durations, postprocessing settings, and loss function overrides. Postprocessing entries in this file take precedence over those in `expes.yaml`.

### datasets.yaml

House-level data splits for each dataset and appliance. Defines which houses are used for training/validation (temporal 80/20 split within each house) and which are held out for testing.

### models.yaml

Architecture-specific hyperparameters for NILMFormer and all 15 baselines, including layer counts, hidden dimensions, dropout rates, and learning rates.

## Datasets

The framework supports three public NILM datasets:

| Dataset | Houses | Sampling | Appliances | Reference |
|---------|--------|----------|------------|-----------|
| **UKDALE** | 5 | 6 seconds | Fridge, Kettle, Dishwasher, Washing Machine, Microwave | Kelly & Knottenbelt, 2015 |
| **REFIT** | 20 | 8 seconds | Fridge, Kettle, Dishwasher, Washing Machine, Microwave | Murray et al., 2017 |
| **REDD** | 6 | 1 second | Fridge, Dishwasher, Washing Machine, Microwave | Kolter & Johnson, 2011 |

Place dataset files under the `data/` directory (or configure `data_path` in `expes.yaml`):

- **UKDALE**: `data/UKDALE/house_1/`, `house_2/`, etc.
- **REFIT**: `data/REFIT/RAW_DATA_CLEAN/CLEAN_House1.csv`, etc.
- **REDD**: `data/REDD/house_1/`, `house_2/`, etc.

## Results

### UKDALE Multi-Device (V7.5i, 25 epochs)

Test set F1 scores using NILMFormer with multi-device joint training:

| Device | Test F1 | Validation F1 |
|--------|---------|---------------|
| Fridge | 0.817 | 0.774 |
| Kettle | 0.703 | 0.504 |
| Dishwasher | 0.662 | 0.494 |
| Microwave | 0.295 | 0.158 |
| Washing Machine | 0.280 | 0.351 |

## Outputs

Experiment results are saved under `result/` (configurable via `result_path` in `expes.yaml`), including:

- **val_report.jsonl**: Per-epoch validation metrics with per-device breakdowns.
- **val_compare.html**: Interactive visualization of validation curves.
- **FINAL_EVAL_JSON**: Test set evaluation results with postprocessing applied.
- **TensorBoard logs**: Training curves under `log/tensorboard/`.

To view validation results interactively:

```bash
streamlit run scripts/streamlit_val_viewer.py
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
