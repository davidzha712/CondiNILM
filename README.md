# CondiNILM

A research codebase for Non-Intrusive Load Monitoring (NILM), featuring a unified training/evaluation pipeline for both single-appliance and multi-appliance joint learning, with NILMFormer and multiple NILM/TSER baseline models.

## Features

- **Unified training pipeline** for single-device and multi-device joint learning
- **NILMFormer** architecture alongside multiple NILM and TSER baselines (BERT4NILM, BiGRU, BiLSTM, CNN1D, DiffNILM, DResNets, EnerGFormer, FCN, STNilm, TSILNet, UNetNILM)
- **Flexible configuration** via YAML files with OmegaConf
- **Hyperparameter optimization** with Optuna integration
- **Interactive visualization** using Streamlit for validation results
- Support for **UKDALE** and **REFIT** datasets

## Installation

**Python**: 3.10+ recommended (tested with 3.12).

This project does **not** support `uv` installation. Use `python + venv + pip`.

### 1. Create virtual environment

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2. Install dependencies

Install PyTorch first (choose the appropriate command for your CUDA/CPU environment):

```bash
pip install -U torch torchvision torchaudio
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The default data root directory is specified in `configs/expes.yaml` under `data_path` (default: `data/`).

- **UKDALE**: Reads from `${data_path}/UKDALE/house_1`, `house_2`, etc.
- **REFIT**: Reads from `${data_path}/REFIT/RAW_DATA_CLEAN/` for cleaned CSVs.

Modify `configs/expes.yaml` if your data is stored elsewhere.

## Usage

The entry script is `scripts/run_one_expe.py`:

```bash
python scripts/run_one_expe.py -h
```

### Single-appliance training

```bash
python scripts/run_one_expe.py \
  --dataset UKDALE \
  --sampling_rate 1min \
  --window_size 128 \
  --appliance WashingMachine \
  --name_model NILMFormer
```

### Multi-appliance training

Use `multi` or a comma-separated list of appliance names:

```bash
python scripts/run_one_expe.py \
  --dataset UKDALE \
  --sampling_rate 1min \
  --window_size 128 \
  --appliance Kettle,Fridge \
  --name_model NILMFormer
```

### Hyperparameter optimization

```bash
python scripts/run_optuna_search.py
```

### Interactive visualization

```bash
streamlit run scripts/streamlit_val_viewer.py
```

## Output

Outputs are saved to the path specified in `configs/expes.yaml` under `result_path` (default: `result/`):

- `val_compare.html` -- Validation curve visualization (filterable by device, model, epoch)
- `val_report.jsonl` -- Per-epoch validation records with overall and per-device metrics

## Project Structure

```
CondiNILM/
├── configs/                 # Experiment configurations (YAML)
│   ├── expes.yaml
│   ├── models.yaml
│   └── datasets.yaml
├── scripts/                 # Entry scripts
│   ├── run_one_expe.py
│   ├── run_optuna_search.py
│   └── streamlit_val_viewer.py
├── src/
│   ├── baselines/           # NILM and TSER baseline models
│   │   ├── nilm/
│   │   └── tser/
│   ├── helpers/             # Training, evaluation, preprocessing, metrics
│   └── nilmformer/          # NILMFormer implementation
├── assets/                  # Image resources
├── docs/                    # Documentation
├── environment_mac.yaml     # Conda environment reference (macOS)
├── environment_win.yaml     # Conda environment reference (Windows)
├── requirements.txt         # pip dependencies
└── README.md
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
