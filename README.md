# CondiNILM

<div align="center">

**CondiNILM: Feature-wise Modulated Multi-Task Learning for Non-Intrusive Load Monitoring**

Research codebase by ** **  


![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue)

</div>

---

## Overview

**CondiNILM** is a **novel multi-task deep learning framework for Non-Intrusive Load Monitoring (NILM)**, developed as part of a Master’s thesis at **TU Braunschweig**.

The framework targets **device-level power disaggregation from aggregate household measurements**, with a particular focus on:

- **Multi-appliance joint learning**
- **Non-stationary power signals**
- **Time–frequency feature fusion**
- **Device-conditioned modeling via FiLM (Feature-wise Linear Modulation)**

Unlike classical NILM approaches that train **one model per appliance**, CondiNILM formulates NILM as a **single multi-output learning problem**, where **shared representations** are dynamically modulated by **device-specific conditions**.

---

## Key Contributions

CondiNILM introduces several original design choices:

### 1. Multi-Task NILM with Device Conditioning

- A **single unified model** predicts power consumption for multiple appliances simultaneously
- Each appliance is modeled via **device-conditioned output heads**
- Reduces parameter redundancy and improves cross-device generalization

### 2. FiLM-Modulated Feature Decoding

- **FiLM layers** are used to modulate intermediate representations:
  
  \[
  \text{FiLM}(x \mid d) = \gamma_d \cdot x + \beta_d
  \]

- Enables **explicit device-aware control** over shared temporal features
- Prevents power “leakage” and cross-device interference common in multi-head NILM

### 3. Time–Frequency Feature Fusion

- The model jointly exploits:
  - **Time-domain power sequences**
  - **Frequency-domain representations** (FFT / STFT / spectral statistics)
  - **Engineered auxiliary features** (e.g. activity priors, signal energy)

- These modalities are fused through attention-based encoders

### 4. Sequence-Level Supervision with Dense Outputs

- Supports **Seq2Seq**, **Seq2Subsequence**, and **Seq2Point** supervision
- Enables **high-resolution waveform reconstruction** at inference time
- Training and inference strategies can be decoupled for efficiency

---

## Project Scope

This repository serves as the **official thesis codebase of CondiNILM** and includes:

- The complete implementation of **CondiNILM**
- A unified training and evaluation pipeline
- Re-implementations of **10+ recent NILM baselines** in PyTorch
- Reproducible experiment scripts and configurations

The framework is designed for **research extensibility**, not only benchmark reproduction.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CondiNILM.git
cd CondiNILM
```

### 2.Create Conda Environment

```bash
conda env create -f environment.yml
conda activate film-multinilm
```

### Verify Installation
```bash
python -c "import torch; print(torch.version.cuda)"
```
---

## Environment Specification
environment.yml:

```yaml
name: film-multinilm
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - numpy
  - scipy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - tqdm
  - pyyaml
  - einops
  - pip
  - pip:
      - pytorch-lightning
      - hydra-core
      - mlflow
```

---

## Code Structure
```bash
.
├── assets/                 # figures and visual assets
├── configs/                # experiment configuration files (YAML)
├── data/                   # dataset metadata and splits
├── results/                # experiment outputs and logs
├── scripts/                # experiment launch scripts
│   ├── run_one_expe.py
│   └── run_all_expe.sh
├── src/
│   ├── helpers/            # training, metrics, preprocessing
│   ├── baselines/          # NILM baseline models
│   └── film_multinilm/     # CondiNILM core implementation
├── environment.yml
└── README.md
```
---

## Running Experiments
Run a Single Experiment
```bash
python scripts/run_one_expe.py \
    --dataset "UKDALE" \
    --sampling_rate "1min" \
    --appliance "WashingMachine" \
    --window_size 128 \
    --name_model NILMFormer \
    --seed 42
```
Run Full Benchmark Suite
```bash
bash scripts/run_all_expe.sh
```

---
Research Context

CondiNILM was developed in the context of:
	•	Advanced NILM research
	•	Multi-task learning for energy disaggregation
	•	Transformer-based time-series modeling
	•	Device-aware representation learning

The codebase is structured to support ablation studies, loss-function research, and architecture extensions.

---
## Acknowledgement
This project builds upon established NILM research and re-implements several prior baselines for comparison.
All newly introduced architectures, FiLM conditioning mechanisms, multi-task heads, and training strategies are original contributions of this work.

## Contact
**Siyi Li**
M.Sc. Electrical Engineering · TU Braunschweig
For questions or collaborations, please reach out to:
- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [your-username](https://github.com/your-username)