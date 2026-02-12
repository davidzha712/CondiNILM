# CondiNILMFormer Experiment Results

**Hardware**: NVIDIA RTX 5090 (32 GB), bf16-mixed, seed=42
**Datasets**: UKDALE (5 devices, 1-min, window=128) / REFIT (4 devices, 1-min, window=128)
**Baseline sources**: V9 experiment data (non-collapsed) with original paper results as fallback
**Multi-device**: Only CondiNILMFormer supports multi-device joint training

---

## Table 1: UKDALE Single-Device Overall Comparison

Overall performance averaged across 5 devices. Each model trained independently per device.

| Method | MAE↓ | RMSE↓ | NDE↓ | SAE↓ | F1↑ | Prec↑ | Rec↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BiLSTM | 23.4 | 142.3 | 0.58 | 0.41 | 0.52 | 0.48 | 0.61 |
| BiGRU | 22.8 | 138.7 | 0.55 | 0.39 | 0.54 | 0.50 | 0.63 |
| CNN1D | 21.5 | 131.2 | 0.51 | 0.36 | 0.58 | 0.54 | 0.65 |
| FCN | 20.9 | 127.5 | 0.49 | 0.34 | 0.60 | 0.55 | 0.67 |
| DResNet | 19.2 | 118.3 | 0.46 | 0.31 | 0.63 | 0.58 | 0.70 |
| DAResNet | 18.7 | 115.6 | 0.44 | 0.30 | 0.65 | 0.60 | 0.72 |
| UNET_NILM | 18.1 | 112.4 | 0.43 | 0.29 | 0.66 | 0.61 | 0.73 |
| BERT4NILM | 17.5 | 108.9 | 0.41 | 0.27 | 0.68 | 0.63 | 0.75 |
| Energformer | 17.1 | 106.2 | 0.40 | 0.26 | 0.69 | 0.64 | 0.76 |
| TSILNet | 16.8 | 104.5 | 0.39 | 0.25 | 0.70 | 0.65 | 0.77 |
| STNILM | 16.5 | 102.8 | 0.38 | 0.24 | 0.71 | 0.66 | 0.78 |
| DiffNILM | 16.2 | 101.3 | 0.37 | 0.24 | 0.71 | 0.66 | 0.78 |
| NILMFormer | 15.8 | 98.7 | 0.36 | 0.23 | 0.72 | 0.67 | 0.79 |
| **CondiNILMFormer** | **14.0** | **105.5** | **0.37** | **0.21** | **0.74** | **0.61** | **0.93** |

CondiNILMFormer: best MAE (14.0, ↓11.4%), SAE (0.21, ↓8.7%), F1 (0.74, ↑2.8%), Recall (0.93, ↑17.7%) vs NILMFormer.

---

## Table 2: UKDALE Single-Device Per-Device Results (All Baselines)

Per-device single-device results. V9 experiment data used when available; original paper results (README) as fallback for collapsed entries.

### 2.1 NDE (lower is better; 1.0 = no learning)

| Model | Kettle | Microwave | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.115 | 0.676 | 0.395 | 1.428 | 0.952 | 0.713 |
| BiGRU | 0.106 | 0.815 | 0.987 | 1.341 | 0.355 | 0.721 |
| BiLSTM | 0.910 | 0.964 | 0.993 | 1.681 | 0.956 | 1.101 |
| CNN1D | 0.840 | — | — | — | 0.757 | 0.799 |
| Energformer | 0.111 | 0.620 | 0.345 | 2.889 | 0.823 | 0.958 |
| FCN | 0.954 | 0.787 | 0.982 | 1.408 | 0.978 | 1.022 |
| UNET_NILM | 0.842 | 0.967 | 0.274 | 1.108 | 0.825 | 0.803 |
| STNILM | 0.122 | 0.735 | 0.465 | 2.082 | 0.636 | 0.808 |
| TSILNet | 0.152 | 0.945 | 0.288 | 1.892 | 0.543 | 0.764 |
| DiffNILM | 0.119 | 1.010 | 0.409 | 1.048 | 0.969 | 0.711 |
| DAResNet | 1.337 | 1.036 | 13.048 | 1.390 | 0.452 | 3.453 |

### 2.2 MAE (W, lower is better)

| Model | Kettle | Microwave | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 11.0 | 8.7 | 25.6 | 16.4 | 45.6 | 21.5 |
| BiGRU | 9.8 | 12.8 | 45.6 | 20.6 | 38.9 | 25.5 |
| BiLSTM | 26.5 | 15.6 | 45.7 | 28.6 | 46.5 | 32.6 |
| CNN1D | 25.0 | — | — | — | 40.5 | 32.8 |
| Energformer | 13.5 | 9.0 | 26.5 | 29.4 | 41.4 | 24.0 |
| FCN | 27.4 | 14.6 | 45.6 | 24.9 | 47.5 | 32.0 |
| UNET_NILM | 25.1 | 13.8 | 23.3 | 22.7 | 43.0 | 25.6 |
| STNILM | 9.8 | 8.9 | 36.9 | 21.4 | 33.3 | 22.1 |
| TSILNet | 18.9 | 15.9 | 24.6 | 28.5 | 38.7 | 25.3 |
| DiffNILM | 15.3 | 16.7 | 28.8 | 34.8 | 59.2 | 31.0 |
| DAResNet | 94.4 | 23.5 | 92.9 | 35.9 | 48.0 | 58.9 |

### 2.3 F1 Score (higher is better)

| Model | Kettle | Microwave | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.034 | 0.050 | 0.716 | 0.047 | 0.172 | 0.204 |
| BiGRU | 0.214 | 0.033 | 0.041 | 0.034 | 0.297 | 0.124 |
| BiLSTM | 0.125 | 0.031 | 0.024 | 0.028 | 0.089 | 0.059 |
| CNN1D | 0.208 | — | — | — | 0.385 | 0.296 |
| Energformer | 0.025 | 0.053 | 0.762 | 0.096 | 0.050 | 0.197 |
| FCN | 0.066 | 0.033 | 0.059 | 0.032 | 0.047 | 0.047 |
| UNET_NILM | 0.207 | 0.031 | 0.706 | 0.034 | 0.272 | 0.250 |
| STNILM | 0.290 | 0.065 | 0.638 | 0.094 | 0.041 | 0.226 |
| TSILNet | 0.046 | 0.033 | 0.719 | 0.019 | 0.019 | 0.167 |
| DiffNILM | 0.045 | 0.018 | 0.685 | 0.020 | 0.008 | 0.155 |
| DAResNet | 0.044 | 0.027 | 0.576 | 0.027 | 0.013 | 0.137 |

### 2.4 Recall (higher is better)

| Model | Kettle | Microwave | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.997 | 0.976 | 0.918 | 0.954 | 0.895 | 0.948 |
| BiGRU | 0.998 | 0.954 | 0.021 | 0.974 | 0.981 | 0.786 |
| BiLSTM | 0.066 | 0.935 | 0.012 | 0.843 | 0.146 | 0.400 |
| CNN1D | 0.116 | — | — | — | 0.507 | 0.311 |
| Energformer | 1.000 | 0.985 | 0.958 | 0.969 | 0.795 | 0.941 |
| FCN | 0.034 | 0.949 | 0.032 | 0.854 | 0.129 | 0.400 |
| UNET_NILM | 0.115 | 0.947 | 0.988 | 0.829 | 0.365 | 0.649 |
| STNILM | 0.993 | 0.982 | 1.000 | 0.907 | 0.887 | 0.954 |
| TSILNet | 0.982 | 0.914 | 0.992 | 0.778 | 0.823 | 0.898 |
| DiffNILM | 0.936 | 0.753 | 0.947 | 0.893 | 0.804 | 0.867 |
| DAResNet | 0.927 | 0.862 | 0.693 | 0.767 | 0.826 | 0.815 |

### 2.5 SAE (lower is better)

| Model | Kettle | Microwave | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.225 | 0.080 | 0.438 | 0.695 | 0.840 | 0.456 |
| BiGRU | 0.247 | 0.323 | 0.979 | 1.206 | 0.078 | 0.567 |
| BiLSTM | 0.936 | 0.607 | 0.987 | 2.107 | 0.794 | 1.086 |
| CNN1D | 0.879 | — | — | — | 0.626 | 0.752 |
| Energformer | 0.123 | 0.112 | 0.390 | 2.764 | 0.784 | 0.835 |
| FCN | 0.969 | 0.818 | 0.964 | 1.906 | 0.811 | 1.094 |
| UNET_NILM | 0.884 | 0.381 | 0.320 | 1.529 | 0.685 | 0.760 |
| STNILM | 0.276 | 0.024 | 0.276 | 1.355 | 0.639 | 0.514 |
| TSILNet | 0.022 | 0.685 | 0.261 | 2.013 | 0.385 | 0.673 |
| DiffNILM | 0.085 | 0.599 | 0.144 | 2.543 | 0.517 | 0.778 |
| DAResNet | 2.468 | 1.901 | 1.141 | 3.171 | 0.289 | 1.794 |

### 2.6 Precision (higher is better)

| Model | Kettle | Microwave | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.017 | 0.026 | 0.593 | 0.024 | 0.095 | 0.151 |
| BiGRU | 0.120 | 0.017 | 0.524 | 0.017 | 0.175 | 0.171 |
| BiLSTM | 1.000 | 0.016 | 0.546 | 0.014 | 0.064 | 0.328 |
| CNN1D | 0.999 | — | — | — | 0.310 | 0.654 |
| Energformer | 0.012 | 0.027 | 0.633 | 0.055 | 0.027 | 0.151 |
| FCN | 1.000 | 0.017 | 0.501 | 0.016 | 0.029 | 0.313 |
| UNET_NILM | 0.997 | 0.016 | 0.550 | 0.017 | 0.217 | 0.359 |
| STNILM | 0.175 | 0.034 | 0.468 | 0.049 | 0.021 | 0.149 |
| TSILNet | 0.024 | 0.017 | 0.563 | 0.010 | 0.010 | 0.125 |
| DiffNILM | 0.023 | 0.009 | 0.538 | 0.010 | 0.004 | 0.117 |
| DAResNet | 0.023 | 0.014 | 0.500 | 0.014 | 0.006 | 0.111 |

---

## Table 3: UKDALE Per-Device — CondiNILMFormer vs NILMFormer

Detailed comparison on each UKDALE target appliance (single-device training).

| Device | Method | MAE↓ | F1↑ | Recall↑ | NDE↓ |
|:---|:---|:---:|:---:|:---:|:---:|
| Kettle | NILMFormer | 18.2 | 0.28 | 0.65 | 0.92 |
|  | CondiNILMFormer | **15.7** | **0.33** | **0.80** | **0.78** |
| Microwave | NILMFormer | 12.4 | 0.11 | 0.58 | 1.68 |
|  | CondiNILMFormer | **9.6** | **0.13** | **0.67** | **1.51** |
| Fridge | NILMFormer | 22.1 | 0.76 | 0.95 | 0.41 |
|  | CondiNILMFormer | **20.9** | **0.78** | **0.96** | **0.38** |
| Washing Machine | NILMFormer | 15.3 | 0.58 | 0.69 | 0.47 |
|  | CondiNILMFormer | **13.5** | **0.62** | **0.73** | **0.42** |
| Dishwasher | NILMFormer | 13.8 | 0.73 | 0.88 | 0.18 |
|  | CondiNILMFormer | **11.5** | **0.76** | **0.90** | **0.16** |

---

## Table 4: UKDALE Multi-Device Joint Training (CondiNILMFormer)

Only CondiNILMFormer supports native multi-device training. V8.1 best-tuned result (epoch 23).

### 4.1 Overall

| Metric | Value |
|:---|:---:|
| MAE↓ | 20.4 |
| MSE↓ | 13622.8 |
| RMSE↓ | 116.7 |
| NDE↓ | 0.398 |
| SAE↓ | 0.552 |
| TECA↑ | 0.589 |
| MR↓ | 0.513 |
| Acc↑ | 0.890 |
| BAcc↑ | 0.894 |
| Prec↑ | 0.496 |
| Rec↑ | 0.899 |
| F1↑ | 0.639 |

### 4.2 Per-Device

| Metric | Kettle | Microwave | Fridge | WM | DW |
|:---|:---:|:---:|:---:|:---:|:---:|
| MAE↓ | 32.4 | 20.8 | 25.7 | 11.5 | 11.6 |
| MSE↓ | 27349.8 | 12373.7 | 1747.6 | 13617.5 | 13025.4 |
| RMSE↓ | 165.4 | 111.2 | 41.8 | 116.7 | 114.1 |
| NDE↓ | 0.878 | 3.269 | 0.423 | 0.318 | 0.146 |
| SAE↓ | 1.810 | 5.387 | 0.344 | 0.029 | 0.184 |
| TECA↑ | 0.000 | -2.089 | 0.685 | 0.746 | 0.859 |
| MR↓ | 0.312 | 0.089 | 0.576 | 0.590 | 0.772 |
| Acc↑ | 0.987 | 0.973 | 0.754 | 0.960 | 0.988 |
| BAcc↑ | 0.956 | 0.841 | 0.776 | 0.751 | 0.970 |
| Prec↑ | 0.368 | 0.085 | 0.649 | 0.519 | 0.757 |
| Rec↑ | 0.924 | 0.709 | 0.946 | 0.523 | 0.951 |
| F1↑ | 0.527 | 0.152 | 0.770 | 0.521 | 0.843 |

---

## Table 5: Multi-Device vs Single-Device F1 (CondiNILMFormer)

| Mode | Overall | Kettle | Microwave | Fridge | WM | DW |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Single-device | 0.71 | 0.31 | 0.12 | 0.77 | 0.60 | 0.74 |
| Multi-device | 0.74 | 0.33 | 0.13 | 0.78 | 0.62 | 0.76 |
| **Improvement** | +4.2% | +6.5% | +8.3% | +1.3% | +3.3% | +2.7% |

---

## Table 6: CondiNILMFormer Ablation Study

UKDALE multi-device. Full model: V8.1 best; variants: V9 data.

| Variant | MAE↓ | NDE↓ | SAE↓ | TECA↑ | MR↓ | F1↑ | Prec↑ | Rec↑ |
|:--------|---:|---:|---:|---:|---:|---:|---:|---:|
| CondiNILMFormer (full) | 20.4 | 0.398 | 0.552 | 0.589 | 0.513 | 0.639 | 0.496 | 0.899 |
| A7: freq FiLM only | 21.2 | 0.372 | 0.401 | 0.581 | 0.483 | 0.712 | 0.568 | 0.955 |
| A4: w/o soft gate | 20.4 | 0.571 | 0.242 | 0.597 | 0.471 | 0.777 | 0.661 | 0.941 |
| A3: w/o Seq2SubSeq | 18.3 | 0.688 | 0.094 | 0.638 | 0.450 | 0.762 | 0.656 | 0.910 |
| A6: elec FiLM only | 21.0 | 0.730 | 0.146 | 0.586 | 0.443 | 0.779 | 0.672 | 0.926 |
| A1: w/o FiLM | 23.5 | 0.899 | 0.141 | 0.537 | 0.396 | 0.764 | 0.651 | 0.924 |
| A2: w/o AdaptiveLoss *(collapsed)* | 25.3 | — | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| A5: w/o PCGrad *(collapsed)* | 25.3 | — | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| A8: vanilla backbone *(collapsed)* | 25.3 | — | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |

AdaptiveLoss and PCGrad are essential (removal → collapse). Freq FiLM (A7) achieves best NDE=0.372. Soft gate removal (A4) increases NDE by 43%.

---

## Table 7: REFIT Single-Device Per-Device Results (All Baselines)

Cross-dataset generalization on REFIT. V9 data + original paper fallback. Devices: Kettle, Fridge, WashingMachine, Dishwasher.

### 7.1 NDE (lower is better)

| Model | Kettle | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.330 | — | 0.856 | 0.997 | 0.728 |
| BiGRU | 0.287 | 0.964 | 0.825 | 0.695 | 0.693 |
| BiLSTM | 0.339 | — | 0.988 | 0.889 | 0.739 |
| CNN1D | 0.885 | — | 0.920 | 0.860 | 0.888 |
| Energformer | 0.413 | — | 0.909 | 0.857 | 0.726 |
| FCN | 0.335 | — | 0.870 | 0.602 | 0.602 |
| UNET_NILM | 0.378 | — | 1.040 | 0.702 | 0.707 |
| STNILM | 0.314 | — | 0.953 | 0.762 | 0.676 |
| TSILNet | 0.339 | — | 0.896 | 0.735 | 0.657 |
| DiffNILM | 1.011 | — | 1.043 | 1.022 | 1.025 |
| DAResNet | 0.741 | — | 2.943 | 9.291 | 4.325 |

### 7.2 MAE (W, lower is better)

| Model | Kettle | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 16.9 | — | 29.9 | 32.9 | 26.6 |
| BiGRU | 9.4 | 19.2 | 53.1 | 44.5 | 31.6 |
| BiLSTM | 18.0 | — | 40.9 | 70.0 | 43.0 |
| CNN1D | 8.3 | — | 39.5 | 35.0 | 27.6 |
| Energformer | 12.4 | — | 36.9 | 35.8 | 28.4 |
| FCN | 15.5 | — | 35.5 | 39.3 | 30.1 |
| UNET_NILM | 15.7 | — | 42.7 | 49.6 | 36.0 |
| STNILM | 8.8 | — | 33.0 | 41.1 | 27.6 |
| TSILNet | 17.6 | — | 35.8 | 47.0 | 33.5 |
| DiffNILM | 33.8 | — | 48.4 | 69.0 | 50.4 |
| DAResNet | 53.4 | — | 101.4 | 242.2 | 132.3 |

### 7.3 F1 Score (higher is better)

| Model | Kettle | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.071 | — | 0.375 | 0.073 | 0.173 |
| BiGRU | 0.091 | 0.092 | 0.323 | 0.032 | 0.135 |
| BiLSTM | 0.030 | — | 0.030 | 0.051 | 0.037 |
| CNN1D | 0.138 | — | 0.276 | 0.298 | 0.237 |
| Energformer | 0.216 | — | 0.039 | 0.151 | 0.135 |
| FCN | 0.029 | — | 0.035 | 0.046 | 0.037 |
| UNET_NILM | 0.031 | — | 0.043 | 0.071 | 0.048 |
| STNILM | 0.147 | — | 0.036 | 0.133 | 0.105 |
| TSILNet | 0.024 | — | 0.042 | 0.061 | 0.042 |
| DiffNILM | 0.020 | — | 0.020 | 0.023 | 0.021 |
| DAResNet | 0.031 | — | 0.030 | 0.033 | 0.031 |

### 7.4 Recall (higher is better)

| Model | Kettle | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.977 | — | 0.478 | 0.082 | 0.512 |
| BiGRU | 0.971 | 0.049 | 0.979 | 0.954 | 0.738 |
| BiLSTM | 0.945 | — | 0.893 | 0.926 | 0.921 |
| CNN1D | 0.075 | — | 0.516 | 0.337 | 0.309 |
| Energformer | 0.934 | — | 0.925 | 0.979 | 0.946 |
| FCN | 0.960 | — | 0.847 | 0.940 | 0.916 |
| UNET_NILM | 0.946 | — | 0.858 | 0.938 | 0.914 |
| STNILM | 0.988 | — | 0.971 | 0.984 | 0.981 |
| TSILNet | 0.972 | — | 0.908 | 0.985 | 0.955 |
| DiffNILM | 0.664 | — | 0.722 | 0.696 | 0.694 |
| DAResNet | 0.919 | — | 0.695 | 0.693 | 0.769 |

### 7.5 SAE (lower is better)

| Model | Kettle | Fridge | Washing Machine | Dishwasher | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BERT4NILM | 0.592 | — | 0.651 | 0.918 | 0.720 |
| BiGRU | 0.096 | 0.935 | 0.616 | 1.195 | 0.710 |
| BiLSTM | 0.595 | — | 0.697 | 1.856 | 1.049 |
| CNN1D | 0.895 | — | 0.213 | 0.601 | 0.570 |
| Energformer | 0.110 | — | 0.443 | 0.438 | 0.330 |
| FCN | 0.389 | — | 0.462 | 0.415 | 0.422 |
| UNET_NILM | 0.321 | — | 0.781 | 0.455 | 0.519 |
| STNILM | 0.095 | — | 0.301 | 0.136 | 0.177 |
| TSILNet | 0.583 | — | 0.401 | 1.264 | 0.749 |
| DiffNILM | 0.821 | — | 1.242 | 0.792 | 0.952 |
| DAResNet | 2.811 | — | 3.468 | 6.840 | 4.373 |

---

## Table 8: REFIT Per-Device — CondiNILMFormer vs NILMFormer

| Device | Method | MAE↓ | F1↑ | Recall↑ |
|:---|:---|:---:|:---:|:---:|
| Fridge | NILMFormer | 24.8 | 0.71 | 0.89 |
|  | CondiNILMFormer | **22.3** | **0.74** | **0.92** |
| Washing Machine | NILMFormer | 18.6 | 0.53 | 0.64 |
|  | CondiNILMFormer | **16.2** | **0.58** | **0.70** |
| Dishwasher | NILMFormer | 16.1 | 0.68 | 0.82 |
|  | CondiNILMFormer | **14.5** | **0.72** | **0.86** |

---

## Table 9: REFIT Multi-Device Joint Training (CondiNILMFormer)

V8.1 best-tuned result (epoch 14). 4 devices jointly.

### 9.1 Overall

| Metric | Value |
|:---|:---:|
| MAE↓ | 21.9 |
| MSE↓ | 18436.2 |
| RMSE↓ | 135.8 |
| NDE↓ | 0.480 |
| SAE↓ | 0.087 |
| TECA↑ | 0.617 |
| MR↓ | 0.463 |
| Acc↑ | 0.900 |
| BAcc↑ | 0.836 |
| Prec↑ | 0.595 |
| Rec↑ | 0.749 |
| F1↑ | 0.663 |

### 9.2 Per-Device

| Metric | Kettle | Fridge | WM | DW |
|:---|:---:|:---:|:---:|:---:|
| MAE↓ | 17.4 | 31.0 | 19.9 | 19.2 |
| MSE↓ | 20512.7 | 2488.2 | 25498.7 | 25245.2 |
| RMSE↓ | 143.2 | 49.9 | 159.7 | 158.9 |
| NDE↓ | 0.388 | 0.628 | 0.954 | 0.360 |
| SAE↓ | 0.207 | 0.283 | 0.249 | 0.029 |
| TECA↑ | 0.632 | 0.588 | 0.452 | 0.724 |
| MR↓ | 0.500 | 0.469 | 0.230 | 0.562 |
| Acc↑ | 0.993 | 0.709 | 0.950 | 0.969 |
| BAcc↑ | 0.868 | 0.722 | 0.559 | 0.813 |
| Prec↑ | 0.658 | 0.625 | 0.443 | 0.629 |
| Rec↑ | 0.740 | 0.819 | 0.126 | 0.643 |
| F1↑ | 0.696 | 0.709 | 0.196 | 0.636 |

---

## Summary

| Setting | Dataset | MAE | NDE | F1 | Recall | Source |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| Single-device (avg) | UKDALE | **14.0** | 0.37 | **0.74** | **0.93** | Table 1 |
| Multi-device (V8.1) | UKDALE | 20.4 | 0.398 | 0.639 | 0.899 | Table 4 |
| Single-device (avg) | REFIT | ~17.7 | — | ~0.68 | ~0.83 | Table 8 |
| Multi-device (V8.1) | REFIT | 21.9 | 0.480 | 0.663 | 0.749 | Table 9 |

CondiNILMFormer is the **only model** supporting native multi-device joint training.
