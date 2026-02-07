# NILM 多设备优化报告

> 生成时间: 2026-01-30
> 数据集: UKDALE
> 模型: NILMFormer

## 1. 问题总览

| 设备 | 类型 | 单设备F1 | 多设备F1 | 核心问题 | 优先级 |
|------|------|----------|----------|----------|--------|
| Microwave | sparse_high_power | 0.29 | 0.08 | OFF惩罚过重，Gate完全关闭 | P0 |
| WashingMachine | long_cycle | 0.68 | 0.55 | Gate饱和，alpha_off过低 | P0 |
| Fridge | cycling_low_power | 0.81 | 0.75 | Precision下降，Gate过松 | P1 |
| Dishwasher | long_cycle | 0.73 | 0.83 | OFF能量泄漏（可接受） | P2 |

## 2. 根因分析

### 2.1 共性问题

1. **多设备训练时 `allow_manual_tuning=False`**
   - 设备专用参数不生效，所有设备使用通用参数
   - 稀疏设备(Microwave)被高频设备(Fridge)的梯度主导

2. **Gate机制参数问题**
   - Microwave: `gate_floor=0.001` 过低，Gate几乎完全关闭
   - WashingMachine: Gate饱和到0.9999，无法区分ON/OFF
   - Fridge: `gate_floor=0.05` 过高，OFF状态也有能量泄漏

3. **ON/OFF权重失衡**
   - 稀疏设备: `alpha_on` 不够高，无法检测稀疏事件
   - 长周期设备: `alpha_off` 不够高，OFF状态预测不准

### 2.2 设备特定问题

| 设备 | 问题 | 表现 |
|------|------|------|
| Microwave | `w_off_fp=0.75` 过高 | 模型不敢预测ON，F1仅0.08 |
| Microwave | `gate_bias=-3.1` 过负 | Gate概率仅0.04 |
| WashingMachine | `alpha_off=0.25` 过低 | Gate饱和到0.9999 |
| WashingMachine | Recall过高(0.98) Precision过低(0.10) | 边界检测失败 |
| Fridge | `w_recall=0.28` 过高 | Recall=0.96但Precision=0.61 |
| Fridge | `gate_floor=0.05` 过高 | OFF状态21%能量泄漏 |

## 3. 优化方案

### 3.1 device_config.py 参数调整

```python
# ============ sparse_high_power (Microwave, Kettle) ============
DEVICE_TYPE_BASE_PARAMS["sparse_high_power"] = {
    "alpha_on": 12.0,           # 7.0 -> 12.0 (大幅提高ON权重)
    "alpha_off": 0.15,          # 0.3 -> 0.15 (降低OFF权重)
    "lambda_zero": 0.001,       # 0.01 -> 0.001 (降低零惩罚)
    "lambda_sparse": 0.001,     # 0.01 -> 0.001 (降低稀疏惩罚)
    "lambda_off_hard": 0.005,   # 0.02 -> 0.005 (降低OFF硬约束)
    "lambda_on_recall": 3.0,    # 1.5 -> 3.0 (提高召回权重)
    "on_recall_margin": 0.95,   # 0.9 -> 0.95 (提高召回阈值)
    "lambda_gate_cls": 0.1,
    "lambda_energy": 0.08,      # 0.03 -> 0.08 (提高能量回归)
    "off_margin": 0.005,        # 0.02 -> 0.005 (降低OFF阈值)
}

DEVICE_TYPE_GATE_CONFIG["sparse_high_power"] = {
    "gate_soft_scale": 2.0,     # 0.5 -> 2.0 (提高Gate锐度)
    "gate_floor": 0.03,         # 0.01 -> 0.03 (提高Gate下限)
    "gate_duty_weight": 0.0,
}

# ============ long_cycle (WashingMachine, Dishwasher) ============
DEVICE_TYPE_BASE_PARAMS["long_cycle"] = {
    "alpha_on": 5.0,            # 4.0 -> 5.0 (提高ON权重)
    "alpha_off": 2.5,           # 1.5 -> 2.5 (大幅提高OFF权重!)
    "lambda_zero": 0.08,        # 0.1 -> 0.08
    "lambda_sparse": 0.015,     # 0.02 -> 0.015
    "lambda_off_hard": 0.12,    # 0.08 -> 0.12 (提高OFF硬约束)
    "lambda_on_recall": 0.6,    # 0.8 -> 0.6 (降低召回权重)
    "on_recall_margin": 0.65,   # 0.6 -> 0.65
    "lambda_gate_cls": 0.2,     # 0.1 -> 0.2 (增强Gate监督)
    "lambda_energy": 0.12,      # 0.15 -> 0.12
    "off_margin": 0.015,        # 0.02 -> 0.015
}

DEVICE_TYPE_GATE_CONFIG["long_cycle"] = {
    "gate_soft_scale": 1.5,     # 1.0 -> 1.5 (提高Gate锐度)
    "gate_floor": 0.005,        # 0.02 -> 0.005 (降低Gate下限!)
    "gate_duty_weight": 0.02,   # 0.0 -> 0.02
}

# ============ cycling_low_power (Fridge) ============
DEVICE_TYPE_BASE_PARAMS["cycling_low_power"] = {
    "alpha_on": 2.8,            # 3.0 -> 2.8
    "alpha_off": 2.0,           # 保持
    "lambda_zero": 0.02,
    "lambda_sparse": 0.005,
    "lambda_off_hard": 0.25,    # 0.12 -> 0.25 (大幅提高OFF硬约束!)
    "lambda_on_recall": 0.8,    # 1.0 -> 0.8 (降低召回权重)
    "on_recall_margin": 0.6,    # 0.7 -> 0.6
    "lambda_gate_cls": 0.18,    # 0.15 -> 0.18
    "lambda_energy": 0.2,
    "off_margin": 0.008,        # 0.02 -> 0.008 (收紧OFF margin)
}

DEVICE_TYPE_GATE_CONFIG["cycling_low_power"] = {
    "gate_soft_scale": 2.5,     # 2.0 -> 2.5 (提高Gate锐度)
    "gate_floor": 0.008,        # 0.01 -> 0.008 (降低Gate下限)
    "gate_duty_weight": 0.02,
}
```

### 3.2 trainer.py 多设备训练修复

关键修改：在多设备训练时也启用设备特定参数

```python
# 修改 trainer.py 中的逻辑
# 将 allow_manual_tuning 的条件从 "单设备时" 改为 "始终启用"

# 原代码:
# allow_manual_tuning = len(self.appliance_names) == 1

# 修改为:
allow_manual_tuning = True  # 始终启用设备特定调参
```

### 3.3 expes.yaml 后处理参数调整

```yaml
postprocess_per_device:
  microwave:
    postprocess_threshold: 100    # 200 -> 100 (降低阈值)
    postprocess_min_on_steps: 1

  Kettle:
    postprocess_threshold: 20
    postprocess_min_on_steps: 2

  Fridge:
    postprocess_threshold: 25     # 18 -> 25 (提高阈值过滤噪声)
    postprocess_min_on_steps: 4   # 3 -> 4

  WashingMachine:
    postprocess_threshold: 25     # 20 -> 25
    postprocess_min_on_steps: 8   # 5 -> 8 (增加最小ON时长)

  Dishwasher:
    postprocess_threshold: 15     # 20 -> 15 (降低阈值捕获低功率阶段)
    postprocess_min_on_steps: 8   # 6 -> 8
```

## 4. Optuna搜索空间建议

### 4.1 Microwave专用搜索空间

```yaml
Microwave_HPO:
  loss_alpha_on:
    type: loguniform
    low: 8.0
    high: 15.0
  loss_alpha_off:
    type: loguniform
    low: 0.1
    high: 0.5
  loss_lambda_on_recall:
    type: loguniform
    low: 2.5
    high: 5.0
  gate_floor:
    type: loguniform
    low: 0.02
    high: 0.08
  gate_soft_scale:
    type: uniform
    low: 1.5
    high: 3.5
  postprocess_threshold:
    type: int
    low: 80
    high: 150
```

### 4.2 WashingMachine专用搜索空间

```yaml
WashingMachine_HPO:
  loss_alpha_on:
    type: loguniform
    low: 4.0
    high: 8.0
  loss_alpha_off:
    type: loguniform
    low: 2.0
    high: 4.0     # 关键：提高OFF权重范围
  loss_lambda_on_recall:
    type: loguniform
    low: 0.4
    high: 1.5
  gate_floor:
    type: loguniform
    low: 0.003
    high: 0.015
  gate_soft_scale:
    type: uniform
    low: 1.2
    high: 2.5
  postprocess_min_on_steps:
    type: int
    low: 6
    high: 12
```

### 4.3 Fridge专用搜索空间

```yaml
Fridge_HPO:
  loss_lambda_off_hard:
    type: loguniform
    low: 0.15
    high: 0.4
  loss_lambda_on_recall:
    type: loguniform
    low: 0.5
    high: 1.2
  gate_floor:
    type: loguniform
    low: 0.005
    high: 0.02
  gate_soft_scale:
    type: uniform
    low: 2.0
    high: 3.5
  gate_bias:
    type: uniform
    low: -3.5
    high: -1.5
```

## 5. 实施步骤

### 第一步：应用基础参数修改 (P0)
1. 修改 `device_config.py` 中的 `DEVICE_TYPE_BASE_PARAMS`
2. 修改 `device_config.py` 中的 `DEVICE_TYPE_GATE_CONFIG`
3. 修改 `trainer.py` 启用多设备时的设备特定参数

### 第二步：验证修改效果
```bash
python scripts/run_experiment.py \
    --dataset UKDALE \
    --appliance "Kettle,Fridge,WashingMachine,Dishwasher,Microwave" \
    --name_model NILMFormer \
    --sampling_rate 1min \
    --window_size 256 \
    --epochs 25
```

### 第三步：针对问题设备运行Optuna优化
```bash
# Microwave优化 (最需要)
python scripts/run_optuna_search.py \
    --dataset UKDALE \
    --appliance Microwave \
    --n_trials 50 \
    --study_name "Microwave_optimized"

# WashingMachine优化
python scripts/run_optuna_search.py \
    --dataset UKDALE \
    --appliance WashingMachine \
    --n_trials 30 \
    --study_name "WashingMachine_optimized"
```

### 第四步：应用最优参数并重新训练多设备模型

## 6. 预期改进

| 设备 | 当前多设备F1 | 预期F1 | 改进幅度 |
|------|-------------|--------|----------|
| Microwave | 0.08 | 0.35+ | +337% |
| WashingMachine | 0.55 | 0.70+ | +27% |
| Fridge | 0.75 | 0.82+ | +9% |
| Dishwasher | 0.83 | 0.85+ | +2% |
| **整体平均** | 0.55 | 0.68+ | +24% |

## 7. 关键文件路径

- 设备配置: `src/helpers/device_config.py`
- 训练器: `src/helpers/trainer.py`
- 实验配置: `configs/expes.yaml`
- 数据集参数: `configs/dataset_params.yaml`
- HPO搜索空间: `configs/hpo_search_spaces.yaml`
