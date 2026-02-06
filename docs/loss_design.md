# CondiNILM Loss 设计与计算详解

本文聚焦当前仓库中默认 loss（`multi_nilm` / `AdaptiveDeviceLoss`）的设计动机、参数来源、数学形式与计算流程，给出与代码一致的公式与实现片段。

## 1. 总体设计目标

- 兼顾功率回归与状态检测，避免仅优化分类或仅优化回归导致的偏差。
- 针对不同设备形态（稀疏高功率、周期性低功率、长周期等）使用不同超参，而不是用单一损失全局硬套。
- 通过统计量自动派生参数，减少人工调参负担。
- 在多设备联合训练中抑制“稀疏设备被忽略/常开设备主导”的现象。

## 2. loss_type 与默认入口

默认 loss 为 `multi_nilm`，由 `src/helpers/expes.py` 选择并实例化 `AdaptiveDeviceLoss`。同时将其它训练模块中的辅助惩罚项置零，避免重复约束。

```python
if loss_type == "multi_nilm":
    criterion = AdaptiveDeviceLoss(
        n_devices=n_app,
        device_stats=device_stats,
        warmup_epochs=warmup_epochs,
        output_ratio=output_ratio,
        config_overrides=config_overrides if config_overrides else None,
    )
    state_zero_penalty_weight = 0.0
    zero_run_kernel = 0
    zero_run_ratio = 0.0
    off_high_agg_penalty_weight = 0.0
    off_state_penalty_weight = 0.0
    off_state_margin = 0.0
    off_state_long_penalty_weight = 0.0
    off_state_long_kernel = 0
    off_state_long_margin = 0.0
```

## 3. 参数来源与覆盖链路

loss 参数是多层融合的结果：

1) `configs/expes.yaml` 提供全局默认值  
2) `configs/dataset_params.yaml` 提供 dataset 级默认项与设备先验  
3) `scripts/run_one_expe.py::_configure_nilm_loss_hyperparams` 基于统计量生成 `device_stats_for_loss` 与 `loss_params_per_device`  
4) `src/helpers/expes.py` 将部分全局超参作为 `config_overrides` 缩放到 AdaptiveDeviceLoss  
5) `AdaptiveDeviceLoss` 对每个设备基于统计量与 `device_type` 派生最终参数

## 4. 设备统计量与类型分类

统计量由数据中功率/状态直接计算：

- `duty_cycle`：ON 占比  
- `peak_power` / `mean_on` / `cv_on`  
- `mean_event_duration` / `n_events`

分类函数为 `src/helpers/device_config.py::classify_device_type`，将设备归为：

- `sparse_high_power`
- `cycling_low_power`
- `cycling_infrequent`
- `frequent_switching`
- `long_cycle`
- `always_on`
- `sparse_medium_power`

这一分类决定参数初始区间与 gate/后处理默认值。

## 5. seq2subseq 的监督区域

为了削弱窗口边界效应，loss 只在中心区域监督：

```python
def _crop_center(self, x, ratio):
    if ratio >= 1.0:
        return x
    L = x.shape[-1]
    crop_len = int(L * ratio)
    crop_len = max(1, crop_len)
    start = (L - crop_len) // 2
    end = start + crop_len
    return x[..., start:end]
```

`output_ratio` 在 `expes.yaml` 配置，默认 0.627。

## 6. 每设备损失结构与数学形式

设目标功率序列为 \(y_t\)，预测为 \(\hat{y}_t\)，阈值为 \(\tau\)，软 ON 概率：

\[
p_{\text{on}}(t)=\sigma\left(\frac{y_t-\tau}{T}\right),\quad p_{\text{off}}(t)=1-p_{\text{on}}(t)
\]

主回归项（SmoothL1 按 ON/OFF 分别加权）：

\[
\mathcal{L}_{\text{main}}=\alpha_{\text{on}}\cdot \frac{\sum_t \ell(\hat{y}_t,y_t)p_{\text{on}}(t)}{\sum_t p_{\text{on}}(t)}+\alpha_{\text{off}}\cdot \frac{\sum_t \ell(\hat{y}_t,y_t)p_{\text{off}}(t)}{\sum_t p_{\text{off}}(t)}
\]

ON 召回项：

\[
\mathcal{L}_{\text{recall}}=\frac{\sum_t \max(r\cdot y_t-\hat{y}_t,0)p_{\text{on}}(t)}{\sum_t p_{\text{on}}(t)}
\]

OFF 误报项：

\[
\mathcal{L}_{\text{off}}=\frac{\sum_t \max(\hat{y}_t-\delta,0)p_{\text{off}}(t)}{\sum_t p_{\text{off}}(t)}
\]

ON 相对功率误差：

\[
\mathcal{L}_{\text{on\_power}}=\frac{\sum_{t:y_t>\tau}\frac{|\hat{y}_t-y_t|}{y_t+\epsilon}}{\sum_t \mathbb{1}[y_t>\tau]}
\]

能量回归项：

\[
\mathcal{L}_{\text{energy}}=\left|\frac{\sum_t \hat{y}_t-\sum_t y_t}{\sum_t |y_t|+\epsilon}\right|
\]

最终损失：

\[
\mathcal{L}=w_{\text{main}}\mathcal{L}_{\text{main}}+w_{\text{global}}\mathbb{E}_t[\ell(\hat{y}_t,y_t)]+w_{\text{recall}}\mathcal{L}_{\text{recall}}+w_{\text{off}}\mathcal{L}_{\text{off}}+w_{\text{on\_power}}\mathcal{L}_{\text{on\_power}}+w_{\text{energy}}\mathcal{L}_{\text{energy}}
\]

## 7. 与代码一致的计算流程

`AdaptiveDeviceLoss._compute_cycling_loss` 的结构如下：

```python
def _compute_cycling_loss(self, pred, target, params):
    threshold = params["threshold"]
    alpha_on = params["alpha_on"]
    alpha_off = params["alpha_off"]
    soft_temp = max(threshold * 2.0, 0.02)
    p_on = torch.sigmoid((target - threshold) / soft_temp)
    p_off = 1.0 - p_on
    point_loss = self.base_loss(pred, target)
    loss_on = (point_loss * p_on).sum() / (p_on.sum() + 1e-6)
    loss_off = (point_loss * p_off).sum() / (p_off.sum() + 1e-6)
    loss_main = alpha_on * loss_on + alpha_off * loss_off
    loss_global = point_loss.mean()
    w_recall = float(params.get("w_recall", 0.1))
    recall_coef = 0.15 + 1.5 * w_recall
    on_deficit = torch.relu(recall_coef * target - pred) * p_on
    on_recall_loss = on_deficit.sum() / (p_on.sum() + 1e-6)
    off_margin = float(params.get("off_margin", 0.01))
    off_fp_loss = torch.relu(pred - off_margin) * p_off
    off_fp_loss = off_fp_loss.sum() / (p_off.sum() + 1e-6)
    on_mask = (target > threshold).float()
    on_power_loss = torch.abs(pred - target) / (target + 1e-6) * on_mask
    on_power_loss = on_power_loss.sum() / (on_mask.sum() + 1e-6)
    pred_energy = pred.sum(dim=-1)
    target_energy = target.sum(dim=-1)
    energy_loss = torch.abs(pred_energy - target_energy) / (target_energy.abs() + 1e-6)
    w_main = float(params.get("w_main", 0.7))
    w_global = float(params.get("w_global", 0.1))
    w_off_fp = float(params.get("w_off_fp", 0.1))
    w_on_power = float(params.get("w_on_power", 0.03))
    w_energy = float(params.get("w_energy", 0.1))
    return (w_main * loss_main + w_global * loss_global +
            w_recall * on_recall_loss + w_off_fp * off_fp_loss +
            w_on_power * on_power_loss + w_energy * energy_loss)
```

## 8. 设备权重与多设备平衡

多设备联合训练时，`device_weights` 会根据设备类型与 duty cycle 生成，并进行归一化，避免稀疏设备或常开设备主导训练。

\[
\sum_{c=1}^{C} w_c = C,\quad \mathcal{L}_{\text{batch}}=\frac{\sum_c w_c \mathcal{L}_c}{\sum_c w_c}
\]

## 9. 动态调参与反塌缩机制

除了 AdaptiveDeviceLoss 自身，还存在基于验证指标的动态调参：

- `AdaptiveLossTuner` 会根据 OFF 非零率、OFF 长段比例、Recall 等指标调整 `lambda_off_hard/off_margin/gate_floor/lambda_on_recall`。
- 早期若检测到“预测全零”，会降低过强惩罚并提升 `gate_floor` 以恢复学习。

## 10. 配置示例

```yaml
loss_type: multi_nilm
output_ratio: 0.6273735795474594
loss_lambda_off_hard: 0.04867828600398974
loss_lambda_sparse: 0.003128418891709596
loss_off_margin: 0.02
loss_lambda_on_recall: 2.0672824354689343
loss_alpha_on: 3.0738331012839955
loss_alpha_off: 0.24591256383397345
loss_on_recall_margin: 0.759652616275792
loss_lambda_energy: 1.105851072569646
gate_soft_scale: 1.0563694013090654
gate_floor: 0.011203254019178552
```

## 11. 关键代码位置索引

- AdaptiveDeviceLoss 结构与计算：`src/helpers/trainer.py`
- loss 初始化与 config_overrides：`src/helpers/expes.py`
- 设备统计与类型分类：`scripts/run_one_expe.py`、`src/helpers/device_config.py`
- 动态调参逻辑：`src/helpers/loss_tuning.py`
