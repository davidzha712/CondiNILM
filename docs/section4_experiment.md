# 第四部分：实验设计、实现细节与结果产出（论文级复现实验说明）

本节给出 CondiNILM 仓库中“实验如何被定义、如何被执行、如何被记录与评估”的可复现描述。内容严格对应现有配置与实现：配置入口为 `configs/*.yaml`，运行入口为 `scripts/run_one_expe.py`，训练/评估骨架位于 `src/helpers/expes.py` 与 `src/helpers/trainer.py`，数据缩放与缓存位于 `src/helpers/dataset.py` 与 `scripts/run_one_expe.py`。

## 1. 实验目标与对比设置

- 任务定义：输入聚合功率序列（aggregate），输出目标设备的功率序列（power）与/或状态序列（state），以完成 NILM 负荷分解与设备开关检测。
- 研究目标：在统一数据预处理、统一训练流程、统一评估指标下，对比 NILMFormer 与多类 NILM/TSER 基线在不同数据集、不同设备、不同窗口长度下的性能差异，并验证多设备联合学习（Multi）在稀疏设备与长周期设备上的泛化能力。
- 对比维度：
  - 数据集维度：UKDALE / REFIT / REDD。
  - 设备维度：不同电气行为（稀疏高功率、低功率周期、频繁开关、长周期、近似常开等）。
  - 时序设置：采样率（如 30s、1min）、窗口长度（如 128/256/512 或 day/week/month）、窗口重叠比例 overlap。
  - 学习范式：单设备训练 vs 多设备联合训练（`--appliance multi` 或 `--appliance a,b,c`）。

## 2. 配置体系与加载顺序（对应你文件第 12 行的更准确版本）

### 2.1 配置文件职责划分

- `configs/expes.yaml`：全局训练默认参数与通用实验开关（如 `epochs`、`batch_size`、早停 `p_es`、学习率 warmup、`output_ratio`、缓存/结果路径字段等）。
- `configs/models.yaml`：各模型的结构与训练超参（如对应 `model_training_param.lr/wd`、模型特有参数等），以“模型名”为键。
- `configs/datasets.yaml`：每个数据集下每个设备（appliance）的房屋划分（`ind_house_train_val` / `ind_house_test`）、阈值与数据路径相关配置，以“数据集名→设备名”为键。
- `configs/dataset_params.yaml`：dataset 级与 appliance 级的补充参数（训练/损失的默认值、设备阈值范围、on/off 最小持续步数、以及部分设备的 `device_type` 先验等），在运行时由 `DatasetParamsManager` 提供默认覆盖与校验逻辑。

### 2.2 run_one_expe.py 的合并/覆盖优先级

在 `scripts/run_one_expe.py` 中，实验配置以“字典 update”的方式合并，优先级（后者覆盖前者）为：

- 先载入全局默认：读取 `configs/expes.yaml` 得到 `expes_config`。
- 再合并模型超参：读取 `configs/models.yaml`，用选定模型条目覆盖 `expes_config`（`expes_config.update(baselines_config[model_key])`）。
- 再合并数据集/设备配置：读取 `configs/datasets.yaml`，用选定设备条目覆盖 `expes_config`（单设备直接 `update`；多设备会构造一个合并后的 `base_entry`，并设置 `app` 列表与 `appliance_group_members`）。
- 最后应用命令行覆盖：`--loss_type/--overlap/--epochs/--batch_size/--ind_house_*` 会再次覆盖同名字段。

此外，在进入 `launch_one_experiment(expes_config)` 后，会由 `DatasetParamsManager` 将 `configs/dataset_params.yaml` 中的 dataset 级默认训练参数/损失参数补齐到 `expes_config`（仅在缺失时补齐训练参数；loss 参数会写入并可在缓存加载后再次写入）。

## 3. 数据准备、数据格式与切片逻辑

### 3.1 数据目录与 DataBuilder

- 默认数据根目录为 `data/`，并期望存在子目录 `data/UKDALE/`、`data/REFIT/`、`data/REDD/`。
- 各数据集的数据读取与窗口切片由 `src/helpers/preprocessing.py` 中的 DataBuilder 族完成，并在 `scripts/run_one_expe.py` 中按 `expes_config.dataset` 分支调用（UKDALE/REFIT/REDD 各自路径与字段一致性由对应 DataBuilder 处理）。

### 3.2 NILM 统一张量格式（训练输入/标签）

预处理后 NILM 数据统一为 4D 张量（numpy）：

- 形状：`[N_windows, 1 + M, 2, window_size]`
  - 第 2 维 `1 + M`：第 0 通道固定为 aggregate，其余通道为目标设备（单设备时 M=1，多设备联合训练时 M>1）。
  - 第 3 维 `2`：`0=power`、`1=state`。
  - 第 4 维：窗口时间长度 `window_size`。

这个约定与 `src/helpers/dataset.py` 的 `NILMscaler` 一致：它假设 `data[:, 0, 0, :]` 是聚合功率，`data[:, n_app, 0, :]` 是第 n 个设备功率，并对其做缩放。

### 3.3 窗口切片与 overlap→stride

- `overlap=0`：无重叠，步长 `window_stride = window_size`。
- `0 < overlap < 1`：部分重叠，步长为 `window_stride = round(window_size * (1 - overlap))`，并做最小值保护（至少为 1）。
- 该设置在 `scripts/run_one_expe.py` 中用于构造 DataBuilder（例如 UKDALE 分支会将 `window_stride` 传给 `UKDALE_DataBuilder`），从而决定样本数量与样本间相关性。

### 3.4 训练/验证/测试划分（避免时间泄漏）

实验划分优先使用房屋（house）级别的隔离：

- 先按 `configs/datasets.yaml` 中配置的 `ind_house_train_val` 与 `ind_house_test` 构造 train/test 房屋集合。
- 再在 train 房屋内切出验证集：对于无重叠窗口，使用固定比例（例如 20%）切分；对于重叠窗口，使用时间块切分函数（`split_train_valid_timeblock_nilmdataset`）来减少时间泄漏与窗口强相关。

### 3.5 多设备联合训练（Multi）的一致性修正

当 `--appliance multi` 或 `--appliance a,b,c` 时，`scripts/run_one_expe.py` 会：

- 将 `expes_config.app` 设置为设备列表（用于 DataBuilder mask）。
- 设置 `expes_config.appliance="Multi"` 并记录 `appliance_group_members`，用于后续可视化与指标的按设备命名。
- 在构建数据后，会以“实际数据 shape”为准同步 `expes_config.app` 与 `c_out`，避免“配置设备名”和“数据通道”不一致影响可视化/评估。

对 REDD 的 Multi 模式，代码提供一个默认的推荐房屋划分：如果用户未指定房屋，将使用 `[1,2]` 作为 train/val、`[3]` 作为 test，以提高设备共同可用性并减少设备缺失导致的训练失败。

## 4. 缓存与归一化（保证同配置可复用）

### 4.1 缓存文件与 cache key

预处理后的数据会被缓存到 `data_cache/`，以避免重复切片与缩放。缓存 key 由多项配置组成（用于“同配置命中缓存，不同配置自动分离”）：

- 必选要素：dataset、appliance（或设备列表）、房屋划分（train/test）、sampling_rate、window_size、seed、`power_scaling_type`、`appliance_scaling_type`、overlap，以及在 DiffNILM 情况下额外追加模型名标识。
- 缓存命中时，会直接加载 `tuple_data/scaler/cutoff/threshold` 并进入训练阶段，同时会把缓存中的 `app_list` 恢复到 `expes_config.app`，确保多设备命名正确。

### 4.2 NILMscaler 的缩放策略（power 与 appliance 可不同）

`src/helpers/dataset.py` 的 `NILMscaler` 支持以下缩放方式：

- `power_scaling_type`：`StandardScaling` / `MinMax` / `MeanScaling` / `MeanMaxScaling` / `MaxScaling` 或者直接给一个整数作为固定缩放因子。
- `appliance_scaling_type`：同上，此外支持 `SameAsPower`（设备功率使用与 aggregate 相同的缩放参数）。
- 训练时先对 train 数据 `fit_transform`，再对 valid/test `transform`；评估/可视化时可用 `inverse_transform` 或 `inverse_transform_appliance` 还原到瓦特量纲。

### 4.3 cutoff 与 threshold 的一致性处理（影响 loss/后处理）

训练脚本会在缓存加载后将 `cutoff`、`threshold` 写回 `expes_config`，并执行一次“阈值归一化修正”：

- 例如 `loss_threshold` 会从瓦特单位除以 `cutoff` 变为缩放后单位；
- 部分 loss 超参（如 `loss_soft_temp_raw`、`loss_edge_eps_raw`、`loss_energy_floor_raw`、`loss_off_margin_raw`）也会按 `cutoff` 做归一化，保证“在不同缩放设置下 loss 超参仍处于合理数量级”。

## 5. 模型集合与输出通道（单设备 vs 多设备）

### 5.1 基线模型集合

`configs/models.yaml` 中定义了多类基线模型与 NILMFormer，可被 `--name_model` 选择：

- NILM 基线：BiLSTM、BiGRU、CNN1D、UNetNILM、FCN、DAResNet、DResNet、BERT4NILM、DiffNILM、TSILNet、Energformer、STNILM。
- TSER 基线：ConvNet、ResNet、Inception（以 TSER 的数据格式与 LightningModule 分支训练）。
- NILMFormer：`src/nilmformer/` 下的实现，通过 `NILMFormerConfig` + `NILMFormer` 构建。

### 5.2 动态输出通道 c_out

当 `expes_config.app` 是列表（多设备）时，`scripts/run_one_expe.py` 会调用 `get_dynamic_output_channels(app_list, dataset_name, params_manager)`，并将结果写入 `expes_config["c_out"]`，用于模型最后输出维度与损失函数维度对齐。

## 6. 训练流程（Lightning 训练骨架、采样策略与 checkpoint）

### 6.1 训练入口与训练骨架

- 数据准备完成后，`scripts/run_one_expe.py` 调用 `launch_models_training(tuple_data, scaler, expes_config)`。
- `src/helpers/expes.py` 负责：
  - 构建 `NILMDataset`（或 TSER/DiffNILM/STNILM 对应的数据封装）。
  - 构建 DataLoader，并根据操作系统设置 `num_workers`（Windows 默认会退化到 `0`，避免多进程数据加载问题）。
  - 构建回调：验证指标写入与可视化 HTML 产出。
  - 构建 LightningModule：根据模型名与 `loss_type` 选择不同训练模块（Seq2Seq / TSER / DiffNILM / STNILM）。

### 6.2 稀疏设备的“窗口平衡采样”（防止全 OFF 崩溃）

当满足以下条件时，训练会启用 `WeightedRandomSampler` 对“含 ON 事件窗口”进行过采样：

- `balance_window_sampling=True`（默认值由配置决定）。
- `name_model == NILMFormer` 且 `loss_type == multi_nilm`。
- 训练集中 `on_window_frac`（窗口内状态和>0 的比例）小于某个阈值（`balance_window_on_frac_threshold`）。

采样权重会被设置为让 on-window 频率接近 `balance_window_target_on_frac`，并通过 `balance_window_max_ratio` 限制过采样比，避免训练被少数窗口主导。

### 6.3 早停与 checkpoint 命名规则

- 早停：若 `p_es` 不为空，则监控 `val_loss` 并按 `patience=p_es` 早停。
- checkpoint：保存在
  - `checkpoint/{dataset_sampling}/{window_size}/{appliance}/{model_seed}/`
  - 其中 `{dataset_sampling}` 形如 `UKDALE_1min`，`{model_seed}` 形如 `NILMFormer_42`。
  - 保存策略为 `save_top_k=1`（按最小 val_loss）与 `save_last=True`。

## 7. 损失函数设计：loss_type 分支与 multi_nilm（核心）

### 7.1 loss_type 的可选项与默认值

在 `src/helpers/expes.py` 中，`loss_type` 默认取 `multi_nilm`，也可由命令行 `--loss_type` 覆盖。支持：

- `multi_nilm`：AdaptiveDeviceLoss（设备自适应、多设备友好、支持 seq2subseq）。
- `smoothl1`：标准 SmoothL1Loss。
- `mse`：MSELoss。
- `mae`：L1Loss。

### 7.2 multi_nilm 的“设备统计→设备类型→参数”链路

multi_nilm 依赖 `scripts/run_one_expe.py` 在数据就绪后自动计算设备统计，并写入配置：

- 若为单设备：从目标设备通道提取 power/state，计算 `duty_cycle`、`peak_power`、`mean_on`、`cv_on`、`mean_event_duration`、事件频率等，并用 `classify_device_type(...)` 输出 `device_type`。
- 若为多设备：对每个设备通道重复上述过程，构造 `device_stats_for_loss` 列表，并写入：
  - `expes_config["device_type_per_device"]`
  - `expes_config["loss_params_per_device"]`
  - `expes_config["device_stats_for_loss"]`

设备类型与基础参数映射由 `src/helpers/device_config.py` 提供（例如稀疏高功率、低功率周期、长周期等），并会派生 gate、后处理阈值、min_on_steps 等默认项。

### 7.3 AdaptiveDeviceLoss：统一结构 + 设备自适应参数

`src/helpers/trainer.py` 的 `AdaptiveDeviceLoss` 是 multi_nilm 的实现核心，关键点：

- 输入：`pred` 与 `target` 的形状对齐为 `[B, C, L]`（`C` 为设备数）。
- seq2subseq：内部对 `pred/target` 做中心裁剪（`_crop_center`），裁剪比例由 `output_ratio` 控制，用于减弱窗口边界对损失的影响。
- 设备自适应：每个设备基于统计信息决定 `device_type` 与参数集合，并用 `device_weights` 做跨设备损失加权，缓解多设备训练中“稀疏设备被忽略/常开设备主导”的问题。

### 7.4 multi_nilm 与 Trainer 辅助惩罚项的关系

为了避免“同一惩罚重复计算”，在 `loss_type == multi_nilm` 时，`src/helpers/expes.py` 会将 `SeqToSeqLightningModule` 的各类辅助惩罚权重（如 zero-run、off-state 等）直接置零，让损失的所有约束集中在 AdaptiveDeviceLoss 内部完成；在非 multi_nilm 时，这些辅助项由 `SeqToSeqLightningModule` 按配置启用。

### 7.5 multi_nilm 的参数来源与覆盖链路（从 YAML 到 AdaptiveDeviceLoss）

- 基础默认值：`configs/expes.yaml` 中的 `loss_lambda_* / loss_alpha_* / loss_off_margin / loss_on_recall_margin / gate_* / output_ratio` 是默认值，作为“全局初值”进入 `expes_config`。
- dataset 级补齐：`configs/dataset_params.yaml` 中的 `loss` 与 `appliances` 结构会在 `DatasetParamsManager` 中补齐缺省字段（如 `loss_type/output_ratio/anti_collapse_weight` 与设备先验 `device_type`）。
- 统计驱动重写：`scripts/run_one_expe.py::_configure_nilm_loss_hyperparams` 在数据就绪后计算设备统计（`duty_cycle/peak_power/mean_on/cv_on/mean_event_duration/n_events`），输出 `device_type` 并写回 `expes_config`（单设备或多设备均可），形成 `device_stats_for_loss` 与 `loss_params_per_device`。
- AdaptiveDeviceLoss 的最终参数：`src/helpers/expes.py` 将 `loss_alpha_on/off、loss_lambda_on_recall、loss_lambda_energy、gate_soft_scale、gate_floor` 等配置作为 `config_overrides` 传入 AdaptiveDeviceLoss，用于“全局缩放”；而每个设备的“局部参数”仍由统计值与 `device_type` 决定。

换言之，multi_nilm 的参数是“三层融合”的：全局默认值 + dataset 补齐 + 统计驱动修正，最后由 AdaptiveDeviceLoss 完成 per-device 参数化。

### 7.6 AdaptiveDeviceLoss 的损失结构（具体项与作用）

`AdaptiveDeviceLoss` 对每个设备的 loss 采用相同结构、不同参数，形成“稳定结构 + 自适应参数”的组合。其每设备 loss 由以下项组成：

- 主回归项（`w_main`）：在 `p_on/p_off` 的软权重下，对 ON/OFF 区间分别施加 SmoothL1，并用 `alpha_on/alpha_off` 调节不平衡性。
- 全局稳定项（`w_global`）：对全时间步的 SmoothL1 均值施加轻微约束，提升整体稳定性。
- ON 召回项（`w_recall`）：对 ON 区间“预测不足”进行惩罚，抑制全零塌缩，提升稀疏事件召回。
- OFF 误报项（`w_off_fp`）：对 OFF 区间“过大输出”施加惩罚，控制误报。
- ON 功率相对误差（`w_on_power`）：以相对误差约束 ON 区间功率，改善 NDE 与回归质量。
- 能量回归项（`w_energy`）：对窗口总能量差异进行约束，直接优化能量指标（如 MR、SAE）。

关键点是“同结构、不同参数”：在 `device_type` 与统计值的驱动下，`alpha_on/alpha_off、w_*、off_margin` 等会随设备动态变化，保证稀疏设备与长周期设备都能以合适强度优化。

对应数学定义可抽象为（逐设备、逐窗口）：

设目标功率序列为 \(y_t\)，预测为 \(\hat{y}_t\)，阈值为 \(\tau\)，则软 ON 概率
\[
p_{\text{on}}(t) = \sigma\left(\frac{y_t-\tau}{T}\right),\quad p_{\text{off}}(t)=1-p_{\text{on}}(t)
\]
其中 \(T\) 为软化温度。主回归项：
\[
\mathcal{L}_{\text{main}} = \alpha_{\text{on}}\cdot \frac{\sum_t \ell(\hat{y}_t,y_t)p_{\text{on}}(t)}{\sum_t p_{\text{on}}(t)} + \alpha_{\text{off}}\cdot \frac{\sum_t \ell(\hat{y}_t,y_t)p_{\text{off}}(t)}{\sum_t p_{\text{off}}(t)}
\]
ON 召回项（抑制漏检）：
\[
\mathcal{L}_{\text{recall}}=\frac{\sum_t \max\left(r\cdot y_t-\hat{y}_t,0\right)p_{\text{on}}(t)}{\sum_t p_{\text{on}}(t)}
\]
OFF 误报项（抑制虚警）：
\[
\mathcal{L}_{\text{off}}=\frac{\sum_t \max\left(\hat{y}_t-\delta,0\right)p_{\text{off}}(t)}{\sum_t p_{\text{off}}(t)}
\]
ON 相对功率误差：
\[
\mathcal{L}_{\text{on\_power}}=\frac{\sum_{t:y_t>\tau}\frac{|\hat{y}_t-y_t|}{y_t+\epsilon}}{\sum_{t}\mathbb{1}[y_t>\tau]}
\]
能量回归项（窗口能量差异）：
\[
\mathcal{L}_{\text{energy}}=\left|\frac{\sum_t \hat{y}_t-\sum_t y_t}{\sum_t |y_t|+\epsilon}\right|
\]
最终损失：
\[
\mathcal{L} = w_{\text{main}}\mathcal{L}_{\text{main}} + w_{\text{global}}\mathbb{E}_t[\ell(\hat{y}_t,y_t)] + w_{\text{recall}}\mathcal{L}_{\text{recall}} + w_{\text{off}}\mathcal{L}_{\text{off}} + w_{\text{on\_power}}\mathcal{L}_{\text{on\_power}} + w_{\text{energy}}\mathcal{L}_{\text{energy}}
\]

对应实现片段：

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

### 7.7 设备类型与参数派生（device_config + 统计驱动）

设备类型与基础参数由 `src/helpers/device_config.py` 提供，包含 `DEVICE_TYPE_BASE_PARAMS`、`DEVICE_TYPE_GATE_CONFIG`、`DEVICE_TYPE_ZERO_PENALTY_CONFIG`、`DEVICE_TYPE_OFF_PENALTY_CONFIG` 等“类型→参数”映射。其核心逻辑为：

- 通过 `classify_device_type(...)` 用统计量将设备归为 `sparse_high_power / cycling_low_power / cycling_infrequent / frequent_switching / long_cycle / always_on / sparse_medium_power` 等类型。
- 用 `get_device_loss_params(...)` 为特定类型（如低 duty 的 cycling/long_cycle）调整 `lambda_gate_cls/off_margin` 等细节，避免误报或过度抑制。
- 通过 gate 配置与后处理最小持续步数修正，保障对不同设备行为的容错（短脉冲 vs 长持续）。

### 7.8 AdaptiveDeviceLoss 内部的“统计驱动 + 手工纠偏”

在 `src/helpers/trainer.py` 内部，AdaptiveDeviceLoss 的 `_classify_and_derive_params` 除了“统计驱动”外，还对特定设备名进行经验性纠偏（如 `kettle/microwave/fridge/dishwasher/washingmachine`），调整 `w_recall/w_energy/alpha_on/alpha_off/gate_*` 等参数，以解决：

- 稀疏设备 recall 低导致漏检；
- 长周期设备功率回归质量不足；
- gate 过强导致输出过于稀疏；
- OFF 误报过多导致精度下降。

这类“名称纠偏”并不替代统计驱动，而是针对历史实验中表现不稳定的设备进行补偿，体现“实践调优”与“统计自适应”并存的策略。

### 7.9 反塌缩与自适应调参（训练过程中的 best practice）

除了 multi_nilm 本身，训练流程还包含防塌缩与自适应调参机制：

- `anti_collapse_weight`（见 `configs/expes.yaml`）在 `SeqToSeqLightningModule` 内部用于强制避免全零预测，包含能量比率与 ON 召回双重惩罚。
- `src/helpers/loss_tuning.py` 的 `AdaptiveLossTuner` 会基于验证集指标（如 OFF 非零率、OFF 长段比例、Recall）动态调整 `lambda_off_hard/off_margin/gate_floor/lambda_on_recall` 等，重点针对 cycling 类设备。
- 早期塌缩恢复逻辑：若早期 epoch 检测到“预测全零”，会降低过强的惩罚并抬高 gate floor，帮助模型重新学习。

以上机制共同构成“静态自适应（统计驱动）+ 动态自适应（训练反馈）”的损失设计范式，使 multi_nilm 在稀疏设备与周期设备上更稳定。

### 7.10 参数与配置示例（对应 YAML 与关键代码）

`configs/expes.yaml` 中的 loss 关键默认值示例：

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

AdaptiveDeviceLoss 的初始化与 config_overrides 传递：

```python
config_overrides = {}
lambda_energy = float(getattr(expes_config, "loss_lambda_energy", 1.0))
if lambda_energy > 0.0:
    config_overrides["energy_weight_scale"] = lambda_energy
alpha_on = float(getattr(expes_config, "loss_alpha_on", 1.0))
if alpha_on > 0.0:
    config_overrides["alpha_on_scale"] = alpha_on / 2.0
alpha_off = float(getattr(expes_config, "loss_alpha_off", 1.0))
if alpha_off > 0.0:
    config_overrides["alpha_off_scale"] = alpha_off
lambda_recall = float(getattr(expes_config, "loss_lambda_on_recall", 1.0))
if lambda_recall > 0.0:
    config_overrides["recall_weight_scale"] = lambda_recall
gate_soft_scale = getattr(expes_config, "gate_soft_scale", None)
if gate_soft_scale is not None:
    config_overrides["gate_soft_scale"] = float(gate_soft_scale)
gate_floor = getattr(expes_config, "gate_floor", None)
if gate_floor is not None:
    config_overrides["gate_floor"] = float(gate_floor)
criterion = AdaptiveDeviceLoss(
    n_devices=n_app,
    device_stats=device_stats,
    warmup_epochs=warmup_epochs,
    output_ratio=output_ratio,
    config_overrides=config_overrides if config_overrides else None,
)
```

## 8. 后处理策略（阈值抑制与最小持续步数）

- 后处理的基本思想：将网络输出映射到二值状态，并抑制短暂毛刺与噪声。
- 设备类型驱动：`apply_device_type_config_defaults(...)` 会为不同 device_type 给出合理的 `postprocess_threshold` 与 `postprocess_min_on_steps`，并对 cycling 设备给出不同 gate/penalty 默认值。
- 多设备时的 per-device：当存在 `device_stats_for_loss` 时，会构造 `postprocess_per_device` 字典，为每个设备写入独立阈值与最小 ON 步数，使稀疏高功率设备与低功率周期设备能使用不同后处理强度。

## 9. 评估指标体系与记录格式

### 9.1 回归、能量与分类指标

指标实现位于 `src/helpers/metrics.py` 的 `NILMmetrics`，包含：

- 回归指标：MAE、MSE、RMSE。
- 能量类指标：TECA、NDE、SAE、MR（与论文写法常用的能量分配/归一化误差一致）。
- 分类指标：Accuracy、Balanced Accuracy、Precision、Recall、F1（通过状态序列 `state` 与预测状态对齐后计算）。

### 9.2 window 级与逐时刻级评估

仓库同时支持：

- 逐时刻（timestamp）级的误差度量：直接在展开的时间轴上计算。
- window 级的能量聚合评估：例如 `eval_win_energy_aggregation` 会在窗口维度聚合后对能量分配质量做评估，用于减少窗口拼接造成的噪声。

### 9.3 结果记录文件

训练过程中会通过回调产出：

- `val_report.jsonl`：每个 epoch 追加/覆盖一条验证记录（避免同 epoch 重复），包含整体指标与 per-device 指标。
- `val_compare.html`：验证集曲线可视化（支持选择设备、模型、epoch），便于检查模型是否出现“全零塌缩/过度平滑/严重误报”等现象。

## 10. 实验执行方式（单次与批量）

### 10.1 单次实验（单设备）

示例（以 UKDALE 为例）：

```bash
uv run -m scripts.run_one_expe \
  --dataset UKDALE \
  --sampling_rate 1min \
  --window_size 256 \
  --appliance Kettle \
  --name_model NILMFormer \
  --seed 0
```

### 10.2 单次实验（多设备）

示例（选择多个设备联合训练）：

```bash
uv run -m scripts.run_one_expe \
  --dataset UKDALE \
  --sampling_rate 1min \
  --window_size 256 \
  --appliance "Kettle,Microwave,Fridge" \
  --name_model NILMFormer \
  --loss_type multi_nilm \
  --seed 0
```

### 10.3 批量实验

- 全量组合脚本：`scripts/run_all_expe.sh`，会遍历 dataset×appliance×window×model×seed 的组合，并调用 `uv run -m scripts.run_one_expe`。
- 数据集专用脚本：
  - `scripts/run_ukdale_all.py`：对 UKDALE 的所有设备跑同一配置。
  - `scripts/run_redd_all.py`：对 REDD 的设备列表按预置房屋组合批量运行。

## 11. 复现清单（论文写作可直接引用）

- 数据集与设备：UKDALE/REFIT/REDD；设备名与组合方式（单设备 vs Multi）。
- 数据切片：采样率、窗口长度、overlap、window_stride 计算方式。
- 数据划分：train/val/test 的房屋划分与时间块切分策略。
- 缩放方式：`power_scaling_type`、`appliance_scaling_type`、cutoff/threshold 的归一化规则。
- 训练设置：batch_size、epochs、early stopping、学习率与 scheduler、随机种子、是否启用窗口平衡采样。
- 损失函数：`loss_type` 与 multi_nilm 的设备统计与参数生成方式、`output_ratio`（seq2subseq）的中心监督比例。
- 输出与评估：checkpoint 目录规则、`val_report.jsonl` 与 `val_compare.html` 的生成与内容、指标集合与按设备记录方式。
