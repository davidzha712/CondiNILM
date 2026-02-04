# CondiNILM

面向 NILM（Non-Intrusive Load Monitoring，非侵入式负荷分解）的研究代码库，包含：
- 统一的训练/评估流程（单设备与 Multi 多设备联合学习）
- NILMFormer 与多种 NILM / TSER 基线模型
- 预处理、指标评估、可视化与实验脚本

**Python**：建议 3.10+（已在 Windows 环境下以 3.12 运行）

## 安装（仅支持原生 Python）

本项目**不支持 uv 安装**。请使用 **python + venv + pip**。

### 1) 创建虚拟环境

Windows（PowerShell）：
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Linux/macOS（bash/zsh）：
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) 安装依赖

先安装 PyTorch（请根据你的 CUDA/CPU 环境选择官方命令）：
```bash
pip install -U torch torchvision torchaudio
python -c "import torch; print(torch.__version__)"
```

然后安装其余 Python 依赖（覆盖训练、评估与脚本所需的常用依赖）：
```bash
pip install -U numpy pandas scipy scikit-learn tqdm pyyaml omegaconf matplotlib tensorboard
pip install -U pytorch-lightning lightning
pip install -U streamlit optuna optuna-dashboard
```

## 数据准备

默认数据根目录来自 [expes.yaml](file:///c:/Users/Workstation/Workspace/CondiNILM/configs/expes.yaml) 的 `data_path`（默认 `data/`）。

- UKDALE：脚本会读取 `${data_path}/UKDALE/house_1`、`house_2`… 这样的目录结构
- REFIT：脚本会读取 `${data_path}/REFIT/RAW_DATA_CLEAN/` 下的清洗 CSV

如果你的数据不在上述路径，请修改 `configs/expes.yaml` 的 `data_path`。

## 运行实验

入口脚本为 [run_one_expe.py](file:///c:/Users/Workstation/Workspace/CondiNILM/scripts/run_one_expe.py)，参数以 `-h` 输出为准：
```bash
python scripts/run_one_expe.py -h
```

### 单设备训练示例

Windows（PowerShell）：
```bash
python scripts/run_one_expe.py ^
  --dataset UKDALE ^
  --sampling_rate 1min ^
  --window_size 128 ^
  --appliance WashingMachine ^
  --name_model NILMFormer
```

Linux/macOS（bash/zsh）：
```bash
python scripts/run_one_expe.py \
  --dataset UKDALE \
  --sampling_rate 1min \
  --window_size 128 \
  --appliance WashingMachine \
  --name_model NILMFormer
```

### Multi（多设备）训练示例
`--appliance` 支持 `multi` 或逗号分隔的设备列表：

Windows（PowerShell）：
```bash
python scripts/run_one_expe.py ^
  --dataset UKDALE ^
  --sampling_rate 1min ^
  --window_size 128 ^
  --appliance Kettle,Fridge ^
  --name_model NILMFormer
```

Linux/macOS（bash/zsh）：
```bash
python scripts/run_one_expe.py \
  --dataset UKDALE \
  --sampling_rate 1min \
  --window_size 128 \
  --appliance Kettle,Fridge \
  --name_model NILMFormer
```

## 输出与可视化

输出根目录来自 `configs/expes.yaml` 的 `result_path`（默认 `result/`）。

典型输出包括：
- `val_compare.html`：验证集曲线可视化（支持选择设备、模型、epoch）
- `val_report.jsonl`：每个 epoch 一条验证记录，包含整体指标与 per-device 指标

此外，可以用 Streamlit 查看保存的验证片段：
```bash
streamlit run scripts/streamlit_val_viewer.py
```

## 目录结构（当前仓库真实结构）

```text
.
├── assets/                  # 图片资源
├── configs/                 # 实验配置（YAML）
├── scripts/                 # 运行脚本（run_one_expe 等）
├── src/
│   ├── baselines/           # NILM / TSER 基线模型
│   ├── helpers/             # 训练、评估、预处理、指标等
│   └── nilmformer/          # NILMFormer 实现
├── environment_win.yaml     # 历史环境记录（不作为安装入口）
├── environment_mac.yaml     # 历史环境记录（不作为安装入口）
└── README.md
```

## 说明

本仓库偏研究用途：代码与配置以可复现实验和快速迭代为目标，默认不提供“一键安装”打包发布形式。
