#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Experiments Helpers

#
#################################################################################################################

import os
import torch
import logging
import platform
import numpy as np
import json
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

from src.helpers.trainer import (
    EAECLoss,
    GAEAECLoss,
    SeqToSeqLightningModule,
    TserLightningModule,
    DiffNILMLightningModule,
    STNILMLightningModule,
)
from src.helpers.dataset import NILMDataset, TSDatasetScaling
from src.helpers.metrics import NILMmetrics, eval_win_energy_aggregation
from src.helpers.loss_tuning import AdaptiveLossTuner


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==== SotA NILM baselines ==== #
# Recurrent-based
from src.baselines.nilm.bilstm import BiLSTM
from src.baselines.nilm.bigru import BiGRU

# Conv-based
from src.baselines.nilm.fcn import FCN
from src.baselines.nilm.cnn1d import CNN1D
from src.baselines.nilm.unetnilm import UNetNiLM
from src.baselines.nilm.dresnets import DAResNet, DResNet
from src.baselines.nilm.diffnilm import DiffNILM
from src.baselines.nilm.tsilnet import TSILNet

# Transformer-based
from src.baselines.nilm.bert4nilm import BERT4NILM
from src.baselines.nilm.stnilm import STNILM
from src.baselines.nilm.energformer import Energformer


# ==== SotA TSER baselines ==== #
from src.baselines.tser.convnet import ConvNet
from src.baselines.tser.resnet import ResNet
from src.baselines.tser.inceptiontime import Inception

# ==== NILMFormer ==== #
from src.nilmformer.congif import NILMFormerConfig
from src.nilmformer.model import NILMFormer


def get_device():
    system = platform.system()
    if system == "Darwin":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if system in ["Windows", "Linux"]:
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            return "cuda"
        return "cpu"
    return "cpu"


def get_model_instance(name_model, c_in, window_size, **kwargs):
    """
    Get model instances
    """
    if name_model == "BiGRU":
        inst = BiGRU(c_in=c_in, **kwargs)
    elif name_model == "BiLSTM":
        inst = BiLSTM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "CNN1D":
        inst = CNN1D(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "UNetNILM":
        inst = UNetNiLM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "FCN":
        inst = FCN(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "BERT4NILM":
        inst = BERT4NILM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "STNILM":
        inst = STNILM(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "DResNet":
        inst = DResNet(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "DAResNet":
        inst = DAResNet(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "DiffNILM":
        inst = DiffNILM(**kwargs)
    elif name_model == "TSILNet":
        inst = TSILNet(c_in=c_in, window_size=window_size, **kwargs)
    elif name_model == "Energformer":
        inst = Energformer(c_in=c_in, **kwargs)
    elif name_model == "ConvNet":
        inst = ConvNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "ResNet":
        inst = ResNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "Inception":
        inst = Inception(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "NILMFormer":
        cfg = kwargs.copy()
        c_out = int(cfg.get("c_out", 1))
        cfg["c_out"] = c_out
        inst = NILMFormer(NILMFormerConfig(c_in=1, c_embedding=c_in - 1, **cfg))
    else:
        raise ValueError("Model name {} unknown".format(name_model))

    return inst


def  _save_val_data(model_trainer, valid_loader, scaler, expes_config, epoch_idx):
    device = getattr(model_trainer, "device", None)
    if device is None and hasattr(model_trainer, "model"):
        try:
            device = next(model_trainer.model.parameters()).device
        except StopIteration:
            device = None
    if device is None and hasattr(model_trainer, "parameters"):
        try:
            device = next(model_trainer.parameters()).device
        except StopIteration:
            device = None
    if device is None:
        device = expes_config.device
    if hasattr(model_trainer, "criterion") and hasattr(model_trainer, "model"):
        model = model_trainer
    elif hasattr(model_trainer, "model"):
        model = model_trainer.model
    else:
        model = model_trainer
    agg_concat = []
    target_concat = None
    pred_concat = None
    timestamps_concat = []
    dataset = getattr(valid_loader, "dataset", None)
    has_time = (
        dataset is not None
        and hasattr(dataset, "st_date")
        and getattr(dataset, "freq", None) is not None
        and hasattr(dataset, "L")
    )
    sample_idx = 0
    timestamps_set = set()
    with torch.no_grad():
        for batch in valid_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    ts_agg, appl = batch[0], batch[1]
                else:
                    continue
            else:
                continue
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            appl_t = appl.float().to(device)
            pred_t = model(ts_agg_t)
            threshold_small_values = float(getattr(expes_config, "threshold", 0.0))
            threshold_postprocess = float(
                getattr(expes_config, "postprocess_threshold", threshold_small_values)
            )
            min_on_steps = int(getattr(expes_config, "postprocess_min_on_steps", 0))
            gate_logits = None
            if hasattr(model, "model") and hasattr(model.model, "forward_with_gate"):
                try:
                    _power_raw, gate_logits = model.model.forward_with_gate(ts_agg_t)
                except Exception:
                    gate_logits = None
            elif hasattr(model, "forward_with_gate"):
                try:
                    _power_raw, gate_logits = model.forward_with_gate(ts_agg_t)
                except Exception:
                    gate_logits = None

            agg_t = scaler.inverse_transform_agg_power(ts_agg_t[:, 0:1, :])
            target_t = scaler.inverse_transform_appliance(appl_t)
            pred_inv_raw = scaler.inverse_transform_appliance(pred_t)
            agg_t = torch.nan_to_num(agg_t, nan=0.0, posinf=0.0, neginf=0.0)
            target_t = torch.nan_to_num(target_t, nan=0.0, posinf=0.0, neginf=0.0)
            pred_inv_raw = torch.nan_to_num(
                pred_inv_raw, nan=0.0, posinf=0.0, neginf=0.0
            )
            pred_inv_raw = torch.clamp(pred_inv_raw, min=0.0)
            pred_inv = pred_inv_raw.clone()
            pred_inv[pred_inv < threshold_postprocess] = 0.0
            if min_on_steps > 1:
                pred_inv = suppress_short_activations(
                    pred_inv, threshold_postprocess, min_on_steps
                )
            if gate_logits is not None and bool(
                getattr(expes_config, "postprocess_use_gate", True)
            ):
                try:
                    gate_logits = torch.nan_to_num(
                        gate_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    soft_scale = float(getattr(expes_config, "gate_soft_scale", 1.0))
                    pscale = getattr(expes_config, "postprocess_gate_soft_scale", None)
                    if pscale is None:
                        pscale = max(soft_scale, 1.0) * 3.0
                    pscale = float(pscale)
                    if not np.isfinite(pscale) or pscale <= 0.0:
                        pscale = 1.0
                    gate_prob_sharp = torch.sigmoid(
                        torch.clamp(gate_logits * pscale, min=-50.0, max=50.0)
                    )
                    k = int(
                        getattr(
                            expes_config,
                            "postprocess_gate_kernel",
                            getattr(expes_config, "state_zero_kernel", max(min_on_steps, 0)),
                        )
                    )
                    pred_inv = suppress_long_off_with_gate(
                        pred_inv,
                        gate_prob_sharp,
                        k,
                        float(getattr(expes_config, "postprocess_gate_avg_threshold", 0.35)),
                        float(getattr(expes_config, "postprocess_gate_max_threshold", 0.55)),
                    )
                except Exception:
                    pass
            pred_inv = torch.nan_to_num(pred_inv, nan=0.0, posinf=0.0, neginf=0.0)
            agg_np = agg_t.detach().cpu().numpy()
            target_np = target_t.detach().cpu().numpy()
            pred_np = pred_inv.detach().cpu().numpy()
            batch_size = agg_np.shape[0]
            if target_concat is None:
                n_app = target_np.shape[1]
                target_concat = [[] for _ in range(n_app)]
                pred_concat = [[] for _ in range(n_app)]
            for b in range(batch_size):
                if has_time:
                    idx = sample_idx + b
                    start = dataset.st_date[idx]
                    tmp = pd.date_range(
                        start=start,
                        periods=agg_np.shape[2],
                        freq=dataset.freq,
                    )
                    ts_list = tmp.strftime("%Y-%m-%d %H:%M:%S").tolist()
                    for t, ts_str in enumerate(ts_list):
                        if ts_str in timestamps_set:
                            continue
                        timestamps_set.add(ts_str)
                        timestamps_concat.append(ts_str)
                        agg_concat.append(float(agg_np[b, 0, t]))
                        for j in range(target_np.shape[1]):
                            target_concat[j].append(float(target_np[b, j, t]))
                            pred_concat[j].append(float(pred_np[b, j, t]))
                else:
                    for t in range(agg_np.shape[2]):
                        agg_concat.append(float(agg_np[b, 0, t]))
                        for j in range(target_np.shape[1]):
                            target_concat[j].append(float(target_np[b, j, t]))
                            pred_concat[j].append(float(pred_np[b, j, t]))
            sample_idx += batch_size
    if target_concat is None:
        return
    appliance_name = getattr(expes_config, "appliance", None)
    n_app = len(target_concat)
    result_root = os.path.dirname(
        os.path.dirname(os.path.dirname(expes_config.result_path))
    )
    group_dir = os.path.join(
        result_root,
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
    )
    if appliance_name is not None:
        group_dir = os.path.join(group_dir, str(appliance_name))
    os.makedirs(group_dir, exist_ok=True)
    html_path = os.path.join(group_dir, "val_compare.html")
    model_name = expes_config.name_model
    payload = {}
    if os.path.isfile(html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_text = f.read()
            marker = "const payload = "
            idx = html_text.find(marker)
            if idx != -1:
                idx += len(marker)
                end_idx = html_text.find(";", idx)
                if end_idx != -1:
                    payload_str = html_text[idx:end_idx].strip()
                    payload = json.loads(payload_str)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["agg"] = agg_concat
    if timestamps_concat:
        payload["timestamps"] = timestamps_concat
    target_all = payload.get("target", [])
    appliance_names = payload.get("appliance_names", [])
    if not isinstance(target_all, list):
        target_all = []
    if not isinstance(appliance_names, list):
        appliance_names = []
    group_members = getattr(expes_config, "appliance_group_members", None)
    display_names = []
    if isinstance(group_members, (list, tuple)) and len(group_members) == n_app:
        display_names = [str(x) for x in group_members]
    elif isinstance(appliance_name, str) and n_app == 1:
        display_names = [appliance_name]
    else:
        display_names = [str(j) for j in range(n_app)]
    name_to_index = {str(name): idx for idx, name in enumerate(appliance_names)}
    for j in range(n_app):
        name_j = display_names[j]
        key_j = str(name_j)
        if key_j in name_to_index:
            idx = name_to_index[key_j]
            if idx < len(target_all):
                target_all[idx] = target_concat[j]
            else:
                target_all.append(target_concat[j])
                appliance_names.append(name_j)
                name_to_index[key_j] = len(target_all) - 1
        else:
            target_all.append(target_concat[j])
            appliance_names.append(name_j)
            name_to_index[key_j] = len(target_all) - 1
    payload["target"] = target_all
    if appliance_names:
        payload["appliance_names"] = appliance_names
    total_n = len(target_all)
    models = payload.get("models", {})
    if not isinstance(models, dict):
        models = {}
    for name_m, data_m in list(models.items()):
        if not isinstance(data_m, dict):
            data_m = {}
        runs = data_m.get("runs")
        if isinstance(runs, list):
            epoch_to_run = {}
            for run in runs:
                if not isinstance(run, dict):
                    continue
                pred_list = run.get("pred", [])
                if not isinstance(pred_list, list):
                    pred_list = []
                if len(pred_list) < total_n:
                    pred_list.extend([None] * (total_n - len(pred_list)))
                else:
                    pred_list = pred_list[:total_n]
                run["pred"] = pred_list
                epoch_val = run.get("epoch")
                if isinstance(epoch_val, (int, float)):
                    epoch_val = int(epoch_val)
                else:
                    epoch_val = -1
                run["epoch"] = epoch_val
                epoch_to_run[epoch_val] = run
            data_m["runs"] = list(epoch_to_run.values())
        else:
            epoch_val = data_m.get("epoch")
            if isinstance(epoch_val, (int, float)):
                epoch_val = int(epoch_val)
            else:
                epoch_val = -1
            pred_list = data_m.get("pred", [])
            if not isinstance(pred_list, list):
                pred_list = []
            if len(pred_list) < total_n:
                pred_list.extend([None] * (total_n - len(pred_list)))
            else:
                pred_list = pred_list[:total_n]
            data_m["runs"] = [{"epoch": epoch_val, "pred": pred_list}]
        models[name_m] = data_m
    epoch_int = int(epoch_idx)
    model_data = models.get(model_name, {})
    runs = model_data.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    target_run = None
    for run in runs:
        if isinstance(run, dict) and run.get("epoch") == epoch_int:
            target_run = run
            break
    if target_run is None:
        pred_list = [None] * total_n
        target_run = {"epoch": epoch_int, "pred": pred_list}
        runs.append(target_run)
    else:
        pred_list = target_run.get("pred", [])
        if not isinstance(pred_list, list):
            pred_list = [None] * total_n
        if len(pred_list) < total_n:
            pred_list.extend([None] * (total_n - len(pred_list)))
        else:
            pred_list = pred_list[:total_n]
        target_run["pred"] = pred_list
    for j in range(n_app):
        pred_list[start_idx + j] = pred_concat[j]
    target_run["pred"] = pred_list
    model_data["runs"] = runs
    models[model_name] = model_data
    payload["models"] = models
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NILM Validation Set Power Curve</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h1>NILM Validation Set Power Curve</h1>
<p>{expes_config.dataset} - {expes_config.appliance} - {expes_config.sampling_rate} - window_size={expes_config.window_size}</p>
<label>Models:
<select id="modelSelect" multiple size="5"></select>
</label>
<label>Appliance:
<select id="appSelect"></select>
</label>
<label>Epoch:
<select id="epochSelect"></select>
</label>
<div>
  <label><input type="checkbox" id="showAgg" checked> Agg</label>
  <label><input type="checkbox" id="showTarget" checked> Target</label>
  <label><input type="checkbox" id="showPred" checked> Pred</label>
</div>
<div id="plot"></div>
<script>
  const payload = {json.dumps(payload)};
  let currentLayout = null;
  function init() {{
    const agg = payload.agg || [];
    const target = payload.target || [];
    const models = payload.models || {{}};
    const modelNames = Object.keys(models);
    const nApp = target.length;
    const applianceNames = payload.appliance_names || [];
    const timestamps = payload.timestamps || [];

    const modelSelect = document.getElementById('modelSelect');
    const appSelect = document.getElementById('appSelect');
    const epochSelect = document.getElementById('epochSelect');
    const showAgg = document.getElementById('showAgg');
    const showTarget = document.getElementById('showTarget');
    const showPred = document.getElementById('showPred');

    const predColors = [
      '#ff7f0e',
      '#1f77b4',
      '#d62728',
      '#9467bd',
      '#8c564b',
      '#e377c2',
      '#7f7f7f',
      '#bcbd22',
      '#17becf'
    ];

    const epochsSet = new Set();
    for (let i = 0; i < modelNames.length; i++) {{
      const mData = models[modelNames[i]];
      if (!mData || !Array.isArray(mData.runs)) {{
        continue;
      }}
      for (let r = 0; r < mData.runs.length; r++) {{
        const run = mData.runs[r];
        if (!run || typeof run.epoch !== 'number') {{
          continue;
        }}
        epochsSet.add(run.epoch);
      }}
    }}
    const epochs = Array.from(epochsSet).sort((a, b) => a - b);

    for (let i = 0; i < modelNames.length; i++) {{
      const opt = document.createElement('option');
      opt.value = modelNames[i];
      opt.text = modelNames[i];
      modelSelect.appendChild(opt);
    }}

    for (let j = 0; j < nApp; j++) {{
      const opt = document.createElement('option');
      opt.value = j;
      const name = applianceNames[j] || j.toString();
      opt.text = name;
      appSelect.appendChild(opt);
    }}

    for (let i = 0; i < epochs.length; i++) {{
      const opt = document.createElement('option');
      opt.value = epochs[i];
      opt.text = 'epoch ' + epochs[i];
      epochSelect.appendChild(opt);
    }}

    function makePlot() {{
      const aj = parseInt(appSelect.value);
      if (Number.isNaN(aj)) {{
        return;
      }}
      const epochValue = parseInt(epochSelect.value);
      if (Number.isNaN(epochValue)) {{
        return;
      }}
      const x = Array.from({{length: agg.length}}, (_, i) => i);
      const hasTs = timestamps.length === agg.length;
      const hoverTemplate = hasTs
        ? '%{{customdata}}<br>%{{y}}<extra>%{{fullData.name}}</extra>'
        : '%{{x}}<br>%{{y}}<extra>%{{fullData.name}}</extra>';
      const data = [];
      if (showAgg.checked) {{
        data.push({{
          x: x,
          y: agg,
          name: 'Aggregate',
          mode: 'lines',
          line: {{color: '#7f7f7f'}},
          customdata: hasTs ? timestamps : null,
          hovertemplate: hoverTemplate,
        }});
      }}
      if (showTarget.checked && target[aj]) {{
        data.push({{
          x: x,
          y: target[aj],
          name: 'Target',
          mode: 'lines',
          line: {{color: '#2ca02c'}},
          customdata: hasTs ? timestamps : null,
          hovertemplate: hoverTemplate,
        }});
      }}
      if (showPred.checked) {{
        const selectedModels = Array.from(modelSelect.selectedOptions).map(opt => opt.value);
        for (let k = 0; k < selectedModels.length; k++) {{
          const mj = selectedModels[k];
          const modelData = models[mj];
          if (!modelData || !Array.isArray(modelData.runs)) {{
            continue;
          }}
          let run = null;
          for (let r = 0; r < modelData.runs.length; r++) {{
            const candidate = modelData.runs[r];
            if (candidate && candidate.epoch === epochValue) {{
              run = candidate;
              break;
            }}
          }}
          if (!run || !run.pred || !run.pred[aj]) {{
            continue;
          }}
          data.push({{
            x: x,
            y: run.pred[aj],
            name: 'Prediction: ' + mj + ' (epoch ' + epochValue + ')',
            mode: 'lines',
            line: {{color: predColors[k % predColors.length]}},
            customdata: hasTs ? timestamps : null,
            hovertemplate: hoverTemplate,
          }});
        }}
      }}
      let layout;
      const baseLayout = {{
        margin: {{t: 60, r: 20, b: 40, l: 60}},
        legend: {{
          orientation: 'h',
          x: 0.5,
          xanchor: 'center',
          y: 1.1
        }}
      }};
      if (currentLayout && currentLayout.xaxis && currentLayout.yaxis) {{
        layout = {{
          xaxis: {{range: currentLayout.xaxis.range, title: 'Time index'}},
          yaxis: {{range: currentLayout.yaxis.range, title: 'Power'}},
          ...baseLayout
        }};
      }} else {{
        layout = {{
          xaxis: {{title: 'Time index'}},
          yaxis: {{title: 'Power'}},
          ...baseLayout
        }};
      }}
      Plotly.newPlot('plot', data, layout).then(function(gd) {{
        currentLayout = gd.layout;
      }});
    }}

    epochSelect.addEventListener('change', makePlot);
    modelSelect.addEventListener('change', makePlot);
    appSelect.addEventListener('change', makePlot);
    showAgg.addEventListener('change', makePlot);
    showTarget.addEventListener('change', makePlot);
    showPred.addEventListener('change', makePlot);

    if (nApp > 0) {{
      appSelect.value = '0';
    }}
    if (epochs.length > 0) {{
      epochSelect.value = String(epochs[epochs.length - 1]);
    }}
    if (nApp > 0 && modelNames.length > 0 && epochs.length > 0) {{
      for (let i = 0; i < modelSelect.options.length; i++) {{
        modelSelect.options[i].selected = true;
      }}
      makePlot();
    }}
  }}
  window.addEventListener('load', init);
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


class ValidationHTMLCallback(pl.Callback):
    def __init__(self, valid_loader, scaler, expes_config):
        super().__init__()
        self.valid_loader = valid_loader
        self.scaler = scaler
        self.expes_config = expes_config

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        epoch_idx = int(trainer.current_epoch)
        _save_val_data(
            pl_module, self.valid_loader, self.scaler, self.expes_config, epoch_idx
        )


class ValidationNILMMetricCallback(pl.Callback):
    def __init__(self, valid_loader, scaler, expes_config):
        super().__init__()
        self.valid_loader = valid_loader
        self.scaler = scaler
        self.expes_config = expes_config
        self.metrics = NILMmetrics()
        self.adaptive_tuner = AdaptiveLossTuner()

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        device = pl_module.device
        threshold_small_values = float(self.expes_config.threshold)
        threshold_postprocess = float(
            getattr(self.expes_config, "postprocess_threshold", threshold_small_values)
        )
        min_on_steps = int(getattr(self.expes_config, "postprocess_min_on_steps", 0))
        off_run_min_len = int(
            getattr(self.expes_config, "state_zero_kernel", max(min_on_steps, 0))
        )
        y = np.array([])
        y_hat = np.array([])
        y_win = np.array([])
        y_hat_win = np.array([])
        y_state = np.array([])
        per_device_data = None
        stats = {
            "pred_scaled_sum": 0.0,
            "pred_scaled_sumsq": 0.0,
            "pred_scaled_max": 0.0,
            "pred_scaled_n": 0,
            "pred_scaled_nan_n": 0,
            "pred_raw_sum": 0.0,
            "pred_raw_sumsq": 0.0,
            "pred_raw_max": 0.0,
            "pred_raw_n": 0,
            "pred_raw_zero_n": 0,
            "pred_post_sum": 0.0,
            "pred_post_sumsq": 0.0,
            "pred_post_max": 0.0,
            "pred_post_n": 0,
            "pred_post_zero_n": 0,
            "target_sum": 0.0,
            "target_sumsq": 0.0,
            "target_max": 0.0,
            "target_n": 0,
            "gate_prob_sum": 0.0,
            "gate_prob_sumsq": 0.0,
            "gate_prob_n": 0,
        }
        off_stats = {
            "off_pred_sum": 0.0,
            "off_pred_max": 0.0,
            "off_pred_nonzero_rate_sum": 0.0,
            "off_pred_nonzero_rate_n": 0,
            "off_long_run_pred_sum": 0.0,
            "off_long_run_pred_max": 0.0,
            "off_long_run_total_len": 0,
        }
        off_stats_raw = {
            "off_pred_sum": 0.0,
            "off_pred_max": 0.0,
            "off_pred_nonzero_rate_sum": 0.0,
            "off_pred_nonzero_rate_n": 0,
            "off_long_run_pred_sum": 0.0,
            "off_long_run_pred_max": 0.0,
            "off_long_run_total_len": 0,
        }
        with torch.no_grad():
            for batch in self.valid_loader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        ts_agg, appl, state = batch[0], batch[1], batch[2]
                    elif len(batch) == 2:
                        ts_agg, appl = batch
                        state = None
                    else:
                        continue
                else:
                    continue
                pl_module.eval()
                ts_agg_t = ts_agg.float().to(device)
                target = appl.float().to(device)
                pred = pl_module(ts_agg_t)
                pred_scaled = torch.nan_to_num(
                    pred.float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_scaled_flat = torch.flatten(pred_scaled).detach().cpu().numpy()
                pred_scaled_flat = np.nan_to_num(
                    pred_scaled_flat.astype(np.float64),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                stats["pred_scaled_sum"] += float(pred_scaled_flat.sum())
                stats["pred_scaled_sumsq"] += float((pred_scaled_flat**2).sum())
                pred_scaled_max = float(pred_scaled_flat.max()) if pred_scaled_flat.size else 0.0
                stats["pred_scaled_max"] = max(stats["pred_scaled_max"], pred_scaled_max)
                stats["pred_scaled_n"] += int(pred_scaled_flat.size)
                stats["pred_scaled_nan_n"] += int((~np.isfinite(pred_scaled_flat)).sum())
                target_inv = self.scaler.inverse_transform_appliance(target)
                pred_inv_raw = self.scaler.inverse_transform_appliance(pred)
                target_inv = torch.nan_to_num(
                    target_inv, nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_inv_raw = torch.nan_to_num(
                    pred_inv_raw, nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_inv_raw = torch.clamp(pred_inv_raw, min=0.0)
                pred_inv = pred_inv_raw.clone()
                pred_inv[pred_inv < threshold_postprocess] = 0
                if min_on_steps > 1:
                    pred_inv = suppress_short_activations(
                        pred_inv, threshold_postprocess, min_on_steps
                    )
                if hasattr(pl_module, "model") and hasattr(pl_module.model, "forward_with_gate"):
                    try:
                        _power_raw, gate_logits = pl_module.model.forward_with_gate(ts_agg_t)
                        soft_scale = float(getattr(pl_module, "gate_soft_scale", 1.0))
                        post_scale = float(
                            getattr(
                                self.expes_config,
                                "postprocess_gate_soft_scale",
                                max(float(getattr(pl_module, "gate_soft_scale", 1.0)), 1.0) * 3.0,
                            )
                        )
                        if not np.isfinite(post_scale) or post_scale <= 0.0:
                            post_scale = 1.0
                        gate_logits = torch.nan_to_num(
                            gate_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                        )
                        gate_prob_stats = torch.sigmoid(
                            torch.clamp(gate_logits * soft_scale, min=-50.0, max=50.0)
                        )
                        gate_prob_np = torch.flatten(gate_prob_stats).detach().cpu().numpy()
                        gate_prob_np = np.nan_to_num(
                            gate_prob_np.astype(np.float64),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        stats["gate_prob_sum"] += float(gate_prob_np.sum())
                        stats["gate_prob_sumsq"] += float((gate_prob_np**2).sum())
                        stats["gate_prob_n"] += int(gate_prob_np.size)
                        gate_prob_sharp = torch.sigmoid(
                            torch.clamp(gate_logits * post_scale, min=-50.0, max=50.0)
                        )
                        use_gate_pp = bool(
                            getattr(self.expes_config, "postprocess_use_gate", True)
                        )
                        if use_gate_pp:
                            k = int(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_kernel",
                                    off_run_min_len,
                                )
                            )
                            gate_avg_thr = float(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_avg_threshold",
                                    0.35,
                                )
                            )
                            gate_max_thr = float(
                                getattr(
                                    self.expes_config,
                                    "postprocess_gate_max_threshold",
                                    0.55,
                                )
                            )
                            pred_inv = suppress_long_off_with_gate(
                                pred_inv,
                                gate_prob_sharp,
                                k,
                                gate_avg_thr,
                                gate_max_thr,
                            )
                    except Exception:
                        pass
                pred_inv = torch.nan_to_num(pred_inv, nan=0.0, posinf=0.0, neginf=0.0)
                pred_raw_np = torch.flatten(pred_inv_raw).detach().cpu().numpy()
                pred_post_np = torch.flatten(pred_inv).detach().cpu().numpy()
                target_np = torch.flatten(target_inv).detach().cpu().numpy()
                pred_raw_np = np.nan_to_num(
                    pred_raw_np.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_post_np = np.nan_to_num(
                    pred_post_np.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
                target_np = np.nan_to_num(
                    target_np.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
                stats["pred_raw_sum"] += float(pred_raw_np.sum())
                stats["pred_raw_sumsq"] += float((pred_raw_np**2).sum())
                pred_raw_max = float(pred_raw_np.max()) if pred_raw_np.size else 0.0
                stats["pred_raw_max"] = max(stats["pred_raw_max"], pred_raw_max)
                stats["pred_raw_n"] += int(pred_raw_np.size)
                stats["pred_raw_zero_n"] += int((pred_raw_np <= 0.0).sum())
                stats["pred_post_sum"] += float(pred_post_np.sum())
                stats["pred_post_sumsq"] += float((pred_post_np**2).sum())
                pred_post_max = float(pred_post_np.max()) if pred_post_np.size else 0.0
                stats["pred_post_max"] = max(stats["pred_post_max"], pred_post_max)
                stats["pred_post_n"] += int(pred_post_np.size)
                stats["pred_post_zero_n"] += int((pred_post_np <= 0.0).sum())
                stats["target_sum"] += float(target_np.sum())
                stats["target_sumsq"] += float((target_np**2).sum())
                target_max = float(target_np.max()) if target_np.size else 0.0
                stats["target_max"] = max(stats["target_max"], target_max)
                stats["target_n"] += int(target_np.size)

                target_3d = target_inv.detach().cpu().numpy()
                pred_post_3d = pred_inv.detach().cpu().numpy()
                pred_raw_3d = pred_inv_raw.detach().cpu().numpy()
                target_3d = np.nan_to_num(
                    target_3d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_post_3d = np.nan_to_num(
                    pred_post_3d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_raw_3d = np.nan_to_num(
                    pred_raw_3d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
                off_s = _off_run_stats(
                    target_3d,
                    pred_post_3d,
                    threshold_small_values,
                    off_run_min_len,
                    pred_thr=0.0,
                )
                off_stats["off_pred_sum"] += float(off_s["off_pred_sum"])
                off_stats["off_pred_max"] = max(
                    float(off_stats["off_pred_max"]), float(off_s["off_pred_max"])
                )
                off_stats["off_pred_nonzero_rate_sum"] += float(
                    off_s["off_pred_nonzero_rate"]
                )
                off_stats["off_pred_nonzero_rate_n"] += 1
                off_stats["off_long_run_pred_sum"] += float(
                    off_s["off_long_run_pred_sum"]
                )
                off_stats["off_long_run_pred_max"] = max(
                    float(off_stats["off_long_run_pred_max"]),
                    float(off_s["off_long_run_pred_max"]),
                )
                off_stats["off_long_run_total_len"] += int(
                    off_s["off_long_run_total_len"]
                )
                off_s_raw = _off_run_stats(
                    target_3d,
                    pred_raw_3d,
                    threshold_small_values,
                    off_run_min_len,
                    pred_thr=threshold_small_values,
                )
                off_stats_raw["off_pred_sum"] += float(off_s_raw["off_pred_sum"])
                off_stats_raw["off_pred_max"] = max(
                    float(off_stats_raw["off_pred_max"]), float(off_s_raw["off_pred_max"])
                )
                off_stats_raw["off_pred_nonzero_rate_sum"] += float(
                    off_s_raw["off_pred_nonzero_rate"]
                )
                off_stats_raw["off_pred_nonzero_rate_n"] += 1
                off_stats_raw["off_long_run_pred_sum"] += float(
                    off_s_raw["off_long_run_pred_sum"]
                )
                off_stats_raw["off_long_run_pred_max"] = max(
                    float(off_stats_raw["off_long_run_pred_max"]),
                    float(off_s_raw["off_long_run_pred_max"]),
                )
                off_stats_raw["off_long_run_total_len"] += int(
                    off_s_raw["off_long_run_total_len"]
                )

                target_win = target_inv.sum(dim=-1)
                pred_win = pred_inv.sum(dim=-1)
                target_np_all = target_inv.detach().cpu().numpy()
                pred_np_all = pred_inv.detach().cpu().numpy()
                target_win_np_all = target_win.detach().cpu().numpy()
                pred_win_np_all = pred_win.detach().cpu().numpy()
                target_flat = target_np_all.reshape(-1)
                pred_flat = pred_np_all.reshape(-1)
                target_win_flat = target_win_np_all.reshape(-1)
                pred_win_flat = pred_win_np_all.reshape(-1)
                state_np_all = None
                state_flat = np.array([])
                if state is not None:
                    state_np_all = state.detach().cpu().numpy()
                    state_flat = state_np_all.reshape(-1)
                y = np.concatenate((y, target_flat)) if y.size else target_flat
                y_hat = np.concatenate((y_hat, pred_flat)) if y_hat.size else pred_flat
                y_win = np.concatenate((y_win, target_win_flat)) if y_win.size else target_win_flat
                y_hat_win = np.concatenate((y_hat_win, pred_win_flat)) if y_hat_win.size else pred_win_flat
                y_state = np.concatenate((y_state, state_flat)) if y_state.size else state_flat
                if state_np_all is not None and target_np_all.ndim == 3:
                    if per_device_data is None:
                        n_app = target_np_all.shape[1]
                        per_device_data = {
                            "y": [np.array([]) for _ in range(n_app)],
                            "y_hat": [np.array([]) for _ in range(n_app)],
                            "y_win": [np.array([]) for _ in range(n_app)],
                            "y_hat_win": [np.array([]) for _ in range(n_app)],
                            "y_state": [np.array([]) for _ in range(n_app)],
                        }
                    n_app = target_np_all.shape[1]
                    for j in range(n_app):
                        y_j = target_np_all[:, j, :].reshape(-1)
                        y_hat_j = pred_np_all[:, j, :].reshape(-1)
                        y_win_j = target_win_np_all[:, j].reshape(-1)
                        y_hat_win_j = pred_win_np_all[:, j].reshape(-1)
                        y_state_j = state_np_all[:, j, :].reshape(-1)
                        if y_j.size:
                            arr = per_device_data["y"][j]
                            per_device_data["y"][j] = np.concatenate((arr, y_j)) if arr.size else y_j
                        if y_hat_j.size:
                            arr = per_device_data["y_hat"][j]
                            per_device_data["y_hat"][j] = np.concatenate((arr, y_hat_j)) if arr.size else y_hat_j
                        if y_win_j.size:
                            arr = per_device_data["y_win"][j]
                            per_device_data["y_win"][j] = np.concatenate((arr, y_win_j)) if arr.size else y_win_j
                        if y_hat_win_j.size:
                            arr = per_device_data["y_hat_win"][j]
                            per_device_data["y_hat_win"][j] = (
                                np.concatenate((arr, y_hat_win_j)) if arr.size else y_hat_win_j
                            )
                        if y_state_j.size:
                            arr = per_device_data["y_state"][j]
                            per_device_data["y_state"][j] = (
                                np.concatenate((arr, y_state_j)) if arr.size else y_state_j
                            )
        if not y.size:
            return
        y_hat_state = (
            (y_hat > threshold_postprocess).astype(int)
            if y_state.size
            else None
        )
        metrics_timestamp = self.metrics(
            y=y,
            y_hat=y_hat,
            y_state=y_state if y_state.size else None,
            y_hat_state=y_hat_state,
        )
        metrics_win = self.metrics(y=y_win, y_hat=y_hat_win)
        metrics_timestamp_per_device = {}
        metrics_win_per_device = {}
        if per_device_data is not None:
            n_app = len(per_device_data["y"])
            for j in range(n_app):
                y_j = per_device_data["y"][j]
                y_hat_j = per_device_data["y_hat"][j]
                if y_j.size and y_hat_j.size:
                    y_state_j = per_device_data["y_state"][j]
                    y_hat_state_j = (
                        (y_hat_j > threshold_postprocess).astype(int) if y_state_j.size else None
                    )
                    metrics_timestamp_per_device[str(j)] = self.metrics(
                        y=y_j,
                        y_hat=y_hat_j,
                        y_state=y_state_j if y_state_j.size else None,
                        y_hat_state=y_hat_state_j,
                    )
                y_win_j = per_device_data["y_win"][j]
                y_hat_win_j = per_device_data["y_hat_win"][j]
                if y_win_j.size and y_hat_win_j.size:
                    metrics_win_per_device[str(j)] = self.metrics(
                        y=y_win_j,
                        y_hat=y_hat_win_j,
                    )

        target_sum = float(stats["target_sum"])
        pred_post_sum = float(stats["pred_post_sum"])
        if not np.isfinite(target_sum):
            target_sum = 0.0
        if not np.isfinite(pred_post_sum):
            pred_post_sum = 0.0
        pred_post_zero_rate = float(stats["pred_post_zero_n"]) / float(
            max(stats["pred_post_n"], 1)
        )
        energy_ratio = pred_post_sum / float(max(target_sum, 1e-6))
        collapse_flag = bool(pred_post_zero_rate >= 0.995 or energy_ratio <= 0.02)
        postprocess_zeroed_flag = bool(
            float(stats["pred_raw_max"]) > 0.0 and float(stats["pred_post_max"]) <= 0.0
        )
        gate_prob_mean = None
        if stats["gate_prob_n"] > 0:
            gate_prob_sum = float(stats["gate_prob_sum"])
            gate_prob_n = float(stats["gate_prob_n"])
            if np.isfinite(gate_prob_sum) and gate_prob_n > 0:
                gate_prob_mean = gate_prob_sum / gate_prob_n
        off_pred_sum = float(off_stats["off_pred_sum"])
        off_long_run_pred_sum = float(off_stats["off_long_run_pred_sum"])
        if not np.isfinite(off_pred_sum):
            off_pred_sum = 0.0
        if not np.isfinite(off_long_run_pred_sum):
            off_long_run_pred_sum = 0.0
        off_energy_ratio = off_pred_sum / float(max(target_sum, 1e-6))
        off_long_run_energy_ratio = off_long_run_pred_sum / float(max(target_sum, 1e-6))
        off_pred_nonzero_rate = None
        if off_stats["off_pred_nonzero_rate_n"] > 0:
            off_pred_nonzero_rate = float(off_stats["off_pred_nonzero_rate_sum"]) / float(
                off_stats["off_pred_nonzero_rate_n"]
            )
        off_pred_sum_raw = float(off_stats_raw["off_pred_sum"])
        off_long_run_pred_sum_raw = float(off_stats_raw["off_long_run_pred_sum"])
        if not np.isfinite(off_pred_sum_raw):
            off_pred_sum_raw = 0.0
        if not np.isfinite(off_long_run_pred_sum_raw):
            off_long_run_pred_sum_raw = 0.0
        off_energy_ratio_raw = off_pred_sum_raw / float(max(target_sum, 1e-6))
        off_long_run_energy_ratio_raw = off_long_run_pred_sum_raw / float(max(target_sum, 1e-6))
        off_pred_nonzero_rate_raw = None
        if off_stats_raw["off_pred_nonzero_rate_n"] > 0:
            off_pred_nonzero_rate_raw = float(off_stats_raw["off_pred_nonzero_rate_sum"]) / float(
                off_stats_raw["off_pred_nonzero_rate_n"]
            )

        result_root = os.path.dirname(
            os.path.dirname(os.path.dirname(self.expes_config.result_path))
        )
        group_dir = os.path.join(
            result_root,
            "{}_{}".format(self.expes_config.dataset, self.expes_config.sampling_rate),
            str(self.expes_config.window_size),
        )
        appliance_name = getattr(self.expes_config, "appliance", None)
        if appliance_name is not None:
            group_dir = os.path.join(group_dir, str(appliance_name))
        os.makedirs(group_dir, exist_ok=True)
        record = {
            "epoch": int(trainer.current_epoch),
            "model": str(self.expes_config.name_model),
            "dataset": str(self.expes_config.dataset),
            "appliance": str(appliance_name) if appliance_name is not None else None,
            "sampling_rate": str(self.expes_config.sampling_rate),
            "window_size": int(self.expes_config.window_size),
            "threshold": float(threshold_small_values),
            "threshold_postprocess": float(threshold_postprocess),
            "min_on_steps": int(min_on_steps),
            "loss_threshold": float(getattr(self.expes_config, "loss_threshold", threshold_small_values)),
            "metrics_timestamp": metrics_timestamp,
            "metrics_win": metrics_win,
            "metrics_timestamp_per_device": metrics_timestamp_per_device,
            "metrics_win_per_device": metrics_win_per_device,
            "pred_scaled_max": float(stats["pred_scaled_max"]),
            "pred_raw_max": float(stats["pred_raw_max"]),
            "pred_post_max": float(stats["pred_post_max"]),
            "pred_post_zero_rate": pred_post_zero_rate,
            "postprocess_zeroed_flag": postprocess_zeroed_flag,
            "target_max": float(stats["target_max"]),
            "energy_ratio": energy_ratio,
            "off_energy_ratio": off_energy_ratio,
            "off_energy_ratio_raw": off_energy_ratio_raw,
            "off_pred_max": float(off_stats["off_pred_max"]),
            "off_pred_max_raw": float(off_stats_raw["off_pred_max"]),
            "off_pred_nonzero_rate": off_pred_nonzero_rate,
            "off_pred_nonzero_rate_raw": off_pred_nonzero_rate_raw,
            "off_run_min_len": int(off_run_min_len),
            "off_long_run_energy_ratio": off_long_run_energy_ratio,
            "off_long_run_energy_ratio_raw": off_long_run_energy_ratio_raw,
            "off_long_run_pred_max": float(off_stats["off_long_run_pred_max"]),
            "off_long_run_pred_max_raw": float(off_stats_raw["off_long_run_pred_max"]),
            "off_long_run_total_len": int(off_stats["off_long_run_total_len"]),
            "gate_prob_mean": gate_prob_mean,
            "collapse_flag": collapse_flag,
        }
        _append_jsonl(os.path.join(group_dir, "val_report.jsonl"), record)
        logging.info("VAL_REPORT_JSON: %s", json.dumps(record, ensure_ascii=False))

        # Adaptive loss tuning using AdaptiveLossTuner
        try:
            device_type = str(getattr(self.expes_config, "device_type", "") or "")
            appliance_name = str(getattr(self.expes_config, "appliance", "") or "")

            # Handle early collapse
            self.adaptive_tuner.handle_early_collapse(
                pl_module, record, device_type, appliance_name, int(trainer.current_epoch)
            )

            # Tune from metrics if not collapsed
            if not bool(record.get("collapse_flag", False)):
                self.adaptive_tuner.tune_from_metrics(
                    pl_module, record, metrics_timestamp, device_type, appliance_name
                )
        except Exception:
            pass

        writer = None
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            writer = trainer.logger.experiment
        if writer is None:
            return
        epoch_idx = int(trainer.current_epoch)
        for name, value in metrics_timestamp.items():
            writer.add_scalar("valid_timestamp/" + name, float(value), epoch_idx)
        for name, value in metrics_win.items():
            writer.add_scalar("valid_win/" + name, float(value), epoch_idx)
        for idx, mdict in metrics_timestamp_per_device.items():
            for name, value in mdict.items():
                writer.add_scalar(
                    "valid_timestamp/" + name + "_app" + str(idx),
                    float(value),
                    epoch_idx,
                )
        for idx, mdict in metrics_win_per_device.items():
            for name, value in mdict.items():
                writer.add_scalar(
                    "valid_win/" + name + "_app" + str(idx),
                    float(value),
                    epoch_idx,
                )


def suppress_short_activations(pred_inv, threshold_small_values, min_on_steps):
    if min_on_steps <= 1:
        return pred_inv
    if pred_inv.dim() != 3:
        return pred_inv
    pred_np = pred_inv.detach().cpu().numpy()
    b, c, t = pred_np.shape
    for i in range(b):
        for j in range(c):
            series = pred_np[i, j]
            on_mask = series >= float(threshold_small_values)
            if not on_mask.any():
                continue
            idx = np.nonzero(on_mask)[0]
            if idx.size == 0:
                continue
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for seg in splits:
                if seg.size < min_on_steps:
                    series[seg] = 0.0
    return torch.from_numpy(pred_np).to(pred_inv.device)


def _pad_same_1d(x, kernel_size):
    k = int(kernel_size)
    if k <= 1:
        return x
    total = k - 1
    left = total // 2
    right = total - left
    if left <= 0 and right <= 0:
        return x
    return F.pad(x, (left, right), mode="replicate")


def suppress_long_off_with_gate(pred_inv, gate_prob, kernel_size, gate_avg_thr, gate_max_thr):
    if pred_inv is None or gate_prob is None:
        return pred_inv
    if not torch.is_tensor(pred_inv) or not torch.is_tensor(gate_prob):
        return pred_inv
    if pred_inv.dim() != 3 or gate_prob.dim() != 3:
        return pred_inv
    if pred_inv.size() != gate_prob.size():
        return pred_inv
    k = int(kernel_size)
    if k <= 1:
        return pred_inv
    k = min(k, int(pred_inv.size(-1)))
    if k <= 1:
        return pred_inv
    gate_prob = torch.nan_to_num(gate_prob.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = gate_prob.reshape(-1, 1, gate_prob.size(-1))
    x = _pad_same_1d(x, k)
    gate_avg = F.avg_pool1d(x, kernel_size=k, stride=1, padding=0)
    gate_max = F.max_pool1d(x, kernel_size=k, stride=1, padding=0)
    gate_avg = gate_avg.reshape_as(gate_prob)
    gate_max = gate_max.reshape_as(gate_prob)
    gate_avg_thr = float(gate_avg_thr)
    gate_max_thr = float(gate_max_thr)
    mask = (gate_avg < gate_avg_thr) & (gate_max < gate_max_thr)
    if not mask.any():
        return pred_inv
    out = pred_inv.clone()
    out[mask] = 0.0
    return out


def _off_run_stats(target_np, pred_np, thr, min_len, pred_thr=None):
    off_mask = target_np <= float(thr)
    pred_off = pred_np[off_mask]
    if pred_thr is None:
        pred_thr = 0.0
    else:
        pred_thr = float(pred_thr)
    out = {
        "off_pred_sum": float(pred_off.sum()) if pred_off.size else 0.0,
        "off_pred_max": float(pred_off.max()) if pred_off.size else 0.0,
        "off_pred_nonzero_rate": float((pred_off > pred_thr).mean()) if pred_off.size else 0.0,
        "off_long_run_pred_sum": 0.0,
        "off_long_run_pred_max": 0.0,
        "off_long_run_total_len": 0,
    }
    min_len = int(min_len)
    if min_len <= 1:
        return out
    if target_np.ndim != 3 or pred_np.ndim != 3:
        return out
    b, c, l = target_np.shape
    long_sum = 0.0
    long_max = 0.0
    long_len = 0
    for i in range(b):
        for j in range(c):
            off_row = off_mask[i, j]
            if not off_row.any():
                continue
            idx = np.nonzero(off_row)[0]
            if idx.size == 0:
                continue
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for seg in splits:
                if seg.size < min_len:
                    continue
                vals = pred_np[i, j, seg]
                if vals.size == 0:
                    continue
                long_sum += float(vals.sum())
                long_max = max(long_max, float(vals.max()))
                long_len += int(seg.size)
    out["off_long_run_pred_sum"] = float(long_sum)
    out["off_long_run_pred_max"] = float(long_max)
    out["off_long_run_total_len"] = int(long_len)
    return out


def evaluate_nilm_split(
    model,
    data_loader,
    scaler,
    threshold_small_values,
    device,
    save_outputs,
    mask,
    log_dict,
    min_on_duration_steps=0,
    expes_config=None,
):
    metrics_helper = NILMmetrics()
    y = np.array([])
    y_hat = np.array([])
    y_win = np.array([])
    y_hat_win = np.array([])
    y_state = np.array([])
    per_device_data = None
    threshold_postprocess = float(threshold_small_values)
    off_run_min_len = int(max(int(min_on_duration_steps or 0), 0))
    postprocess_use_gate = True
    post_gate_kernel = int(off_run_min_len)
    post_gate_avg_thr = 0.35
    post_gate_max_thr = 0.55
    post_gate_soft_scale = None
    gate_floor = 0.0
    gate_soft_scale = 1.0
    if expes_config is not None:
        threshold_postprocess = float(
            getattr(expes_config, "postprocess_threshold", threshold_postprocess)
        )
        postprocess_use_gate = bool(getattr(expes_config, "postprocess_use_gate", True))
        off_run_min_len = int(
            getattr(
                expes_config,
                "postprocess_gate_kernel",
                getattr(expes_config, "state_zero_kernel", off_run_min_len),
            )
        )
        post_gate_kernel = int(getattr(expes_config, "postprocess_gate_kernel", off_run_min_len))
        post_gate_avg_thr = float(
            getattr(expes_config, "postprocess_gate_avg_threshold", post_gate_avg_thr)
        )
        post_gate_max_thr = float(
            getattr(expes_config, "postprocess_gate_max_threshold", post_gate_max_thr)
        )
        post_gate_soft_scale = getattr(expes_config, "postprocess_gate_soft_scale", None)
        gate_floor = float(getattr(expes_config, "gate_floor", gate_floor))
        gate_soft_scale = float(getattr(expes_config, "gate_soft_scale", gate_soft_scale))
    with torch.no_grad():
        iterator = data_loader
        try:
            total = len(data_loader)
        except TypeError:
            total = None
        iterator = tqdm(iterator, total=total, desc=mask, leave=False)
        for batch in iterator:
            if not isinstance(batch, (list, tuple)):
                continue
            if len(batch) >= 3:
                ts_agg, appl, state = batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                ts_agg, appl = batch
                state = None
            else:
                continue
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            target = appl.float().to(device)
            pred = None
            gate_logits = None
            if hasattr(model, "forward_with_gate"):
                try:
                    power_raw, gate_logits = model.forward_with_gate(ts_agg_t)
                    gate_logits = torch.nan_to_num(
                        gate_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    soft_scale = float(gate_soft_scale)
                    if not np.isfinite(soft_scale) or soft_scale <= 0.0:
                        soft_scale = 1.0
                    gate_prob = torch.sigmoid(
                        torch.clamp(gate_logits * soft_scale, min=-50.0, max=50.0)
                    )
                    gf = float(gate_floor)
                    if not np.isfinite(gf):
                        gf = 0.0
                    gf = min(max(gf, 0.0), 1.0)
                    pred = power_raw * (gf + (1.0 - gf) * gate_prob)
                except Exception:
                    pred = None
                    gate_logits = None
            if pred is None:
                pred = model(ts_agg_t)
            target_inv = scaler.inverse_transform_appliance(target)
            pred_inv = scaler.inverse_transform_appliance(pred)
            pred_inv = torch.clamp(pred_inv, min=0.0)
            pred_inv[pred_inv < threshold_postprocess] = 0
            if min_on_duration_steps and min_on_duration_steps > 1:
                pred_inv = suppress_short_activations(
                    pred_inv, threshold_postprocess, min_on_duration_steps
                )
            if (
                postprocess_use_gate
                and gate_logits is not None
                and pred_inv is not None
                and pred_inv.dim() == 3
                and int(post_gate_kernel or 0) > 1
            ):
                try:
                    pscale = post_gate_soft_scale
                    if pscale is None:
                        pscale = max(float(gate_soft_scale), 1.0) * 3.0
                    pscale = float(pscale)
                    if not np.isfinite(pscale) or pscale <= 0.0:
                        pscale = 1.0
                    gate_prob_sharp = torch.sigmoid(
                        torch.clamp(gate_logits * pscale, min=-50.0, max=50.0)
                    )
                    pred_inv = suppress_long_off_with_gate(
                        pred_inv,
                        gate_prob_sharp,
                        int(post_gate_kernel),
                        float(post_gate_avg_thr),
                        float(post_gate_max_thr),
                    )
                except Exception:
                    pass
            target_win = target_inv.sum(dim=-1)
            pred_win = pred_inv.sum(dim=-1)
            target_np_all = target_inv.detach().cpu().numpy()
            pred_np_all = pred_inv.detach().cpu().numpy()
            target_win_np_all = target_win.detach().cpu().numpy()
            pred_win_np_all = pred_win.detach().cpu().numpy()
            target_flat = target_np_all.reshape(-1)
            pred_flat = pred_np_all.reshape(-1)
            target_win_flat = target_win_np_all.reshape(-1)
            pred_win_flat = pred_win_np_all.reshape(-1)
            state_np_all = None
            state_flat = np.array([])
            if state is not None:
                state_np_all = state.detach().cpu().numpy()
                state_flat = state_np_all.reshape(-1)
            y = np.concatenate((y, target_flat)) if y.size else target_flat
            y_hat = np.concatenate((y_hat, pred_flat)) if y_hat.size else pred_flat
            y_win = (
                np.concatenate((y_win, target_win_flat))
                if y_win.size
                else target_win_flat
            )
            y_hat_win = (
                np.concatenate((y_hat_win, pred_win_flat))
                if y_hat_win.size
                else pred_win_flat
            )
            y_state = np.concatenate((y_state, state_flat)) if y_state.size else state_flat
            if state_np_all is not None and target_np_all.ndim == 3:
                if per_device_data is None:
                    n_app = target_np_all.shape[1]
                    per_device_data = {
                        "y": [np.array([]) for _ in range(n_app)],
                        "y_hat": [np.array([]) for _ in range(n_app)],
                        "y_win": [np.array([]) for _ in range(n_app)],
                        "y_hat_win": [np.array([]) for _ in range(n_app)],
                        "y_state": [np.array([]) for _ in range(n_app)],
                    }
                n_app = target_np_all.shape[1]
                for j in range(n_app):
                    y_j = target_np_all[:, j, :].reshape(-1)
                    y_hat_j = pred_np_all[:, j, :].reshape(-1)
                    y_win_j = target_win_np_all[:, j].reshape(-1)
                    y_hat_win_j = pred_win_np_all[:, j].reshape(-1)
                    y_state_j = state_np_all[:, j, :].reshape(-1)
                    if y_j.size:
                        arr = per_device_data["y"][j]
                        per_device_data["y"][j] = np.concatenate((arr, y_j)) if arr.size else y_j
                    if y_hat_j.size:
                        arr = per_device_data["y_hat"][j]
                        per_device_data["y_hat"][j] = np.concatenate((arr, y_hat_j)) if arr.size else y_hat_j
                    if y_win_j.size:
                        arr = per_device_data["y_win"][j]
                        per_device_data["y_win"][j] = np.concatenate((arr, y_win_j)) if arr.size else y_win_j
                    if y_hat_win_j.size:
                        arr = per_device_data["y_hat_win"][j]
                        per_device_data["y_hat_win"][j] = (
                            np.concatenate((arr, y_hat_win_j)) if arr.size else y_hat_win_j
                        )
                    if y_state_j.size:
                        arr = per_device_data["y_state"][j]
                        per_device_data["y_state"][j] = (
                            np.concatenate((arr, y_state_j)) if arr.size else y_state_j
                        )
    if not y.size:
        return {}, {}
    y_hat_state = (
        (y_hat > threshold_postprocess).astype(int) if y_state.size else None
    )
    metrics_timestamp = metrics_helper(
        y=y,
        y_hat=y_hat,
        y_state=y_state if y_state.size else None,
        y_hat_state=y_hat_state,
    )
    metrics_win = metrics_helper(y=y_win, y_hat=y_hat_win)
    metrics_timestamp_per_device = {}
    metrics_win_per_device = {}
    if per_device_data is not None:
        n_app = len(per_device_data["y"])
        for j in range(n_app):
            y_j = per_device_data["y"][j]
            y_hat_j = per_device_data["y_hat"][j]
            if y_j.size and y_hat_j.size:
                y_state_j = per_device_data["y_state"][j]
                y_hat_state_j = (
                    (y_hat_j > threshold_postprocess).astype(int) if y_state_j.size else None
                )
                metrics_timestamp_per_device[str(j)] = metrics_helper(
                    y=y_j,
                    y_hat=y_hat_j,
                    y_state=y_state_j if y_state_j.size else None,
                    y_hat_state=y_hat_state_j,
                )
            y_win_j = per_device_data["y_win"][j]
            y_hat_win_j = per_device_data["y_hat_win"][j]
            if y_win_j.size and y_hat_win_j.size:
                metrics_win_per_device[str(j)] = metrics_helper(
                    y=y_win_j,
                    y_hat=y_hat_win_j,
                )
    log_dict[mask + "_timestamp"] = metrics_timestamp
    log_dict[mask + "_win"] = metrics_win
    if metrics_timestamp_per_device:
        log_dict[mask + "_timestamp_per_device"] = metrics_timestamp_per_device
    if metrics_win_per_device:
        log_dict[mask + "_win_per_device"] = metrics_win_per_device
    if save_outputs:
        log_dict[mask + "_yhat"] = y_hat
        if y_hat_win.size:
            log_dict[mask + "_yhat_win"] = y_hat_win
    return metrics_timestamp, metrics_win


def nilm_model_training(inst_model, tuple_data, scaler, expes_config):
    expes_config.device = get_device()
    ckpt_path = expes_config.result_path + ".pt"

    if expes_config.name_model == "DiffNILM":
        train_dataset = NILMDataset(
            tuple_data[0],
            st_date=tuple_data[4],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
        valid_dataset = NILMDataset(
            tuple_data[1],
            st_date=tuple_data[5],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
        test_dataset = NILMDataset(
            tuple_data[2],
            st_date=tuple_data[6],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
    else:
        list_exo = expes_config.list_exo_variables
        train_dataset = NILMDataset(
            tuple_data[0],
            st_date=tuple_data[4],
            list_exo_variables=list_exo,
            freq=expes_config.sampling_rate,
        )

        valid_dataset = NILMDataset(
            tuple_data[1],
            st_date=tuple_data[5],
            list_exo_variables=list_exo,
            freq=expes_config.sampling_rate,
        )

        test_dataset = NILMDataset(
            tuple_data[2],
            st_date=tuple_data[6],
            list_exo_variables=list_exo,
            freq=expes_config.sampling_rate,
        )

    default_loss_type = "ga_eaec" if expes_config.name_model == "NILMFormer" else "eaec"
    loss_type = str(getattr(expes_config, "loss_type", default_loss_type))
    if loss_type in ["eaec", "ga_eaec"] and expes_config.name_model == "NILMFormer":
        p_es_eaec = getattr(expes_config, "p_es_eaec", None)
        if p_es_eaec is not None:
            expes_config.p_es = p_es_eaec
        p_rlr_eaec = getattr(expes_config, "p_rlr_eaec", None)
        if p_rlr_eaec is not None:
            expes_config.p_rlr = p_rlr_eaec
        n_warmup_eaec = getattr(expes_config, "n_warmup_epochs_eaec", None)
        if n_warmup_eaec is not None:
            expes_config.n_warmup_epochs = n_warmup_eaec
        training_param_eaec = getattr(expes_config, "model_training_param_eaec", None)
        if training_param_eaec is not None:
            expes_config.model_training_param = training_param_eaec
        warmup_type_eaec = getattr(expes_config, "warmup_type_eaec", None)
        if warmup_type_eaec is not None:
            expes_config.warmup_type = warmup_type_eaec
        neg_penalty_weight_eaec = getattr(expes_config, "neg_penalty_weight_eaec", None)
        if neg_penalty_weight_eaec is not None:
            expes_config.neg_penalty_weight = neg_penalty_weight_eaec
        rlr_factor_eaec = getattr(expes_config, "rlr_factor_eaec", None)
        if rlr_factor_eaec is not None:
            expes_config.rlr_factor = rlr_factor_eaec
        rlr_min_lr_eaec = getattr(expes_config, "rlr_min_lr_eaec", None)
        if rlr_min_lr_eaec is not None:
            expes_config.rlr_min_lr = rlr_min_lr_eaec
        gate_cls_weight_eaec = getattr(expes_config, "gate_cls_weight_eaec", None)
        if gate_cls_weight_eaec is not None:
            expes_config.gate_cls_weight = gate_cls_weight_eaec
        gate_window_weight_eaec = getattr(
            expes_config, "gate_window_weight_eaec", None
        )
        if gate_window_weight_eaec is not None:
            expes_config.gate_window_weight = gate_window_weight_eaec
        gate_focal_gamma_eaec = getattr(expes_config, "gate_focal_gamma_eaec", None)
        if gate_focal_gamma_eaec is not None:
            expes_config.gate_focal_gamma = gate_focal_gamma_eaec

    train_sampler = None
    try:
        balance_window_sampling = bool(
            getattr(expes_config, "balance_window_sampling", True)
        )
    except Exception:
        balance_window_sampling = False
    if (
        balance_window_sampling
        and str(getattr(expes_config, "name_model", "")).lower() == "nilmformer"
        and loss_type in ["eaec", "ga_eaec"]
    ):
        try:
            train_states = tuple_data[0][:, 1, 1, :]
            on_window_mask = (train_states.sum(axis=-1) > 0).astype(np.float32)
            on_window_frac = float(on_window_mask.mean()) if on_window_mask.size else 0.0
            sparse_threshold = float(
                getattr(expes_config, "balance_window_on_frac_threshold", 0.25)
            )
            if 0.0 < on_window_frac < sparse_threshold:
                target_on_frac = float(
                    getattr(expes_config, "balance_window_target_on_frac", 0.5)
                )
                target_on_frac = min(max(target_on_frac, 0.05), 0.95)
                w_on = target_on_frac / float(max(on_window_frac, 1e-6))
                w_off = (1.0 - target_on_frac) / float(max(1.0 - on_window_frac, 1e-6))
                max_ratio = float(getattr(expes_config, "balance_window_max_ratio", 100.0))
                max_ratio = max(max_ratio, 1.0)
                ratio = float(w_on) / float(max(w_off, 1e-12))
                if ratio > max_ratio:
                    w_on = w_off * max_ratio
                weights_np = np.where(on_window_mask > 0.5, w_on, w_off).astype(
                    np.float32
                )
                weights = torch.from_numpy(weights_np)
                train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights,
                    num_samples=int(weights.numel()),
                    replacement=True,
                )
                logging.info(
                    "Enable balanced window sampling: on_window_frac=%.4f target_on_frac=%.2f",
                    on_window_frac,
                    target_on_frac,
                )
            else:
                logging.info(
                    "Skip balanced window sampling: on_window_frac=%.4f",
                    on_window_frac,
                )
        except Exception as e:
            logging.warning("Could not build balanced sampler: %s", e)

    num_workers = getattr(expes_config, "num_workers", 0)
    if os.name == "nt" and num_workers:
        num_workers = 0
    persistent_workers = num_workers > 0
    pin_memory = expes_config.device == "cuda"
    batch_size = expes_config.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    metric_callback = ValidationNILMMetricCallback(valid_loader, scaler, expes_config)
    html_callback = ValidationHTMLCallback(valid_loader, scaler, expes_config)
    callbacks = [metric_callback, html_callback]
    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=expes_config.p_es, mode="min"
            )
        )
    ckpt_root = os.path.join(
        "checkpoint",
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
        expes_config.appliance,
        "{}_{}".format(expes_config.name_model, expes_config.seed),
    )
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_name = "ckpt"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_root,
        filename=ckpt_name + "_{epoch:03d}",
    )
    callbacks.append(checkpoint_callback)
    if expes_config.name_model == "DiffNILM":
        lightning_module = DiffNILMLightningModule(inst_model)
    elif expes_config.name_model == "STNILM":
        lightning_module = STNILMLightningModule(
            inst_model,
            learning_rate=float(expes_config.model_training_param.lr),
            weight_decay=float(expes_config.model_training_param.wd),
            patience_rlr=expes_config.p_rlr,
            n_warmup_epochs=expes_config.n_warmup_epochs,
            warmup_type=getattr(expes_config, "warmup_type", "linear"),
        )
    else:
        default_loss_type = "ga_eaec" if expes_config.name_model == "NILMFormer" else "eaec"
        loss_type = str(getattr(expes_config, "loss_type", default_loss_type))
        threshold_loss_raw = float(
            getattr(expes_config, "loss_threshold", expes_config.threshold)
        )
        threshold_loss = threshold_loss_raw
        loss_scale_denom = None
        if threshold_loss_raw > 1.5 and scaler is not None and getattr(scaler, "is_fitted", False):
            try:
                scaling_type = getattr(scaler, "appliance_scaling_type", None)
                n_app = int(getattr(scaler, "n_appliance", 0))
                if scaling_type is not None and n_app > 0:
                    if scaling_type == "SameAsPower":
                        offset = float(getattr(scaler, "power_stat1", 0.0))
                        denom = float(getattr(scaler, "power_stat2", 1.0))
                        if getattr(scaler, "power_scaling_type", None) == "MinMax":
                            denom = float(getattr(scaler, "power_stat2", 1.0)) - float(
                                getattr(scaler, "power_stat1", 0.0)
                            )
                    else:
                        offset = float(getattr(scaler, "appliance_stat1", [0.0])[0])
                        denom = float(getattr(scaler, "appliance_stat2", [1.0])[0])
                        if scaling_type == "MinMax":
                            denom = float(getattr(scaler, "appliance_stat2", [1.0])[0]) - float(
                                getattr(scaler, "appliance_stat1", [0.0])[0]
                            )
                    if denom == 0.0:
                        denom = 1.0
                    threshold_loss = max(0.0, (threshold_loss_raw - offset) / denom)
                    loss_scale_denom = float(denom)
            except Exception:
                threshold_loss = threshold_loss_raw
        try:
            cutoff = float(getattr(expes_config, "cutoff", 0.0) or 0.0)
            if cutoff > 0.0 and threshold_loss_raw > 1.5:
                threshold_loss = threshold_loss_raw / cutoff
        except Exception:
            threshold_loss = threshold_loss_raw
        if loss_type in ["eaec", "ga_eaec"]:
            alpha_on = float(getattr(expes_config, "loss_alpha_on", 3.0))
            alpha_off = float(getattr(expes_config, "loss_alpha_off", 1.0))
            lambda_grad = float(getattr(expes_config, "loss_lambda_grad", 0.5))
            lambda_energy = float(getattr(expes_config, "loss_lambda_energy", 0.5))
            soft_temp_raw = float(getattr(expes_config, "loss_soft_temp", 10.0))
            edge_eps_raw = float(getattr(expes_config, "loss_edge_eps", 5.0))
            if loss_scale_denom is not None:
                soft_temp = soft_temp_raw / loss_scale_denom if soft_temp_raw > 1.5 else soft_temp_raw
                edge_eps = edge_eps_raw / loss_scale_denom if edge_eps_raw > 1.5 else edge_eps_raw
            else:
                soft_temp = soft_temp_raw
                edge_eps = edge_eps_raw
            lambda_sparse = float(getattr(expes_config, "loss_lambda_sparse", 0.0))
            lambda_zero = float(getattr(expes_config, "loss_lambda_zero", 0.0))
            center_ratio = float(getattr(expes_config, "loss_center_ratio", 1.0))
            energy_floor_default = threshold_loss * float(expes_config.window_size) * 0.1
            if hasattr(expes_config, "loss_energy_floor"):
                energy_floor = float(expes_config.loss_energy_floor)
            elif hasattr(expes_config, "loss_energy_floor_raw") and getattr(
                expes_config, "cutoff", None
            ):
                energy_floor = float(expes_config.loss_energy_floor_raw) / float(
                    expes_config.cutoff
                )
                if loss_scale_denom is not None and float(expes_config.loss_energy_floor_raw) > 1.5:
                    energy_floor = float(expes_config.loss_energy_floor_raw) / loss_scale_denom
            else:
                energy_floor = energy_floor_default
            if loss_type == "eaec":
                criterion = EAECLoss(
                    threshold=threshold_loss,
                    alpha_on=alpha_on,
                    alpha_off=alpha_off,
                    lambda_grad=lambda_grad,
                    lambda_energy=lambda_energy,
                    soft_temp=soft_temp,
                    edge_eps=edge_eps,
                    energy_floor=energy_floor,
                    lambda_sparse=lambda_sparse,
                    lambda_zero=lambda_zero,
                    center_ratio=center_ratio,
                )
            else:
                # OFF false-positive penalty parameters (mild setting)
                lambda_off_hard = float(getattr(expes_config, "loss_lambda_off_hard", 0.1))
                off_margin = float(getattr(expes_config, "loss_off_margin", 0.02))
                # ON missed-detection penalty parameters (prevents all-zero outputs)
                lambda_on_recall = float(getattr(expes_config, "loss_lambda_on_recall", 0.3))
                on_recall_margin = float(getattr(expes_config, "loss_on_recall_margin", 0.5))
                # Gate classification parameters
                lambda_gate_cls = float(getattr(expes_config, "loss_lambda_gate_cls", 0.1))
                gate_focal_gamma = float(getattr(expes_config, "loss_gate_focal_gamma", 2.0))
                criterion = GAEAECLoss(
                    threshold=threshold_loss,
                    alpha_on=alpha_on,
                    alpha_off=alpha_off,
                    lambda_grad=lambda_grad,
                    lambda_energy=lambda_energy,
                    soft_temp=soft_temp,
                    edge_eps=edge_eps,
                    energy_floor=energy_floor,
                    lambda_sparse=lambda_sparse,
                    lambda_zero=lambda_zero,
                    center_ratio=center_ratio,
                    lambda_off_hard=lambda_off_hard,
                    off_margin=off_margin,
                    lambda_on_recall=lambda_on_recall,
                    on_recall_margin=on_recall_margin,
                    lambda_gate_cls=lambda_gate_cls,
                    gate_focal_gamma=gate_focal_gamma,
                )
        elif loss_type == "smoothl1":
            criterion = nn.SmoothL1Loss()
        elif loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        state_zero_penalty_weight = float(
            getattr(expes_config, "state_zero_penalty_weight", 0.0)
        )
        zero_run_kernel = int(getattr(expes_config, "state_zero_kernel", 0))
        zero_run_ratio = float(getattr(expes_config, "state_zero_ratio", 0.8))
        off_high_agg_penalty_weight = float(
            getattr(expes_config, "off_high_agg_penalty_weight", 0.0)
        )
        off_state_penalty_weight = float(
            getattr(expes_config, "off_state_penalty_weight", 0.0)
        )
        off_state_margin = float(getattr(expes_config, "off_state_margin", 0.0))
        off_state_long_penalty_weight = float(
            getattr(expes_config, "off_state_long_penalty_weight", 0.0)
        )
        off_state_long_kernel = int(getattr(expes_config, "off_state_long_kernel", 0))
        off_state_long_margin = float(
            getattr(expes_config, "off_state_long_margin", off_state_margin)
        )
        lightning_module = SeqToSeqLightningModule(
            inst_model,
            learning_rate=float(expes_config.model_training_param.lr),
            weight_decay=float(expes_config.model_training_param.wd),
            criterion=criterion,
            patience_rlr=expes_config.p_rlr,
            n_warmup_epochs=expes_config.n_warmup_epochs,
            warmup_type=getattr(expes_config, "warmup_type", "linear"),
            output_stats_warmup_epochs=int(
                getattr(expes_config, "output_stats_warmup_epochs", 0)
            ),
            output_stats_ramp_epochs=int(
                getattr(expes_config, "output_stats_ramp_epochs", 0)
            ),
            output_stats_mean_max=float(
                getattr(expes_config, "output_stats_mean_max", 0.0)
            ),
            output_stats_std_max=float(getattr(expes_config, "output_stats_std_max", 0.0)),
            neg_penalty_weight=float(getattr(expes_config, "neg_penalty_weight", 0.1)),
            rlr_factor=float(getattr(expes_config, "rlr_factor", 0.1)),
            rlr_min_lr=float(getattr(expes_config, "rlr_min_lr", 0.0)),
            state_zero_penalty_weight=state_zero_penalty_weight,
            zero_run_kernel=zero_run_kernel,
            zero_run_ratio=zero_run_ratio,
            loss_threshold=threshold_loss,
            off_high_agg_penalty_weight=off_high_agg_penalty_weight,
            off_state_penalty_weight=off_state_penalty_weight,
            off_state_margin=off_state_margin,
            off_state_long_penalty_weight=off_state_long_penalty_weight,
            off_state_long_kernel=off_state_long_kernel,
            off_state_long_margin=off_state_long_margin,
            gate_cls_weight=float(getattr(expes_config, "gate_cls_weight", 0.0)),
            gate_window_weight=float(
                getattr(expes_config, "gate_window_weight", 0.0)
            ),
            gate_focal_gamma=float(getattr(expes_config, "gate_focal_gamma", 2.0)),
            gate_soft_scale=float(getattr(expes_config, "gate_soft_scale", 1.0)),
            gate_floor=float(getattr(expes_config, "gate_floor", 0.1)),
            gate_duty_weight=float(getattr(expes_config, "gate_duty_weight", 0.0)),
        )
    accelerator = "cpu"
    devices = 1
    device_cfg = str(getattr(expes_config, "device", "auto")).lower()
    if device_cfg == "cpu":
        accelerator = "cpu"
    elif device_cfg == "cuda":
        if torch.cuda.is_available():
            accelerator = "gpu"
        else:
            logging.warning(
                "Device set to 'cuda' but CUDA is not available. Falling back to CPU."
            )
            accelerator = "cpu"
    elif device_cfg == "mps":
        accelerator = "mps"
    else:
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    expes_config.device = accelerator if accelerator != "gpu" else "cuda"
    precision = "32"
    try:
        user_precision = getattr(expes_config, "precision", None)
    except Exception:
        user_precision = None
    if user_precision is not None:
        precision = str(user_precision)
    elif accelerator == "gpu":
        try:
            device_type = str(getattr(expes_config, "device_type", "") or "")
        except Exception:
            device_type = ""
        try:
            appliance_name = str(getattr(expes_config, "appliance", "") or "")
        except Exception:
            appliance_name = ""
        try:
            loss_type_for_prec = str(getattr(expes_config, "loss_type", "") or "")
        except Exception:
            loss_type_for_prec = ""
        force_fp32 = bool(
            str(getattr(expes_config, "name_model", "")).lower() == "nilmformer"
            and loss_type_for_prec in ("eaec", "ga_eaec")
            and (
                device_type in ("frequent_switching", "cycling_low_power")
                or appliance_name.lower() in ("fridge",)
            )
        )
        if force_fp32:
            precision = "32"
        else:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                precision = "bf16-mixed"
            else:
                precision = "16-mixed"
    tb_root = os.path.join("log", "tensorboard")
    os.makedirs(tb_root, exist_ok=True)
    tb_name = "{}_{}_{}_{}_{}".format(
        expes_config.dataset,
        expes_config.appliance,
        expes_config.sampling_rate,
        expes_config.window_size,
        expes_config.name_model,
    )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_root, name=tb_name)
    resume_flag = bool(getattr(expes_config, "resume", False))
    ckpt_path_resume = None
    if resume_flag:
        ckpt_last_candidates = [
            os.path.join(ckpt_root, "last.ckpt"),
            os.path.join(ckpt_root, ckpt_name + "_last.ckpt"),
        ]
        ckpt_last = None
        for cand in ckpt_last_candidates:
            if os.path.isfile(cand):
                ckpt_last = cand
                break
        if ckpt_last is not None:
            ckpt_path_resume = ckpt_last
            logging.info("Resume training from last checkpoint: %s", ckpt_last)
        else:
            logging.info(
                "Resume flag is set but no last checkpoint found at %s, train from scratch.",
                ckpt_last_candidates[0],
            )
    loss_type = str(getattr(expes_config, "loss_type", "eaec"))
    gradient_clip_val = 0.0
    if loss_type in ["eaec", "ga_eaec"]:
        gradient_clip_val = float(getattr(expes_config, "gradient_clip_val_eaec", 1.0))
    accumulate_grad_batches = int(
        getattr(expes_config, "accumulate_grad_batches", 1)
    )
    if accumulate_grad_batches < 1:
        accumulate_grad_batches = 1
    max_epochs = int(expes_config.epochs)
    if ckpt_path_resume is not None:
        try:
            ckpt_meta = torch.load(ckpt_path_resume, weights_only=False, map_location="cpu")
            ckpt_epoch = ckpt_meta.get("epoch", None)
            if ckpt_epoch is None:
                ckpt_epoch = (
                    ckpt_meta.get("loops", {})
                    .get("fit_loop", {})
                    .get("epoch_progress", {})
                    .get("current", {})
                    .get("completed", None)
                )
            if ckpt_epoch is not None:
                ckpt_epoch = int(ckpt_epoch)
                if max_epochs <= ckpt_epoch:
                    max_epochs = (ckpt_epoch + 1) + max(1, int(expes_config.epochs))
        except Exception:
            pass
    trainer_kwargs = dict(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=tb_logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = pl.Trainer(**trainer_kwargs)
    if ckpt_path_resume is not None:
        logging.info("Start model training with explicit resume.")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=ckpt_path_resume,
        )
    else:
        logging.info("Start model training from scratch (no resume).")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    best_model_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_model_path:
        try:
            ckpt = torch.load(best_model_path, weights_only=False, map_location="cpu")
            lightning_module.load_state_dict(ckpt["state_dict"])
        except Exception as e:
            logging.warning(
                "Could not load best checkpoint %s, keeping latest weights: %s",
                best_model_path,
                e,
            )
    inst_model.to(expes_config.device)
    if getattr(checkpoint_callback, "best_model_score", None) is not None:
        best_loss = float(checkpoint_callback.best_model_score)
    else:
        best_loss = float("inf")
    eval_log = {}
    skip_final_eval = getattr(expes_config, "skip_final_eval", False)
    if not skip_final_eval:
        logging.info("Eval model...")
        logging.info("Eval valid split metrics...")
        min_on_steps = int(getattr(expes_config, "postprocess_min_on_steps", 0))
        evaluate_nilm_split(
            inst_model,
            valid_loader,
            scaler,
            expes_config.threshold,
            expes_config.device,
            True,
            "valid_metrics",
            eval_log,
            min_on_steps,
            expes_config,
        )
        logging.info("Eval test split metrics...")
        evaluate_nilm_split(
            inst_model,
            test_loader,
            scaler,
            expes_config.threshold,
            expes_config.device,
            True,
            "test_metrics",
            eval_log,
            min_on_steps,
            expes_config,
        )
        if expes_config.name_model == "DiffNILM":
            eval_win_energy_aggregation(
                tuple_data[2],
                tuple_data[6],
                inst_model,
                expes_config.device,
                scaler=scaler,
                metrics=NILMmetrics(round_to=5),
                window_size=expes_config.window_size,
                freq=expes_config.sampling_rate,
                cosinbase=False,
                new_range=(-0.5, 0.5),
                mask_metric="test_metrics",
                list_exo_variables=["hour", "dow", "month"],
                threshold_small_values=expes_config.threshold,
                log_dict=eval_log,
            )
        else:
            eval_win_energy_aggregation(
                tuple_data[2],
                tuple_data[6],
                inst_model,
                expes_config.device,
                scaler=scaler,
                metrics=NILMmetrics(round_to=5),
                window_size=expes_config.window_size,
                freq=expes_config.sampling_rate,
                mask_metric="test_metrics",
                list_exo_variables=expes_config.list_exo_variables,
                threshold_small_values=expes_config.threshold,
                log_dict=eval_log,
            )

        writer = tb_logger.experiment
        if hasattr(lightning_module, "best_epoch") and lightning_module.best_epoch >= 0:
            epoch_idx = int(lightning_module.best_epoch)
        else:
            epoch_idx = int(lightning_module.current_epoch)
        for log_key, log_val in eval_log.items():
            if not (
                log_key.startswith("valid_metrics") or log_key.startswith("test_metrics")
            ):
                continue
            if isinstance(log_val, dict):
                if log_key.startswith("valid_"):
                    split = "valid"
                else:
                    split = "test"
                sub = log_key[len(split + "_metrics") :]
                for name, value in log_val.items():
                    if isinstance(value, (int, float, np.floating)):
                        tag = split + sub + "/" + name
                        writer.add_scalar(tag, float(value), epoch_idx)
    else:
        logging.info("Skip final eval metrics.")
    result_root = os.path.dirname(
        os.path.dirname(os.path.dirname(expes_config.result_path))
    )
    html_path = os.path.join(
        result_root,
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
        "val_compare.html",
    )
    logging.info(
        "Training and eval completed! Best checkpoint: %s, TensorBoard logdir: %s, HTML: %s",
        best_model_path,
        os.path.join(tb_root, tb_name),
        html_path,
    )
    result = {
        "best_loss": float(best_loss),
        "valid_timestamp": eval_log.get("valid_metrics_timestamp", {}),
        "valid_win": eval_log.get("valid_metrics_win", {}),
        "test_timestamp": eval_log.get("test_metrics_timestamp", {}),
        "test_win": eval_log.get("test_metrics_win", {}),
    }
    return result


def tser_model_training(inst_model, tuple_data, expes_config):
    expes_config.device = get_device()
    train_dataset = TSDatasetScaling(tuple_data[0][0], tuple_data[0][1])
    valid_dataset = TSDatasetScaling(tuple_data[1][0], tuple_data[1][1])
    test_dataset = TSDatasetScaling(tuple_data[2][0], tuple_data[2][1])

    num_workers = getattr(expes_config, "num_workers", 0)
    if os.name == "nt" and num_workers:
        num_workers = 0
    persistent_workers = num_workers > 0
    pin_memory = expes_config.device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    lightning_module = TserLightningModule(
        inst_model,
        learning_rate=float(expes_config.model_training_param.lr),
        weight_decay=float(expes_config.model_training_param.wd),
        criterion=nn.MSELoss(),
        patience_rlr=expes_config.p_rlr,
        n_warmup_epochs=expes_config.n_warmup_epochs,
        warmup_type=getattr(expes_config, "warmup_type", "linear"),
    )
    accelerator = "cpu"
    devices = 1
    if expes_config.device == "cuda" and torch.cuda.is_available():
        accelerator = "gpu"
    elif expes_config.device == "mps":
        accelerator = "mps"
    precision = "32"
    if accelerator == "gpu":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    tb_root = os.path.join("log", "tensorboard")
    os.makedirs(tb_root, exist_ok=True)
    tb_name = "{}_{}_{}_{}_{}".format(
        expes_config.dataset,
        expes_config.appliance,
        expes_config.sampling_rate,
        expes_config.window_size,
        expes_config.name_model,
    )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_root, name=tb_name)
    callbacks = []
    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=expes_config.p_es, mode="min"
            )
        )
    ckpt_root = os.path.join(
        "checkpoint",
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
        expes_config.appliance,
        "{}_{}".format(expes_config.name_model, expes_config.seed),
    )
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_name = "ckpt"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_root,
        filename=ckpt_name + "_{epoch:03d}",
    )
    callbacks.append(checkpoint_callback)
    resume_flag = bool(getattr(expes_config, "resume", False))
    ckpt_path_resume = None
    if resume_flag:
        ckpt_last_candidates = [
            os.path.join(ckpt_root, "last.ckpt"),
            os.path.join(ckpt_root, ckpt_name + "_last.ckpt"),
        ]
        ckpt_last = None
        for cand in ckpt_last_candidates:
            if os.path.isfile(cand):
                ckpt_last = cand
                break
        if ckpt_last is not None:
            ckpt_path_resume = ckpt_last
            logging.info("Resume TSER training from last checkpoint: %s", ckpt_last)
        else:
            logging.info(
                "Resume flag is set for TSER but no last checkpoint found at %s, train from scratch.",
                ckpt_last_candidates[0],
            )
    max_epochs = int(expes_config.epochs)
    if ckpt_path_resume is not None:
        try:
            ckpt_meta = torch.load(ckpt_path_resume, weights_only=False, map_location="cpu")
            ckpt_epoch = ckpt_meta.get("epoch", None)
            if ckpt_epoch is None:
                ckpt_epoch = (
                    ckpt_meta.get("loops", {})
                    .get("fit_loop", {})
                    .get("epoch_progress", {})
                    .get("current", {})
                    .get("completed", None)
                )
            if ckpt_epoch is not None:
                ckpt_epoch = int(ckpt_epoch)
                if max_epochs <= ckpt_epoch:
                    max_epochs = (ckpt_epoch + 1) + max(1, int(expes_config.epochs))
        except Exception:
            pass
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=tb_logger,
    )
    if ckpt_path_resume is not None:
        logging.info("Start TSER model training with explicit resume.")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=ckpt_path_resume,
        )
    else:
        logging.info("Start TSER model training from scratch (no resume).")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    best_model_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_model_path:
        try:
            ckpt = torch.load(best_model_path, weights_only=False, map_location="cpu")
            lightning_module.load_state_dict(ckpt["state_dict"])
        except Exception as e:
            logging.warning(
                "Could not load best checkpoint %s, keeping latest weights: %s",
                best_model_path,
                e,
            )
    inst_model.to(expes_config.device)
    logging.info(
        "Training and eval completed! Best checkpoint: %s, TensorBoard logdir: %s",
        best_model_path,
        os.path.join(tb_root, tb_name),
    )
    return None


def launch_models_training(data_tuple, scaler, expes_config):
    if "cutoff" in expes_config.model_kwargs:
        expes_config.model_kwargs.cutoff = expes_config.cutoff

    if "threshold" in expes_config.model_kwargs:
        expes_config.model_kwargs.threshold = expes_config.threshold

    if expes_config.name_model == "NILMFormer" and scaler is not None:
        try:
            n_app = int(getattr(scaler, "n_appliance", 1) or 1)
        except Exception:
            n_app = 1
        if n_app < 1:
            n_app = 1
        try:
            expes_config.model_kwargs["c_out"] = n_app
        except Exception:
            try:
                tmp_kwargs = dict(expes_config.model_kwargs)
                tmp_kwargs["c_out"] = n_app
                expes_config.model_kwargs = tmp_kwargs
            except Exception:
                pass

    model_instance = get_model_instance(
        name_model=expes_config.name_model,
        c_in=(1 + 2 * len(expes_config.list_exo_variables)),
        window_size=expes_config.window_size,
        **expes_config.model_kwargs,
    )

    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        result = tser_model_training(model_instance, data_tuple, expes_config)
    else:
        result = nilm_model_training(model_instance, data_tuple, scaler, expes_config)

    del model_instance
    return result
