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
        inst = NILMFormer(NILMFormerConfig(c_in=1, c_embedding=c_in - 1, **kwargs))
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
    if hasattr(model_trainer, "model"):
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
        for ts_agg, appl, _ in valid_loader:
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            appl_t = appl.float().to(device)
            pred_t = model(ts_agg_t)
            pred_t = torch.clamp(pred_t, min=0.0)
            agg_np = scaler.inverse_transform_agg_power(
                ts_agg_t[:, 0:1, :].detach().cpu().numpy()
            )
            target_np = scaler.inverse_transform_appliance(
                appl_t.detach().cpu().numpy()
            )
            pred_np = scaler.inverse_transform_appliance(
                pred_t.detach().cpu().numpy()
            )
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
    if appliance_name is not None:
        existing_indices = [
            i for i, name in enumerate(appliance_names) if name == appliance_name
        ]
    else:
        existing_indices = []
    if existing_indices:
        start_idx = existing_indices[0]
        required_len = start_idx + n_app
        if len(target_all) < required_len:
            target_all.extend([[] for _ in range(required_len - len(target_all))])
        for j in range(n_app):
            target_all[start_idx + j] = target_concat[j]
    else:
        start_idx = len(target_all)
        for j in range(n_app):
            target_all.append(target_concat[j])
            if appliance_name is not None:
                appliance_names.append(appliance_name)
            else:
                appliance_names.append(str(start_idx + j))
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
      if (currentLayout && currentLayout.xaxis && currentLayout.yaxis) {{
        layout = {{
          xaxis: {{range: currentLayout.xaxis.range}},
          yaxis: {{range: currentLayout.yaxis.range}},
          margin: {{t: 40}}
        }};
      }} else {{
        layout = {{
          xaxis: {{title: 'Time index'}},
          yaxis: {{title: 'Power'}},
          margin: {{t: 40}}
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

    def on_validation_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        threshold_small_values = self.expes_config.threshold
        min_on_steps = int(getattr(self.expes_config, "postprocess_min_on_steps", 0))
        y = np.array([])
        y_hat = np.array([])
        y_win = np.array([])
        y_hat_win = np.array([])
        y_state = np.array([])
        with torch.no_grad():
            for ts_agg, appl, state in self.valid_loader:
                pl_module.eval()
                ts_agg_t = ts_agg.float().to(device)
                target = appl.float().to(device)
                pred = pl_module(ts_agg_t)
                pred = torch.clamp(pred, min=0.0)
                target_inv = self.scaler.inverse_transform_appliance(target)
                pred_inv = self.scaler.inverse_transform_appliance(pred)
                pred_inv[pred_inv < threshold_small_values] = 0
                if min_on_steps > 1:
                    pred_inv = suppress_short_activations(
                        pred_inv, threshold_small_values, min_on_steps
                    )
                target_win = target_inv.sum(dim=-1)
                pred_win = pred_inv.sum(dim=-1)
                target_flat = torch.flatten(target_inv).detach().cpu().numpy()
                pred_flat = torch.flatten(pred_inv).detach().cpu().numpy()
                target_win_flat = (
                    torch.flatten(target_win).detach().cpu().numpy()
                )
                pred_win_flat = torch.flatten(pred_win).detach().cpu().numpy()
                state_flat = state.flatten().detach().cpu().numpy()
                y = (
                    np.concatenate((y, target_flat))
                    if y.size
                    else target_flat
                )
                y_hat = (
                    np.concatenate((y_hat, pred_flat))
                    if y_hat.size
                    else pred_flat
                )
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
                y_state = (
                    np.concatenate((y_state, state_flat))
                    if y_state.size
                    else state_flat
                )
        if not y.size:
            return
        y_hat_state = (
            (y_hat > threshold_small_values).astype(int)
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
):
    metrics_helper = NILMmetrics()
    y = np.array([])
    y_hat = np.array([])
    y_win = np.array([])
    y_hat_win = np.array([])
    y_state = np.array([])
    with torch.no_grad():
        iterator = data_loader
        try:
            total = len(data_loader)
        except TypeError:
            total = None
        iterator = tqdm(iterator, total=total, desc=mask, leave=False)
        for ts_agg, appl, state in iterator:
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            target = appl.float().to(device)
            pred = model(ts_agg_t)
            pred = torch.clamp(pred, min=0.0)
            target_inv = scaler.inverse_transform_appliance(target)
            pred_inv = scaler.inverse_transform_appliance(pred)
            pred_inv[pred_inv < threshold_small_values] = 0
            if min_on_duration_steps and min_on_duration_steps > 1:
                pred_inv = suppress_short_activations(
                    pred_inv, threshold_small_values, min_on_duration_steps
                )
            target_win = target_inv.sum(dim=-1)
            pred_win = pred_inv.sum(dim=-1)
            target_flat = torch.flatten(target_inv).detach().cpu().numpy()
            pred_flat = torch.flatten(pred_inv).detach().cpu().numpy()
            target_win_flat = torch.flatten(target_win).detach().cpu().numpy()
            pred_win_flat = torch.flatten(pred_win).detach().cpu().numpy()
            state_flat = state.flatten().detach().cpu().numpy()
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
            y_state = (
                np.concatenate((y_state, state_flat)) if y_state.size else state_flat
            )
    if not y.size:
        return {}, {}
    y_hat_state = (
        (y_hat > threshold_small_values).astype(int) if y_state.size else None
    )
    metrics_timestamp = metrics_helper(
        y=y,
        y_hat=y_hat,
        y_state=y_state if y_state.size else None,
        y_hat_state=y_hat_state,
    )
    metrics_win = metrics_helper(y=y_win, y_hat=y_hat_win)
    log_dict[mask + "_timestamp"] = metrics_timestamp
    log_dict[mask + "_win"] = metrics_win
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

    loss_type = str(getattr(expes_config, "loss_type", "eaec"))
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

    num_workers = getattr(expes_config, "num_workers", 0)
    persistent_workers = num_workers > 0
    pin_memory = expes_config.device == "cuda"
    batch_size = expes_config.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
        loss_type = str(getattr(expes_config, "loss_type", "eaec"))
        threshold_loss = float(
            getattr(expes_config, "loss_threshold", expes_config.threshold)
        )
        if loss_type in ["eaec", "ga_eaec"]:
            alpha_on = float(getattr(expes_config, "loss_alpha_on", 3.0))
            alpha_off = float(getattr(expes_config, "loss_alpha_off", 1.0))
            lambda_grad = float(getattr(expes_config, "loss_lambda_grad", 0.5))
            lambda_energy = float(getattr(expes_config, "loss_lambda_energy", 0.5))
            soft_temp = float(getattr(expes_config, "loss_soft_temp", 10.0))
            edge_eps = float(getattr(expes_config, "loss_edge_eps", 5.0))
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
        lightning_module = SeqToSeqLightningModule(
            inst_model,
            learning_rate=float(expes_config.model_training_param.lr),
            weight_decay=float(expes_config.model_training_param.wd),
            criterion=criterion,
            patience_rlr=expes_config.p_rlr,
            n_warmup_epochs=expes_config.n_warmup_epochs,
            warmup_type=getattr(expes_config, "warmup_type", "linear"),
            neg_penalty_weight=float(getattr(expes_config, "neg_penalty_weight", 0.1)),
            rlr_factor=float(getattr(expes_config, "rlr_factor", 0.1)),
            rlr_min_lr=float(getattr(expes_config, "rlr_min_lr", 0.0)),
            state_zero_penalty_weight=state_zero_penalty_weight,
            zero_run_kernel=zero_run_kernel,
            zero_run_ratio=zero_run_ratio,
            loss_threshold=threshold_loss,
            off_high_agg_penalty_weight=off_high_agg_penalty_weight,
            gate_cls_weight=float(getattr(expes_config, "gate_cls_weight", 0.0)),
            gate_window_weight=float(
                getattr(expes_config, "gate_window_weight", 0.0)
            ),
            gate_focal_gamma=float(getattr(expes_config, "gate_focal_gamma", 2.0)),
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
    resume_flag = bool(getattr(expes_config, "resume", False))
    ckpt_path_resume = None
    if resume_flag:
        ckpt_last = os.path.join(ckpt_root, ckpt_name + "_last.ckpt")
        if os.path.isfile(ckpt_last):
            ckpt_path_resume = ckpt_last
            logging.info("Resume training from last checkpoint: %s", ckpt_last)
        else:
            logging.info(
                "Resume flag is set but no last checkpoint found at %s, train from scratch.",
                ckpt_last,
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
    trainer_kwargs = dict(
        max_epochs=expes_config.epochs,
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
            ckpt = torch.load(best_model_path, weights_only=False)
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
        ckpt_last = os.path.join(ckpt_root, ckpt_name + "_last.ckpt")
        if os.path.isfile(ckpt_last):
            ckpt_path_resume = ckpt_last
            logging.info("Resume TSER training from last checkpoint: %s", ckpt_last)
        else:
            logging.info(
                "Resume flag is set for TSER but no last checkpoint found at %s, train from scratch.",
                ckpt_last,
            )
    trainer = pl.Trainer(
        max_epochs=expes_config.epochs,
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
            ckpt = torch.load(best_model_path, weights_only=False)
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
