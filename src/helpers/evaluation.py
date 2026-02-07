#################################################################################################################
#
# Evaluation helpers extracted from src/helpers/expes.py
#
#################################################################################################################

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections.abc import Sequence, Mapping

from src.helpers.metrics import NILMmetrics
from src.helpers.postprocess import (
    suppress_short_activations,
    suppress_long_off_with_gate,
)
from src.helpers.inference import _crop_center_tensor


def _coerce_appliance_names(expes_config, n_app, fallback_name=None):
    from src.helpers.experiment import _coerce_appliance_names as _impl
    return _impl(expes_config, n_app, fallback_name)


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
    result_root = os.path.dirname(
        os.path.dirname(os.path.dirname(expes_config.result_path))
    )
    group_dir = os.path.join(
        result_root,
        "{}_{}".format(expes_config.dataset, expes_config.sampling_rate),
        str(expes_config.window_size),
    )
    appliance_name = getattr(expes_config, "appliance", None)
    if appliance_name is not None:
        group_dir = os.path.join(group_dir, str(appliance_name))
    os.makedirs(group_dir, exist_ok=True)
    html_path = os.path.join(group_dir, "val_compare.html")
    payload = {}
    existing_agg = payload.get("agg")
    existing_target = payload.get("target")
    existing_timestamps = payload.get("timestamps")
    existing_appliance_names = payload.get("appliance_names")
    can_reuse_static = (
        isinstance(existing_agg, list)
        and isinstance(existing_target, list)
        and (not has_time or isinstance(existing_timestamps, list))
    )
    reuse_static = False
    timestamp_to_index = None
    global_idx = 0
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
            # seq2subseq: crop OUTPUTS to center region (model still sees full context)
            output_ratio = float(getattr(expes_config, "output_ratio", 1.0))
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

            # seq2subseq: crop outputs to center region (model sees full context, we only evaluate center)
            if output_ratio < 1.0:
                pred_t = _crop_center_tensor(pred_t, output_ratio)
                appl_t = _crop_center_tensor(appl_t, output_ratio)
                ts_agg_t = _crop_center_tensor(ts_agg_t, output_ratio)
                if gate_logits is not None:
                    gate_logits = _crop_center_tensor(gate_logits, output_ratio)

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
            if pred_inv.dim() == 3:
                per_device_cfg = getattr(expes_config, "postprocess_per_device", None)
                if isinstance(per_device_cfg, Mapping) and per_device_cfg:
                    per_device_cfg_norm = {
                        str(k).strip().lower(): v for k, v in per_device_cfg.items()
                    }
                    n_app = int(pred_inv.size(1))
                    device_names = _coerce_appliance_names(
                        expes_config, n_app, appliance_name
                    )
                    for j in range(n_app):
                        name_j = (
                            device_names[j] if j < len(device_names) else str(j)
                        )
                        cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                        thr_j = float(threshold_postprocess)
                        min_on_j = int(min_on_steps)
                        if isinstance(cfg_j, Mapping):
                            thr_j = float(
                                cfg_j.get("postprocess_threshold", thr_j)
                            )
                            min_on_j = int(
                                cfg_j.get(
                                    "postprocess_min_on_steps", min_on_j
                                )
                            )
                        ch = pred_inv[:, j : j + 1, :]
                        ch[ch < thr_j] = 0.0
                        if min_on_j > 1:
                            ch = suppress_short_activations(
                                ch, thr_j, min_on_j
                            )
                        pred_inv[:, j : j + 1, :] = ch
                else:
                    pred_inv[pred_inv < threshold_postprocess] = 0.0
                    if min_on_steps > 1:
                        pred_inv = suppress_short_activations(
                            pred_inv, threshold_postprocess, min_on_steps
                        )
            else:
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
            if target_concat is None and pred_concat is None:
                n_app = target_np.shape[1]
                if can_reuse_static and isinstance(existing_appliance_names, list):
                    if isinstance(existing_target, list) and len(existing_target) == n_app:
                        reuse_static = True
                if reuse_static:
                    if isinstance(existing_agg, list):
                        agg_concat = existing_agg
                    if isinstance(existing_target, list):
                        target_concat = existing_target
                    if has_time and isinstance(existing_timestamps, list):
                        timestamps_concat = existing_timestamps
                        timestamp_to_index = {
                            ts: i for i, ts in enumerate(existing_timestamps)
                        }
                    total_len = len(agg_concat)
                    pred_concat = [[0.0] * total_len for _ in range(n_app)]
                else:
                    target_concat = [[] for _ in range(n_app)]
                    pred_concat = [[] for _ in range(n_app)]
            if reuse_static:
                if has_time and timestamp_to_index is not None and dataset is not None:
                    for b in range(batch_size):
                        idx_sample = sample_idx + b
                        start = dataset.st_date[idx_sample]
                        tmp = pd.date_range(
                            start=start,
                            periods=pred_np.shape[2],
                            freq=dataset.freq,
                        )
                        ts_list = tmp.strftime("%Y-%m-%d %H:%M:%S").tolist()
                        for t, ts_str in enumerate(ts_list):
                            if ts_str in timestamps_set:
                                continue
                            timestamps_set.add(ts_str)
                            pos = timestamp_to_index.get(ts_str)
                            if pos is None or pos >= len(pred_concat[0]):
                                continue
                            for j in range(pred_np.shape[1]):
                                pred_concat[j][pos] = float(pred_np[b, j, t])
                else:
                    for b in range(batch_size):
                        for t in range(pred_np.shape[2]):
                            if global_idx >= len(pred_concat[0]):
                                break
                            for j in range(pred_np.shape[1]):
                                pred_concat[j][global_idx] = float(pred_np[b, j, t])
                            global_idx += 1
                sample_idx += batch_size
                continue
            for b in range(batch_size):
                if has_time and dataset is not None:
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
    n_app = len(target_concat)
    model_name = expes_config.name_model
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
    display_names = _coerce_appliance_names(expes_config, n_app, appliance_name)
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
    name_to_index = {str(name): idx for idx, name in enumerate(appliance_names)}
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
        key_j = str(display_names[j])
        idx = name_to_index.get(key_j)
        if idx is None or idx >= total_n:
            continue
        pred_list[idx] = pred_concat[j]
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
    criterion=None,
):
    metrics_helper = NILMmetrics()
    y = np.array([], dtype=np.float32)
    y_hat = np.array([], dtype=np.float32)
    y_win = np.array([], dtype=np.float32)
    y_hat_win = np.array([], dtype=np.float32)
    y_state = np.array([], dtype=np.int8)
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
    # seq2subseq: ratio of center region to evaluate (model sees full context)
    output_ratio = 1.0
    if expes_config is not None:
        output_ratio = float(getattr(expes_config, "output_ratio", 1.0))
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
                    # V7.5: Use per-device learned gate params to match training
                    _applied_perdevice = False
                    if (criterion is not None
                            and hasattr(criterion, "gate_soft_scales")
                            and hasattr(criterion, "gate_biases")
                            and hasattr(criterion, "gate_floors")):
                        _ss = criterion.gate_soft_scales.detach().to(gate_logits.device)
                        _bi = criterion.gate_biases.detach().to(gate_logits.device)
                        _fl = criterion.gate_floors.detach().to(gate_logits.device)
                        C = gate_logits.size(1)
                        if _ss.numel() == C and _bi.numel() == C and _fl.numel() == C:
                            _ss = torch.clamp(_ss, min=0.5, max=6.0).view(1, C, 1)
                            _fl = torch.clamp(_fl, min=0.01, max=0.5).view(1, C, 1)
                            _bi = _bi.view(1, C, 1)
                            gl = gate_logits.float()
                            if hasattr(criterion, "_gate_logits_floors"):
                                _glf = criterion._gate_logits_floors
                                if len(_glf) == C:
                                    _glf_t = torch.tensor(
                                        _glf, device=gl.device, dtype=gl.dtype
                                    ).view(1, C, 1)
                                    gl = torch.maximum(gl, _glf_t)
                            gl_adj = gl * _ss + _bi
                            gate_prob = torch.sigmoid(
                                torch.clamp(gl_adj, min=-50.0, max=50.0)
                            )
                            pred = power_raw * (_fl + (1.0 - _fl) * gate_prob)
                            _applied_perdevice = True
                    if not _applied_perdevice:
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
            # seq2subseq: crop outputs to center region (model sees full context, we only evaluate center)
            if output_ratio < 1.0:
                pred = _crop_center_tensor(pred, output_ratio)
                target = _crop_center_tensor(target, output_ratio)
                if gate_logits is not None:
                    gate_logits = _crop_center_tensor(gate_logits, output_ratio)
                if state is not None:
                    state = _crop_center_tensor(state.float(), output_ratio)
            target_inv = scaler.inverse_transform_appliance(target)
            pred_inv = scaler.inverse_transform_appliance(pred)
            pred_inv = torch.clamp(pred_inv, min=0.0)
            per_device_cfg = None
            if expes_config is not None:
                per_device_cfg = getattr(expes_config, "postprocess_per_device", None)
            if pred_inv.dim() == 3 and isinstance(per_device_cfg, Mapping) and per_device_cfg:
                per_device_cfg_norm = {
                    str(k).strip().lower(): v for k, v in per_device_cfg.items()
                }
                n_app = int(pred_inv.size(1))
                device_names = _coerce_appliance_names(
                    expes_config, n_app, getattr(expes_config, "appliance", None) if expes_config is not None else None
                )
                for j in range(n_app):
                    name_j = device_names[j] if j < len(device_names) else str(j)
                    cfg_j = per_device_cfg_norm.get(str(name_j).strip().lower())
                    thr_j = float(threshold_postprocess)
                    min_on_j = int(min_on_duration_steps or 0)
                    if isinstance(cfg_j, Mapping):
                        thr_j = float(cfg_j.get("postprocess_threshold", thr_j))
                        min_on_j = int(cfg_j.get("postprocess_min_on_steps", min_on_j))
                    ch = pred_inv[:, j : j + 1, :]
                    ch[ch < thr_j] = 0.0
                    if min_on_j > 1:
                        ch = suppress_short_activations(ch, thr_j, min_on_j)
                    pred_inv[:, j : j + 1, :] = ch
            else:
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
                    # V7.5: Use per-device learned params for postprocess gate too
                    _pp_perdevice = False
                    if (criterion is not None
                            and hasattr(criterion, "gate_soft_scales")
                            and hasattr(criterion, "gate_biases")):
                        _ss = criterion.gate_soft_scales.detach().to(gate_logits.device)
                        _bi = criterion.gate_biases.detach().to(gate_logits.device)
                        C = gate_logits.size(1)
                        if _ss.numel() == C and _bi.numel() == C:
                            # Use 3x learned scale for sharp postprocess gate
                            _ss_sharp = torch.clamp(_ss * 3.0, min=1.5, max=18.0).view(1, C, 1)
                            _bi = _bi.view(1, C, 1)
                            gl = gate_logits.float()
                            if hasattr(criterion, "_gate_logits_floors"):
                                _glf = criterion._gate_logits_floors
                                if len(_glf) == C:
                                    _glf_t = torch.tensor(
                                        _glf, device=gl.device, dtype=gl.dtype
                                    ).view(1, C, 1)
                                    gl = torch.maximum(gl, _glf_t)
                            gl_adj = gl * _ss_sharp + _bi
                            gate_prob_sharp = torch.sigmoid(
                                torch.clamp(gl_adj, min=-50.0, max=50.0)
                            )
                            _pp_perdevice = True
                    if not _pp_perdevice:
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
            target_np_all = target_inv.detach().cpu().numpy().astype(np.float32, copy=False)
            pred_np_all = pred_inv.detach().cpu().numpy().astype(np.float32, copy=False)
            target_win_np_all = target_win.detach().cpu().numpy().astype(np.float32, copy=False)
            pred_win_np_all = pred_win.detach().cpu().numpy().astype(np.float32, copy=False)
            target_flat = target_np_all.reshape(-1)
            pred_flat = pred_np_all.reshape(-1)
            target_win_flat = target_win_np_all.reshape(-1)
            pred_win_flat = pred_win_np_all.reshape(-1)
            state_np_all = None
            state_flat = np.array([], dtype=np.int8)
            if state is not None:
                state_np_all = state.detach().cpu().numpy().astype(np.int8, copy=False)
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
                    "y": [np.array([], dtype=np.float32) for _ in range(n_app)],
                    "y_hat": [np.array([], dtype=np.float32) for _ in range(n_app)],
                    "y_win": [np.array([], dtype=np.float32) for _ in range(n_app)],
                    "y_hat_win": [np.array([], dtype=np.float32) for _ in range(n_app)],
                    "y_state": [np.array([], dtype=np.int8) for _ in range(n_app)],
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
    # Safety: ensure y_state matches y_hat length (can differ with output_ratio cropping)
    if y_state.size and y_state.size != y_hat.size:
        # Crop y_state to match y_hat if needed (center crop like output_ratio)
        if y_state.size > y_hat.size:
            ratio = y_hat.size / y_state.size
            crop_len = y_hat.size
            start = (y_state.size - crop_len) // 2
            y_state = y_state[start:start + crop_len]
        else:
            # y_state shorter than y_hat - this shouldn't happen, but handle gracefully
            y_state = np.array([])
    # FIX: Use threshold_postprocess instead of 0 for consistent state determination
    # The postprocessing already set values < threshold to 0, so any remaining > 0 values
    # are above threshold. Using a small epsilon (0.01) to handle floating point noise.
    y_hat_state = (y_hat > 0.01).astype(int) if y_state.size else None
    metrics_timestamp = metrics_helper(
        y=y,
        y_hat=y_hat,
        y_state=y_state if y_state.size else None,
        y_hat_state=y_hat_state,
    )
    metrics_win = metrics_helper(y=y_win, y_hat=y_hat_win)
    metrics_timestamp_per_device = {}
    metrics_win_per_device = {}

    # Build per-device threshold map for state calculation
    per_device_thr_map = {}
    if expes_config is not None:
        try:
            per_device_cfg = getattr(expes_config, "postprocess_per_device", None)
            if isinstance(per_device_cfg, Mapping):
                per_device_cfg_norm = {str(k).strip().lower(): v for k, v in per_device_cfg.items()}
                per_device_thr_map = {
                    k: float(v.get("postprocess_threshold", threshold_postprocess))
                    if isinstance(v, Mapping) else threshold_postprocess
                    for k, v in per_device_cfg_norm.items()
                }
        except Exception:
            pass

    if per_device_data is not None:
        n_app = len(per_device_data["y"])
        device_names = _coerce_appliance_names(
            expes_config, n_app, getattr(expes_config, "appliance", None) if expes_config is not None else None
        )
        for j in range(n_app):
            y_j = per_device_data["y"][j]
            y_hat_j = per_device_data["y_hat"][j]
            if y_j.size and y_hat_j.size:
                y_state_j = per_device_data["y_state"][j]
                # Safety: ensure y_state_j matches y_hat_j length
                if y_state_j.size and y_state_j.size != y_hat_j.size:
                    if y_state_j.size > y_hat_j.size:
                        crop_len = y_hat_j.size
                        start = (y_state_j.size - crop_len) // 2
                        y_state_j = y_state_j[start:start + crop_len]
                    else:
                        y_state_j = np.array([])
                # FIX: Use device-specific threshold for state calculation
                name_j = device_names[j] if j < len(device_names) else str(j)
                thr_j = per_device_thr_map.get(str(name_j).strip().lower(), threshold_postprocess)
                # Use small epsilon after postprocessing (values already thresholded)
                y_hat_state_j = (y_hat_j > 0.01).astype(int) if y_state_j.size else None
                metrics_timestamp_per_device[str(device_names[j])] = metrics_helper(
                    y=y_j,
                    y_hat=y_hat_j,
                    y_state=y_state_j if y_state_j.size else None,
                    y_hat_state=y_hat_state_j,
                )
            y_win_j = per_device_data["y_win"][j]
            y_hat_win_j = per_device_data["y_hat_win"][j]
            if y_win_j.size and y_hat_win_j.size:
                metrics_win_per_device[str(device_names[j])] = metrics_helper(
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
