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
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch.nn as nn

from src.helpers.trainer import (
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


def _save_val_streamlit_data(model_trainer, valid_loader, scaler, expes_config, epoch_idx):
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
                agg_concat.extend(agg_np[b, 0, :].tolist())
                for j in range(target_np.shape[1]):
                    target_concat[j].extend(target_np[b, j, :].tolist())
                    pred_concat[j].extend(pred_np[b, j, :].tolist())
    if target_concat is None:
        return
    n_app = len(target_concat)
    group_dir = os.path.dirname(expes_config.result_path)
    json_path = os.path.join(group_dir, "val_compare.json")
    html_path = os.path.join(group_dir, "val_compare.html")
    model_name = expes_config.name_model
    payload = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    payload["agg"] = agg_concat
    payload["target"] = target_concat
    appliance_name = getattr(expes_config, "appliance", None)
    if appliance_name is not None:
        payload["appliance_names"] = [appliance_name] * n_app
    models = payload.get("models", {})
    models[model_name] = {
        "epoch": int(epoch_idx),
        "pred": pred_concat,
    }
    payload["models"] = models
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    json_data = json.dumps(payload)
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
<div>
  <label><input type="checkbox" id="showAgg" checked> Agg</label>
  <label><input type="checkbox" id="showTarget" checked> Target</label>
  <label><input type="checkbox" id="showPred" checked> Pred</label>
</div>
<div id="plot"></div>
<script>
  const payload = {json_data};
  const agg = payload.agg;
  const target = payload.target;
  const models = payload.models || {{}};
  const modelNames = Object.keys(models);
  const nApp = target.length;
  const applianceNames = payload.appliance_names || [];

  const modelSelect = document.getElementById('modelSelect');
  const appSelect = document.getElementById('appSelect');
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

  function makePlot() {{
    const aj = parseInt(appSelect.value);
    const x = Array.from({{length: agg.length}}, (_, i) => i);
    const data = [];
    if (showAgg.checked) {{
      data.push({{
        x: x,
        y: agg,
        name: 'Aggregate',
        mode: 'lines',
        line: {{color: '#7f7f7f'}},
      }});
    }}
    if (showTarget.checked) {{
      data.push({{
        x: x,
        y: target[aj],
        name: 'Target',
        mode: 'lines',
        line: {{color: '#2ca02c'}},
      }});
    }}
    if (showPred.checked) {{
      const selectedModels = Array.from(modelSelect.selectedOptions).map(opt => opt.value);
      for (let k = 0; k < selectedModels.length; k++) {{
        const mj = selectedModels[k];
        const modelData = models[mj];
        if (!modelData || !modelData.pred || !modelData.pred[aj]) {{
          continue;
        }}
        data.push({{
          x: x,
          y: modelData.pred[aj],
          name: 'Prediction: ' + mj,
          mode: 'lines',
          line: {{color: predColors[k % predColors.length]}},
        }});
      }}
    }}
    const layout = {{
      xaxis: {{title: 'Time index'}},
      yaxis: {{title: 'Power'}},
      margin: {{t: 40}}
    }};
    Plotly.newPlot('plot', data, layout);
  }}

  modelSelect.addEventListener('change', makePlot);
  appSelect.addEventListener('change', makePlot);
  showAgg.addEventListener('change', makePlot);
  showTarget.addEventListener('change', makePlot);
  showPred.addEventListener('change', makePlot);

  if (nApp > 0) {{
    appSelect.value = '0';
  }}
  if (nApp > 0 && modelNames.length > 0) {{
    for (let i = 0; i < modelSelect.options.length; i++) {{
      modelSelect.options[i].selected = true;
    }}
    makePlot();
  }}
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
        epoch_idx = trainer.current_epoch + 1
        _save_val_streamlit_data(
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


def evaluate_nilm_split(
    model,
    data_loader,
    scaler,
    threshold_small_values,
    device,
    save_outputs,
    mask,
    log_dict,
):
    metrics_helper = NILMmetrics()
    y = np.array([])
    y_hat = np.array([])
    y_win = np.array([])
    y_hat_win = np.array([])
    y_state = np.array([])
    with torch.no_grad():
        for ts_agg, appl, state in data_loader:
            model.eval()
            ts_agg_t = ts_agg.float().to(device)
            target = appl.float().to(device)
            pred = model(ts_agg_t)
            pred = torch.clamp(pred, min=0.0)
            target_inv = scaler.inverse_transform_appliance(target)
            pred_inv = scaler.inverse_transform_appliance(pred)
            pred_inv[pred_inv < threshold_small_values] = 0
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


def _plot_loss_history(loss_train, loss_valid, path_prefix):
    if not loss_train and not loss_valid:
        return
    epochs = range(len(loss_train))
    plt.figure()
    if loss_train:
        plt.plot(epochs, loss_train, label="Train loss")
    if loss_valid:
        plt.plot(range(len(loss_valid)), loss_valid, label="Valid loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    path_fig = path_prefix + "_loss.png"
    try:
        plt.savefig(path_fig)
    finally:
        plt.close()


def nilm_model_training(inst_model, tuple_data, scaler, expes_config):
    expes_config.device = get_device()
    ckpt_path = expes_config.result_path + ".pt"
    previous_best_loss = None
    previous_ckpt = None
    if os.path.isfile(ckpt_path):
        try:
            previous_ckpt = torch.load(ckpt_path, weights_only=False)
        except Exception as e:
            logging.warning(
                "Could not load existing checkpoint %s, start from scratch: %s",
                ckpt_path,
                e,
            )
            previous_ckpt = None
        else:
            state_dict = None
            if "best_model_state_dict" in previous_ckpt:
                state_dict = previous_ckpt["best_model_state_dict"]
            elif "model_state_dict" in previous_ckpt:
                state_dict = previous_ckpt["model_state_dict"]
            if state_dict is not None:
                try:
                    inst_model.load_state_dict(state_dict)
                except RuntimeError as e:
                    logging.warning(
                        "Checkpoint %s is incompatible with current model, skip warm-start: %s",
                        ckpt_path,
                        e,
                    )
                    previous_ckpt = None
                else:
                    if "value_best_loss" in previous_ckpt:
                        previous_best_loss = float(previous_ckpt["value_best_loss"])

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

    num_workers = getattr(expes_config, "num_workers", 0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metric_callback = ValidationNILMMetricCallback(valid_loader, scaler, expes_config)
    callbacks = [metric_callback]
    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=expes_config.p_es, mode="min"
            )
        )
    if expes_config.name_model == "DiffNILM":
        lightning_module = DiffNILMLightningModule(inst_model)
    elif expes_config.name_model == "STNILM":
        lightning_module = STNILMLightningModule(
            inst_model,
            learning_rate=expes_config.model_training_param.lr,
            weight_decay=expes_config.model_training_param.wd,
            patience_rlr=expes_config.p_rlr,
            n_warmup_epochs=expes_config.n_warmup_epochs,
            warmup_type=getattr(expes_config, "warmup_type", "linear"),
        )
    else:
        lightning_module = SeqToSeqLightningModule(
            inst_model,
            learning_rate=expes_config.model_training_param.lr,
            weight_decay=expes_config.model_training_param.wd,
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
    tb_root = os.path.join("log", "tensorboard")
    os.makedirs(tb_root, exist_ok=True)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_root, name="")
    trainer = pl.Trainer(
        max_epochs=expes_config.epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=tb_logger,
    )
    logging.info("Model training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    if lightning_module.best_model_state_dict is not None:
        inst_model.load_state_dict(lightning_module.best_model_state_dict)
    inst_model.to(expes_config.device)
    ckpt_root = os.path.join("checkpoint")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_root,
        "{}_{}_{}_{}_{}".format(
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            expes_config.window_size,
            expes_config.name_model,
        ),
    )
    class EvalContainer:
        def __init__(self, model, device, path_checkpoint):
            self.model = model
            self.device = device
            self.log = {}
            self.path_checkpoint = path_checkpoint
            self.best_loss = lightning_module.best_val_loss
            self.loss_train_history = lightning_module.loss_train_history
            self.loss_valid_history = lightning_module.loss_valid_history
            self.passed_epochs = len(lightning_module.loss_train_history)

        def save(self):
            torch.save(self.log, self.path_checkpoint + ".pt")

    eval_trainer = EvalContainer(inst_model, expes_config.device, ckpt_path)
    eval_trainer.log["best_model_state_dict"] = inst_model.state_dict()
    logging.info("Eval model...")
    evaluate_nilm_split(
        inst_model,
        valid_loader,
        scaler,
        expes_config.threshold,
        expes_config.device,
        True,
        "valid_metrics",
        eval_trainer.log,
    )
    evaluate_nilm_split(
        inst_model,
        test_loader,
        scaler,
        expes_config.threshold,
        expes_config.device,
        True,
        "test_metrics",
        eval_trainer.log,
    )

    if expes_config.name_model == "DiffNILM":
        eval_win_energy_aggregation(
            tuple_data[2],
            tuple_data[6],
            eval_trainer,
            scaler=scaler,
            metrics=NILMmetrics(round_to=5),
            window_size=expes_config.window_size,
            freq=expes_config.sampling_rate,
            cosinbase=False,
            new_range=(-0.5, 0.5),
            mask_metric="test_metrics",
            list_exo_variables=["hour", "dow", "month"],
            threshold_small_values=expes_config.threshold,
            save_results=False,
        )
    else:
        eval_win_energy_aggregation(
            tuple_data[2],
            tuple_data[6],
            eval_trainer,
            scaler=scaler,
            metrics=NILMmetrics(round_to=5),
            window_size=expes_config.window_size,
            freq=expes_config.sampling_rate,
            mask_metric="test_metrics",
            list_exo_variables=expes_config.list_exo_variables,
            threshold_small_values=expes_config.threshold,
            save_results=False,
        )

    writer = tb_logger.experiment
    if hasattr(lightning_module, "best_epoch") and lightning_module.best_epoch >= 0:
        epoch_idx = int(lightning_module.best_epoch)
    else:
        epoch_idx = int(lightning_module.current_epoch)
    for log_key, log_val in eval_trainer.log.items():
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
    new_best_loss = eval_trainer.best_loss
    if previous_best_loss is None or new_best_loss < previous_best_loss:
        eval_trainer.save()
    elif previous_ckpt is not None:
        torch.save(previous_ckpt, ckpt_path)
    try:
        _plot_loss_history(
            eval_trainer.loss_train_history,
            eval_trainer.loss_valid_history,
            expes_config.result_path,
        )
    except Exception as e:
        logging.warning("Could not plot NILM loss history: %s", e)
    try:
        if hasattr(lightning_module, "best_epoch") and lightning_module.best_epoch >= 0:
            epoch_idx_html = int(lightning_module.best_epoch) + 1
        else:
            epoch_idx_html = int(lightning_module.current_epoch) + 1
        _save_val_streamlit_data(
            inst_model,
            valid_loader,
            scaler,
            expes_config,
            epoch_idx_html,
        )
    except Exception as e:
        logging.warning("Could not save NILM validation HTML: %s", e)
    logging.info(
        "Training and eval completed! Model weights and log save at: {}.pt".format(
            expes_config.result_path
        )
    )
    result = {
        "best_loss": float(eval_trainer.best_loss),
        "valid_timestamp": eval_trainer.log.get("valid_metrics_timestamp", {}),
        "valid_win": eval_trainer.log.get("valid_metrics_win", {}),
        "test_timestamp": eval_trainer.log.get("test_metrics_timestamp", {}),
        "test_win": eval_trainer.log.get("test_metrics_win", {}),
    }
    return result


def tser_model_training(inst_model, tuple_data, expes_config):
    expes_config.device = get_device()
    train_dataset = TSDatasetScaling(tuple_data[0][0], tuple_data[0][1])
    valid_dataset = TSDatasetScaling(tuple_data[1][0], tuple_data[1][1])
    test_dataset = TSDatasetScaling(tuple_data[2][0], tuple_data[2][1])

    num_workers = getattr(expes_config, "num_workers", 0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=expes_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    lightning_module = TserLightningModule(
        inst_model,
        learning_rate=expes_config.model_training_param.lr,
        weight_decay=expes_config.model_training_param.wd,
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
    tb_root = os.path.join("log", "tensorboard")
    os.makedirs(tb_root, exist_ok=True)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_root, name="")
    callbacks = []
    if expes_config.p_es is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=expes_config.p_es, mode="min"
            )
        )
    trainer = pl.Trainer(
        max_epochs=expes_config.epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=tb_logger,
    )
    logging.info("Model training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    if lightning_module.best_model_state_dict is not None:
        inst_model.load_state_dict(lightning_module.best_model_state_dict)
    inst_model.to(expes_config.device)
    ckpt_root = os.path.join("checkpoint")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_root,
        "{}_{}_{}_{}_{}".format(
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            expes_config.window_size,
            expes_config.name_model,
        ),
    )

    class EvalContainer:
        def __init__(self, model, device, path_checkpoint):
            self.model = model
            self.device = device
            self.log = {}
            self.path_checkpoint = path_checkpoint
            self.best_loss = lightning_module.best_val_loss
            self.loss_train_history = lightning_module.loss_train_history
            self.loss_valid_history = lightning_module.loss_valid_history
            self.passed_epochs = len(lightning_module.loss_train_history)

        def save(self):
            torch.save(self.log, self.path_checkpoint + ".pt")

    eval_trainer = EvalContainer(inst_model, expes_config.device, ckpt_path)

    def evaluate_tser_split(data_loader, mask):
        y = np.array([])
        y_hat = np.array([])
        with torch.no_grad():
            for ts_agg, target in data_loader:
                inst_model.eval()
                ts_agg_t = ts_agg.float().to(expes_config.device)
                target_t = target.float().to(expes_config.device)
                if target_t.dim() == 1:
                    target_t = target_t.unsqueeze(1)
                pred = inst_model(ts_agg_t)
                target_flat = torch.flatten(target_t).detach().cpu().numpy()
                pred_flat = torch.flatten(pred).detach().cpu().numpy()
                y_local = target_flat
                y_hat_local = pred_flat
                y_nonlocal = y
                y_hat_nonlocal = y_hat
                y_nonlocal = (
                    np.concatenate((y_nonlocal, y_local)) if y_nonlocal.size else y_local
                )
                y_hat_nonlocal = (
                    np.concatenate((y_hat_nonlocal, y_hat_local))
                    if y_hat_nonlocal.size
                    else y_hat_local
                )
                y = y_nonlocal
                y_hat = y_hat_nonlocal
        if not y.size:
            return
        metrics_helper = NILMmetrics()
        metrics_win = metrics_helper(y=y, y_hat=y_hat)
        eval_trainer.log[mask + "_win"] = metrics_win

    logging.info("Eval model...")
    evaluate_tser_split(valid_loader, "valid_metrics")
    evaluate_tser_split(test_loader, "test_metrics")
    eval_trainer.log["best_model_state_dict"] = inst_model.state_dict()
    eval_trainer.save()
    try:
        _plot_loss_history(
            eval_trainer.loss_train_history,
            eval_trainer.loss_valid_history,
            expes_config.result_path,
        )
    except Exception as e:
        logging.warning("Could not plot TSER loss history: %s", e)
    logging.info(
        "Training and eval completed! Model weights and log save at: {}".format(
            expes_config.result_path
        )
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
