"""Experiment helpers and model factory -- CondiNILM.

Author: Siyi Li
"""

import os
import torch
import logging
import platform
import numpy as np
import json
import pytorch_lightning as pl
from collections.abc import Sequence

from src.helpers.trainer import (
    AdaptiveDeviceLoss,
    SeqToSeqLightningModule,
    TserLightningModule,
    DiffNILMLightningModule,
    STNILMLightningModule,
)
from src.helpers.dataset import NILMDataset, TSDatasetScaling
from src.helpers.metrics import NILMmetrics, eval_win_energy_aggregation
from src.helpers.loss_tuning import AdaptiveLossTuner


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    epoch_new = record.get("epoch", None)
    model_new = record.get("model", None)
    appliance_new = record.get("appliance", None)

    def _norm_epoch(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    epoch_new_norm = _norm_epoch(epoch_new)
    existing_lines = []

    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        obj = json.loads(line_stripped)
                    except (ValueError, json.JSONDecodeError):
                        existing_lines.append(line_stripped)
                        continue
                    epoch_old_norm = _norm_epoch(obj.get("epoch", None))
                    model_old = obj.get("model", None)
                    appliance_old = obj.get("appliance", None)
                    if (
                        epoch_new_norm is not None
                        and epoch_old_norm is not None
                        and epoch_old_norm == epoch_new_norm
                        and model_old == model_new
                        and appliance_old == appliance_new
                    ):
                        continue
                    existing_lines.append(json.dumps(_to_jsonable(obj), ensure_ascii=False))
        except (OSError, UnicodeDecodeError):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            return

    existing_lines.append(json.dumps(_to_jsonable(record), ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        for line in existing_lines:
            f.write(line + "\n")


def _sanitize_tb_tag(value):
    s = str(value)
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return s


def _coerce_appliance_names(expes_config, n_app, fallback_name=None):
    """Return appliance names from config, falling back to numeric indices.

    If the config provides fewer names than n_app, the available names are
    used and the remaining slots are filled with numeric indices.
    """
    app_names = None
    group_members = None
    if expes_config is not None:
        app_names = getattr(expes_config, "app", None)
        group_members = getattr(expes_config, "appliance_group_members", None)

    n_app = int(n_app)
    result = []

    # Try to use app_names first
    if app_names is not None and not isinstance(app_names, str):
        try:
            if isinstance(app_names, Sequence):
                app_list = [str(x) for x in list(app_names)]
                if len(app_list) == n_app:
                    return app_list
                # Partial match: use available names
                result = app_list[:n_app]
        except (TypeError, ValueError):
            pass

    # Try group_members if app_names didn't work
    if not result and group_members is not None and not isinstance(group_members, str):
        try:
            if isinstance(group_members, Sequence):
                group_list = [str(x) for x in list(group_members)]
                if len(group_list) == n_app:
                    return group_list
                # Partial match: use available names
                result = group_list[:n_app]
        except (TypeError, ValueError):
            pass

    # Single device fallback
    if not result and isinstance(fallback_name, str) and fallback_name and n_app == 1:
        return [str(fallback_name)]

    # Fill remaining slots with numeric indices
    if result:
        for j in range(len(result), n_app):
            result.append(str(j))
        return result

    # Complete fallback to numeric indices
    return [str(j) for j in range(n_app)]


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
from src.nilmformer.config import NILMFormerConfig
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
            except RuntimeError:
                pass
            return "cuda"
        return "cpu"
    return "cpu"


def _get_num_workers(num_workers):
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        cpu_count = 1

    if isinstance(num_workers, int):
        if num_workers <= 0:
            return 0
        if num_workers > cpu_count:
            return cpu_count
        return num_workers

    # Windows spawn-based multiprocessing pickles the Dataset to each worker,
    # which can exceed pipe buffer limits for large datasets. Default to 0.
    if platform.system() == "Windows":
        return 0

    base = max(1, cpu_count - 1)
    if base > 16:
        base = 16
    return base


def _set_default_thread_env():
    if platform.system() != "Windows":
        return
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "1")))
    except RuntimeError:
        pass


def _dataloader_worker_init(worker_id):
    _set_default_thread_env()


def get_model_instance(name_model, c_in, window_size, **kwargs):
    """Instantiate a model by name, forwarding only the kwargs it accepts."""
    def _filter_kwargs(cls, extra_kwargs):
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_var_keyword:
            return extra_kwargs  # Model accepts **kwargs, pass everything
        return {k: v for k, v in extra_kwargs.items() if k in valid_params}

    if name_model == "BiGRU":
        inst = BiGRU(c_in=c_in, window_size=window_size, **_filter_kwargs(BiGRU, kwargs))
    elif name_model == "BiLSTM":
        inst = BiLSTM(c_in=c_in, window_size=window_size, **_filter_kwargs(BiLSTM, kwargs))
    elif name_model == "CNN1D":
        inst = CNN1D(c_in=c_in, window_size=window_size, **_filter_kwargs(CNN1D, kwargs))
    elif name_model in ("UNetNILM", "UNET_NILM"):
        inst = UNetNiLM(c_in=c_in, window_size=window_size, **_filter_kwargs(UNetNiLM, kwargs))
    elif name_model == "FCN":
        inst = FCN(c_in=c_in, window_size=window_size, **_filter_kwargs(FCN, kwargs))
    elif name_model == "BERT4NILM":
        inst = BERT4NILM(c_in=c_in, window_size=window_size, **_filter_kwargs(BERT4NILM, kwargs))
    elif name_model == "STNILM":
        inst = STNILM(c_in=c_in, window_size=window_size, **_filter_kwargs(STNILM, kwargs))
    elif name_model == "DResNet":
        inst = DResNet(c_in=c_in, window_size=window_size, **_filter_kwargs(DResNet, kwargs))
    elif name_model == "DAResNet":
        inst = DAResNet(c_in=c_in, window_size=window_size, **_filter_kwargs(DAResNet, kwargs))
    elif name_model == "DiffNILM":
        inst = DiffNILM(**_filter_kwargs(DiffNILM, kwargs))
    elif name_model == "TSILNet":
        inst = TSILNet(c_in=c_in, window_size=window_size, **_filter_kwargs(TSILNet, kwargs))
    elif name_model == "Energformer":
        inst = Energformer(c_in=c_in, **_filter_kwargs(Energformer, kwargs))
    elif name_model == "ConvNet":
        inst = ConvNet(in_channels=1, nb_class=1, **_filter_kwargs(ConvNet, kwargs))
    elif name_model == "ResNet":
        inst = ResNet(in_channels=1, nb_class=1, **_filter_kwargs(ResNet, kwargs))
    elif name_model == "Inception":
        inst = Inception(in_channels=1, nb_class=1, **_filter_kwargs(Inception, kwargs))
    elif name_model == "NILMFormer":
        cfg = kwargs.copy()
        c_out = int(cfg.get("c_out", 1))
        cfg["c_out"] = c_out
        inst = NILMFormer(NILMFormerConfig(c_in=1, c_embedding=c_in - 1, **cfg))
    else:
        raise ValueError("Model name {} unknown".format(name_model))

    return inst


# Backward-compatible re-exports
from src.helpers.inference import (
    _crop_center_tensor, create_sliding_windows,
    stitch_center_predictions, inference_seq2subseq,
)
from src.helpers.postprocess import (
    suppress_short_activations, _pad_same_1d,
    suppress_long_off_with_gate, _off_run_stats,
)
from src.helpers.callbacks import (
    ValidationHTMLCallback, RobustLossEpochCallback,
    ValidationNILMMetricCallback,
)
from src.helpers.evaluation import evaluate_nilm_split, _save_val_data
from src.helpers.training import (
    nilm_model_training, tser_model_training, launch_models_training,
)
