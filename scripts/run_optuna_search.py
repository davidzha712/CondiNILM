import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import optuna
import yaml
from omegaconf import OmegaConf

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.helpers.utils import create_dir
from src.helpers.dataset_params import DatasetParamsManager, validate_appliances_for_dataset
from scripts.run_one_expe import launch_one_experiment


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_dataset_keys(datasets_all: Dict[str, Any], dataset_arg: str) -> List[str]:
    if not dataset_arg:
        raise ValueError("dataset is required")
    if dataset_arg.strip().lower() in ("all", "*"):
        return list(datasets_all.keys())
    dataset_key_map = {k.lower(): k for k in datasets_all.keys()}
    dataset_key = dataset_key_map.get(dataset_arg.strip().lower())
    if dataset_key is None:
        available = ", ".join(sorted(datasets_all.keys()))
        raise ValueError(f"Dataset {dataset_arg} unknown. Choices: {available}")
    return [dataset_key]


def _select_multi_keys(dataset_config: Dict[str, Any], appliance_arg: str) -> List[str]:
    if appliance_arg is None or str(appliance_arg).strip().lower() == "multi":
        return list(dataset_config.keys())
    appliance_str = str(appliance_arg).strip()
    if "," not in appliance_str:
        raise ValueError("Only multi-device mode is supported in this script")
    requested = [s.strip().lower() for s in appliance_str.split(",") if s.strip()]
    if len(requested) <= 1:
        raise ValueError("Multi-device mode requires at least two appliances")
    appliance_key_map = {k.lower(): k for k in dataset_config.keys()}
    selected_keys = []
    for name in requested:
        key = appliance_key_map.get(name)
        if key is None:
            available = ", ".join(sorted(dataset_config.keys()))
            raise ValueError(
                f"Appliance {name} unknown for dataset. Available: {available}"
            )
        selected_keys.append(key)
    return selected_keys


def _build_multi_entry(
    dataset_config: Dict[str, Any],
    selected_keys: List[str],
    params_manager: DatasetParamsManager,
    dataset_key: str,
) -> Dict[str, Any]:
    base_entry: Dict[str, Any] = {}
    for k in selected_keys:
        cfg_k = dataset_config[k]
        for ck, cv in cfg_k.items():
            if ck == "app":
                continue
            if ck not in base_entry:
                base_entry[ck] = cv
    app_list: List[str] = []
    for k in selected_keys:
        app_val = dataset_config[k].get("app", k)
        if isinstance(app_val, list):
            app_list.extend(list(app_val))
        else:
            app_list.append(app_val)
    valid_apps = validate_appliances_for_dataset(app_list, dataset_key, params_manager)
    base_entry["app"] = valid_apps if valid_apps else app_list
    base_entry["appliance_group_members"] = selected_keys
    return base_entry


def _parse_search_space(path: str, model_name: str) -> Dict[str, Any]:
    data = _load_yaml(path)
    model_key_map = {k.lower(): k for k in data.keys()}
    key = model_key_map.get(model_name.strip().lower())
    if key is None:
        available = ", ".join(sorted(data.keys()))
        raise ValueError(f"Search space for {model_name} not found. Choices: {available}")
    return data[key] or {}


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    if not isinstance(spec, dict):
        return spec
    ptype = str(spec.get("type", "")).lower()
    if ptype == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if ptype == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if ptype == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step)
    if ptype == "categorical":
        return trial.suggest_categorical(name, list(spec.get("choices", [])))
    if "value" in spec:
        return spec["value"]
    return spec


def _sample_params(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        params[name] = _suggest_param(trial, name, spec)
    d_model = int(params.get("d_model", 0) or 0)
    n_head = int(params.get("n_head", 0) or 0)
    if d_model and n_head and d_model % n_head != 0:
        valid = [h for h in range(2, 17, 2) if d_model % h == 0]
        params["n_head"] = valid[0] if valid else max(1, n_head)
    if d_model and d_model % 4 != 0:
        d_model = int(round(d_model / 4.0) * 4)
        params["d_model"] = max(4, d_model)
    return params


def _apply_model_params(expes_config: Dict[str, Any], params: Dict[str, Any]) -> None:
    model_kwargs = dict(expes_config.get("model_kwargs", {}) or {})
    for k in [
        "d_model",
        "n_encoder_layers",
        "n_head",
        "dp_rate",
        "pffn_ratio",
        "kernel_size",
        "kernel_size_head",
    ]:
        if k in params:
            model_kwargs[k] = params[k]
    expes_config["model_kwargs"] = model_kwargs
    model_training = dict(expes_config.get("model_training_param", {}) or {})
    if "lr" in params:
        model_training["lr"] = params["lr"]
    if "wd" in params:
        model_training["wd"] = params["wd"]
    expes_config["model_training_param"] = model_training


def _split_loss_params(params: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "loss_lambda_energy",
        "loss_alpha_on",
        "loss_alpha_off",
        "loss_lambda_on_recall",
        "loss_on_recall_margin",
        "loss_lambda_sparse",
        "loss_lambda_off_hard",
        "output_ratio",
        "gate_soft_scale",
        "gate_floor",
    ]
    return {k: params[k] for k in keys if k in params}


def _build_expes_config(
    dataset_key: str,
    dataset_config: Dict[str, Any],
    model_key: str,
    model_config: Dict[str, Any],
    sampling_rate: str,
    window_size: str,
    seed: int,
    appliance_arg: str,
    trial_number: int,
    params: Dict[str, Any],
    params_manager: DatasetParamsManager,
    base_config: Dict[str, Any],
) -> OmegaConf:
    expes_config = dict(base_config)
    expes_config.update(model_config)
    selected_keys = _select_multi_keys(dataset_config, appliance_arg)
    multi_entry = _build_multi_entry(dataset_config, selected_keys, params_manager, dataset_key)
    expes_config.update(multi_entry)
    expes_config["dataset"] = dataset_key
    expes_config["sampling_rate"] = str(sampling_rate).strip().lower()
    try:
        window_size_val = int(window_size)
    except Exception:
        window_size_val = window_size
    expes_config["window_size"] = window_size_val
    expes_config["seed"] = int(seed)
    expes_config["name_model"] = model_key
    expes_config["resume"] = False
    expes_config["loss_type"] = "multi_nilm"
    _apply_model_params(expes_config, params)
    loss_override = _split_loss_params(params)
    if loss_override:
        expes_config["hpo_override"] = loss_override
    appliance_tag = f"Multi_T{trial_number}"
    expes_config["appliance"] = appliance_tag
    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset_key}_{expes_config['sampling_rate']}/")
    result_path = create_dir(f"{result_path}{expes_config['window_size']}/")
    expes_config = OmegaConf.create(expes_config)
    expes_config.result_path = (
        f"{result_path}{expes_config.name_model}_{expes_config.seed}_t{trial_number}"
    )
    return expes_config


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _best_score_from_records(
    records: List[Dict[str, Any]],
    score_mode: str = "weighted_f1"
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate best score from validation records.

    score_mode options:
        - "min_f1": minimize 1 - min(F1) across devices
        - "mean_f1": minimize 1 - mean(F1) across devices
        - "weighted_f1": minimize 1 - weighted_mean(F1), with higher weights for difficult devices

    ENHANCED (v4): Added precision penalty for Microwave/Kettle to prevent
    configurations that achieve high recall but terrible precision (lots of FP).
    """
    # Device weights: higher for difficult devices (Microwave, Kettle)
    DEVICE_WEIGHTS = {
        "microwave": 2.0,
        "Microwave": 2.0,
        "kettle": 1.5,
        "Kettle": 1.5,
        "fridge": 1.0,
        "Fridge": 1.0,
        "washingmachine": 1.0,
        "WashingMachine": 1.0,
        "washing_machine": 1.0,
        "dishwasher": 1.0,
        "Dishwasher": 1.0,
    }

    # Devices that need precision penalty (prone to false positives)
    PRECISION_PENALTY_DEVICES = {"microwave", "kettle"}
    PRECISION_THRESHOLD = 0.10  # Minimum acceptable precision

    best_score = None
    best_meta: Dict[str, Any] = {}
    for record in records:
        per_device = record.get("metrics_timestamp_per_device") or record.get("metrics_win_per_device")
        f1_vals: List[float] = []
        f1_weighted_sum = 0.0
        weight_sum = 0.0
        worst_name = None
        worst_val = None
        device_f1s: Dict[str, float] = {}
        device_precisions: Dict[str, float] = {}
        precision_penalty = 0.0

        if isinstance(per_device, dict) and per_device:
            for name, mdict in per_device.items():
                try:
                    val = float(mdict.get("F1_SCORE"))
                except Exception:
                    continue
                f1_vals.append(val)
                device_f1s[name] = val
                weight = DEVICE_WEIGHTS.get(name, 1.0)
                f1_weighted_sum += val * weight
                weight_sum += weight
                if worst_val is None or val < worst_val:
                    worst_val = val
                    worst_name = str(name)

                # Track precision for penalty calculation
                try:
                    precision = float(mdict.get("PRECISION", 0.0))
                    device_precisions[name] = precision

                    # Add precision penalty for Microwave/Kettle with low precision
                    # This prevents configurations that achieve recall by predicting
                    # everything as ON (which gives high recall but terrible precision)
                    if name.lower() in PRECISION_PENALTY_DEVICES and precision < PRECISION_THRESHOLD:
                        # Penalty: (threshold - precision) * weight * 2.0
                        # E.g., if precision=0.003 and threshold=0.10, penalty = 0.097 * 2.0 * 2.0 = 0.388
                        penalty = (PRECISION_THRESHOLD - precision) * weight * 2.0
                        precision_penalty += penalty
                except Exception:
                    pass

        if not f1_vals:
            metrics = record.get("metrics_timestamp") or record.get("metrics_win") or {}
            if "F1_SCORE" in metrics:
                try:
                    f1_vals = [float(metrics.get("F1_SCORE"))]
                    f1_weighted_sum = f1_vals[0]
                    weight_sum = 1.0
                except Exception:
                    f1_vals = []

        if not f1_vals:
            continue

        min_f1 = float(min(f1_vals))
        mean_f1 = float(sum(f1_vals) / len(f1_vals))
        weighted_f1 = float(f1_weighted_sum / weight_sum) if weight_sum > 0 else mean_f1

        # Select score based on mode
        if score_mode == "min_f1":
            score = 1.0 - min_f1
        elif score_mode == "mean_f1":
            score = 1.0 - mean_f1
        else:  # weighted_f1 (default)
            score = 1.0 - weighted_f1

        # Add precision penalty for low-precision devices
        score += precision_penalty

        if bool(record.get("collapse_flag", False)):
            score += 1.0

        if best_score is None or score < best_score:
            best_score = score
            best_meta = {
                "epoch": int(record.get("epoch", -1)),
                "min_f1": min_f1,
                "mean_f1": mean_f1,
                "weighted_f1": weighted_f1,
                "worst_device": worst_name if isinstance(worst_name, str) else None,
                "device_f1s": device_f1s,
                "device_precisions": device_precisions,
                "precision_penalty": precision_penalty,
                "energy_ratio": float(record.get("energy_ratio", 0.0)),
                "off_energy_ratio": float(record.get("off_energy_ratio", 0.0)),
                "pred_raw_max": float(record.get("pred_raw_max", 0.0)),
                "target_max": float(record.get("target_max", 0.0)),
            }
    if best_score is None:
        return 1e9, {}
    return float(best_score), best_meta


def _get_group_dir(expes_config: OmegaConf) -> str:
    result_root = os.path.dirname(os.path.dirname(os.path.dirname(expes_config.result_path)))
    return os.path.join(
        result_root,
        f"{expes_config.dataset}_{expes_config.sampling_rate}",
        str(expes_config.window_size),
        str(expes_config.appliance),
    )


def _build_study_name(
    dataset_key: str, model_key: str, sampling_rate: str, window_size: str, prefix: str
) -> str:
    parts = [prefix, dataset_key, "multi", model_key, str(sampling_rate), str(window_size)]
    return "_".join([p for p in parts if p])


def run_study_for_dataset(
    dataset_key: str,
    datasets_all: Dict[str, Any],
    models_all: Dict[str, Any],
    args: argparse.Namespace,
    params_manager: DatasetParamsManager,
) -> None:
    dataset_config = datasets_all[dataset_key]
    model_key_map = {k.lower(): k for k in models_all.keys()}
    model_key = model_key_map.get(args.name_model.strip().lower())
    if model_key is None:
        available = ", ".join(sorted(models_all.keys()))
        raise ValueError(f"Model {args.name_model} unknown. Choices: {available}")
    model_config = models_all[model_key]
    base_config = _load_yaml("configs/expes.yaml")
    if args.epochs is not None:
        base_config["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        base_config["batch_size"] = int(args.batch_size)
    space = _parse_search_space(args.search_space, model_key)
    sampling_tag = args.sampling_rate
    window_tag = args.window_size
    if "sampling_rate" in space:
        sampling_tag = "var"
    if "window_size" in space:
        window_tag = "var"
    storage_dir = args.storage_dir
    os.makedirs(storage_dir, exist_ok=True)
    study_name = (
        args.study_name
        or _build_study_name(
            dataset_key, model_key, sampling_tag, window_tag, args.study_prefix
        )
    )
    storage = f"sqlite:///{os.path.join(storage_dir, study_name)}.db"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, space)
        trial_seed = int(args.seed) + int(trial.number)
        sampling_rate = params.get("sampling_rate", args.sampling_rate)
        window_size = params.get("window_size", args.window_size)
        if args.lock_sampling_rate:
            sampling_rate = args.sampling_rate
        if args.lock_window_size:
            window_size = args.window_size
        expes_config = _build_expes_config(
            dataset_key=dataset_key,
            dataset_config=dataset_config,
            model_key=model_key,
            model_config=model_config,
            sampling_rate=sampling_rate,
            window_size=window_size,
            seed=trial_seed,
            appliance_arg=args.appliance,
            trial_number=trial.number,
            params=params,
            params_manager=params_manager,
            base_config=base_config,
        )
        launch_one_experiment(expes_config)
        group_dir = _get_group_dir(expes_config)
        records = _read_jsonl(os.path.join(group_dir, "val_report.jsonl"))
        # Use weighted_f1 score mode (prioritizes difficult devices like Microwave)
        score, meta = _best_score_from_records(records, score_mode="weighted_f1")
        trial.set_user_attr("group_dir", group_dir)
        for k, v in meta.items():
            trial.set_user_attr(k, v)
        return float(score)

    study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--sampling_rate", required=True, type=str)
    parser.add_argument("--window_size", required=True, type=str)
    parser.add_argument("--appliance", type=str, default="multi")
    parser.add_argument("--name_model", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lock_sampling_rate", action="store_true")
    parser.add_argument("--lock_window_size", action="store_true")
    parser.add_argument("--search_space", type=str, default="configs/hpo_search_spaces.yaml")
    parser.add_argument("--storage_dir", type=str, default="optuna_studies")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--study_prefix", type=str, default="study")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    datasets_all = _load_yaml("configs/datasets.yaml")
    models_all = _load_yaml("configs/models.yaml")
    params_manager = DatasetParamsManager()
    dataset_keys = _resolve_dataset_keys(datasets_all, args.dataset)
    for dataset_key in dataset_keys:
        run_study_for_dataset(dataset_key, datasets_all, models_all, args, params_manager)


if __name__ == "__main__":
    main()
