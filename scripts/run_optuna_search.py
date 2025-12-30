#################################################################################################################
#
# @description : Optuna search for NILMFormer experiments
#
#################################################################################################################

import argparse
import logging
import os

logging.getLogger("torch.utils.flop_counter").disabled = True

import numpy as np
import optuna
import torch
import yaml

from omegaconf import OmegaConf

from src.helpers.expes import launch_models_training
from src.helpers.preprocessing import (
    REFIT_DataBuilder,
    UKDALE_DataBuilder,
    nilmdataset_to_tser,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    split_train_valid_timeblock_nilmdataset,
)
from src.helpers.dataset import NILMscaler
from src.helpers.utils import create_dir


def load_base_configs():
    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)
    with open("configs/datasets.yaml", "r") as f:
        datasets_config = yaml.safe_load(f)
    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)
    with open("configs/hpo_search_spaces.yaml", "r") as f:
        hpo_spaces = yaml.safe_load(f)
    return expes_config, datasets_config, baselines_config, hpo_spaces


def build_exp_config(
    base_exp,
    datasets_config,
    baselines_config,
    dataset,
    appliance,
    name_model,
    sampling_rate,
    window_size,
    seed,
):
    if dataset not in datasets_config:
        raise ValueError(
            "Dataset {} unknown. Only 'UKDALE' and 'REFIT' available.".format(dataset)
        )
    dataset_cfg = datasets_config[dataset]
    if name_model not in baselines_config:
        raise ValueError(
            "Model {} unknown. List of implemented baselines: {}".format(
                name_model, list(baselines_config.keys())
            )
        )
    if appliance not in dataset_cfg:
        raise ValueError(
            "Appliance {} unknown. List of available appliances (for selected {} dataset): {}".format(
                appliance, dataset, list(dataset_cfg.keys())
            )
        )
    cfg = dict(base_exp)
    cfg.update(baselines_config[name_model])
    cfg.update(dataset_cfg[appliance])
    cfg["dataset"] = dataset
    cfg["appliance"] = appliance
    cfg["window_size"] = window_size
    cfg["sampling_rate"] = sampling_rate
    cfg["seed"] = seed
    cfg["name_model"] = name_model
    result_path = create_dir(cfg["result_path"])
    result_path = create_dir(f"{result_path}{dataset}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")
    cfg = OmegaConf.create(cfg)
    cfg.result_path = f"{result_path}{cfg.name_model}_{cfg.seed}"
    return cfg


def suggest_from_space(trial, model_name, hpo_spaces, base_cfg):
    space = hpo_spaces.get(model_name, {})
    lr = base_cfg.model_training_param.lr
    wd = base_cfg.model_training_param.wd
    if "lr" in space:
        spec = space["lr"]
        if spec["type"] == "loguniform":
            lr = trial.suggest_float(
                "lr", float(spec["low"]), float(spec["high"]), log=True
            )
        else:
            lr = trial.suggest_float(
                "lr", float(spec["low"]), float(spec["high"])
            )
    if "wd" in space:
        spec = space["wd"]
        if spec["type"] == "loguniform":
            wd = trial.suggest_float(
                "wd", float(spec["low"]), float(spec["high"]), log=True
            )
        else:
            wd = trial.suggest_float(
                "wd", float(spec["low"]), float(spec["high"])
            )
    base_cfg.model_training_param.lr = float(lr)
    base_cfg.model_training_param.wd = float(wd)

    train_keys = [
        "batch_size",
        "epochs",
        "n_warmup_epochs",
        "warmup_type",
        "sampling_rate",
        "window_size",
    ]
    for tkey in train_keys:
        t_spec = space.get(tkey)
        if t_spec is None:
            continue
        if t_spec["type"] == "int":
            low = int(t_spec["low"])
            high = int(t_spec["high"])
            step = int(t_spec["step"]) if "step" in t_spec else 1
            val = trial.suggest_int(tkey, low, high, step=step)
        elif t_spec["type"] == "loguniform":
            val = trial.suggest_float(
                tkey, float(t_spec["low"]), float(t_spec["high"]), log=True
            )
        elif t_spec["type"] == "categorical":
            val = trial.suggest_categorical(tkey, t_spec["choices"])
        else:
            val = trial.suggest_float(
                tkey, float(t_spec["low"]), float(t_spec["high"])
            )
        setattr(base_cfg, tkey, val)

    if model_name == "NILMFormer":
        model_kwargs = dict(base_cfg.model_kwargs)
        d_spec = space.get("d_model")
        if d_spec is not None:
            d_low = int(d_spec["low"])
            d_high = int(d_spec["high"])
            d_step = int(d_spec.get("step", 4))
            d_model = trial.suggest_int("d_model", d_low, d_high, step=d_step)
        else:
            d_model = model_kwargs.get("d_model", base_cfg.model_kwargs.d_model)
        model_kwargs["d_model"] = int(d_model)
        h_spec = space.get("n_head")
        if h_spec is not None:
            h_low = int(h_spec["low"])
            h_high = int(h_spec["high"])
            candidates = [h for h in range(h_low, h_high + 1) if int(d_model) % h == 0]
            if not candidates:
                candidates = [h_low]
            n_head = trial.suggest_categorical("n_head", candidates)
            model_kwargs["n_head"] = int(n_head)
        for key, spec in space.items():
            if key in ["lr", "wd", "d_model", "n_head"] + train_keys:
                continue
            if spec["type"] == "int":
                low = int(spec["low"])
                high = int(spec["high"])
                step = int(spec["step"]) if "step" in spec else 1
                val = trial.suggest_int(key, low, high, step=step)
            elif spec["type"] == "loguniform":
                val = trial.suggest_float(
                    key, float(spec["low"]), float(spec["high"]), log=True
                )
            elif spec["type"] == "categorical":
                val = trial.suggest_categorical(key, spec["choices"])
            else:
                val = trial.suggest_float(
                    key, float(spec["low"]), float(spec["high"])
                )
            model_kwargs[key] = val
        base_cfg.model_kwargs = OmegaConf.create(model_kwargs)
        return base_cfg

    model_kwargs = dict(base_cfg.model_kwargs)
    for key, spec in space.items():
        if key in ["lr", "wd"] + train_keys:
            continue
        if spec["type"] == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec["step"]) if "step" in spec else 1
            val = trial.suggest_int(key, low, high, step=step)
        elif spec["type"] == "loguniform":
            val = trial.suggest_float(
                key, float(spec["low"]), float(spec["high"]), log=True
            )
        elif spec["type"] == "categorical":
            val = trial.suggest_categorical(key, spec["choices"])
        else:
            val = trial.suggest_float(
                key, float(spec["low"]), float(spec["high"])
            )
        model_kwargs[key] = val
    base_cfg.model_kwargs = OmegaConf.create(model_kwargs)
    return base_cfg


def prepare_data(expes_config):
    np.random.seed(seed=expes_config.seed)
    overlap = getattr(expes_config, "overlap", 0.0)
    overlap_str = "ov{}".format(str(overlap).replace(".", "p"))
    cache_key = "{}_{}_{}_{}_{}_{}_{}".format(
        expes_config.dataset,
        expes_config.appliance,
        expes_config.sampling_rate,
        expes_config.window_size,
        expes_config.power_scaling_type,
        expes_config.appliance_scaling_type,
        overlap_str,
    )
    if expes_config.name_model == "DiffNILM":
        cache_key += "_DiffNILM"
    cache_key = cache_key.replace("/", "-")
    cache_dir = os.path.join("data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_key + ".pt")
    if os.path.isfile(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        tuple_data = cache["tuple_data"]
        scaler = cache["scaler"]
        expes_config.cutoff = cache["cutoff"]
        expes_config.threshold = cache["threshold"]
        return tuple_data, scaler
    if expes_config.dataset == "UKDALE":
        overlap = getattr(expes_config, "overlap", 0.0)
        if overlap == 0:
            window_stride = expes_config.window_size
        else:
            if not (0 < overlap < 1):
                raise ValueError(
                    "Invalid overlap value {}. Expected 0 or 0 < overlap < 1.".format(
                        overlap
                    )
                )
            window_stride = max(
                1, int(round(expes_config.window_size * (1.0 - float(overlap))))
            )

        data_builder = UKDALE_DataBuilder(
            data_path=f"{expes_config.data_path}/UKDALE/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            window_stride=window_stride,
        )
        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])
        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size
        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_train_val
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_test
        )
        overlap = getattr(expes_config, "overlap", 0.0)
        if overlap == 0:
            data_train, st_date_train, data_valid, st_date_valid = (
                split_train_test_nilmdataset(
                    data_train,
                    st_date_train,
                    perc_house_test=0.2,
                    seed=expes_config.seed,
                )
            )
        else:
            if not (0 < overlap < 1):
                raise ValueError(
                    "Invalid overlap value {}. Expected 0 or 0 < overlap < 1.".format(
                        overlap
                    )
                )
            data_train, st_date_train, data_valid, st_date_valid = (
                split_train_valid_timeblock_nilmdataset(
                    data_train,
                    st_date_train,
                    perc_valid=0.2,
                    window_size=expes_config.window_size,
                    window_stride=data_builder.window_stride,
                )
            )
    elif expes_config.dataset == "REFIT":
        data_builder = REFIT_DataBuilder(
            data_path=f"{expes_config.data_path}/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
        )
        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=expes_config.house_with_app_i
        )
        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size
        data_train, st_date_train, data_test, st_date_test = (
            split_train_test_pdl_nilmdataset(
                data.copy(), st_date.copy(), nb_house_test=2, seed=expes_config.seed
            )
        )
        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=expes_config.seed
            )
        )
    else:
        raise ValueError("Unsupported dataset {}".format(expes_config.dataset))
    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    data = scaler.fit_transform(data)
    expes_config.cutoff = float(scaler.appliance_stat2[0])
    expes_config.threshold = data_builder.appliance_param[expes_config.app][
        "min_threshold"
    ]
    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        X, y = nilmdataset_to_tser(data)
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)
        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)
        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )
    else:
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)
        tuple_data = (
            data_train,
            data_valid,
            data_test,
            data,
            st_date_train,
            st_date_valid,
            st_date_test,
            st_date,
        )
    cache = {
        "tuple_data": tuple_data,
        "scaler": scaler,
        "cutoff": expes_config.cutoff,
        "threshold": expes_config.threshold,
    }
    torch.save(cache, cache_path)
    return tuple_data, scaler


def objective_factory(
    base_exp, datasets_cfg, models_cfg, hpo_spaces, dataset, appliance, model, sampling_rate, window_size, seed
):
    def objective(trial):
        cfg = build_exp_config(
            base_exp,
            datasets_cfg,
            models_cfg,
            dataset,
            appliance,
            model,
            sampling_rate,
            window_size,
            seed,
        )
        cfg = suggest_from_space(trial, model, hpo_spaces, cfg)
        tuple_data, scaler = prepare_data(cfg)
        metrics = launch_models_training(tuple_data, scaler, cfg)
        loss = metrics["best_loss"] if metrics is not None else np.inf
        trial.set_user_attr("metrics", metrics)
        return loss

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for NILMFormer experiments.")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--appliance", required=True, type=str)
    parser.add_argument("--name_model", required=True, type=str)
    parser.add_argument("--sampling_rate", required=True, type=str)
    parser.add_argument("--window_size", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n_trials", required=False, type=int, default=20)
    parser.add_argument("--storage", required=False, type=str, default=None)
    parser.add_argument("--study_name", required=False, type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    base_exp, datasets_cfg, models_cfg, hpo_spaces = load_base_configs()
    try:
        window_size = int(args.window_size)
    except ValueError:
        window_size = args.window_size
    study_name = args.study_name
    if study_name is None:
        study_name = "{}_{}_{}_{}_{}".format(
            args.dataset,
            args.appliance,
            args.name_model,
            args.sampling_rate,
            args.window_size,
        )
    storage = args.storage
    if storage is None:
        storage_dir = os.path.join("results", "optuna")
        os.makedirs(storage_dir, exist_ok=True)
        db_path = os.path.join(storage_dir, study_name + ".db")
        storage = "sqlite:///{}".format(os.path.abspath(db_path))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    objective = objective_factory(
        base_exp,
        datasets_cfg,
        models_cfg,
        hpo_spaces,
        args.dataset,
        args.appliance,
        args.name_model,
        args.sampling_rate,
        window_size,
        args.seed,
    )
    study.optimize(objective, n_trials=args.n_trials)
    best = study.best_trial
    logging.info("Best value: %s", best.value)
    logging.info("Best params: %s", best.params)
    logging.info("Best metrics: %s", best.user_attrs.get("metrics"))


if __name__ == "__main__":
    main()
