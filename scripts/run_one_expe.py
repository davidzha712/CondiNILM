#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Experiments
#
#################################################################################################################

import argparse
import os
import yaml
import logging

logging.getLogger("torch.utils.flop_counter").disabled = True

import numpy as np
import torch

from omegaconf import OmegaConf

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
    split_train_valid_timeblock_nilmdataset,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training


def _configure_nilm_loss_hyperparams(expes_config, data, threshold):
    try:
        if data.ndim != 4 or data.shape[1] < 2:
            return
        power = data[:, 1, 0, :].astype(np.float32)
        flat = power.reshape(-1)
    except Exception:
        return
    if flat.size == 0:
        return
    thr = float(threshold)
    on_mask = flat > thr
    duty_cycle = float(on_mask.mean())
    if flat.size > 1:
        diff_all = np.abs(np.diff(flat))
    else:
        diff_all = np.zeros(1, dtype=np.float32)
    on_values = flat[on_mask]
    off_values = flat[~on_mask]
    if on_values.size > 1:
        diff_on = np.abs(np.diff(on_values))
    else:
        diff_on = diff_all
    if off_values.size > 1:
        diff_off = np.abs(np.diff(off_values))
    else:
        diff_off = diff_all
    if diff_off.size > 0:
        noise_level = float(np.quantile(diff_off, 0.9))
    else:
        noise_level = float(np.quantile(diff_all, 0.9))
    if diff_on.size > 0:
        edge_level = float(np.quantile(diff_on, 0.9))
    else:
        edge_level = float(np.quantile(diff_all, 0.9))
    ratio = edge_level / (noise_level + 1e-6)
    if not np.isfinite(ratio):
        ratio = 1.0
    ratio_clipped = min(max(ratio, 1.0), 10.0)
    lambda_grad = 0.2 + (0.8 - 0.2) * (ratio_clipped - 1.0) / 9.0
    if duty_cycle < 0.01:
        alpha_on = 6.0
        alpha_off = 0.5
    elif duty_cycle < 0.05:
        alpha_on = 4.5
        alpha_off = 0.8
    elif duty_cycle < 0.15:
        alpha_on = 3.0
        alpha_off = 1.0
    else:
        alpha_on = 2.0
        alpha_off = 1.0
    if duty_cycle < 0.01:
        lambda_energy = 0.01
    elif duty_cycle < 0.05:
        lambda_energy = 0.03
    elif duty_cycle < 0.15:
        lambda_energy = 0.08
    else:
        lambda_energy = 0.20

    soft_temp_raw = max(0.25 * thr, 2.0 * noise_level, 1.0)
    edge_eps_raw = max(3.0 * noise_level, 0.5 * edge_level, 0.1 * thr, 1.0)

    try:
        energy_all = power.sum(axis=-1)
        if energy_all.size > 0:
            window_on = (power > thr).any(axis=-1)
            energy_on = energy_all[window_on]
            if energy_on.size > 0:
                base_floor = float(np.quantile(energy_on, 0.1))
            else:
                base_floor = float(np.quantile(energy_all, 0.5))
            energy_floor_raw = max(
                0.1 * thr * power.shape[-1],
                0.05 * base_floor,
            )
        else:
            energy_floor_raw = thr * power.shape[-1] * 0.1
    except Exception:
        energy_floor_raw = thr * power.shape[-1] * 0.1

    expes_config["loss_alpha_on"] = float(alpha_on)
    expes_config["loss_alpha_off"] = float(alpha_off)
    expes_config["loss_lambda_grad"] = float(lambda_grad)
    expes_config["loss_lambda_energy"] = float(lambda_energy)
    expes_config["loss_soft_temp_raw"] = float(soft_temp_raw)
    expes_config["loss_edge_eps_raw"] = float(edge_eps_raw)
    expes_config["loss_energy_floor_raw"] = float(energy_floor_raw)
    duty = float(duty_cycle)
    if duty <= 0.002:
        lambda_zero = 1.0
        lambda_sparse = 0.10
    elif duty <= 0.01:
        lambda_zero = 0.8
        lambda_sparse = 0.08
    elif duty <= 0.05:
        lambda_zero = 0.5
        lambda_sparse = 0.05
    elif duty <= 0.3:
        lambda_zero = 0.2
        lambda_sparse = 0.02
    else:
        lambda_zero = 0.08
        lambda_sparse = 0.008
    expes_config["loss_lambda_zero"] = float(lambda_zero)
    expes_config["loss_lambda_sparse"] = float(lambda_sparse)


def get_cache_path(expes_config: OmegaConf):
    overlap = getattr(expes_config, "overlap", 0.0)
    overlap_str = "ov{}".format(str(overlap).replace(".", "p"))
    if getattr(expes_config, "name_model", None) == "DiffNILM":
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
            overlap_str,
            "DiffNILM",
        ]
    else:
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
            overlap_str,
        ]
    key = "_".join(str(x) for x in key_elements)
    key = key.replace("/", "-")
    cache_dir = os.path.join("data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, key + ".pt")


def launch_one_experiment(expes_config: OmegaConf):
    np.random.seed(seed=expes_config.seed)

    cache_path = get_cache_path(expes_config)
    if os.path.isfile(cache_path):
        logging.info("Load cached preprocessed data from %s", cache_path)
        cache = torch.load(cache_path, weights_only=False)
        tuple_data = cache["tuple_data"]
        scaler = cache["scaler"]
        expes_config.cutoff = cache["cutoff"]
        expes_config.threshold = cache["threshold"]
        return launch_models_training(tuple_data, scaler, expes_config)

    logging.info("Process data ...")
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

    logging.info("             ... Done.")

    threshold = data_builder.appliance_param[expes_config.app]["min_threshold"]
    expes_config.threshold = threshold
    _configure_nilm_loss_hyperparams(expes_config, data, threshold)

    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    data = scaler.fit_transform(data)

    expes_config.cutoff = float(scaler.appliance_stat2[0])
    if expes_config.cutoff and expes_config.cutoff > 0:
        expes_config["loss_threshold"] = float(expes_config.threshold) / float(
            expes_config.cutoff
        )
        if "loss_soft_temp_raw" in expes_config:
            expes_config["loss_soft_temp"] = float(expes_config.loss_soft_temp_raw) / float(
                expes_config.cutoff
            )
        if "loss_edge_eps_raw" in expes_config:
            expes_config["loss_edge_eps"] = float(expes_config.loss_edge_eps_raw) / float(
                expes_config.cutoff
            )

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

    return launch_models_training(tuple_data, scaler, expes_config)


def main(
    dataset,
    sampling_rate,
    window_size,
    appliance,
    name_model,
    resume,
    no_final_eval,
    loss_type=None,
    epochs=None,
):
    """
    Main function to load configuration, update it with parameters,
    and launch an experiment.

    Args:
        dataset (str): Name of the dataset (case-insensitive, e.g. UKDALE or REFIT).
        sampling_rate (str): Selected sampling rate (case-insensitive, e.g. 30s, 1min).
        window_size (int or str): Size of the window (converted to int if possible not day, week or month).
        appliance (str): Selected appliance (case-insensitive).
        name_model (str): Name of the model to use for the experiment (case-insensitive).
    """

    seed = 42

    try:
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r") as f:
        datasets_all = yaml.safe_load(f)
        dataset_key_map = {k.lower(): k for k in datasets_all.keys()}
        dataset_key = dataset_key_map.get(str(dataset).strip().lower())
        if dataset_key is None:
            available = ", ".join(sorted(datasets_all.keys()))
            raise ValueError(
                "Dataset {} unknown. Available datasets (case-insensitive): {}. Use -h to see argument help.".format(
                    dataset, available
                )
            )
        datasets_config = datasets_all[dataset_key]

    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)

        model_key_map = {k.lower(): k for k in baselines_config.keys()}
        model_key = model_key_map.get(str(name_model).strip().lower())
        if model_key is None:
            available = ", ".join(sorted(baselines_config.keys()))
            raise ValueError(
                "Model {} unknown. Available models (case-insensitive): {}. Use -h to see argument help.".format(
                    name_model, available
                )
            )
        expes_config.update(baselines_config[model_key])

    appliance_key_map = {k.lower(): k for k in datasets_config.keys()}
    appliance_key = appliance_key_map.get(str(appliance).strip().lower())
    if appliance_key is None:
        available = ", ".join(sorted(datasets_config.keys()))
        logging.error("Appliance '%s' not found in datasets_config.", appliance)
        raise ValueError(
            "Appliance {} unknown for dataset {}. Available appliances (case-insensitive): {}. Use -h to see argument help.".format(
                appliance, dataset_key, available
            )
        )
    expes_config.update(datasets_config[appliance_key])

    sampling_rate = str(sampling_rate).strip().lower()

    logging.info("---- Run experiments with provided parameters ----")
    logging.info("      Dataset: %s", dataset_key)
    logging.info("      Sampling Rate: %s", sampling_rate)
    logging.info("      Window Size: %s", window_size)
    logging.info("      Appliance : %s", appliance_key)
    logging.info("      Model: %s", model_key)
    logging.info("      Seed: %s", seed)
    logging.info("--------------------------------------------------")

    expes_config["dataset"] = dataset_key
    expes_config["appliance"] = appliance_key
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = model_key
    expes_config["resume"] = bool(resume)
    expes_config["skip_final_eval"] = bool(no_final_eval)
    if loss_type is not None:
        expes_config["loss_type"] = str(loss_type)
    if epochs is not None:
        expes_config["epochs"] = int(epochs)

    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset_key}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")

    expes_config = OmegaConf.create(expes_config)

    expes_config.result_path = (
        f"{result_path}{expes_config.name_model}_{expes_config.seed}"
    )

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    launch_one_experiment(expes_config)


if __name__ == "__main__":
    with open("configs/datasets.yaml", "r") as f:
        _datasets_all = yaml.safe_load(f)
    with open("configs/models.yaml", "r") as f:
        _models_all = yaml.safe_load(f)
    _dataset_choices = ", ".join(sorted(_datasets_all.keys()))
    _model_choices = ", ".join(sorted(_models_all.keys()))
    _appliance_hints = []
    if "REFIT" in _datasets_all:
        _appliance_hints.append(
            "REFIT: " + ", ".join(sorted(_datasets_all["REFIT"].keys()))
        )
    if "UKDALE" in _datasets_all:
        _appliance_hints.append(
            "UKDALE: " + ", ".join(sorted(_datasets_all["UKDALE"].keys()))
        )
    _appliance_help = " | ".join(_appliance_hints) if _appliance_hints else ""

    parser = argparse.ArgumentParser(
        description=(
            "NILMFormer Experiments. Use -h to see valid options for each argument."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (non-case-insensitive). Choices: {}.".format(_dataset_choices),
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate (non-case-insensitive), e.g. '30s', '1min', '10min'.",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, e.g. '128' or 'day.",
    )
    parser.add_argument(
        "--appliance",
        required=True,
        type=str,
        help=(
            "Selected appliance (non-case-insensitive). Available by dataset: {}.".format(
                _appliance_help
            )
        ),
    )
    parser.add_argument(
        "--name_model",
        required=True,
        type=str,
        help="Name of the model for training (non-case-insensitive). Choices: {}.".format(
            _model_choices
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint for the same experiment if available.",
    )
    parser.add_argument(
        "--no_final_eval",
        action="store_true",
        help="Skip final full evaluation (keep visualization HTML only).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help=(
            "Loss type for NILM baselines. Choices: "
            "'eaec', 'smoothl1', 'mse', 'mae'."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs defined in configs/expes.yaml.",
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        resume=args.resume,
        no_final_eval=args.no_final_eval,
        loss_type=args.loss_type,
        epochs=args.epochs,
    )
