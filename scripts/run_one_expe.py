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
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training


def get_cache_path(expes_config: OmegaConf):
    if getattr(expes_config, "name_model", None) == "DiffNILM":
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
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
        data_builder = UKDALE_DataBuilder(
            data_path=f"{expes_config.data_path}/UKDALE/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
        )

        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_train
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_test
        )

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_nilmdataset(
                data_train,
                st_date_train,
                perc_house_test=0.2,
                seed=expes_config.seed,
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

    return launch_models_training(tuple_data, scaler, expes_config)


def main(dataset, sampling_rate, window_size, appliance, name_model, resume, no_final_eval):
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

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        resume=args.resume,
        no_final_eval=args.no_final_eval,
    )
