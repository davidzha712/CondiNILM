"""Batch UKDALE training across appliances -- CondiNILM.

Author: Siyi Li
"""

import argparse
import logging
import os
import sys
import subprocess
import platform
import yaml


def get_ukdale_appliances():
    with open("configs/datasets.yaml", "r") as f:
        datasets_all = yaml.safe_load(f)
    dataset_key_map = {k.lower(): k for k in datasets_all.keys()}
    dataset_key = dataset_key_map.get("ukdale")
    if dataset_key is None:
        available = ", ".join(sorted(datasets_all.keys()))
        raise ValueError(
            "Dataset UKDALE not found in configs/datasets.yaml. Available datasets: {}".format(
                available
            )
        )
    appliances_config = datasets_all[dataset_key]
    appliances = list(appliances_config.keys())
    if not appliances:
        raise ValueError("No appliances found for dataset {}.".format(dataset_key))
    return dataset_key, appliances


def parse_appliance_list(all_appliances, appliances_str):
    if appliances_str.strip().lower() == "all":
        return all_appliances
    mapping = {k.lower(): k for k in all_appliances}
    selected = []
    for name in appliances_str.split(","):
        name = name.strip()
        if not name:
            continue
        key = mapping.get(name.lower())
        if key is None:
            available = ", ".join(sorted(all_appliances))
            raise ValueError(
                "Appliance {} unknown for UKDALE. Available appliances: {}.".format(
                    name, available
                )
            )
        if key not in selected:
            selected.append(key)
    if not selected:
        selected = all_appliances
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Run NILM training on all UKDALE appliances using run_experiment."
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate, for example '1min' or '30s'.",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, for example '128'.",
    )
    parser.add_argument(
        "--name_model",
        required=True,
        type=str,
        help="Model name passed to scripts.run_experiment.",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        default="all",
        help="Comma-separated list of UKDALE appliances or 'all' to run all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Forwarded to scripts.run_experiment to resume from checkpoint.",
    )
    parser.add_argument(
        "--no_final_eval",
        action="store_true",
        help="Forwarded to scripts.run_experiment to skip final heavy eval.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Forwarded to scripts.run_experiment to override max epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Forwarded to scripts.run_experiment to override batch size.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help=(
            "Forwarded to scripts.run_experiment. Choices: "
            "'eaec', 'smoothl1', 'mse', 'mae'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to scripts.run_experiment (default: 42).",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop batch at first failed subprocess (default: continue).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataset_key, all_appliances = get_ukdale_appliances()
    appliances = parse_appliance_list(all_appliances, args.appliances)

    logging.info(
        "Running UKDALE batch training. Dataset=%s, Sampling=%s, Window=%s, Model=%s",
        dataset_key,
        args.sampling_rate,
        args.window_size,
        args.name_model,
    )
    logging.info("Target appliances: %s", ", ".join(appliances))

    results = []
    for appliance in appliances:
        logging.info(
            "Start training appliance %s for dataset %s.", appliance, dataset_key
        )
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_experiment",
            "--dataset",
            dataset_key,
            "--sampling_rate",
            args.sampling_rate,
            "--window_size",
            args.window_size,
            "--appliance",
            appliance,
            "--name_model",
            args.name_model,
            "--seed",
            str(int(args.seed)),
            "--epochs",
            str(int(args.epochs)),
        ]
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(int(args.batch_size))])
        if args.resume:
            cmd.append("--resume")
        if args.no_final_eval:
            cmd.append("--no_final_eval")
        if args.loss_type is not None:
            cmd.extend(["--loss_type", args.loss_type])
        logging.info("Launch subprocess: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        results.append((appliance, int(result.returncode)))
        if result.returncode != 0:
            logging.error(
                "Subprocess for appliance %s exited with code %s.",
                appliance,
                result.returncode,
            )
            if args.stop_on_error:
                logging.error("Stop batch due to --stop_on_error.")
                break

    failed = [(a, rc) for a, rc in results if rc != 0]
    if failed:
        logging.error("Batch summary: %s/%s failed.", len(failed), len(results))
        for appliance, rc in failed:
            logging.error("  - %s: exit_code=%s", appliance, rc)
        sys.exit(1)
    logging.info("Batch summary: all %s appliances completed successfully.", len(results))


if __name__ == "__main__":
    main()
