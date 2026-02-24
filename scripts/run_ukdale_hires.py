"""UKDALE high-resolution (6s) experiment launcher.

Purpose: Verify NILMFormer performance at original 6s sampling rate
with reduced data to keep compute comparable to 1min baseline.

Design:
  - sampling_rate=6s (original UKDALE resolution, no downsampling)
  - window_size=480 (480 × 6s = 48 min, covers 2-3 fridge cycles)
  - batch_size=256 (VRAM constraint: 256×480² ≈ 1.76× current attention memory)
  - overlap=0.5 (reduced from 0.75 to limit window count)
  - limit_train/val_batches=0.15 (only 15% of data for quick verification)
  - epochs=25, train_num_crops=2

VRAM estimate: ~30GB peak (bf16), fits RTX 5090 32GB.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import yaml

HIRES_DEFAULTS = dict(
    sampling_rate="6s",
    window_size=480,          # 480 × 6s = 48 min
    batch_size=256,           # VRAM safe for 32GB GPU
    epochs=25,
    overlap=0.5,
    # These go through hpo_override_json → expes_config
    hpo_extras=dict(
        limit_train_batches=0.15,
        limit_val_batches=0.15,
        train_num_crops=2,
        train_crop_ratio=0.75,
        train_crop_event_bias=0.8,
        early_stopping_patience=8,
        check_val_every_n_epoch=1,   # More frequent validation with less data
    ),
)


def get_ukdale_appliances():
    with open("configs/datasets.yaml", "r", encoding="utf-8") as f:
        datasets_all = yaml.safe_load(f)
    dataset_key_map = {k.lower(): k for k in datasets_all.keys()}
    dataset_key = dataset_key_map.get("ukdale")
    if dataset_key is None:
        raise ValueError("Dataset UKDALE not found in configs/datasets.yaml")
    appliances = list(datasets_all[dataset_key].keys())
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
            raise ValueError(f"Unknown appliance '{name}'. Available: {available}")
        if key not in selected:
            selected.append(key)
    return selected if selected else all_appliances


def main():
    parser = argparse.ArgumentParser(
        description="Run UKDALE high-resolution (6s) experiments for NILMFormer."
    )
    parser.add_argument(
        "--appliances", type=str, default="all",
        help="Comma-separated appliance list or 'all'. Default: all.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=f"Override batch size (default: {HIRES_DEFAULTS['batch_size']}). "
             "Reduce to 128 if OOM.",
    )
    parser.add_argument(
        "--window_size", type=int, default=None,
        help=f"Override window size (default: {HIRES_DEFAULTS['window_size']}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Override max epochs (default: {HIRES_DEFAULTS['epochs']}).",
    )
    parser.add_argument(
        "--limit_data", type=float, default=None,
        help="Override limit_train/val_batches fraction (default: 0.15).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint.",
    )
    parser.add_argument(
        "--no_final_eval", action="store_true",
        help="Skip final heavy evaluation.",
    )
    parser.add_argument(
        "--stop_on_error", action="store_true",
        help="Stop at first failed appliance.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_key, all_appliances = get_ukdale_appliances()
    appliances = parse_appliance_list(all_appliances, args.appliances)

    sampling_rate = HIRES_DEFAULTS["sampling_rate"]
    window_size = args.window_size or HIRES_DEFAULTS["window_size"]
    batch_size = args.batch_size or HIRES_DEFAULTS["batch_size"]
    epochs = args.epochs or HIRES_DEFAULTS["epochs"]

    hpo_extras = dict(HIRES_DEFAULTS["hpo_extras"])
    if args.limit_data is not None:
        hpo_extras["limit_train_batches"] = args.limit_data
        hpo_extras["limit_val_batches"] = args.limit_data

    hpo_json = json.dumps(hpo_extras)

    logging.info("=" * 60)
    logging.info("UKDALE High-Resolution (6s) Experiment")
    logging.info("=" * 60)
    logging.info("  Sampling rate : %s", sampling_rate)
    logging.info("  Window size   : %d (= %d min)", window_size, window_size * 6 // 60)
    logging.info("  Batch size    : %d", batch_size)
    logging.info("  Epochs        : %d", epochs)
    logging.info("  Train data    : %.0f%%", hpo_extras["limit_train_batches"] * 100)
    logging.info("  Val data      : %.0f%%", hpo_extras["limit_val_batches"] * 100)
    logging.info("  Crops         : %d", hpo_extras["train_num_crops"])
    logging.info("  Overlap       : %.2f", HIRES_DEFAULTS["overlap"])
    logging.info("  Appliances    : %s", ", ".join(appliances))
    logging.info("  Tag           : hires_6s")
    logging.info("=" * 60)

    results = []
    for appliance in appliances:
        logging.info(">>> Training %s ...", appliance)

        cmd = [
            sys.executable, "-m", "scripts.run_experiment",
            "--dataset", dataset_key,
            "--sampling_rate", sampling_rate,
            "--window_size", str(window_size),
            "--appliance", appliance,
            "--name_model", "NILMFormer",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--overlap", str(HIRES_DEFAULTS["overlap"]),
            "--experiment_tag", "hires_6s",
            "--hpo_override_json", hpo_json,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.no_final_eval:
            cmd.append("--no_final_eval")

        logging.info("CMD: %s", " ".join(cmd))

        if args.dry_run:
            logging.info("[DRY RUN] Skipped.")
            results.append((appliance, 0))
            continue

        result = subprocess.run(cmd, check=False)
        rc = int(result.returncode)
        results.append((appliance, rc))

        if rc != 0:
            logging.error("!!! %s FAILED (exit code %d)", appliance, rc)
            if args.stop_on_error:
                logging.error("Stopping due to --stop_on_error.")
                break
        else:
            logging.info("<<< %s completed successfully.", appliance)

    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    failed = [(a, rc) for a, rc in results if rc != 0]
    for appliance, rc in results:
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        logging.info("  %-20s %s", appliance, status)
    if failed:
        logging.error("%d/%d failed.", len(failed), len(results))
        sys.exit(1)
    else:
        logging.info("All %d appliances completed successfully.", len(results))


if __name__ == "__main__":
    main()
