"""Quick single-device regression test on UKDALE and REFIT.

Uses the same config as multi-device training but with individual devices.
10% data for fast verification.
"""

import argparse
import json
import logging
import subprocess
import sys
import yaml

PYTHON = sys.executable

# UKDALE: same config as expes.yaml (batch=2048, 1min, window=128)
UKDALE_DEVICES = ["Fridge", "Kettle", "Microwave", "Dishwasher", "WashingMachine"]
# REFIT: batch=128 per dataset_params.yaml
REFIT_DEVICES = ["Fridge", "Kettle", "Dishwasher", "WashingMachine"]

HPO_OVERRIDES = {
    "limit_train_batches": 0.1,
    "limit_val_batches": 1.0,  # full val (single-device val sets are tiny)
}


def run_experiment(dataset, device, epochs, batch_size, tag, dry_run=False):
    hpo_json = json.dumps(HPO_OVERRIDES)
    cmd = [
        PYTHON, "-m", "scripts.run_experiment",
        "--dataset", dataset,
        "--sampling_rate", "1min",
        "--window_size", "128",
        "--appliance", device,
        "--name_model", "NILMFormer",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--experiment_tag", tag,
        "--hpo_override_json", hpo_json,
    ]
    logging.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Quick single-device regression on UKDALE + REFIT")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset (UKDALE or REFIT). Default: both.")
    parser.add_argument("--device", type=str, default=None,
                        help="Run only this device. Default: all.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tag = "single_10pct"
    results = []

    jobs = []
    if args.dataset is None or args.dataset.upper() == "UKDALE":
        for d in UKDALE_DEVICES:
            if args.device is None or args.device.lower() == d.lower():
                jobs.append(("UKDALE", d, 2048))
    if args.dataset is None or args.dataset.upper() == "REFIT":
        for d in REFIT_DEVICES:
            if args.device is None or args.device.lower() == d.lower():
                jobs.append(("REFIT", d, 128))

    logging.info("=" * 60)
    logging.info("Single-Device Quick Regression (10%% data, %d epochs)", args.epochs)
    logging.info("Jobs: %d", len(jobs))
    for ds, dev, bs in jobs:
        logging.info("  %s / %s (batch=%d)", ds, dev, bs)
    logging.info("=" * 60)

    for ds, dev, bs in jobs:
        logging.info(">>> %s / %s ...", ds, dev)
        rc = run_experiment(ds, dev, args.epochs, bs, tag, args.dry_run)
        results.append((ds, dev, rc))
        if rc != 0:
            logging.error("!!! %s / %s FAILED (rc=%d)", ds, dev, rc)
        else:
            logging.info("<<< %s / %s OK", ds, dev)

    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for ds, dev, rc in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        logging.info("  %-8s %-20s %s", ds, dev, status)
    failed = [x for x in results if x[2] != 0]
    if failed:
        logging.error("%d/%d failed.", len(failed), len(results))
        sys.exit(1)
    logging.info("All %d experiments completed.", len(results))


if __name__ == "__main__":
    main()
