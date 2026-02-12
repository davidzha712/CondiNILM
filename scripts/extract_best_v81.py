"""Extract best-epoch results from V8.1 val_report.jsonl for UKDALE and REFIT."""
import json

def extract_best(path, label):
    """Find epoch with lowest NDE and print all 12 metrics."""
    best_nde = 999
    best_data = None
    all_epochs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            m = d["metrics_timestamp"]
            nde = m["NDE"]
            all_epochs.append((d["epoch"], nde, m["F1_SCORE"], m["MAE"]))
            if nde < best_nde:
                best_nde = nde
                best_data = d

    print(f"\n{'='*80}")
    print(f"{label} - Best epoch: {best_data['epoch']} (NDE={best_nde:.3f})")
    print(f"{'='*80}")

    m = best_data["metrics_timestamp"]
    print(f"\nOverall (all 12 metrics):")
    for k in ["MAE", "MSE", "RMSE", "NDE", "SAE", "TECA", "MR",
              "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL", "F1_SCORE"]:
        print(f"  {k:20s} = {m[k]:.4f}")

    pd = best_data["metrics_timestamp_per_device"]
    for dev, dm in sorted(pd.items()):
        print(f"\n  {dev}:")
        for k in ["MAE", "MSE", "RMSE", "NDE", "SAE", "TECA", "MR",
                  "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL", "F1_SCORE"]:
            print(f"    {k:20s} = {dm[k]:.4f}")

    # Show top-5 epochs by NDE
    all_epochs.sort(key=lambda x: x[1])
    print(f"\nTop-5 epochs by NDE:")
    for ep, nde, f1, mae in all_epochs[:5]:
        print(f"  epoch={ep:2d}  NDE={nde:.3f}  F1={f1:.3f}  MAE={mae:.1f}")

    # Also save as JSON
    return {
        "epoch": best_data["epoch"],
        "overall": m,
        "per_device": pd,
    }

uk = extract_best(
    r"C:\Users\Workstation\Workspace\CondiNILM\best_experiments\UKDALE_V8.1_best\result\val_report.jsonl",
    "UKDALE V8.1"
)
rf = extract_best(
    r"C:\Users\Workstation\Workspace\CondiNILM\best_experiments\REFIT_V8.1_best\result\val_report.jsonl",
    "REFIT V8.1"
)

# Save
out = {"UKDALE_V8.1": uk, "REFIT_V8.1": rf}
with open(r"C:\Users\Workstation\Workspace\CondiNILM\scripts\v81_best.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to scripts/v81_best.json")
