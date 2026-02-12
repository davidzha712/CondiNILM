"""Extract per-device metrics from all multi-device experiments."""
import json, re, glob, os

dirs = [
    ("rerun", r"C:\Users\Workstation\Workspace\CondiNILM\logs\rerun_collapsed_20260211_130756"),
    ("V9", r"C:\Users\Workstation\Workspace\CondiNILM\logs\comparison_20260210_005222"),
]

multi_patterns = ["T2_*_multi", "T3_*_multi_controlled", "T5_multi_*",
                  "T4_A1_*", "T4_A2_*", "T4_A3_*", "T4_A4_*",
                  "T4_A5_*", "T4_A6_*", "T4_A7_*", "T4_A8_*"]

seen = set()
all_data = {}

for src, d in dirs:
    for pat in multi_patterns:
        for log_path in sorted(glob.glob(os.path.join(d, f"{pat}.log"))):
            name = os.path.splitext(os.path.basename(log_path))[0]
            if name in seen:
                continue
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                test_line = None
                for line in f:
                    if "FINAL_EVAL_JSON" in line and '"test"' in line:
                        test_line = line
            if test_line:
                m = re.search(r'FINAL_EVAL_JSON: ({.*})', test_line)
                if m:
                    data = json.loads(m.group(1))
                    per_dev = data.get("per_device", {})
                    overall = data.get("overall", {})
                    if per_dev:
                        seen.add(name)
                        all_data[name] = {"source": src, "overall": overall, "per_device": per_dev}
                        print(f"\n{name} ({src}):")
                        print(f"  Overall: NDE={overall.get('NDE',-1):.3f}  MAE={overall.get('MAE',-1):.1f}  "
                              f"RMSE={overall.get('RMSE',-1):.1f}  F1={overall.get('F1_SCORE',0):.3f}  "
                              f"SAE={overall.get('SAE',-1):.3f}")
                        for dev, dm in sorted(per_dev.items()):
                            print(f"  {dev:<20} NDE={dm.get('NDE',-1):.3f}  MAE={dm.get('MAE',-1):.1f}  "
                                  f"RMSE={dm.get('RMSE',-1):.1f}  F1={dm.get('F1_SCORE',0):.3f}  "
                                  f"SAE={dm.get('SAE',-1):.3f}")

# Save
out = os.path.join(r"C:\Users\Workstation\Workspace\CondiNILM\scripts", "per_device_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out}")
