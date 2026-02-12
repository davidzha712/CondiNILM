"""Quick check of FINAL_EVAL_JSON structure for multi-device experiments."""
import json, re, glob, os

dirs = [
    r"C:\Users\Workstation\Workspace\CondiNILM\logs\rerun_collapsed_20260211_130756",
    r"C:\Users\Workstation\Workspace\CondiNILM\logs\comparison_20260210_005222",
]

multi_patterns = ["T2_*_multi", "T3_*_multi_controlled", "T5_multi_*"]

for d in dirs:
    for pat in multi_patterns:
        for log_path in sorted(glob.glob(os.path.join(d, f"{pat}.log"))):
            name = os.path.splitext(os.path.basename(log_path))[0]
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "FINAL_EVAL_JSON" in line and '"test"' in line:
                        m = re.search(r'FINAL_EVAL_JSON: ({.*})', line)
                        if m:
                            data = json.loads(m.group(1))
                            keys = [k for k in data.keys() if k not in ("overall", "split")]
                            per_dev = {k: v for k, v in data.items()
                                      if k not in ("overall", "split") and isinstance(v, dict) and "NDE" in v}
                            if per_dev:
                                print(f"\n{name} ({os.path.basename(d)[:10]}):")
                                for dev, dm in sorted(per_dev.items()):
                                    print(f"  {dev:<20} NDE={dm.get('NDE',-1):.3f}  MAE={dm.get('MAE',-1):.1f}  "
                                          f"RMSE={dm.get('RMSE',-1):.1f}  F1={dm.get('F1_SCORE',0):.3f}  "
                                          f"SAE={dm.get('SAE',-1):.3f}  TECA={dm.get('TECA',-1):.3f}")
                            else:
                                print(f"\n{name}: no per-device (keys: {keys})")
