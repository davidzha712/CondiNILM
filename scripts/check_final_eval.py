"""Check FINAL_EVAL_JSON test results from any rerun log."""
import json, re, sys, glob, os

log_dir = r"C:\Users\Workstation\Workspace\CondiNILM\logs\rerun_collapsed_20260211_130756"
pattern = sys.argv[1] if len(sys.argv) > 1 else "T2_*"
logs = sorted(glob.glob(os.path.join(log_dir, f"{pattern}.log")))

for log_path in logs:
    name = os.path.splitext(os.path.basename(log_path))[0]
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r'FINAL_EVAL_JSON: ({.*})', line)
            if m and '"test"' in line:
                data = json.loads(m.group(1))
                o = data.get("overall", {})
                print(f"\n=== {name} (test) ===")
                print(f"  Overall: NDE={o.get('NDE',-1):.3f}  MAE={o.get('MAE',-1):.1f}  F1={o.get('F1_SCORE',-1):.3f}")
                # Per-device
                per_dev = data.get("per_device", {})
                if not per_dev:
                    per_dev = {k: v for k, v in data.items()
                              if k not in ("overall", "split") and isinstance(v, dict) and "NDE" in v}
                for dev, dm in sorted(per_dev.items()):
                    print(f"  {dev:<18}: NDE={dm.get('NDE',-1):.3f}  MAE={dm.get('MAE',-1):.1f}  F1={dm.get('F1_SCORE',0):.3f}")
