"""Extract FINAL_EVAL_JSON test metrics from rerun logs."""
import json, os, re, glob

log_dir = r"C:\Users\Workstation\Workspace\CondiNILM\logs\rerun_collapsed_20260211_130756"
logs = sorted(glob.glob(os.path.join(log_dir, "*.log")))

print(f"{'Experiment':<45} {'NDE':>6} {'MAE':>8} {'F1':>6} {'Status'}")
print("-" * 80)

for log_path in logs:
    name = os.path.splitext(os.path.basename(log_path))[0]
    test_line = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "FINAL_EVAL_JSON" in line and '"test"' in line:
                test_line = line
    if test_line:
        m = re.search(r'FINAL_EVAL_JSON: ({.*})', test_line)
        if m:
            data = json.loads(m.group(1))
            o = data.get("overall", {})
            nde = o.get("NDE", -1)
            mae = o.get("MAE", -1)
            f1 = o.get("F1_SCORE", -1)
            status = "LEARNED" if nde < 0.95 else "COLLAPSED" if nde >= 1.0 else "WEAK"
            print(f"{name:<45} {nde:>6.3f} {mae:>8.1f} {f1:>6.3f} {status}")
        else:
            print(f"{name:<45} PARSE_ERROR")
    else:
        print(f"{name:<45} RUNNING/NO_RESULT")
