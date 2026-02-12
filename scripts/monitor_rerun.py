"""Monitor rerun experiments until completion, then extract all results."""
import json, re, glob, os, time

LOG_DIR = r"C:\Users\Workstation\Workspace\CondiNILM\logs\rerun_collapsed_20260211_130756"
EXPECTED_TOTAL = 34  # Total experiments in rerun

def count_completed():
    logs = glob.glob(os.path.join(LOG_DIR, "*.log"))
    completed = 0
    running = 0
    for log_path in logs:
        has_final = False
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "FINAL_EVAL_JSON" in line and '"test"' in line:
                    has_final = True
                    break
        if has_final:
            completed += 1
        else:
            running += 1
    return len(logs), completed, running

def extract_results():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    results = []
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
                results.append((name, nde, mae, f1, status))
                # Per device
                per_dev = data.get("per_device", {})
                if not per_dev:
                    per_dev = {k: v for k, v in data.items()
                              if k not in ("overall", "split") and isinstance(v, dict) and "NDE" in v}
                for dev, dm in sorted(per_dev.items()):
                    dn = dm.get("NDE", -1)
                    dm_mae = dm.get("MAE", -1)
                    df1 = dm.get("F1_SCORE", 0)
                    results.append((f"  {name}_{dev}", dn, dm_mae, df1, ""))
    return results

while True:
    total, completed, running = count_completed()
    print(f"[{time.strftime('%H:%M:%S')}] Logs: {total}/{EXPECTED_TOTAL}, "
          f"Completed: {completed}, Running: {running}")

    if total >= EXPECTED_TOTAL and running == 0:
        print("\nAll experiments completed!")
        break

    if running == 0 and total > 0:
        # All existing logs have FINAL_EVAL_JSON, but fewer than expected
        # Main process may still be starting new experiments
        pass

    time.sleep(300)  # Check every 5 minutes

# Final extraction
print("\n" + "=" * 100)
print("FINAL RERUN RESULTS")
print("=" * 100)
print(f"{'Experiment':<50} {'NDE':>8} {'MAE':>8} {'F1':>8} {'Status':>10}")
print("-" * 90)

results = extract_results()
for name, nde, mae, f1, status in results:
    if name.startswith("  "):
        print(f"{name:<50} {nde:>8.3f} {mae:>8.1f} {f1:>8.3f}")
    else:
        print(f"{name:<50} {nde:>8.3f} {mae:>8.1f} {f1:>8.3f} {status:>10}")

# Summary
learned = sum(1 for n, nde, _, _, _ in results if not n.startswith("  ") and nde < 0.95)
weak = sum(1 for n, nde, _, _, _ in results if not n.startswith("  ") and 0.95 <= nde < 1.0)
collapsed = sum(1 for n, nde, _, _, _ in results if not n.startswith("  ") and nde >= 1.0)
top_level = [r for r in results if not r[0].startswith("  ")]
print(f"\nSummary: {len(top_level)} experiments")
print(f"  Learned (NDE < 0.95): {learned}")
print(f"  Weak (0.95 <= NDE < 1.0): {weak}")
print(f"  Collapsed (NDE >= 1.0): {collapsed}")
