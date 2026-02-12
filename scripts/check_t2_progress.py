"""Quick script to check T2_NILMFormer_multi training progress."""
import json, re

log = r"C:\Users\Workstation\Workspace\CondiNILM\logs\rerun_collapsed_20260211_130756\T2_NILMFormer_multi.log"

with open(log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r'VAL_REPORT_JSON: ({.*})', line)
        if m:
            data = json.loads(m.group(1))
            ep = data.get("epoch", "?")
            mt = data.get("metrics_timestamp", {})
            nde = mt.get("NDE", -1)
            f1 = mt.get("F1_SCORE", -1)
            mae = mt.get("MAE", -1)
            cf = data.get("collapse_flag", None)
            print(f"Epoch {ep:>2}: NDE={nde:.3f}  MAE={mae:.1f}  F1={f1:.3f}  collapsed={cf}")

        m2 = re.search(r'FINAL_EVAL_JSON: ({.*})', line)
        if m2:
            data = json.loads(m2.group(1))
            split = data.get("split", "?")
            o = data.get("overall", {})
            nde = o.get("NDE", -1)
            f1 = o.get("F1_SCORE", -1)
            mae = o.get("MAE", -1)
            print(f"\n>>> FINAL_EVAL ({split}): NDE={nde:.3f}  MAE={mae:.1f}  F1={f1:.3f}")
            per_dev = {k: v for k, v in data.items() if k != "overall" and isinstance(v, dict) and "NDE" in v}
            for dev, dm in sorted(per_dev.items()):
                print(f"    {dev:<18}: NDE={dm['NDE']:.3f}  MAE={dm['MAE']:.1f}  F1={dm.get('F1_SCORE',0):.3f}")
