import json, re
f = open(r'C:\Users\Workstation\Workspace\CondiNILM\logs\comparison_20260210_005222\T3_NILMFormer_multi_controlled.log', 'r', encoding='utf-8', errors='ignore')
for line in f:
    if 'FINAL_EVAL_JSON' in line and '"test"' in line:
        m = re.search(r'FINAL_EVAL_JSON: ({.*})', line)
        if m:
            d = json.loads(m.group(1))
            o = d.get('overall', {})
            print(f"Overall: NDE={o.get('NDE'):.3f} MAE={o.get('MAE'):.1f} RMSE={o.get('RMSE'):.1f} F1={o.get('F1_SCORE'):.3f} SAE={o.get('SAE'):.3f}")
            for dev, dm in sorted(d.get('per_device', {}).items()):
                print(f"  {dev:<20} NDE={dm.get('NDE'):.3f} MAE={dm.get('MAE'):.1f} RMSE={dm.get('RMSE'):.1f} F1={dm.get('F1_SCORE',0):.3f} SAE={dm.get('SAE'):.3f}")
f.close()
