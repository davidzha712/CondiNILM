"""Summarize HPO trial results -- CondiNILM.

Author: Siyi Li
"""

import json
from pathlib import Path
import argparse

def load_last_record(path: Path):
    last=None
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            last=json.loads(line)
    return last

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='result/UKDALE_1min/128')
    ap.add_argument('--glob', type=str, default='Multi_T*/val_report.jsonl')
    args=ap.parse_args()

    root=Path(args.root)
    rows=[]
    for fp in sorted(root.glob(args.glob)):
        rec=load_last_record(fp)
        if not rec: continue
        mt=rec.get('metrics_timestamp',{})
        per=rec.get('metrics_timestamp_per_device',{})
        rows.append({
            'trial': fp.parts[-2],
            'epoch': rec.get('epoch'),
            'F1': mt.get('F1_SCORE'),
            'MAE': mt.get('MAE'),
            'NDE': mt.get('NDE'),
            'TECA': mt.get('TECA'),
            'kettle_F1': per.get('kettle',{}).get('F1_SCORE'),
            'microwave_F1': per.get('microwave',{}).get('F1_SCORE'),
            'fridge_F1': per.get('fridge',{}).get('F1_SCORE'),
            'dishwasher_F1': per.get('dishwasher',{}).get('F1_SCORE'),
            'washing_machine_F1': per.get('washing_machine',{}).get('F1_SCORE'),
            'energy_ratio': rec.get('energy_ratio'),
            'off_energy_ratio': rec.get('off_energy_ratio'),
            'off_nzr_raw': rec.get('off_pred_nonzero_rate_raw'),
        })

    # sort by F1 descending
    rows_sorted=sorted([r for r in rows if r['F1'] is not None], key=lambda r: r['F1'], reverse=True)
    print('found',len(rows_sorted),'trials')
    print('top 10 by overall F1 (timestamp):')
    for r in rows_sorted[:10]:
        print(r)

if __name__=='__main__':
    main()
