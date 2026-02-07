#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find best 5-device results across all configurations."""
import json
import os
import glob

# Find all Multi_T* results across all configurations
results = []

for report_path in glob.glob('result/UKDALE_*/*/Multi_T*/val_report.jsonl'):
    parts = report_path.replace('\\', '/').split('/')
    sampling_rate = parts[1].replace('UKDALE_', '')
    window_size = parts[2]
    trial = parts[3]

    try:
        with open(report_path, 'r') as f:
            lines = f.readlines()

        # Find best epoch (highest weighted F1)
        best_weighted_f1 = 0
        best_epoch_data = None

        for line in lines:
            data = json.loads(line)
            per_device = data.get('metrics_timestamp_per_device', {})

            # Skip if not 5 devices
            if len(per_device) != 5:
                continue

            # Calculate weighted F1
            weights = {'microwave': 2.0, 'kettle': 1.5, 'fridge': 1.0, 'washing_machine': 1.0, 'dishwasher': 1.0}
            total_weight = sum(weights.values())

            try:
                weighted_f1 = sum(per_device[dev]['F1_SCORE'] * weights[dev] for dev in weights) / total_weight
            except KeyError:
                continue

            if weighted_f1 > best_weighted_f1:
                best_weighted_f1 = weighted_f1
                best_epoch_data = data

        if best_epoch_data:
            results.append((f'{sampling_rate}/{window_size}/{trial}', sampling_rate, window_size, best_weighted_f1, best_epoch_data))
    except Exception as e:
        pass

# Sort by weighted F1
results.sort(key=lambda x: x[3], reverse=True)

print('Top 10 5-device results across ALL configurations:')
print('=' * 80)
for path, sr, ws, wf1, data in results[:10]:
    per_device = data['metrics_timestamp_per_device']
    print(f'{path}: weighted_F1={wf1:.4f}, epoch={data["epoch"]}')
    print(f'  Config: sampling_rate={sr}, window_size={ws}')
    for dev in ['microwave', 'kettle', 'fridge', 'washing_machine', 'dishwasher']:
        m = per_device[dev]
        print(f'    {dev}: F1={m["F1_SCORE"]:.3f}')
    print()
