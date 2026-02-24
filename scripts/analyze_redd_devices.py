"""REDD dataset device availability analysis -- CondiNILM."""

import pandas as pd
import numpy as np
import os
import glob
from itertools import combinations

data_path = "C:/Users/Workstation/Workspace/CondiNILM/data/REDD"

# REDD appliance name mapping
REDD_APPLIANCE_MAPPING = {
    "dish washer": "dishwasher",
    "washer dryer": "washing_machine",
    "electric space heater": "electric_heater",
    "electric stove": "cooker",
    "fridge": "fridge",
    "microwave": "microwave",
    "electric furnace": "electric_furnace",
    "CE appliance": "ce_appliance",
    "waste disposal unit": "waste_disposal",
}

print("=" * 80)
print("REDD Dataset Device Analysis - Raw Column Names")
print("=" * 80)

house_columns = {}
house_device_activity = {}

for house in range(1, 7):
    pattern = os.path.join(data_path, f"redd_house{house}_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if csv_files:
        df_first = pd.read_csv(csv_files[0], nrows=10)
        raw_cols = [c for c in df_first.columns if c not in ["Unnamed: 0", "index"]]

        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)
            all_data.append(df)
        house_data = pd.concat(all_data, ignore_index=True)

        standardized_cols = {}
        for col in raw_cols:
            if col in REDD_APPLIANCE_MAPPING:
                standardized_cols[col] = REDD_APPLIANCE_MAPPING[col]
            elif col not in ["main", "aggregate"]:
                standardized_cols[col] = col

        house_columns[house] = set(standardized_cols.values())

        print(f"\n{'='*60}")
        print(f"House {house} - Raw columns: {raw_cols}")
        print(f"House {house} - Standardized devices: {sorted(standardized_cols.values())}")
        print("-" * 60)

        activity = {}
        for raw_col, std_col in standardized_cols.items():
            if raw_col in house_data.columns:
                data_col = house_data[raw_col]
                active_rate = (data_col > 10).mean() * 100
                max_power = data_col.max()
                mean_on = data_col[data_col > 10].mean() if (data_col > 10).any() else 0
                activity[std_col] = {
                    "active_rate": active_rate,
                    "max_power": max_power,
                    "mean_on": mean_on,
                    "raw_name": raw_col
                }
                status = "[OK]" if active_rate > 0.5 else "[SPARSE]" if active_rate > 0.1 else "[NONE]"
                print(f"  {std_col:20s} ({raw_col:20s}): {active_rate:6.2f}% active, max={max_power:7.1f}W {status}")

        house_device_activity[house] = activity
    else:
        print(f"House {house}: NO FILES FOUND")

print("\n" + "=" * 80)
print("Common Devices Analysis")
print("=" * 80)

if len(house_columns) >= 2:
    common_all = set.intersection(*house_columns.values())
    print(f"\nDevices in ALL {len(house_columns)} houses:")
    if common_all:
        for dev in sorted(common_all):
            activities = []
            for h in house_columns.keys():
                act = house_device_activity.get(h, {}).get(dev, {}).get("active_rate", 0)
                activities.append(f"H{h}:{act:.1f}%")
            print(f"  {dev}: {', '.join(activities)}")
    else:
        print("  NONE - No device exists in all houses!")

    print("\n" + "-" * 40)
    print("Common devices by house combinations:")
    print("-" * 40)

    houses_list = sorted(house_columns.keys())

    for n in range(len(houses_list), 1, -1):
        for combo in combinations(houses_list, n):
            common = set.intersection(*[house_columns[h] for h in combo])
            if common:
                active_common = []
                for dev in common:
                    min_activity = min(
                        house_device_activity.get(h, {}).get(dev, {}).get("active_rate", 0)
                        for h in combo
                    )
                    if min_activity > 0.3:
                        active_common.append((dev, min_activity))

                if active_common:
                    print(f"\n  Houses {combo}:")
                    print(f"    All common columns: {sorted(common)}")
                    print(f"    Active devices (>0.3% in all): {[d[0] for d in sorted(active_common)]}")

print("\n" + "=" * 80)
print("RECOMMENDED MULTI-DEVICE TRAINING CONFIGURATIONS")
print("=" * 80)

best_configs = []

for n_houses in range(len(houses_list), 1, -1):
    for combo in combinations(houses_list, n_houses):
        common = set.intersection(*[house_columns[h] for h in combo])

        active_devices = []
        for dev in common:
            activities = [
                house_device_activity.get(h, {}).get(dev, {}).get("active_rate", 0)
                for h in combo
            ]
            min_act = min(activities)
            avg_act = sum(activities) / len(activities)
            if min_act > 0.3:  # Meaningful activity threshold
                active_devices.append({
                    "name": dev,
                    "min_activity": min_act,
                    "avg_activity": avg_act
                })

        if len(active_devices) >= 2:  # At least 2 active devices
            score = len(active_devices) * n_houses
            best_configs.append({
                "houses": combo,
                "devices": [d["name"] for d in sorted(active_devices, key=lambda x: -x["avg_activity"])],
                "device_details": active_devices,
                "n_devices": len(active_devices),
                "n_houses": n_houses,
                "score": score
            })

best_configs.sort(key=lambda x: (x["score"], x["n_devices"]), reverse=True)

print("\nTop 5 configurations (sorted by devices x houses):\n")
for i, cfg in enumerate(best_configs[:5]):
    print(f"{i+1}. Houses {cfg['houses']} ({cfg['n_houses']} houses)")
    print(f"   Devices ({cfg['n_devices']}): {cfg['devices']}")
    for d in cfg["device_details"]:
        print(f"     - {d['name']}: min={d['min_activity']:.2f}%, avg={d['avg_activity']:.2f}%")
    print(f"   Score: {cfg['score']}")
    print()

# Provide concrete training/test split recommendations
print("=" * 80)
print("CONCRETE TRAINING RECOMMENDATIONS")
print("=" * 80)

if best_configs:
    best = best_configs[0]
    houses = list(best["houses"])
    devices = best["devices"]

    if len(houses) >= 3:
        train_houses = houses[:-1]
        test_houses = [houses[-1]]
    else:
        train_houses = houses[:1]
        test_houses = houses[1:]

    print(f"\nBest configuration:")
    print(f"  Devices: {devices}")
    print(f"  Train houses: {train_houses}")
    print(f"  Test houses: {test_houses}")
    print(f"\nCommand example:")
    print(f"  python scripts/run_experiment.py \\")
    print(f"    --dataset REDD \\")
    print(f"    --appliance multi \\")
    print(f"    --app {','.join(devices)} \\")
    print(f"    --ind_house_train_val {','.join(map(str, train_houses))} \\")
    print(f"    --ind_house_test {','.join(map(str, test_houses))}")
