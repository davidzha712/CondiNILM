"""Export best single-device REFIT experiments from TensorBoard logs.

Scans all log/tensorboard/REFIT_* directories (excluding Multi_*),
extracts test metrics, and finds the best experiment per device.
Exports ALL scalar tags to CSV and copies the TB directory.

Fallback: if no test metrics exist for any experiment for a device,
uses best val_loss instead.
"""
import os, csv, glob, shutil
from tensorboard.backend.event_processing import event_accumulator

# 1. Scan all REFIT single-device TB dirs (exclude Multi_*)
tb_base = 'log/tensorboard'
all_dirs = []
for d in glob.glob(os.path.join(tb_base, 'REFIT_*')):
    name = os.path.basename(d)
    if name.startswith('REFIT_Multi_'):
        continue
    # Find all version subdirs
    for v in sorted(glob.glob(os.path.join(d, 'version_*'))):
        all_dirs.append(v)

print(f"Found {len(all_dirs)} version directories across single-device REFIT experiments")

# 2. For each dir, extract device name, model, test F1
DEVICES = ['Dishwasher', 'Fridge', 'Kettle', 'WashingMachine']

results = []
for d in all_dirs:
    parent_name = os.path.basename(os.path.dirname(d))
    parts = parent_name.replace('REFIT_', '', 1)
    try:
        ea = event_accumulator.EventAccumulator(d)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        test_f1 = ea.Scalars('test_timestamp/F1_SCORE')[-1].value if 'test_timestamp/F1_SCORE' in tags else -1
        test_mae = ea.Scalars('test_timestamp/MAE')[-1].value if 'test_timestamp/MAE' in tags else -1
        n_epochs = int(ea.Scalars('epoch')[-1].value) + 1 if 'epoch' in tags else 0

        # val_loss for fallback (lower is better)
        val_loss = ea.Scalars('val_loss')[-1].value if 'val_loss' in tags else float('inf')
        # best (minimum) val_loss across all logged steps
        if 'val_loss' in tags:
            val_loss = min(e.value for e in ea.Scalars('val_loss'))

        # Determine device name - check if parts starts with known device
        device = None
        for dev in DEVICES:
            if parts.startswith(dev + '_') or parts == dev:
                device = dev
                break
        if device:
            version = os.path.basename(d)
            results.append({
                'device': device, 'tb_dir': d, 'name': parts,
                'version': version,
                'test_f1': test_f1, 'test_mae': test_mae, 'epochs': n_epochs,
                'val_loss': val_loss,
                'has_test': test_f1 >= 0,
                'tags': tags
            })
    except Exception as e:
        print(f"  WARN: Failed to process {d}: {e}")

print(f"Parsed {len(results)} experiment versions with recognized devices")
print(f"  Devices found: {sorted(set(r['device'] for r in results))}")

# Show summary of all experiments per device
print("\n--- All experiments scanned ---")
for dev in DEVICES:
    dev_results = [r for r in results if r['device'] == dev]
    dev_results.sort(key=lambda r: r['test_f1'], reverse=True)
    n_with_test = sum(1 for r in dev_results if r['has_test'])
    print(f"\n{dev} ({len(dev_results)} versions, {n_with_test} with test metrics):")
    for r in dev_results:
        f1_str = f"{r['test_f1']:.4f}" if r['test_f1'] >= 0 else "N/A"
        mae_str = f"{r['test_mae']:.1f}" if r['test_mae'] >= 0 else "N/A"
        vl_str = f"{r['val_loss']:.4f}" if r['val_loss'] < float('inf') else "N/A"
        print(f"  F1={f1_str}  MAE={mae_str}  val_loss={vl_str}  E={r['epochs']:3d}  {r['name']}/{r['version']}")

# 3. Find best per device
#    Primary: best test F1 (among experiments that have test metrics)
#    Fallback: best (lowest) val_loss if no test metrics exist for any version
best = {}
for dev in DEVICES:
    dev_results = [r for r in results if r['device'] == dev]
    with_test = [r for r in dev_results if r['has_test']]

    if with_test:
        # Pick by best test F1
        winner = max(with_test, key=lambda r: r['test_f1'])
        winner['selection'] = 'test_f1'
    elif dev_results:
        # Fallback: best val_loss
        winner = min(dev_results, key=lambda r: r['val_loss'])
        winner['selection'] = 'val_loss (fallback - no test metrics)'
        print(f"\n  WARNING: No test metrics for {dev}, using val_loss fallback")
    else:
        print(f"\n  WARNING: No experiments found for {dev}")
        continue
    best[dev] = winner

# 4. Export each best experiment's ALL scalar metrics to CSV
output_base = 'log/best_single_device/REFIT'
os.makedirs(output_base, exist_ok=True)

for dev, info in sorted(best.items()):
    print(f"\n=== {dev} ===")
    print(f"  Best: {info['name']}/{info['version']}")
    print(f"  TB dir: {info['tb_dir']}")
    sel = info.get('selection', 'test_f1')
    if info['has_test']:
        print(f"  Test F1: {info['test_f1']:.4f}, Test MAE: {info['test_mae']:.1f}, Epochs: {info['epochs']}")
    else:
        print(f"  (no test metrics) val_loss: {info['val_loss']:.4f}, Epochs: {info['epochs']}")
    print(f"  Selected by: {sel}")

    # Export all scalars to CSV
    ea = event_accumulator.EventAccumulator(info['tb_dir'])
    ea.Reload()

    dev_dir = os.path.join(output_base, dev)
    os.makedirs(dev_dir, exist_ok=True)

    # Save info
    with open(os.path.join(dev_dir, 'info.txt'), 'w') as f:
        f.write(f"Device: {dev}\n")
        f.write(f"Experiment: {info['name']}\n")
        f.write(f"Version: {info['version']}\n")
        f.write(f"TB Source: {info['tb_dir']}\n")
        f.write(f"Selected by: {sel}\n")
        if info['has_test']:
            f.write(f"Test F1: {info['test_f1']:.4f}\n")
            f.write(f"Test MAE: {info['test_mae']:.1f}\n")
        else:
            f.write(f"Test F1: N/A\n")
            f.write(f"Test MAE: N/A\n")
            f.write(f"Val Loss (best): {info['val_loss']:.4f}\n")
        f.write(f"Epochs: {info['epochs']}\n")

    # Export each scalar tag to a single CSV
    csv_path = os.path.join(dev_dir, 'all_scalars.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'wall_time', 'step', 'epoch_approx', 'value'])

        # Build step->epoch map
        epoch_map = {}
        if 'epoch' in ea.Tags()['scalars']:
            for e in ea.Scalars('epoch'):
                epoch_map[e.step] = int(e.value)

        for tag in sorted(ea.Tags()['scalars']):
            for event in ea.Scalars(tag):
                ep = epoch_map.get(event.step, -1)
                writer.writerow([tag, event.wall_time, event.step, ep, event.value])

    print(f"  Exported to: {csv_path}")

    # Also copy the TB directory
    dest = os.path.join(dev_dir, 'tensorboard')
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(info['tb_dir'], dest)
    print(f"  TB copied to: {dest}")

print("\n\nDone! All best single-device REFIT experiments exported to:", output_base)
