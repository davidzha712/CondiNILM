"""Generate UKDALE labels.dat from metadata -- CondiNILM."""

import os
import yaml

def generate_labels_for_house(metadata_path: str, house_path: str, building_num: int):
    """Generate labels.dat for a single house."""
    yaml_file = os.path.join(metadata_path, f"building{building_num}.yaml")

    if not os.path.exists(yaml_file):
        print(f"  Skipping building{building_num}.yaml - not found")
        return False

    with open(yaml_file, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)

    channel_map = {1: "aggregate"}

    appliances = metadata.get('appliances', [])
    for app in appliances:
        meters = app.get('meters', [])
        name = app.get('original_name', app.get('type', 'unknown'))
        name = name.replace(' ', '_').lower()

        for meter_id in meters:
            if meter_id > 0 and meter_id not in channel_map:
                channel_map[meter_id] = name

    labels_path = os.path.join(house_path, "labels.dat")
    with open(labels_path, 'w') as f:
        for channel_id in sorted(channel_map.keys()):
            f.write(f"{channel_id} {channel_map[channel_id]}\n")

    print(f"  Generated {labels_path} with {len(channel_map)} entries")
    return True

def main():
    data_path = "data/UKDALE"
    metadata_path = os.path.join(data_path, "metadata")

    print("Generating UKDALE labels.dat files from metadata...")
    print(f"Data path: {data_path}")
    print(f"Metadata path: {metadata_path}")

    houses = [d for d in os.listdir(data_path) if d.startswith("house")]
    print(f"Found houses: {houses}")

    for house_dir in sorted(houses):
        house_num = int(house_dir.replace("house_", "").replace("house", ""))
        house_path = os.path.join(data_path, house_dir)

        print(f"\nProcessing {house_dir} (building {house_num})...")
        generate_labels_for_house(metadata_path, house_path, house_num)

if __name__ == "__main__":
    main()
