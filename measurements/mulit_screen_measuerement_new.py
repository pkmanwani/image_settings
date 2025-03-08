import json
import os

# Path to parameters.json
json_file = "../beamlines/awa/2025_02_20/3Screen_HandTweaked/parameters.json"

# Check if parameters.json exists
if not os.path.exists(json_file):
    print(f"Error: {json_file} not found.")
else:
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Process each entry
    for entry in data:
        filename = entry.get("filename", "")
        rx_values = entry.get("Rx_final", [])
        ry_values = entry.get("Ry_final", [])

        if not filename:
            print("Skipping entry with missing filename.")
            continue

        if not (rx_values and ry_values and rx_values[0] != 0 and ry_values[0] != 0):
            print(f"{filename}: Missing or zero Rx or Ry values.")
            continue

        print(f"\n{filename}: Rx = {rx_values[0]}, Ry = {ry_values[0]}")

        # Extract date from filename path
        parts = filename.split("/")
        if len(parts) < 4:
            print(f"Skipping {filename}: Could not extract date.")
            continue

        date_folder = parts[2]  # Extract "2025_02_20" from path
        elements_path = f"../beamlines/awa/{date_folder}/elements.json"

        # Check if elements.json exists
        if not os.path.exists(elements_path):
            print(f"Error: {elements_path} not found.")
            continue

        # Load elements.json
        with open(elements_path, "r") as f:
            elements_data = json.load(f)

        # Find matching Yag
        matched_yag = None
        for yag in elements_data.get("yags", []):
            if any(name in yag["name"] for name in possible_names):
                matched_yag = yag
                break

        if matched_yag:
            print(f"Found Yag: {matched_yag['name']}")
            print(f"Position: {matched_yag['position']} m")
            print(f"Resolution: {matched_yag['res']} mm/pixel")
        else:
            print(f"No matching Yag found in {elements_path}.")
