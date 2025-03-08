import json
import os
def update_parameters():
    # Define the dataset folder path
    dataset_folder = "../beamlines/awa/2025_02_20/DowntheLine"

    # Load parameters.json (dataset)
    parameters_path = os.path.join(dataset_folder, "parameters.json")
    with open(parameters_path, "r") as f:
        dataset = json.load(f)

    # Update dataset with correct resolution if "res" is missing or equals 1
    for entry in dataset:
        if "res" not in entry or entry["res"] == 1:
            print(f"Warning: Resolution missing or set to 1 for {entry['filename']}.")

    # Convert to mm using res
    for entry in dataset:
        if "res" in entry and entry["res"] is not None:  # Ensure "res" is available
            res = entry["res"]
            entry["Rx_mm"] = entry["Rx_final"][0] * res[0]
            entry["Ry_mm"] = entry["Ry_final"][0] * res[0]
            entry["Sx_mm"] = entry["Sx_final"][0] * res[0]
            entry["Sy_mm"] = entry["Sy_final"][0] * res[0]

    # Save updated dataset as parameters_mm.json in the same folder
    output_path = os.path.join(dataset_folder, "parameters_mm.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Updated parameters saved to {output_path}")


if __name__ == "__main__":
    update_parameters()
