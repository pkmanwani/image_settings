import json

def update_parameters():
    # Load parameters.json (dataset)
    with open("2025_02_20/DowntheLine/parameters.json", "r") as f:
        dataset = json.load(f)

    # Load res.json (resolution values)
    with open("../beamlines/awa/res.json", "r") as f:
        res_data = json.load(f)[0]  # Extract the dictionary inside the list

    # Function to extract the correct key from the filename
    def get_res_key(filename):
        for key in res_data.keys():
            if key in filename:
                return key
        return None

    # Update dataset with correct resolution only if "res" does not exist
    for entry in dataset:
        if "res" not in entry:  # Check if "res" is already present
            key = get_res_key(entry["filename"])
            if key and key in res_data:
                entry["res"] = res_data[key]

    # Save updated dataset back to parameters.json
    with open("parameters.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Updated parameters.json successfully!")

    # Load updated parameters.json
    with open("parameters.json", "r") as f:
        dataset = json.load(f)

    # Convert to mm using res
    for entry in dataset:
        if "res" in entry:  # Ensure "res" is available
            res = entry["res"]
            entry["Rx_mm"] = entry["Rx_final"] * res
            entry["Ry_mm"] = entry["Ry_final"] * res
            entry["Sx_mm"] = entry["Sx_final"] * res
            entry["Sy_mm"] = entry["Sy_final"] * res

    # Save updated parameters.json
    with open("parameters.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Updated parameters.json with mm values successfully!")


if __name__ == "__main__":
    update_parameters()