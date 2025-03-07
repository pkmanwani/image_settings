import json

def update_parameters():
    # Load parameters.json (dataset)
    with open("../beamlines/awa/2025_02_20/DowntheLine/parameters.json", "r") as f:
        dataset = json.load(f)

    # Load elements.json (resolution values)
    with open("../beamlines/awa/elements.json", "r") as f:
        elements = json.load(f)

    # Convert elements list to a dictionary mapping names to resolutions
    res_map = {entry["name"].lower(): entry["res"] for entry in elements["yags"]}

    # Account for SlitYag being Yag7
    res_map["slityag"] = res_map.get("yag7", None)

    # Function to extract the correct resolution key from the filename
    def get_res_key(filename):
        for key in res_map.keys():
            if key.lower() in filename.lower():
                return key
        return None

    # Update dataset with correct resolution if "res" is missing or equals 1
    for entry in dataset:
        if "res" not in entry or entry["res"] == 1:
            key = get_res_key(entry["filename"])
            if key and key in res_map and res_map[key] is not None:
                entry["res"] = res_map[key]

    # Save updated dataset back to parameters.json
    with open("parameters.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Updated parameters.json successfully!")

    # Convert to mm using res
    for entry in dataset:
        if "res" in entry:  # Ensure "res" is available
            res = entry["res"]
            entry["Rx_mm"] = entry["Rx_final"][0] * res
            entry["Ry_mm"] = entry["Ry_final"][0] * res
            entry["Sx_mm"] = entry["Sx_final"][0] * res
            entry["Sy_mm"] = entry["Sy_final"][0] * res

    # Save updated parameters.json
    with open("parameters.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Updated parameters.json with mm values successfully!")


if __name__ == "__main__":
    update_parameters()
