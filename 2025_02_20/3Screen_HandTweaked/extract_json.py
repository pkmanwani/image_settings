import json

# Load JSON data from the file
with open("parameters.json", "r") as file:  # Replace with your actual JSON file path
    data = json.load(file)

# Initialize variables
sigx_YAG780 = sigy_YAG780 = None
sigx_YAG198 = sigy_YAG198 = None
sigx_EYG7 = sigy_EYG7 = None

# Extract values based on filenames
for entry in data:
    filename = entry["filename"]
    Rx_final = entry["Rx_final"]
    res = entry["res"]

    scaled_Rx = Rx_final * res  # Multiply by res

    if "SlitYagThreeScreen" in filename:
        sigx_YAG780 = scaled_Rx
        sigy_YAG780 = entry["Ry_final"] * res  # Also scale Ry_final for y

    elif "Yag8ThreeScreen" in filename:
        sigx_YAG198 = scaled_Rx
        sigy_YAG198 = entry["Ry_final"] * res

    elif "Yag6ThreeScreen" in filename:
        sigx_EYG7 = scaled_Rx
        sigy_EYG7 = entry["Ry_final"] * res

