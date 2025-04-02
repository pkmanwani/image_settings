import sys
sys.path.append('../')
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import c, e, m_e
from functions.save_parameters_json_parallel import save_parameters_json_parallel

def magnetization_scan(folder_path, yag_string_slit,yag_string_after_slit,cathode_spot_size,error_cathode_size,current_to_Bfield):
    """
    Perform magnetization scan analysis using beam size data from JSON files.

    Parameters:
    - folder_path (str): Path to the folder containing parameters.json and elements.json.
    - yag_string (str): String to identify the YAGs in elements.json.
    """
    # Load parameters.json
    parameters_path = os.path.join(folder_path, "parameters.json")
    elements_path = os.path.join(os.path.dirname(os.path.dirname(folder_path)), "elements.json")

    if not os.path.exists(parameters_path):
        print(f"Skipping {folder_path}: parameters.json not found.")
        return None
    if not os.path.exists(elements_path):
        print(f"Skipping {folder_path}: elements.json not found.")
        return None

    # Load elements.json
    with open(elements_path, "r") as f:
        elements_data = json.load(f)
        print(f"Loaded: {elements_path}")
        # print(elements_data)
        pz = elements_data.get("beam")[0].get("pz")
        print(f"Beam momentum : {pz}")
        beta_gamma = e * pz / (m_e * c ** 2) + 1
        print(f"beta gamma : {beta_gamma}")

    # Extract quad and YAG positions
    yag_1_position = next((item['position'] for item in elements_data['yags'] if yag_string_slit in item['name']), None)
    yag_2_position = next((item['position'] for item in elements_data['yags'] if yag_string_after_slit in item['name']), None)

    if yag_1_position is None or yag_2_position is None:
        print("Error: YAG 1 or YAG 2 not found in elements.json.")
        return

    # Compute drift length
    drift_length = yag_2_position - yag_1_position
    print(f"Drift length: {drift_length} m")

    # Load parameters.json
    with open(parameters_path, "r") as f:
        data = json.load(f)

        # Extract R1 and R2 from "Magnetized Without Slit" files
    R1 = None
    R2 = None
    angle_magnetized = None
    angle_unmagnetized = None

    for entry in data:
        filename = entry["filename"]
        if "Yag6_Magnetized_WithoutSlit" in filename:
            Rx1 = entry["Rx_final"][0] * entry["res"][0]*1e-3  # Convert to meters
            Ry1 = entry["Ry_final"][0] * entry["res"][0]*1e-3  # Convert to meters
            error_Rx1 = entry["Rx_final"][1] * entry["res"][0]*1e-3
            error_Ry1 = entry["Ry_final"][1] * entry["res"][0] * 1e-3
            R1 = np.sqrt(Rx1**2+Ry1**2)
            error_R1 = (1 / R1) * np.sqrt((Rx1 * error_Rx1) ** 2 + (Ry1 * error_Ry1) ** 2)
        elif "SlitYag_Magnetized_WithoutSlit" in filename:
            Rx2 = entry["Rx_final"][0] * entry["res"][0] * 1e-3  # Convert to meters
            Ry2 = entry["Ry_final"][0] * entry["res"][0] * 1e-3  # Convert to meters
            error_Rx2 = entry["Rx_final"][1] * entry["res"][0] * 1e-3
            error_Ry2 = entry["Ry_final"][1] * entry["res"][0] * 1e-3
            R2 = np.sqrt(Rx2 ** 2 + Ry2 ** 2)
            error_R2 = (1 / R2) * np.sqrt((Rx2 * error_Rx2) ** 2 + (Ry2 * error_Ry2) ** 2)
        elif "SlitYag_Magnetized_Slit" in filename:
            angle_magnetized = entry["angle"][0]
            error_angle_magnetized = entry["angle"][1]
            current_magnetized = entry["current"][0] #Amps
            error_current_magnetized = entry["current"][1]
        elif "SlitYag_Unmagnetized_Slit" in filename:
            angle_unmagnetized = entry["angle"][0]
            error_angle_unmagnetized = entry["angle"][1]

    if None in [R1, R2, angle_magnetized, angle_unmagnetized]:
        print(f"Skipping {folder_path}: Missing required parameters.")
        return None

        # Compute orientation difference
    orientation_diff_degrees = angle_magnetized - angle_unmagnetized
    orientation_diff = np.radians(orientation_diff_degrees)
    print(f'Angle change : {angle_magnetized - angle_unmagnetized}')
    error_orientation_diff_degrees = np.sqrt(error_angle_unmagnetized**2 + error_angle_magnetized**2)
    error_orientation_diff = np.radians(error_orientation_diff_degrees)

    B = current_to_Bfield * current_magnetized / 100
    error_B = current_to_Bfield * error_current_magnetized / 100

    # Prepare results dictionary
    B = current_to_Bfield * current_magnetized
    # Compute Magnetization Lengths
    L = (pz * R1 * R2 * np.abs(np.sin(orientation_diff)) * e) / (drift_length * 2 * m_e * c ** 2)
    error_L = L * np.sqrt(
        (error_R1 / R1) ** 2 + (error_R2 / R2) ** 2 + (error_orientation_diff / np.tan(orientation_diff)) ** 2
    )

    L_m = e * B * (cathode_spot_size ** 2) / (2 * m_e * c)
    error_L_m = L_m * np.sqrt(
        (error_B / B) ** 2 + (2 * error_cathode_size / cathode_spot_size) ** 2
    )

    results = {
        "Folder" : os.path.basename(folder_path),
        "Difference in Angle (degrees)" : [orientation_diff_degrees,error_orientation_diff_degrees],
        "Current (A)" : [current_magnetized,error_current_magnetized],
        "Magnetic field at Cathode (T)" : [B,error_B],
        "Magnetization L (um)": [L * 1e6,error_L*1e6],  # Convert to microns
        "Magnetization cathode L (um)": [L_m * 1e6,error_L_m*1e6]  # Convert to microns
    }

    return results
# Example usage
if __name__ == "__main__":
    base_folder = "../beamlines/awa/2025_02_20/FlatBeam_Magnet/"
    yag_string_slit = "yag6"
    yag_string_after_slit = "yag7"
    cathode_radius = 3.25e-3
    cathode_spot_size = cathode_radius/2
    error_cathode_radius = 0.25e-3
    error_cathode_spot_size=error_cathode_radius/2
    current_to_Bfield = 0.0254e-2
    all_results = []

    for folder in sorted(os.listdir(base_folder)):
        full_path = os.path.join(base_folder, folder)
        if os.path.isdir(full_path):
            result = magnetization_scan(full_path, yag_string_slit, yag_string_after_slit, cathode_spot_size,
                                        error_cathode_spot_size,current_to_Bfield)
            if result:
                all_results.append(result)

    # Final Plot
    if all_results:
        currents = [r["Current (A)"][0] for r in all_results]
        current_error = [r["Current (A)"][1] for r in all_results]
        L_values = [r["Magnetization L (um)"][0] for r in all_results]
        L_errors = [r["Magnetization L (um)"][1] for r in all_results]
        L_m_values = [r["Magnetization cathode L (um)"][0] for r in all_results]
        L_m_errors = [r["Magnetization cathode L (um)"][1] for r in all_results]

        plt.errorbar(currents, L_values, yerr=L_errors,xerr=current_error,fmt='o', label="Beam Magnetization",alpha=0.5)
        plt.errorbar(currents, L_m_values, yerr=L_m_errors, xerr=current_error, fmt='o', label="Cathode Magnetization",alpha=0.5)
        I_scatter = np.linspace(10, 600, 50)
        B_scatter = 0.0254 * I_scatter / 100
        L_scatter = np.multiply(B_scatter, 1e6 * e * (cathode_spot_size ** 2) / (2 * m_e * c))
        plt.plot(I_scatter,L_scatter,label='Expected relation')
        plt.xlabel("Current (A)")
        plt.ylabel(r"Magnetization $\mathcal{L}$ (µm)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(base_folder, "magnetization_summary.png"))
        plt.show()

        # Save the results to a JSON file
        output_json_path = os.path.join(base_folder, "magnetization_results.json")
        with open(output_json_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        print(f"Results saved to {output_json_path}")