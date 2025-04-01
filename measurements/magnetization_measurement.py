import sys
sys.path.append('../')
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import c, e, m_e
from functions.save_parameters_json_parallel import save_parameters_json_parallel

def magnetization_scan(folder_path, yag_string1, yag_string2):
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
        print(f"Error: {parameters_path} not found.")
        return
    if not os.path.exists(elements_path):
        print(f"Error: {elements_path} not found.")
        return

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
    quad_position = next((item['position'] for item in elements_data['quads'] if quad_string in item['name']), None)
    yag_position = next((item['position'] for item in elements_data['yags'] if yag_string in item['name']), None)

    if quad_position is None or yag_position is None:
        print("Error: Quad or YAG not found in elements.json.")
        return

    # Compute drift length
    drift_length = yag_position - quad_position - quad_length / 2

    # Load parameters.json
    with open(parameters_path, "r") as f:
        data = json.load(f)

    quad_values = []
    rx_m = []  # Store beam sizes in meters directly
    ry_m = []
    errors_x_m = []
    errors_y_m = []
    # Extract beam size data
    for entry in data:
        filename = entry["filename"]
        quad = float(filename.split("quad_")[1].split("_")[0])  # Extract quad value
        res = entry["res"][0]
        # Append directly in meters
        rx_m.append(entry["Rx_final"][0] * res * 1e-3)
        ry_m.append(entry["Ry_final"][0] * res * 1e-3)
        errors_x_m.append(entry["Rx_final"][1] * res*1e-3)
        errors_y_m.append(entry["Ry_final"][1] * res*1e-3)
        quad_values.append(quad)

    # Convert to numpy arrays and sort by quad values
    quad_values = np.array(quad_values)
    rx_m = np.array(rx_m)
    ry_m = np.array(ry_m)
    errors_x_m = np.array(errors_x_m)
    errors_y_m = np.array(errors_y_m)

    sort_idx = np.argsort(quad_values)
    quad_values = quad_values[sort_idx]
    rx_m = rx_m[sort_idx]
    ry_m = ry_m[sort_idx]
    errors_x_m = errors_x_m[sort_idx]
    errors_y_m = errors_y_m[sort_idx]

    # Convert quadrupole strength to m^-2
    pCentral = beta_gamma * m_e * c

    quad_strength = np.array(quad_values) / t_m_to_ampere
    kval_x = -quad_strength * 0.299*(e/((pCentral*c * 1e-9)))
    kval_y = quad_strength * 0.299*(e/((pCentral*c * 1e-9)))

    # Use only valid data range
    startx = 1
    endx = -1
    starty = 3
    endy = -1

    kval_x = kval_x[startx:endx]
    kval_y = kval_y[starty:endy]
    rx_m = rx_m[startx:endx]
    ry_m = ry_m[starty:endy]
    errors_x_m = errors_x_m[startx:endx]
    errors_y_m = errors_y_m[starty:endy]

    # Convert beam sizes to meters and square values
    rx_sqr = rx_m ** 2
    ry_sqr = ry_m ** 2

    errors_x_sqr = 2 * rx_m * errors_x_m
    errors_y_sqr = 2 * ry_m * errors_y_m
    # Fit function
    def quad_fit(x, a, b, c):
        return a * x**2 + b * x + c

    # Curve fitting
    params_rx, _ = curve_fit(quad_fit, kval_x, rx_sqr)
    params_ry, _ = curve_fit(quad_fit, kval_y, ry_sqr)

    # Generate fit curves
    kval_fit_x = np.linspace(min(kval_x), max(kval_x), 100)
    rx_fit = quad_fit(kval_fit_x, *params_rx)

    kval_fit_y = np.linspace(min(kval_y), max(kval_y), 100)
    ry_fit = quad_fit(kval_fit_y, *params_ry)

    # Compute Twiss parameters
    sq11_x = params_rx[0] / ((drift_length**2) * (quad_length**2))
    sq12_x = (params_rx[1] - (2 * drift_length * quad_length * sq11_x)) / (2 * (drift_length**2) * quad_length)
    sq22_x = (params_rx[2] - sq11_x - (2 * drift_length * sq12_x)) / (drift_length**2)
    ex = np.sqrt((sq11_x * sq22_x) - (sq12_x**2))
    enx =  beta_gamma* ex

    sq11_y = params_ry[0] / ((drift_length**2) * (quad_length**2))
    sq12_y = (params_ry[1] - (2 * drift_length * quad_length * sq11_y)) / (2 * (drift_length**2) * quad_length)
    sq22_y = (params_ry[2] - sq11_y - (2 * drift_length * sq12_y)) / (drift_length**2)
    ey = np.sqrt((sq11_y * sq22_y) - (sq12_y**2))
    eny = beta_gamma * ey

    alpha_x = -sq12_x / ex
    beta_x = sq11_x / ex
    alpha_y = -sq12_y / ey
    beta_y = sq11_y / ey

    # Print results
    print('========================================')
    print(f'enx: {enx * 1e6:.3f} mm mrad')
    print(f'eny: {eny * 1e6:.3f} mm mrad')
    print('========================================')
    print(f'betax at initial position: {beta_x:.3f} m')
    print(f'betay at initial position: {beta_y:.3f} m')
    print(f'alphax at initial position: {alpha_x:.3f}')
    print(f'alphay at initial position: {alpha_y:.3f}')
    print('========================================')

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))


    axs[0].scatter(kval_x, rx_sqr*1e6 , label=r'$\sigma_x^2$ ($mm^2$)', color='blue')
    # Plot error bars only for nonzero errors
    nonzero_x = errors_x_sqr > 0
    axs[0].errorbar(kval_x[nonzero_x], rx_sqr[nonzero_x]*1e6, yerr=errors_x_sqr[nonzero_x]*1e6, fmt='o', color='blue',
                    capsize=3)

    axs[0].plot(kval_fit_x, rx_fit*1e6, '--', color='blue')
    axs[0].set_xlabel(r'Quadrupole Strength : k ($m^-2$)')
    axs[0].set_ylabel(r'$\sigma_x^2$ ($mm^2$)')
    axs[0].legend()
    axs[0].grid()

    axs[1].scatter(kval_y, ry_sqr*1e6, label=r'$\sigma_y^2$ ($mm^2$)', color='orange')

    nonzero_y = errors_y_sqr > 0
    axs[1].errorbar(kval_y[nonzero_y], ry_sqr[nonzero_y]*1e6, yerr=errors_y_sqr[nonzero_y]*1e6, fmt='o', color='orange',
                 capsize=3)

    axs[1].plot(kval_fit_y, ry_fit*1e6, '--', color='red')

    axs[1].set_xlabel(r'Quadrupole Strength : k ($m^-2$)')
    axs[1].set_ylabel(r'$\sigma_y^2$ ($mm^2$)')
    axs[1].legend()
    axs[1].grid()

    # Save plot in the same directory
    plot_path = os.path.join(folder_path, 'quad_scan_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")

    # Save results in a JSON file
    results = {
        "enx_mm_mrad": enx * 1e6,
        "eny_mm_mrad": eny * 1e6,
        "betax_m": beta_x,
        "betay_m": beta_y,
        "alphax": alpha_x,
        "alphay": alpha_y
    }

    results_path = os.path.join(folder_path, 'quad_scan_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved at: {results_path}")

# Example usage
if __name__ == "__main__":
    folder_path = "../beamlines/awa/2025_02_20/FlatBeam_Magnet/"
    yag_string = "yag7"
    effective_quad_length = 0.12 #effective_length
    t_m_to_ampere = 0.893 #0.893 A gives 1 T/m
    magnetization_scan(folder_path, quad_string, yag_string,effective_quad_length,t_m_to_ampere)

import numpy as np
# Constants
e = 1.602e-19  # Elementary charge in Coulombs
m_e = 9.1093837e-31  # Electron mass in kg
c = 299792458  # Speed of light in m/s
R_c = 3.25e-3  # Spot size of the cathode in meters
D = 2.87  # Distance parameter in meters
pz = 63e6
error_R_c =0.25e-3
res_6 =0.1181
res_7= 0.0433
R1 =np.sqrt(13.856274938771453**2 + 15.988803936271733**2)*res_6*1e-3
R2 =np.sqrt(72.93**2+ 82.446**2)*res_7*1e-3
current = 393
B = 0.0254 * current / 100
L_m = e * B * (R_c ** 2) / 4
L = pz*R1*R2*np.sin(16*np.pi/180)*e/((D*c)*2*m_e*c)
print(L_m*1e6/(2*m_e*c))
print(L)