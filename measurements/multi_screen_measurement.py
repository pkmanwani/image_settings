import sys
sys.path.append('../')
import os
import json
from scipy.optimize import curve_fit
from scipy.linalg import inv
from scipy.constants import c, e, m_e
from sympy import MatrixSymbol, Matrix
import numpy as np
import matplotlib.pyplot as plt
from functions.save_parameters_json_parallel import save_parameters_json_parallel
def Drift(l):
    return Matrix([[1, l, 0, 0,  0, 0],
                    [0, 1,  0, 0,  0, 0],
                    [0, 0,  1, l, 0, 0],
                    [0, 0,  0, 1,  0, 0],
                    [0, 0,  0, 0,  1, 0],
                    [0, 0,  0, 0,  0, 1]])
# Define the parabolic function
def spot_size_evolution(x, sigma_star, x0, beta_star):
    return sigma_star * np.sqrt((1 + ((x - x0 - np.min(x)) / beta_star) ** 2))
def multi_screen_measurement(folder_path,initial_position=0):
    """
    Process the given folder by looking for parameters.json and elements.json,
    extracting and printing relevant information.

    :param folder_path: Path to the folder containing parameters.json.
    :param possible_names: List of possible names to match Yag elements.
    """
    # Paths to JSON files
    parameters_path = os.path.join(folder_path, "parameters.json")
    elements_path = os.path.join(os.path.dirname(os.path.dirname(folder_path)), "elements.json")

    # Check if parameters.json exists
    if not os.path.exists(parameters_path):
        print(f"Error: {parameters_path} not found.")
        return
    # Check if elements.json exists
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
    # Load parameters.json
    with open(parameters_path, "r") as f:
        data = json.load(f)
    names = []
    beam_sizes_x = []
    beam_sizes_y = []
    errors_x = []
    errors_y = []
    #Set position where you want to check the beam
    # Process each entry
    #pz = dataload
    positions = []
    for entry in data:
        filename = entry.get("filename", "")
        rx_values = entry.get("Rx_final", [])
        ry_values = entry.get("Ry_final", [])

        if not filename or not rx_values:
            print("Skipping entry with missing filename or Rx values.")
            continue

        print(f"\n{filename}: Rx = {rx_values[0]}, Ry = {ry_values[0]}")

        #Extract potential Yag name from filename
        filename_parts = filename.split("_")
        potential_yag_name = filename_parts[0] if filename_parts else ""

        # Find matching Yag in elements.json
        matched_yag = None
        for yag in elements_data.get("yags", []):
            if potential_yag_name.lower() in yag["name"]:
                matched_yag = yag
                break

        if matched_yag:
            res = matched_yag['res']
            size_x_mm = rx_values[0] * res
            size_y_mm = ry_values[0] * res
            errors_x_mm = rx_values[1] * res
            errors_y_mm = ry_values[1] * res
            beam_sizes_x.append(size_x_mm*1e-3)
            beam_sizes_y.append(size_y_mm*1e-3)
            errors_x.append(errors_x_mm*1e-3)
            errors_y.append(errors_y_mm*1e-3)
            positions.append(matched_yag['position'])
            names.append(matched_yag['name'])
            print(f"\n{filename}: Rx = {size_x_mm:.4f} mm, Ry = {size_y_mm:.4f} mm")
            print(f"Found Yag: {matched_yag['name']}")
            print(f"Position: {matched_yag['position']} m")
            print(f"Resolution: {matched_yag['res']} mm/pixel")
        else:
            print(f"No matching Yag found for {potential_yag_name} in {elements_path}.")

    # Compute drift distances directly from positions
    #print(positions)
    #print(beam_sizes_x)
    #print(beam_sizes_y)
    print(initial_position)
    drift_distances = [
        positions[i]-initial_position for i in range(0, len(positions))
    ]
    print(drift_distances)
    #print(drift_distances)
    # Compute drift matrices dynamically
    drift_matrices = [Drift(drift_distances[i]) for i in range(0,len(drift_distances))]
    #print(drift_matrices)
    #print(np.shape(drift_matrices))
    # Matrix M for beam size calculation
    M = Matrix([
        [D[0, 0] ** 2, 2 * D[0, 1] * D[1, 1], D[0, 1] ** 2] for D in drift_matrices
    ])
    #print(M)
    # Beam size calculation using beam_sizes_x and beam_sizes_y
    beam_size_matrix_x = Matrix([[beam_sizes_x[i]**2] for i in range(len(beam_sizes_x))])
    beam_size_matrix_y = Matrix([[beam_sizes_y[i]**2] for i in range(len(beam_sizes_y))])
    #print(M.transpose() * M)
    #print((M.transpose() * M).n(30).inv().n(16))
    # Beam size calculation
    #Calculate pseudo inverse
    MM = np.linalg.pinv(np.array(M).astype(np.float64))
    # MM = M.inv()
    #print(np.shape(MM))
    #print(np.shape(beam_size_matrix_x))
    # Solve for Twiss parameters
    #print(n)
    twiss_params_x = Matrix(MM) * beam_size_matrix_x
    twiss_params_y = Matrix(MM) * beam_size_matrix_y
    #print('T')
    #print(np.shape(twiss_params_x))
    #print(twiss_params_x)
    #print(np.shape(twiss_params_y))
    #print(twiss_params_y)
    twiss_params_x = np.array(twiss_params_x).astype(np.float64)
    twiss_params_y = np.array(twiss_params_y).astype(np.float64)
    sig0x_11, sig0x_12, sig0x_22 = twiss_params_x.flatten()
    sig0y_11, sig0y_12, sig0y_22 = twiss_params_y.flatten()
    sig0x_recon = np.sqrt(sig0x_11)
    sig0y_recon = np.sqrt(sig0y_11)
    emitx_recon = np.sqrt(sig0x_11 * sig0x_22 - sig0x_12 ** 2)
    emity_recon = np.sqrt(sig0y_11 * sig0y_22 - sig0y_12 ** 2)
    pCentral = beta_gamma
    enx_recon = emitx_recon * pCentral
    eny_recon = emity_recon * pCentral
    betax_recon = sig0x_11 / emitx_recon
    betay_recon = sig0y_11 / emity_recon
    alphax_recon = -sig0x_12 / emitx_recon
    alphay_recon = -sig0y_12 / emity_recon

    print(f'RMSX: {sig0x_recon * 1e3:.3f} mm, RMSY: {sig0y_recon * 1e3:.3f} mm')
    print(f'enx (pseudoinverse): {enx_recon * 1e6:.3f} mm mrad, eny (pseudoinverse): {eny_recon * 1e6:.3f} mm mrad')
    print(f'betax: {betax_recon:.3f} m, betay: {betay_recon:.3f} m')
    print(f'alphax: {alphax_recon:.3f}, alphay: {alphay_recon:.3f}')

    positions = np.array(positions)
    #print(drift_distances)
    beam_sizes_x = np.array(beam_sizes_x)
    beam_sizes_y = np.array(beam_sizes_y)
    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)

    parametersx, _ = curve_fit(spot_size_evolution, positions, beam_sizes_x)
    fit_sigmax, fit_x0, fit_beta_star_x = parametersx
    parametersy, _ = curve_fit(spot_size_evolution, positions, beam_sizes_y)
    fit_sigmay, fit_y0, fit_beta_star_y = parametersy

    sfit = np.linspace(np.min(positions), np.max(positions), 100)
    envx = fit_sigmax * np.sqrt(1 + ((sfit - fit_x0 - np.min(positions)) / fit_beta_star_x) ** 2)
    envy = fit_sigmay * np.sqrt(1 + ((sfit - fit_y0 - np.min(positions)) / fit_beta_star_y) ** 2)
    emit_fit = fit_sigmax**2/fit_beta_star_x,fit_sigmay**2/fit_beta_star_y
    emit_n_fit = np.multiply(emit_fit,beta_gamma)
    print(f'enx (fit): {emit_n_fit[0] * 1e6:.3f} mm mrad, eny (fit): {emit_n_fit[1] * 1e6:.3f} mm mrad')
    plt.figure()
    plt.scatter(positions, beam_sizes_x, label="X Beam Size",color='blue')
    plt.scatter(positions, beam_sizes_y, label="Y Beam Size",color='orange')
    # Plot error bars only for nonzero errors
    nonzero_x = errors_x > 0
    plt.errorbar(positions[nonzero_x], beam_sizes_x[nonzero_x], yerr=errors_x[nonzero_x], fmt='o', color='blue', capsize=3)

    nonzero_y = errors_y > 0
    plt.errorbar(positions[nonzero_y], beam_sizes_y[nonzero_y], yerr=errors_y[nonzero_y], fmt='o', color='orange', capsize=3)
    plt.plot(sfit, envx, linestyle='dotted', label="X Fit")
    plt.plot(sfit, envy, linestyle='dotted', label="Y Fit")
    plt.axvline(initial_position, linestyle='dotted',color='black', label = 'At position')
    plt.ylabel(r'$\sigma (mm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, 'multiscreen.png'))
    # Prepare results dictionary
    results = {
        "RMSX_mm": sig0x_recon * 1e3,
        "RMSY_mm": sig0y_recon * 1e3,
        "enx_pseudoinverse_mm_mrad": enx_recon * 1e6,
        "eny_pseudoinverse_mm_mrad": eny_recon * 1e6,
        "betax_m": betax_recon,
        "betay_m": betay_recon,
        "alphax": alphax_recon,
        "alphay": alphay_recon,
        "enx_fit_mm_mrad": emit_n_fit[0] * 1e6,
        "eny_fit_mm_mrad": emit_n_fit[1] * 1e6,
        "positions": positions.tolist(),
        "beam_sizes_x": beam_sizes_x.tolist(),
        "beam_sizes_y": beam_sizes_y.tolist(),
        "errors_x": errors_x.tolist(),
        "errors_y": errors_y.tolist()
    }

    # Save the results to a JSON file in the same folder as the plot
    output_json_path = os.path.join(folder_path, "multiscreen_results.json")
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    folder_path = "../beamlines/awa/2025_02_20/ThreeScreen2/"
    initial_position = 17
    #save_parameters_json_parallel(folder_path,roi=True,calc_jitter=True,sigma_size=3)
    multi_screen_measurement(folder_path, initial_position)