import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 15
# Function to load JSON data from a file
def load_json(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {filename}.")
    return None

# Directory containing JSON files
json_directory = 'Slitscan/Magnetization2/Gradient_results'

# Ensure the directory exists
if not os.path.exists(json_directory):
    raise FileNotFoundError(f"Directory {json_directory} does not exist.")

# Group files by their common prefixes
file_groups = defaultdict(list)

for file in os.listdir(json_directory):
    if file.endswith('.json'):
        prefix = file.split('_')[0]  # Get the prefix (e.g., '350', '50', etc.)
        file_groups[prefix].append(file)


# Sort the prefixes to ensure '0' is processed first
sorted_prefixes = sorted(file_groups.keys(), key=lambda x: int(x))
print(sorted_prefixes)
# Constants
e = 1.602e-19  # Elementary charge in Coulombs
m_e = 9.1093837e-31  # Electron mass in kg
c = 299792458  # Speed of light in m/s
R_c = 3.25e-3  # Spot size of the cathode in meters
D = 2.87  # Distance parameter in meters
pz = 63e6
error_R_c =0.25e-3
# Initialize lists to store L_theta and magnetic field values
L_theta_all = []
L_theta_error_all = []
magnetic_field_all = []
std_magnetic_field_all = []
# Initialize theta_0
theta_0 = 0

# Process each group of files
for prefix in sorted_prefixes:
    files = file_groups[prefix]
    R1 = None
    R2 = None
    error_R1=None
    error_R2 = None
    orientation = None
    error_orientation=None
    magnetic_field = None
    std_magnetic_field = None
    for file in files:
        file_path = os.path.join(json_directory, file)
        data = load_json(file_path)
        if data is not None:
            if 'Yag6' in file:
                print("yes6")
                R1 = data.get('Circular Beam Radius', R1)
                error_R1 = data.get('Circular beam radius error', error_R1)
            elif 'SlitYag_withoutSlit' in file:
                print("yes7")
                R2 = data.get('Circular Beam Radius', R2)
                error_R2 = data.get('Circular beam radius error', error_R2)
            elif 'SlitYag_withSlit' in file:
                print("yes7_slit")
                orientation = data.get('Average Orientation', orientation)
                error_orientation = data.get('Std Orientation', error_orientation)
            magnetic_field = data.get('Magnetic field', magnetic_field)
            std_magnetic_field=data.get('Std Magnetic field',std_magnetic_field)

    # Check if all necessary data was loaded for the group
    if R1 is None or R2 is None or orientation is None or magnetic_field is None:
        print(f"One or more required values were not found for prefix {prefix}. Skipping this group.")
        continue

    # Print values for verification
    print(f'Prefix: {prefix}')
    print(f'R1: {R1} mm')
    print(f'R2: {R2} mm')
    print(f'Orientation: {orientation} degrees')
    print(f'Magnetic field: {magnetic_field} T')
    print(f'Std Magnetic field: {std_magnetic_field} T')

    # Calculate theta
    theta = orientation
    #if prefix == '0':
    #    theta_0 = theta
    theta_0 = 0
    print(f'theta_0={theta_0}')
    # Adjust theta based on theta_0
    adjusted_theta = theta - theta_0

    # Example calculation using R1 and R2
    L_theta = (4/9) * R1 * R2 * (1e-6) * np.sin(np.radians(adjusted_theta)) * pz * e / (D * c)
    angle_error = np.cos(np.radians(adjusted_theta))*np.radians(error_orientation)
    L_theta_error = L_theta*(np.sqrt((error_R1/R1)**2 + (error_R2/R2)**2 + (angle_error/np.sin(np.radians(adjusted_theta)))**2))
    L_theta_all.append(L_theta)
    L_theta_error_all.append(L_theta_error)
    magnetic_field_all.append(magnetic_field)
    std_magnetic_field_all.append(std_magnetic_field)
# Calculate magnetization (L_m)
I = np.linspace(10, 600, 50)  # Amps
B = 0.0254 * I / 100  # Magnetic field in Tesla
L = e * B * (R_c ** 2) / 4
error_L = 2*L*error_R_c/R_c
L_scatter = np.multiply(magnetic_field_all,e*(R_c ** 2)/4)
error_L_scatter=[]
for i,points in enumerate(L_scatter):
    error_L_scatter.append(np.multiply(points, np.sqrt(
        (2 * error_R_c / R_c) ** 2 + (std_magnetic_field_all[i] / magnetic_field_all[i]) ** 2)))

L_m = L / (2*m_e * c)
error_L_m = error_L/(2*m_e * c)
L_theta_all_norm = np.multiply(L_theta_all,1/(2*m_e* c))
L_theta_error_all_norm = np.multiply(L_theta_error_all,1/(2*m_e* c))
# Plotting
plt.plot(B, L_m*1e6)
plt.scatter(magnetic_field_all, L_theta_all_norm*1e6,color = "black",s=10)  # Plot the point for the magnetic field
plt.errorbar(magnetic_field_all, L_theta_all_norm*1e6, yerr=L_theta_error_all_norm*1e6,xerr=std_magnetic_field_all,capsize=5, ecolor = "black",linestyle='none')  # Plot the point for the magnetic field
plt.xlabel('Magnetic Field (T)')
plt.ylabel(r'Magnetization $\mathcal{L}$ ($\mu$m)')
plt.savefig('Magnetization.png')
plt.close()

plt.plot(B, L/e)
plt.scatter(magnetic_field_all, np.multiply(L_theta_all,1/e), color = "black",s=10)  # Plot the point for the magnetic field
plt.errorbar(magnetic_field_all, np.multiply(L_theta_all,1/e), yerr=L_theta_error_all,xerr=None,capsize=5, ecolor = "black",linestyle='none')  # Plot the point for the magnetic field
plt.xlabel('Magnetic Field (T)')
plt.ylabel(r'Mean angular momentum <L> (J s)')
plt.savefig('angular_momentum.png')
plt.close()

plt.plot(np.multiply(L,1/e), np.multiply(L,1/e))
plt.scatter(np.multiply(L_scatter,1/e), np.multiply(L_theta_all,1/e), color = "black",s=10)  # Plot the point for the magnetic field
plt.errorbar(np.multiply(L_scatter,1/e), np.multiply(L_theta_all,1/e), yerr=np.multiply(L_theta_error_all,1/e),xerr=np.multiply(error_L_scatter,1/e),capsize=5, ecolor = "black",linestyle='none')  # Plot the point for the magnetic field
plt.xlabel(r' <L> ($p_z$ e $<r_c^2> B_0$) (eV s) ')
plt.ylabel(r'<L> ($p_z$ $<r_1>$ $<r_2>$ $\sin(\theta)$/D) (eV s)')
plt.savefig('angular_momentum_both.png')
plt.close()