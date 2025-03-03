import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Physical constants
e   = 1.602e-19  # Electron charge (C)
m   = 9.11e-31   # Electron mass (kg)
c   = 299792458  # Speed of Light (m/s)
m0  = 511000     # Electron rest mass (eV/c)
mc2 = m0
EMASS = mc2

# Beam parameters
Ebeam1 = 63E6  # Initial energy in GeV
gamma1 = (Ebeam1 + EMASS) / EMASS
beta1  = np.sqrt(1 - (1 / gamma1**2))
P01    = gamma1 * beta1 * mc2
pCentral = P01 / EMASS

# Load parameters from JSON file
json_file_path = 'parameters.json'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Extract necessary values
quad_values = []
rx_mm = []
ry_mm = []

for entry in data:
    filename = entry["filename"]
    quad = float(filename.split("quad_")[1].split("_")[0])  # Extract quad value
    res = entry["res"]
    rx_mm.append(entry["Rx_final"] * res)  # Convert to mm
    ry_mm.append(entry["Ry_final"] * res)  # Convert to mm
    quad_values.append(quad)

# Convert to numpy arrays
quad_values = np.array(quad_values)
rx_mm = np.array(rx_mm)
ry_mm = np.array(ry_mm)

# Sort by quad values
sort_idx = np.argsort(quad_values)
quad_values = quad_values[sort_idx]
rx_mm = rx_mm[sort_idx]
ry_mm = ry_mm[sort_idx]

# Apply moving average smoothing
def moving_average(data, window_size=4):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

rx_mm_smooth = moving_average(rx_mm)
ry_mm_smooth = moving_average(ry_mm)
quad_values_smooth = moving_average(quad_values)

# Convert quadrupole strength to m^-2
quad_strength = np.array(quad_values_smooth) / 0.893
kval_x = -quad_strength * (1 / ((beta1 * Ebeam1 * 1e-9) / 0.299))
kval_y = -quad_strength * (1 / ((beta1 * Ebeam1 * 1e-9) / 0.299))
kval_y = kval_y[2:-1]
ry_mm_smooth = ry_mm_smooth[2:-1]
# Convert beam sizes to meters and square values
rx_sqr = (rx_mm_smooth * 1e-3) ** 2
ry_sqr = (ry_mm_smooth * 1e-3) ** 2

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

quad_length = 0.168
Q6 = 14109.7e-3
YAG7 = 17792.7e-3
YAG6 = 14922.5e-3
drift_length = YAG7 - Q6 - quad_length/2

sq11_x = params_rx[0] / ((drift_length**2) * (quad_length**2))
sq12_x = (params_rx[1] - (2 * drift_length * quad_length * sq11_x)) / (2 * (drift_length**2) * quad_length)
sq22_x = (params_rx[2] - sq11_x - (2 * drift_length * sq12_x)) / (drift_length**2)
ex = np.sqrt((sq11_x * sq22_x) - (sq12_x**2))
enx = pCentral * ex

sq11_y = params_ry[0] / ((drift_length**2) * (quad_length**2))
sq12_y = (params_ry[1] - (2 * drift_length * quad_length * sq11_y)) / (2 * (drift_length**2) * quad_length)
sq22_y = (params_ry[2] - sq11_y - (2 * drift_length * sq12_y)) / (drift_length**2)
ey = np.sqrt((sq11_y * sq22_y) - (sq12_y**2))
eny = pCentral * ey

# Twiss parameters
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

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(kval_x, rx_sqr * 1e6, label=r'$\sigma_x^2$ ($mm^2$)', color='blue')
plt.scatter(kval_y, ry_sqr * 1e6, label=r'$\sigma_y^2$ ($mm^2$)', color='red')
plt.plot(kval_fit_x, rx_fit * 1e6, '--',  color='blue')
plt.plot(kval_fit_y, ry_fit * 1e6, '--',  color='red')
plt.xlabel('Quadrupole Strength : k ($m^-2$)')
plt.ylabel(r'$\sigma$ ($mm^2$)')
plt.legend()
plt.grid()
#plt.title('Quad Scan Analysis')
plt.savefig('quad_scan_plot.png')
plt.show()