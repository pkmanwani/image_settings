import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import inv
from sympy import MatrixSymbol, Matrix
import matplotlib.pyplot as plt

def Drift(l):
    return Matrix([[1, l, 0, 0,  0, 0],
                    [0, 1,  0, 0,  0, 0],
                    [0, 0,  1, l, 0, 0],
                    [0, 0,  0, 1,  0, 0],
                    [0, 0,  0, 0,  1, 0],
                    [0, 0,  0, 0,  0, 1]])

# Constants
electron_mass = 9.109e-31  # kg
electron_charge = 1.602e-19  # C
energy_MeV = 63  # Example beam energy in MeV
energy_J = energy_MeV * 1e6 * electron_charge
gamma = energy_J / (electron_mass * 9e16)
beta = np.sqrt(1 - 1 / gamma ** 2)

# Screen positions in meters
SCREEN_POSITIONS = {
    "Yag4": 11372.85e-3,
    "Yag6": 14922.5e-3,
    "Yag7": 17792.7e-3,  # SlitYAG is YAG7
    "Yag8": 18821.4e-3,
    "Yag5": 16484.6e-3
}

# Drift space
skew = 12255.5e-3
dd0 = 0.01
dd1 = 0.01  # 3.555
dd2 = SCREEN_POSITIONS["Yag5"] - SCREEN_POSITIONS["Yag6"]  # 2.86 # For EYG8
dd3 = SCREEN_POSITIONS["Yag7"] - SCREEN_POSITIONS["Yag5"]  # 2.86 # For EYG8
dd4 = SCREEN_POSITIONS["Yag8"] - SCREEN_POSITIONS["Yag7"]  # 2.86 # For EYG8

# Matrix D1, D2, D3, D4 for the drift between screens
D_0 = dd0
D_1 = dd1
D_2 = D_1 + dd2
D_3 = D_2 + dd3
D_4 = D_3 + dd4
print(D_0, D_1, D_2, D_3, D_4)

D1 = Drift(dd0+dd1)
D2 = Drift(dd0 + dd1)
D3 = Drift(dd0 + dd1 + dd2+dd3)
D4 = Drift(dd0 + dd1 + dd2 + dd3+dd4)

# Matrix M (for beam size calculation)
M = Matrix([
    [D1[0, 0]**2, 2*D1[0, 0]*D1[0, 1], D1[0, 1]**2],
    [D2[0, 0]**2, 2*D2[0, 0]*D2[0, 1], D2[0, 1]**2],
    [D3[0, 0]**2, 2*D3[0, 0]*D3[0, 1], D3[0, 1]**2],
    [D4[0, 0]**2, 2*D4[0, 0]*D4[0, 1], D4[0, 1]**2]
])

# Beam size calculation
MM = (M.transpose() * M).inv() * M.transpose()
#MM = M.inv()

# Define the parabolic function
def spot_size_evolution(x, sigma_star, x0, beta_star):
    return sigma_star * (np.sqrt(1 + ((x - x0) / beta_star) ** 2))

# Load beam size data from JSON file
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    screen_positions = []
    sigx = []
    sigy = []

    for entry in data:
        filename = entry["filename"]
        screen_name = filename.split('ThreeScreen')[0].replace("SlitYag", "Yag7").strip('_')
        if screen_name in SCREEN_POSITIONS:
            screen_positions.append(SCREEN_POSITIONS[screen_name])
            sigx.append(entry["Rx_final"] * entry["res"])
            sigy.append(entry["Ry_final"] * entry["res"])

    return np.array(screen_positions), np.array(sigx), np.array(sigy)

# Process the JSON data (example path)
screen_positions, sigx, sigy = load_data("parameters.json")

# RMS beam size calculation
sig1x = (sigx[0] * 1e-3) ** 2
sig2x = (sigx[1] * 1e-3) ** 2
sig3x = (sigx[2] * 1e-3) ** 2
sig4x = (sigx[3] * 1e-3) ** 2

sig1y = (sigy[0] * 1e-3) ** 2
sig2y = (sigy[1] * 1e-3) ** 2
sig3y = (sigy[2] * 1e-3) ** 2
sig4y = (sigy[3] * 1e-3) ** 2

sig0x_11 = MM[0, 0] * sig1x + MM[0, 1] * sig2x + MM[0, 2] * sig3x + MM[0, 3] * sig4x
sig0x_12 = MM[1, 0] * sig1x + MM[1, 1] * sig2x + MM[1, 2] * sig3x + MM[1, 3] * sig4x
sig0x_22 = MM[2, 0] * sig1x + MM[2, 1] * sig2x + MM[2, 2] * sig3x + MM[2, 3] * sig4x
sig0y_11 = MM[0, 0] * sig1y + MM[0, 1] * sig2y + MM[0, 2] * sig3y + MM[0, 3] * sig4y
sig0y_12 = MM[1, 0] * sig1y + MM[1, 1] * sig2y + MM[1, 2] * sig3y + MM[1, 3] * sig4x
sig0y_22 = MM[2, 0] * sig1y + MM[2, 1] * sig2y + MM[2, 2] * sig3y + MM[2, 3] * sig4x

sig0x_11 = np.array(sig0x_11, dtype='float')
sig0x_12 = np.array(sig0x_12, dtype='float')
sig0x_22 = np.array(sig0x_22, dtype='float')
sig0y_11 = np.array(sig0y_11, dtype='float')
sig0y_12 = np.array(sig0y_12, dtype='float')
sig0y_22 = np.array(sig0y_22, dtype='float')

# RMS beam sizes
sig0x_recon = np.sqrt(sig0x_11)
sig0y_recon = np.sqrt(sig0y_11)

# Emittance
emitx_recon = np.sqrt(sig0x_11 * sig0x_22 - (sig0x_12 ** 2))
emity_recon = np.sqrt(sig0y_11 * sig0y_22 - (sig0y_12 ** 2))

# Normalized emittance
pCentral = 1e-3  # Example momentum (modify as needed)
enx_recon = emitx_recon * pCentral
eny_recon = emity_recon * pCentral

# Twiss Beta function
betax_recon = sig0x_recon ** 2 / emitx_recon
betay_recon = sig0y_recon ** 2 / emity_recon

# Twiss Alpha function
alphax_recon = -sig0x_12 / emitx_recon
alphay_recon = -sig0y_12 / emity_recon

# Display results
print('RMSX is ' + repr(sig0x_recon * 1e3) + ' mm.')
print('RMSY is ' + repr(sig0y_recon * 1e3) + ' mm.')
print('========================================')
print('enx is ' + repr(enx_recon * 1e6) + ' mm mrad.')
print('eny is ' + repr(eny_recon * 1e6) + ' mm mrad.')
print('========================================')
print('betax at the initial position is  ' + repr(betax_recon) + ' m.')
print('betay at the initial position is  ' + repr(betay_recon) + ' m.')
print('alphax at the initial position is ' + repr(alphax_recon) + ' .')
print('alphay at the initial position is ' + repr(alphay_recon) + ' .')

# Parabolic curve fitting for beam size evolution
sigx_YAG = [sigx[0], sigx[1], sigx[2],sigx[3]]
sigy_YAG = [sigy[0], sigy[1], sigy[2],sigy[3]]
sYAG = [D_1, D_2, D_3, D_4]

parametersx, _ = curve_fit(spot_size_evolution, sYAG, sigx_YAG)
fit_sigmax = parametersx[0]
fit_x0 = parametersx[1]
fit_beta_star_x = parametersx[2]

parametersy, _ = curve_fit(spot_size_evolution, sYAG, sigy_YAG)
fit_sigmay = parametersy[0]
fit_y0 = parametersy[1]
fit_beta_star_y = parametersy[2]

# Plotting
sfit = np.linspace(0.0, D_4, 1000)
envx = fit_sigmax * (np.sqrt(1 + ((sfit - fit_x0) / fit_beta_star_x) ** 2))
envy = fit_sigmay * (np.sqrt(1 + ((sfit - fit_y0) / fit_beta_star_y) ** 2))

plt.figure()
plt.scatter(sYAG, sigx_YAG, label="X Beam Size")
plt.scatter(sYAG, sigy_YAG, label="Y Beam Size")
plt.plot(sfit, envx, linestyle='dotted', label="X Fit")
plt.plot(sfit, envy, linestyle='dotted', label="Y Fit")
plt.legend()
plt.grid(True)
plt.savefig('fourscan.png')
