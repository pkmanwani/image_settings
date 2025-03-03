import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy.optimize import curve_fit
import sympy as sym
from sympy import MatrixSymbol, Matrix
from sympy import *
import math
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import matplotlib
# ==========================================================================
# Basic parameters
e   = 1.602e-19   # Electron charge, Coulomb
m   = 9.11e-31    # Electron mass
me  = 0.511e+6    # Electron rest mass (MeV/c)
c   = 299792458   # Speed of Light [m/s]
e0  = 8.85e-12    # Electric permittivity of the free space
mu0 = 4*np.pi*1E-7# Permeability of the free space
mp = 938.272e+6   # proton rest mass (eV/c)
m0 = 511000
mc2 = m0
EMASS = mc2
emass = m0
clite = c

# Energy and gamma
Ebeam1   = 45.3E6 #; %//initial energy in GeV
gamma1   = (Ebeam1 + EMASS) / EMASS
beta1    = np.sqrt(1 - (1 / gamma1**2))
P01      = gamma1 * beta1 * mc2
pCentral = P01 / EMASS

# Load data from JSON file
json_file_path = os.path.join('2024_07_18', 'data_quad_scan', 'all_statistics.json')
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

sigx = data['Sx']['mean']
stdx = data['Sx']['std']
sigy = data['Sy']['mean']
stdy = data['Sy']['std']
current = data['quad_current']

# Convert current to float
current = list(map(float, current))
# Count to T/m
count_tm = np.array(current) /0.893

# T/m to m^-2
kvalh = -np.array(count_tm) * (1 / ((beta1 * Ebeam1 * 1e-9) / 0.299))
kvalv = np.array(count_tm) * (1 / ((beta1 * Ebeam1 * 1e-9) / 0.299))

# =====================================================================
quad_length = 0.12
Q6 = 14109.7e-3
YAG7 = 17792.7e-3
YAG6 = 14922.5e-3
drift_length = YAG6 - Q6

# Convert to meters and square values
sigx_sqr = np.array(sigx) * 1e-3
sigx_sqr = sigx_sqr**2

sigy_sqr = np.array(sigy) * 1e-3
sigy_sqr = sigy_sqr**2

stdx_sqr = np.array(stdx) * 1e-3
stdx_sqr = stdx_sqr**2

stdy_sqr = np.array(stdy) * 1e-3
stdy_sqr = stdy_sqr**2

starth = 0
endh = len(sigx_sqr)

startv = 0
endv = len(sigy_sqr)

sigx_sqr = sigx_sqr[starth:endh]
sigy_sqr = sigy_sqr[startv:endv]
stdx_sqr = stdx_sqr[starth:endh]
stdy_sqr = stdy_sqr[startv:endv]
kvalh = kvalh[starth:endh]
kvalv = kvalv[startv:endv]

# Optimization
def lq(x, a, b, c):
    return (a * (x**2) + b * x + c)

parametersx, covariancex = curve_fit(lq, kvalh, sigx_sqr)
ax = parametersx[0]
bx = parametersx[1]
cx = parametersx[2]

parametersy, covariancey = curve_fit(lq, kvalv, sigy_sqr)
ay = parametersy[0]
by = parametersy[1]
cy = parametersy[2]

kval_fith = np.linspace(min(kvalh), max(kvalh), 500)
kval_fitv = np.linspace(min(kvalv), max(kvalv), 500)
fitx = (ax * (kval_fith**2) + bx * kval_fith + cx)
fity = (ay * (kval_fitv**2) + by * kval_fitv + cy)

# Emittance calculation
sq11_x = ax / ((drift_length**2) * (quad_length**2))
sq12_x = (bx - (2 * drift_length * quad_length * sq11_x)) / (2 * (drift_length**2) * quad_length)
sq21_x = sq12_x
sq22_x = (cx - sq11_x - (2 * drift_length * sq12_x)) / (drift_length**2)

# Calculation of the geometrical emittance
ex = np.sqrt((sq11_x * sq22_x) - (sq12_x**2))

# Calculation of the normalized emittance
enx = (pCentral * ex)

sq11_y = ay / ((drift_length**2) * (quad_length**2))
sq12_y = (by - (2 * drift_length * quad_length * sq11_y)) / (2 * (drift_length**2) * quad_length)
sq12_y = (by - (2 * drift_length * quad_length * sq11_y)) / (2 * (drift_length**2) * quad_length)
sq21_y = sq12_y
sq22_y = (cy - sq11_y - (2 * drift_length * sq12_y)) / (drift_length**2)

# Calculation of the geometrical emittance
ey = np.sqrt((sq11_y * sq22_y) - (sq12_y**2))

# Calculation of the normalized emittance
eny = (pCentral * ey)

# Twiss parameters
alpha_x = -sq12_x / ex
beta_x = sq11_x / ex

alpha_y = -sq12_y / ex
beta_y = sq11_y / ex

print('========================================')
print('enx is ' + repr(enx * 1e6) + ' mm mrad.')
print('eny is ' + repr(eny * 1e6) + ' mm mrad.')
print('========================================')
print('betax at the initial position is ' + repr(beta_x) + ' m.')
print('betay at the initial position is ' + repr(beta_y) + ' m.')
print('alphax at the initial position is ' + repr(alpha_x) + ' .')
print('alphay at the initial position is ' + repr(alpha_y) + ' .')
print('========================================')

# Figure setting
fonts = 25
plt.style.use('classic')
rc = {"font.family": "Arial"}
plt.rcParams.update(rc)

params = {'legend.fontsize': 18,
          'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 25,
          'ytick.labelsize': 25,
          'grid.color': 'k',
          'grid.linestyle': ':',
          'grid.linewidth': 1.5
          }
matplotlib.rcParams.update(params)

ccode1 = [40, 122, 169]
ccode2 = [120, 130, 46]
ccode3 = [120, 175, 59]
ccode4 = [80, 80, 80]

ccode1 = tuple(np.array(ccode1) / 255)
ccode2 = tuple(np.array(ccode2) / 255)
ccode3 = tuple(np.array(ccode3) / 255)
ccode4 = tuple(np.array(ccode4) / 255)

# Linewidth
line_width = 2.3

# Plot for horizontal data
fig, ax1 = plt.subplots()
fig.patch.set_facecolor('white')
fig.set_size_inches(10, 9)

p1 = ax1.scatter(kvalh, np.array(sigx_sqr) * 1e6, s=40)
p2 = ax1.errorbar(kvalh, np.array(sigx_sqr) * 1e6, (stdx_sqr) * 1e6, linewidth=0.5, linestyle='--', label=r'$\sigma_{x}^{2},~\mathrm{Measurement}$')
p3 = ax1.plot(kval_fith, np.array(fitx) * 1e6, '-', linewidth=line_width, label=r'$\sigma_{x}^{2},~\mathrm{Curve~fitting}$')

color = ccode1
ax1.set_xlabel(r'$\mathrm{k~(m^{-2})}$')
ax1.set_ylabel(r'$\sigma_{x}^{2}~\mathrm{(mm^{2})}$', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid()
plt.legend()
fig.tight_layout()
plt.savefig('sigma_x.png')
plt.close()
# Plot for vertical data
fig, ax2 = plt.subplots()
fig.patch.set_facecolor('white')
fig.set_size_inches(10, 9)

p1 = ax2.scatter(kvalv, np.array(sigy_sqr) * 1e6, s=40)
p2 = ax2.errorbar(kvalv, np.array(sigy_sqr) * 1e6, (stdy_sqr) * 1e6, linewidth=0.5, linestyle='--', label=r'$\sigma_{y}^{2},~\mathrm{Measurement}$')
p3 = ax2.plot(kval_fitv, np.array(fity) * 1e6, '-', linewidth=line_width, label=r'$\sigma_{y}^{2},~\mathrm{Curve~fitting}$')

color = ccode1
ax2.set_xlabel(r'$\mathrm{k~(m^{-2})}$')
ax2.set_ylabel(r'$\sigma_{y}^{2}~\mathrm{(mm^{2})}$', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid()
plt.legend()
fig.tight_layout()
plt.savefig('sigma_y.png')