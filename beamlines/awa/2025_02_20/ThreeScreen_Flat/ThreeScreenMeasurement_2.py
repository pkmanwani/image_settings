# =====================================================================
# =====================================================================
# LPS plotting
# =====================================================================
# =====================================================================
# LPS plotting
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
nullfmt = NullFormatter()         # no labels
import os
import matplotlib.pyplot as plt

import sympy as sym
from sympy import MatrixSymbol, Matrix
from sympy import *
import math

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Basic parameters
e   = 1.602e-19   # Electron charge, Coulomb
m   = 9.11e-31    # Electron mass
me  = 0.511e+6    # Electron rest mass (MeV/c)
c   = 299792458   # Speed of Light [m/s]
e0  = 8.85e-12    # Electric permittivity of the free space
mu0 = 4*np.pi*1E-7# Permeability of the free space
mp = 938.272e+6   # proton rest mass (eV/c)
m0 = 511000;
mc2 = m0;
EMASS = mc2;

# Energy and gamma
Ebeam1   = 45.3e6 #; %//initial energy in eV
gamma1   = (Ebeam1+EMASS)/EMASS
beta1    = np.sqrt(1-(1/gamma1**2)) 
P01      = gamma1*beta1*mc2
pCentral    = P01/EMASS;


# ==========================================================================
# ==========================================================================
# Transfer matrix
# Transverse deflecting cavity: This is horizontal TDC matrix

# Initial guess of the kappa
def Quad(kval, lq):
    """return 4 by 4 matrix of horizontal focusing normal quad"""
    return Matrix([[sym.cos(sym.sqrt(kval)*lq),      (1/sym.sqrt(kval))*sym.sin(sym.sqrt(kval)*lq), 0,     0,  0, 0],
                    [(-sym.sqrt(kval))*sym.sin(sym.sqrt(kval)*lq), sym.cos(sym.sqrt(kval)*lq),       0,     0,  0, 0],
                    [0,      0, sym.cosh(sym.sqrt(kval)*lq),      (1/sym.sqrt(kval))*sym.sinh(sym.sqrt(kval)*lq),  0, 0],
                    [0,      0, (-sym.sqrt(kval))*sym.sinh(sym.sqrt(kval)*lq), sym.cosh(sym.sqrt(kval)*lq),  0, 0],
                    [0,      0, 0,     0,  1, 0],
                    [0,      0, 0,     0,  0, 1]])

def Drift(l):
    return Matrix([[1, l, 0, 0,  0, 0],
                    [0, 1,  0, 0,  0, 0],
                    [0, 0,  1, l, 0, 0],
                    [0, 0,  0, 1,  0, 0],
                    [0, 0,  0, 0,  1, 1],
                    [0, 0,  0, 0,  0, 1]])

def Rotation(phi):
    return        Matrix([[np.cos(phi), 0 , np.sin(phi), 0, 0, 0],
        	             [0, np.cos(phi) , 0,  np.sin(phi), 0, 0],
        	             [-np.sin(phi), 0, np.cos(phi), 0, 0, 0],
        	             [0, -np.sin(phi), 0, np.cos(phi), 0, 0],
                         [0, 0,  0, 0,  1, 0],
                         [0, 0,  0, 0,  0, 1]])

# ==========================================================================
# Drift space
#D_z = 0.01
#D_0 = 0.01
#D_1 = D_0+3.50  # Drift from the starting position to middle of
#D_2 = D_1+2.86  # Drift from the starting position to EYG7
#D_3 = D_2+2.48  # Drift from the starting position to YAG186
#D_4 = D_3+0.435 # Drift from the starting position to PG68 (not used camera)
#D_5 = D_4+1.275 # Drift from the starting position to DMA

# ==========================================================================
# Drift space
YAG4 = 11372.85e-3
YAG6 = 14922.5e-3
YAG7 = 17792.7e-3
YAG8 = 18821.4e-3
YAG5 = 16484.6e-3
skew = 12255.5e-3
dd0 = 0.01
dd1 = YAG6-YAG4 # 3.555
dd2 = YAG8-YAG6 # 2.86 # For EYG8
ddskew=0.01+skew-YAG4-(0.15/2)
D_0 =dd0
D_1 = dd0+dd1
D_2 = D_1 + dd2
D1 = (Drift(dd0))
D2 = (Drift(dd0+dd1))
D3 = (Drift(dd0+dd1+dd2))

D0 = Drift(0.01+skew-YAG4-(0.15/2))
#Dz = Drift(0.1)
# ==========================================================================
# Matrix M
M = Matrix([[D1[0,0]**2, 2*D1[0,0]*D1[0,1], D1[0,1]**2],
            [D2[0,0]**2, 2*D2[0,0]*D2[0,1], D2[0,1]**2],
            [D3[0,0]**2, 2*D3[0,0]*D3[0,1], D3[0,1]**2]])

# Beam size calculation
MM = (M.transpose()*M).inv() * M.transpose()
MM = M.inv()

# Define the parabolic function
# Define the Gaussian function
from scipy.optimize import curve_fit
def spot_size_evolution(x, sigma_star, x0, beta_star):
    y = sigma_star*(np.sqrt(1+((x-x0)/(beta_star))**2))
    return y
# ==========================================================================
# ==========================================================================
# Measured beam size
sigx_YAG780 = 1.426925#1.3604 #4.2334 # 5.1012 # DYG4
sigy_YAG780 = 1.53524 # 1.2346 # 0.8385 #1.8574

sigx_YAG198 = 4.011 # 1.7951 #1.880# 4 #2.0842 # YAG689
sigy_YAG198 = 3.6729 # 2.0706 #2.5785 #0.3007

sigx_EYG7 =  8.953  # 2.8789 #5.2995 #1.2153 # YAG1022
sigy_EYG7 = 7.14 # 3.3649 # 4.1881 # 1.0679

#sigx_186 = 3.3240624256144957
#sigy_186 = 3.391803800056596

#sigx_DMA = 4.552377400402452
#sigy_DMA = 4.655794209630551
# ==========================================================================
# ==========================================================================

sig1x = (sigx_YAG780*1e-3)**2
sig2x = (sigx_YAG198*1e-3)**2
sig3x = (sigx_EYG7*1e-3)**2

sig1y = (sigy_YAG780*1e-3)**2
sig2y = (sigy_YAG198*1e-3)**2
sig3y = (sigy_EYG7*1e-3)**2


# ==========================================================================
# RMS beam size calculation
#sig0_11 = MM[0][0]*sig1 + MM[0][1]*sig2 + MM[0][2]*sig3
#sig0_12 = MM[1][0]*sig1 + MM[1][1]*sig2 + MM[1][2]*sig3
#sig0_22 = MM[2][0]*sig1 + MM[2][1]*sig2 + MM[2][2]*sig3
sig0x_11 = MM[0,0]*sig1x + MM[0,1]*sig2x + MM[0,2]*sig3x
sig0x_12 = MM[1,0]*sig1x + MM[1,1]*sig2x + MM[1,2]*sig3x
sig0x_22 = MM[2,0]*sig1x + MM[2,1]*sig2x + MM[2,2]*sig3x
sig0y_11 = MM[0,0]*sig1y + MM[0,1]*sig2y + MM[0,2]*sig3y
sig0y_12 = MM[1,0]*sig1y + MM[1,1]*sig2y + MM[1,2]*sig3y
sig0y_22 = MM[2,0]*sig1y + MM[2,1]*sig2y + MM[2,2]*sig3y


sig0x_11 = np.array(sig0x_11, dtype='float')
sig0x_12 = np.array(sig0x_12, dtype='float')
sig0x_22 = np.array(sig0x_22, dtype='float')
sig0y_11 = np.array(sig0y_11, dtype='float')
sig0y_12 = np.array(sig0y_12, dtype='float')
sig0y_22 = np.array(sig0y_22, dtype='float')

# ==========================================================================
# RMS beam size calculation
sig0x_recon = np.sqrt(sig0x_11)
sig0y_recon = np.sqrt(sig0y_11)

# Emittance
emitx_recon = np.sqrt(sig0x_11*sig0x_22 - (sig0x_12**2))
emity_recon = np.sqrt(sig0y_11*sig0y_22 - (sig0y_12**2))

# Normalized emittance
enx_recon = emitx_recon * pCentral
eny_recon = emity_recon * pCentral

# Twiss Beta function
betax_recon = sig0x_recon**2 / emitx_recon
betay_recon = sig0y_recon**2 / emity_recon

# Twiss Beta function
alphax_recon = -sig0x_12 / emitx_recon
alphay_recon = -sig0y_12 / emity_recon


print('RMSX is ' + repr(sig0x_recon*1e3)+ ' mm.')
print('RMSY is ' + repr(sig0y_recon*1e3)+ ' mm.')
print('========================================')
print('enx is ' + repr(enx_recon*1e6)+ ' mm mrad.')
print('eny is ' + repr(eny_recon*1e6)+ ' mm mrad.')
print('========================================')
print('betax at the initial position is  '+repr(betax_recon)+ ' m.')
print('betay at the initial position is  '+repr(betay_recon)+ ' m.')
print('alphax at the initial position is '+repr(alphax_recon)+ ' .')
print('alphay at the initial position is '+repr(alphay_recon)+ ' .')
print('========================================')


# ==========================================================================
# ==========================================================================
# Parabolic curve fitting
# =====================================================================
# =====================================================================

sigx_YAG= [sigx_YAG780,sigx_YAG198,sigx_EYG7]
sigy_YAG = [sigy_YAG780,sigy_YAG198,sigy_EYG7]
sYAG = [D_0,D_1,D_2]
parametersx, covariancex = curve_fit(spot_size_evolution, sYAG, sigx_YAG)
fit_sigmax = parametersx[0]
fit_x0 = parametersx[1]
fit_beta_star_x = parametersx[2]

parametersy, covariancey = curve_fit(spot_size_evolution, sYAG, sigy_YAG)
fit_sigmay = parametersy[0]
fit_y0 = parametersy[1]
fit_beta_star_y = parametersy[2]


#plt.figure()
#plt.scatter(sYAG, sigx_YAG)
#plt.plot(sfit, envx)

# =====================================================================
# Inverse matrix to calculate beam parameters at YAG780
gammax_recon = (1 + alphax_recon**2) / betax_recon
gammay_recon = (1 + alphay_recon**2) / betay_recon


# Twiss matrix to calculate the Twiss params at the entrance of skew quad
MTx =Matrix([[D0[0,0]**2, -2*D0[0,0]*D0[0,1], D0[0,1]**2],
            [-D0[0,0]*D0[1,0], D0[0,0]*D0[1,1] + D0[0,1]*D0[1,0], -D0[0,1]*D0[1,1]],
            [D0[1,0]**2, -2*D0[1,0]*D0[1,1], D0[1,1]**2]])

MTy =Matrix([[D0[2,2]**2, -2*D0[2,2]*D0[2,3], D0[2,3]**2],
            [-D0[2,2]*D0[3,2], D0[2,2]*D0[3,3] + D0[2,3]*D0[3,2], -D0[2,3]*D0[3,3]],
            [D0[3,2]**2, -2*D0[3,2]*D0[3,3], D0[3,3]**2]])


# Twiss inverse
MTxi = MTx
MTyi = MTy

betaxi = MTxi[0,0]*betax_recon + MTxi[0,1]*alphax_recon + MTxi[0,2]*gammax_recon
alphaxi= MTxi[1,0]*betax_recon + MTxi[1,1]*alphax_recon + MTxi[1,2]*gammax_recon
betayi = MTyi[0,0]*betay_recon + MTyi[0,1]*alphay_recon + MTyi[0,2]*gammay_recon
alphayi= MTyi[1,0]*betay_recon + MTyi[1,1]*alphay_recon + MTyi[1,2]*gammay_recon

betaxi  = np.array(betaxi, dtype='float')
alphaxi = np.array(alphaxi, dtype='float')

betayi  = np.array(betayi, dtype='float')
alphayi = np.array(alphayi, dtype='float')

print('betax at the entrance of SQ1 is  '+repr(betaxi)+ ' m.')
print('betay at the entrance of SQ1 is  '+repr(betayi)+ ' m.')
print('alphax at the entrance of SQ1 is '+repr(alphaxi)+ ' .')
print('alphay at the entrance of SQ1 is '+repr(alphayi)+ ' .')

# =====================================================================
# Initial parameters at the entrance of the second quads

sigx_YAG= [sigx_YAG780,sigx_YAG198,sigx_EYG7]
sigy_YAG = [sigy_YAG780,sigy_YAG198,sigy_EYG7]
sYAG = [D_0,D_1,D_2]
#plt.plot(z,sigmas_x,marker=11,label=r'$\sigma_x$ (mm)',)
#plt.plot(z,sigmas_y,marker=11,label=r'$\sigma_y$ (mm)')
sfit = np.linspace(0.0,D_2, 1000)
envx = fit_sigmax*(np.sqrt(1+((sfit-fit_x0)/fit_beta_star_x)**2))
envy = fit_sigmay*(np.sqrt(1+((sfit-fit_y0)/fit_beta_star_y)**2))
plt.figure()
plt.scatter(sYAG, sigx_YAG)
plt.scatter(sYAG, sigy_YAG)
#plt.plot(sfit, envx,linestyle='dotted')
plt.plot(sfit, envx,linestyle='dotted')
plt.plot(sfit, envy,linestyle='dotted')
plt.axvline(ddskew,linestyle='dotted',color='black',linewidth=2)
plt.legend()
plt.grid(True)
plt.savefig('threescan.png')
