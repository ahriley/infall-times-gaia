import numpy as np
import pandas as pd
from utils import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

G = 1.327*10**11*(1/kpc2km)**3    # kpc^3 / (solar mass * s^2)

# load z=0 halos
halos, subs = load_vl2(scale=1.0, processed=True)

# enclosed mass by particles out to Rvir(z=0)
parts = pd.read_table(VL2_DIR+'vl2subset_relposvel.txt', sep=' ')
parts['r'] = np.sqrt(parts.x**2+parts.y**2+parts.z**2)
rvals = np.logspace(0,3,1000)
mltr = np.array([np.sum(parts.r < rval) for rval in rvals]) * 2.823*10**7

# interpolating function for M(<r)
mltr_interp = interp1d(rvals, mltr, kind='cubic')

def potential(r, R0):
    func = lambda r: G*mltr_interp(r)/r**2
    return -quad(func, r, R0)[0]

props = pd.read_pickle('derived_props/vl2')

# approx potential as NFW, z=0 concentration from Emberson+ 2015
halos['c'] = halo_concentrations('vl2')[0]
halos['Rs'] = halos['Rvir']/halos['c']
halos['rho0'] = halos.Mvir/(4*np.pi*halos.Rs**3)
halos['rho0'] /= np.log(1+halos.c) - (halos.c/(1+halos.c))

# calculate potential multiple ways, save result
R0 = radii_shea()[-1]
props['pot_NFW'] = potentialNFW(subs, halos)
props['pot_NFW_1000'] = potentialNFW_R0(subs, halos, R0=1000)
props['pot_NFW_approx_mltr'] = potentialNFW_R0(subs, halos, R0=R0)
props['pot_mltr_1000'] = [potential(r, R0=10**3)*kpc2km**2 for r in subs.r]
props['pot_mltr'] = [potential(r, R0=R0)*kpc2km**2 for r in subs.r]
props.to_pickle('derived_props/vl2')
