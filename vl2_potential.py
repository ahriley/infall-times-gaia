import numpy as np
import pandas as pd
from utils import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

G = 1.327*10**11*km2kpc**3    # kpc^3 / (solar mass * s^2)

# load z=0 halos
halo = load_vl2(scale=1.0).iloc[0]

# enclosed mass by particles out to Rvir(z=0)
parts = pd.read_table(VL2_DIR+'vl2subset_relposvel.txt', sep=' ')
parts['r'] = np.sqrt(parts.x**2+parts.y**2+parts.z**2)
rvals = np.logspace(0,3,1000)
mltr = np.array([np.sum(parts.r < rval) for rval in rvals]) * 2.823*10**7

# interpolating function for M(<r)
mltr_interp = interp1d(rvals, mltr, kind='cubic')

def potential(r):
    func = lambda r: G*mltr_interp(r)/r**2
    return -quad(func, r, 10**3)[0]

subs = pd.read_pickle('derived_props/vl2')
subs['pot_grav'] = [potential(r)*kpc2km**2 for r in subs.r]
subs.to_pickle('derived_props/vl2')
