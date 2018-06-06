import numpy as np
import pandas as pd
from utils import *
from scipy.integrate import quad

G = 1.327*10**11    # km^3 / (solar mass * s^2)
G2 = G*km2kpc**3

def potential(subhalos, hosts):
    r = subhalos.r*kpc2km
    Rs = hosts.loc[subs['hostID']].Rs.values*kpc2km
    rho0 = hosts.loc[subs['hostID']].rho0.values/(kpc2km**3)
    return -4*np.pi*G*rho0*Rs**3*np.log(1+r/Rs)/r

def mltr(r, rho0, rs):
    return 4*np.pi*rho0*rs**3*(np.log((rs+r)/rs) - r/(rs+r))

def potential2(r, rho0, rs, rmax):
    func = lambda r: G2*mltr(r, rho0, rs)/r**2
    return -quad(func, r, rmax)[0]

for sim in ['iHall_HiRes', 'iScylla_HiRes', 'iKauket_HiRes']:
    nhosts = 2 if '&' in sim else 1
    halos = load_elvis(sim).iloc[0:nhosts]
    subs = pd.read_pickle('derived_props/'+sim)

    # approx potential as NFW, concentrations from ELVIS paper Table 2
    halos['c'] = halo_concentrations(sim)
    halos['Rs'] = halos['Rvir']/halos['c']
    halos['rho0'] = halos.M_dm/(4*np.pi*halos.Rs**3)
    halos['rho0'] /= np.log(1+halos.c) - (halos.c/(1+halos.c))

    # calculate potential two ways, save result
    subs['pot_NFW'] = potential(subs, halos)
    subs['pot_NFW_1Mpc'] = [potential2(rval,halos.rho0,halos.Rs, 10**3)*kpc2km**2 for rval in subs.r]
    subs['pot_NFW_inf'] = [potential2(rval,halos.rho0,halos.Rs, np.inf)*kpc2km**2 for rval in subs.r]
    subs.to_pickle('derived_props/'+sim)
