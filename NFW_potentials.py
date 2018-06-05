import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt

G = 1.327*10**11    # km^3 / (solar mass * s^2)

def potential(subhalos, hosts):
    r = subhalos.r*kpc2km
    Rs = hosts.loc[subs['hostID']].Rs.values*kpc2km
    rho0 = hosts.loc[subs['hostID']].rho0.values/(kpc2km**3)
    return -4*np.pi*G*rho0*Rs**3*np.log(1+r/Rs)/r

for sim in list_of_sims('elvis'):
    halos = load_elvis(sim).iloc[0:2]
    subs = pd.read_pickle('derived_props/'+sim)

    # approx potential as NFW, concentrations from ELVIS paper Table 2
    halos['c'] = halo_concentrations(sim)
    halos['Rs'] = halos['Rvir']/halos['c']
    halos['rho0'] = halos.M_dm/(4*np.pi*halos.Rs**3)
    halos['rho0'] /= np.log(1+halos.c) - (halos.c/(1+halos.c))

    # calculate potential, save result
    subs['pot_NFW'] = potential(subs, halos)
    subs.to_pickle('derived_props/'+sim)
