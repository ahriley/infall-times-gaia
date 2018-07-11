import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from astropy.cosmology import WMAP7
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error as MSE

# grab sims that are wanted
simlist_full = list_of_sims('elvis')
simlist = []
for sim in simlist_full:
    if sim[0] == 'i' and 'HiRes' not in sim:
        simlist.append(sim)
simlist.append('vl2')

slopes, b, Mvir, c, apeak, error, mass_change, Mpeak, Vmax = [[] for i in range(9)]
for sim in simlist:
    if sim == 'vl2':
        halos, subs = load_vl2(scale=1.0, processed=True)
        subs = subs[subs.Vmax > 8]
    else:
        halos, subs = load_elvis(sim=sim, processed=True)
    # subs = subs[subs.nadler2018 > 0.5]
    pot = subs.pot_mltr
    bind = -pot - 0.5*(subs.v_r**2 + subs.v_t**2)
    infall = WMAP7.lookback_time(1/subs.a_acc - 1).value
    infallcut = infall > 2
    bindcut = bind > 0
    bind_ = bind[infallcut & bindcut]
    infall_ = infall[infallcut & bindcut]

    # linear regression
    slope, intercept, r_value = linregress(infall_, np.log10(bind_))[:3]
    pred = intercept + slope*infall_

    # save the params
    slopes.append(slope)
    error.append(MSE(np.log10(bind_),pred))
    b.append(intercept)
    if sim == 'vl2':
        Mvir.append(1.93596e+12)
        Vmax.append(201.033)
        c.append(halo_concentrations('vl2')[0])
        continue
    halo = halos.iloc[0]
    mass_change.append((halo.Mpeak - halo.Mvir)/halo.Mvir)
    Mvir.append(halo.Mvir)
    c.append(halo_concentrations(sim)[0])
    apeak.append(halo.apeak)
    Mpeak.append(halo.Mpeak)
    Vmax.append(halo.Vmax)
slopes = np.array(slopes)
Mvir = np.array(Mvir)
c = np.array(c)
error = np.array(error)
mass_change = np.array(mass_change)
apeak = np.array(apeak)

x = np.log10(Mvir)
color = c
y = slopes
k = {'cmap': 'plasma', 'vmin': np.min(color), 'vmax': np.max(color)}
plt.scatter(x[:-1], y[:-1], c=color[:-1], **k)
plt.scatter(x[-1], y[-1], c=color[-1], marker='*', s=200, **k)
plt.colorbar().set_label(r'Concentration');
plt.title(r'MSE vs. Halo Params ($\Phi_{M(<r)}(500) = 0$)')
plt.xlabel(r'$\log(M_{vir}) [M_\odot]$')
plt.ylabel(r'$d\log(E_{bind})/dt_{infall}$');
plt.savefig('figures/spread_mvir_c.png', bbox_inches='tight')
