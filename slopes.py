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

slopes, Mvir, c, apeak, error, mass_change, Mpeak = [], [], [], [], [], [], []
for sim in simlist:
    subs = pd.read_pickle('derived_props/'+sim)
    bind = -subs.pot_NFW2 - 0.5*(subs.v_r**2 + subs.v_t**2)
    infall = WMAP7.lookback_time(1/subs.a_acc - 1).value
    infallcut = infall > 2
    bindcut = bind > 0
    bind_ = bind[infallcut & bindcut]
    infall_ = infall[infallcut & bindcut]

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(infall_, np.log10(bind_))
    pred = intercept + slope*infall[bindcut]
    """
    plt.scatter(WMAP7.lookback_time(1/subs.a_acc.values - 1)[bindcut],
                np.log10(bind[bindcut]), s=2.0, c=subs.r[bindcut],
                cmap='plasma', vmin=0.0, vmax=370)
    plt.plot(WMAP7.lookback_time(1/subs.a_acc.values - 1).value, intercept + slope*WMAP7.lookback_time(1/subs.a_acc.values - 1).value, 'r')
    plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
    plt.xlim(0.0, WMAP7.lookback_time(np.inf).value)
    plt.ylim(3.4,5.2)
    plt.yticks([3.5,4.0,4.5,5.0])
    plt.xlabel(r'Infall time [$Gyr$]')
    plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]');
    plt.savefig('figures/eachvolume/slope_'+sim+'.png', bbox_inches='tight')
    plt.close()
    """
    # save the params
    halo = load_elvis(sim).iloc[0]
    mass_change.append((halo.Mpeak - halo.M_dm)/halo.M_dm)
    slopes.append(slope)
    Mvir.append(halo.M_dm)
    c.append(halo_concentrations(sim)[0])
    apeak.append(halo.apeak)
    Mpeak.append(halo.Mpeak)
    error.append(MSE(np.log10(bind[bindcut]),pred))
slopes = np.array(slopes)
Mvir = np.array(Mvir)
c = np.array(c)
error = np.array(error)
mass_change = np.array(mass_change)
apeak = np.array(apeak)

plt.scatter(c, slopes, c=np.log10(Mvir), cmap='plasma')
plt.colorbar().set_label(r'log(Mvir) [$M_\odot$]')
plt.xlabel('c')
plt.ylabel('Slope');
plt.savefig('figures/slope_c_mvir.png', bbox_inches='tight')
plt.close()

plt.scatter(c, slopes, c=1/apeak-1, cmap='plasma_r')
plt.plot(12.2, 0.10688449504116676, 'ks')
plt.colorbar().set_label(r'z_peak')
plt.xlabel('c')
plt.ylabel('Slope');
plt.savefig('figures/slope_c_zpeak.png', bbox_inches='tight')
plt.close()

plt.scatter(1/apeak[mass_change>0]-1, mass_change[mass_change>0], c=Mvir[mass_change>0], cmap='plasma')
plt.xlabel('z_peak')
plt.ylabel('(Mpeak-Mvir(z=0)) / Mpeak');
plt.colorbar().set_label(r'Mvir')
plt.savefig('figures/massloss_zpeak.png', bbox_inches='tight')
plt.close()
