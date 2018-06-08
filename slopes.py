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

slopes, Mvir, c, vmax, vpeak, apeak, error = [], [], [], [], [], [], []
for sim in simlist:
    subs = pd.read_pickle('derived_props/'+sim)
    bind = -subs.pot_NFW2.values - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)
    infall = WMAP7.lookback_time(1/subs.a_acc.values - 1).value
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
    slopes.append(slope)
    Mvir.append(halo.M_dm)
    c.append(halo_concentrations(sim)[0])
    vmax.append(halo.Vmax)
    vpeak.append(halo.Vpeak)
    apeak.append(halo.apeak)
    error.append(MSE(np.log10(bind[bindcut]),pred))
slopes = np.array(slopes)
Mvir = np.array(Mvir)
c = np.array(c)
error = np.array(error)

plt.scatter(np.log10(Mvir), error, c=c, cmap='plasma')
plt.colorbar().set_label(r'Concentration')
plt.xlabel('Mvir')
plt.ylabel('Slope');
# plt.savefig('figures/slope_c.png', bbox_inches='tight')
plt.close()

plt.scatter(np.log10(Mvir), slopes, c=c, cmap='plasma')
plt.colorbar().set_label(r'Concentration')
plt.xlabel(r'log(Mvir) [$M_\odot$]')
plt.ylabel('Slope')
# plt.savefig('figures/slope_mvir.png', bbox_inches='tight')
plt.close()
