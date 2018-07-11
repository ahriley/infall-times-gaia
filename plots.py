import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from astropy.cosmology import WMAP7

# Rocha plots for ELVIS
r = np.array([])
for sim in list_of_sims('elvis'):
    halos, subs = load_elvis(sim=sim, processed=True)
    r = np.append(r, subs.r)

max_r = np.max(r)

bind, z, r, v_r = [np.array([]) for i in range(4)]
for sim in list_of_sims('elvis'):
    if sim[0] != 'i' or 'HiRes' in sim:
        continue
    try:
        halos, subs = load_elvis(sim=sim, processed=True)
        subs = subs[subs.nadler2018 > 0.5]
        pot = subs.pot_mltr
    except AttributeError:
        print(sim+" not included")
        continue

    bind_sim = -pot - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)
    bind = np.append(bind, bind_sim)
    z = np.append(z, WMAP7.lookback_time(1/subs.a_acc.values - 1))
    r = np.append(r, subs.r)
    v_r = np.append(v_r, subs.v_r)
    """
    plt.scatter(WMAP7.lookback_time(1/subs.a_acc.values - 1)[bind_sim>0], np.log10(bind_sim[bind_sim>0]), s=2.0, c=subs.r[bind_sim>0], cmap='plasma', vmin=0.0, vmax=max_r)
    plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
    plt.xlim(0.0, WMAP7.lookback_time(np.inf).value)
    plt.ylim(3.4,5.2)
    plt.yticks([3.5,4.0,4.5,5.0])
    plt.xlabel(r'Infall time [$Gyr$]')
    plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]');
    plt.savefig('figures/eachvolume/rocha_fig1_'+sim+'.png', bbox_inches='tight')
    plt.close()

    plt.scatter(subs.r[bind_sim>0], subs.v_r[bind_sim>0], s=2.0, c=WMAP7.lookback_time(1/subs.a_acc.values - 1)[bind_sim>0], cmap='plasma')
    plt.colorbar().set_label(r'Infall time [$Gyr$]')
    plt.xlabel(r'Galactocentric Radius [$kpc$]')
    plt.ylabel(r'Radial Velocity [$km/s$]')
    plt.savefig('figures/eachvolume/rocha_fig3_'+sim+'.png', bbox_inches='tight')
    plt.close()
    """
plt.scatter(z[(bind>0)], np.log10(bind[bind>0]), c=r[bind>0], s=2., cmap='plasma')
plt.ylim(2.5,5.2)
plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
plt.xlabel(r'Infall time [$Gyr$]')
plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]');
plt.savefig('figures/isolated.png', bbox_inches='tight')

"""
# plot for single halo
halos, subs = load_vl2(scale=1.0, processed=True)
# subs = subs[subs.nadler2018 > 0.5]
z = WMAP7.lookback_time(1/subs.a_acc.values - 1)
# z = subs.a_acc.values
r = subs.r.values
v_r = subs.v_r
bind = -subs.pot_mltr_1000.values - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)

plt.scatter(z[bind>0], np.log10(bind[bind>0]), s=10.0, c=r[bind>0], cmap='plasma')
plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
plt.title('VL2')
plt.ylim(3.4,5.2)
plt.yticks([3.5,4.0,4.5,5.0])
plt.xlabel(r'Infall time [$Gyr$]')
plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]');
# plt.savefig('figures/rocha_fig1_iScylla_HiRes.png', bbox_inches='tight')
plt.close()

plt.scatter(r[bind>0], v_r[bind>0], s=2.0, c=z[bind>0], cmap='plasma')
plt.colorbar().set_label(r'Infall time [$Gyr$]')
plt.xlabel(r'Galactocentric Radius [$kpc$]')
plt.ylabel(r'Radial Velocity [$km/s$]');
plt.title('iScylla_HiRes')
# plt.savefig('figures/rocha_fig3_iScylla_HiRes.png', bbox_inches='tight')
plt.close()
# """
