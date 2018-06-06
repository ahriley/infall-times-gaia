import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from astropy.cosmology import WMAP7
import astropy.units as u

subs = pd.read_pickle('derived_props/vl2')
z = WMAP7.lookback_time(1/subs.a_acc.values - 1)
r = subs.r.values
v_r = subs.v_r
bind = -subs.pot_grav.values - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)

plt.scatter(z[(r<400) & (bind>0)], np.log10(bind[(r<400) & (bind>0)]), s=2.0, c=r[(r<400) & (bind>0)], cmap='plasma')
plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
plt.ylim(3.4,5.2)
plt.yticks([3.5,4.0,4.5,5.0])
plt.xlabel(r'Infall time [$Gyr$]')
plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]');
plt.savefig('figures/rocha_fig1_reconstruct.png', bbox_inches='tight')
plt.close()

plt.scatter(r[(r<400) & (bind>0)], v_r[(r<400) & (bind>0)], s=2.0, c=z[(r<400) & (bind>0)], cmap='plasma')
plt.colorbar().set_label(r'Infall time [$Gyr$]')
plt.xlabel(r'Galactocentric Radius [$kpc$]')
plt.ylabel(r'Radial Velocity [$km/s$]');
plt.savefig('figures/rocha_fig3_reconstruct.png', bbox_inches='tight')
plt.close()

"""
# Rocha plots for ELVIS
bind, z, r, v_r = [np.array([]) for i in range(4)]
for sim in list_of_sims('elvis'):
    subs = pd.read_pickle('derived_props/'+sim)
    bind_sim = -subs.pot_NFW.values - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)
    bind = np.append(bind, bind_sim)
    z = np.append(z, WMAP7.lookback_time(1/subs.apeak.values - 1))
    r = np.append(r, subs.r.values)
    v_r = np.append(v_r, subs.v_r)

plt.scatter(z, np.log10(bind), s=2.0, c=r, cmap='plasma')
plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
plt.ylim(3.4,5.2)
plt.yticks([3.5,4.0,4.5,5.0])
plt.xlabel(r'Infall time [$Gyr$]')
plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]');
plt.savefig('figures/rocha_fig1_apeak.png', bbox_inches='tight')
plt.close()

plt.scatter(r, v_r, s=2.0, c=z, cmap='plasma')
plt.colorbar().set_label(r'Infall time [$Gyr$]')
plt.xlabel(r'Galactocentric Radius [$kpc$]')
plt.ylabel(r'Radial Velocity [$km/s$]')
plt.savefig('figures/rocha_fig3.png', bbox_inches='tight')
plt.close()
"""
