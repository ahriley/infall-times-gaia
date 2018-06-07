import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from utils import *
from astropy.cosmology import WMAP7

# grab sims that are wanted
simlist_full = list_of_sims('elvis')
simlist = []
for sim in simlist_full:
    if sim[0] == 'i' and not 'HiRes' in sim:
        simlist.append(sim)

# order by virial mass
Mvir = []
for sim in simlist:
    Mvir.append(np.sum(load_elvis(sim).iloc[0:1].M_dm))
Mvir, simlist = zip(*sorted(zip(Mvir, simlist)))

# get maximum radius of subhalos
r = np.array([])
for sim in list_of_sims('elvis'):
    subs = pd.read_pickle('derived_props/'+sim)
    r = np.append(r, subs.r)
max_r = np.max(r)

# set up constant properties of figure
fig = plt.figure()
ax = plt.axes(xlim=(0.,WMAP7.lookback_time(np.inf).value), ylim=(3.4,5.2))
plt.yticks([3.5,4.0,4.5,5.0])
scat = plt.scatter([], [], s=2.0, c=[], cmap='plasma', vmin=0.0, vmax=max_r)
plt.title("Isolated Halos")
plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
plt.xlabel(r'Infall time [$Gyr$]')
plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]')
sim_text = ax.text(0.65, 0.1, '', transform=ax.transAxes)
mvir_text = ax.text(0.65, 0.05, '', transform=ax.transAxes)

def update_plot(i):
    sim = simlist[i]
    subs = pd.read_pickle('derived_props/'+sim)
    bind = -subs.pot_NFW2.values - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)
    z = WMAP7.lookback_time(1/subs.a_acc.values - 1)[bind>0]
    r = subs.r[bind>0]
    bind = bind[bind>0]
    offset = [[infall.value, np.log10(E)] for E,infall in zip(bind,z)]
    scat.set_offsets(offset)        # positions
    scat.set_array(r)               # colors
    sim_text.set_text(sim)
    mvir_text.set_text('%.2E' % Mvir[i])
    return scat,sim_text,mvir_text

anim = FuncAnimation(fig, update_plot, frames=len(simlist), blit=True, interval=600)
anim.save('isolated.mp4')
# plt.show()
