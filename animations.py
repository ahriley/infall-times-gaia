import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from utils import *
from astropy.cosmology import WMAP7
from scipy.stats import linregress

# grab sims that are wanted
simlist_full = list_of_sims('elvis')
simlist = []
for sim in simlist_full:
    if sim[0] == 'i' and 'HiRes' not in sim:
        simlist.append(sim)

# order by virial mass
sortprop = []
for sim in simlist:
    # sortprop.append(np.sum(load_elvis(sim).iloc[0:1].M_dm))
    sortprop.append(halo_concentrations(sim)[0])
    # sortprop.append(np.sum(load_elvis(sim).iloc[0:1].apeak))
sortprop, simlist = zip(*sorted(zip(sortprop, simlist)))

# get maximum radius of subhalos
r = np.array([])
for sim in list_of_sims('elvis'):
    subs = pd.read_pickle('derived_props/'+sim)
    r = np.append(r, subs.r)
max_r = np.max(r)

# set up constant properties of figure
fig = plt.figure()
ax = plt.axes(xlim=(0.,12.75), ylim=(3.4,5.2))
plt.yticks([3.5,4.0,4.5,5.0])
scat = plt.scatter([], [], s=2.0, c=[], cmap='plasma', vmin=0.0, vmax=max_r)
line, = plt.plot([], [], c='r')
plt.title("Isolated Halos")
plt.colorbar().set_label(r'Galactocentric Radius [$kpc$]')
plt.xlabel(r'Infall time [$Gyr$]')
plt.ylabel(r'log(Binding Energy) [$km^2\ s^{-2}$]')
sim_text = ax.text(0.65, 0.1, '', transform=ax.transAxes)
sortprop_text = ax.text(0.65, 0.05, '', transform=ax.transAxes)
plt.tight_layout()

def update_plot(i):
    sim = simlist[i]
    subs = pd.read_pickle('derived_props/'+sim)
    bind = -subs.pot_NFW2.values - 0.5*(subs.v_r.values**2 + subs.v_t.values**2)
    z = WMAP7.lookback_time(1/subs.a_acc.values - 1)[bind>0].value
    r = subs.r[bind>0]
    bind = bind[bind>0]

    # calculate linear regression
    infallcut = z > 2
    bindcut = bind > 0
    bind_ = bind[infallcut & bindcut]
    infall_ = z[infallcut & bindcut]
    slope, intercept, r_value, p_value, std_err = linregress(infall_, np.log10(bind_))

    # update plot
    offset = [[infall, np.log10(E)] for E,infall in zip(bind,z)]
    scat.set_offsets(offset)        # positions
    scat.set_array(r)               # colors
    line.set_data(z[bindcut], intercept + slope*z[bindcut])
    sim_text.set_text(sim)
    sortprop_text.set_text("c = "'%.2f' % sortprop[i])
    return scat,sim_text,sortprop_text,line

anim = FuncAnimation(fig, update_plot, frames=len(simlist), blit=True, interval=600)
anim.save('animations/isolated_concentration_slope.mp4')
# plt.show()
