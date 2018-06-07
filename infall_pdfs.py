import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from utils import *
from astropy.cosmology import WMAP7

# read in dwarfs yaml
dwarf_file = 'data/fritz.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)
names = list(dwarfs.keys())
ignore = [names.index('Cra I'), names.index('Eri II'), names.index('Phe I')]
for ii in ignore:
    names.remove(names[ii])

# compute stats from MC based on Fritz+ 2018
# ORDER: vx vy vz x y z r theta phi v_r v_theta v_phi v_t
MCsats = np.load('data/MCsample_fritz.npy')
MCsats = np.swapaxes(MCsats,0,1)
MCsats = np.swapaxes(MCsats,1,2)
MCsats = np.delete(MCsats, ignore, axis=0)

# calculate means, sigmas for r, v_r, and v_t
means = np.mean(MCsats, axis=2)[:,[6,9,12]]
sigs = np.sqrt(np.var(MCsats, axis=2)[:,[6,9,12]])
cols = ['r', 'v_r', 'v_t', 'sigma_r', 'sigma_v_r', 'sigma_v_t']
sats = pd.DataFrame(np.concatenate((means,sigs), axis=1), columns=cols, index=names)

# upper and lower bounds
for col in cols[:3]:
    sats[col+'_upper'] = sats[col] + sats['sigma_'+col]
    sats[col+'_lower'] = sats[col] - sats['sigma_'+col]

# grab sims that are wanted
simlist_full = list_of_sims('elvis')
simlist = []
for sim in simlist_full:
    if sim[0] == 'i':
        simlist.append(sim)

# grab ALL subhalos that are desired
r, v_r, v_t, a_acc = [np.array([]) for i in range(4)]
for sim in simlist:
    subs = pd.read_pickle('derived_props/'+sim)
    r = np.append(r, subs.r.values)
    v_r = np.append(v_r, subs.v_r.values)
    v_t = np.append(v_t, subs.v_t.values)
    a_acc = np.append(a_acc, subs.a_acc.values)
dict = {'r': r, 'v_r': v_r, 'v_t': v_t, 'a_acc': a_acc}
subs = pd.DataFrame(dict)

# generate PDFs for each satellite
for name, sat in sats.iterrows():
    rconstraint = (sat.r_lower < subs.r) & (subs.r < sat.r_upper)
    vconstraint = (subs.v_r < sat.v_r_upper) & (sat.v_r_lower < subs.v_r)
    subset = subs[rconstraint & vconstraint]
    subset2 = subset[(sat.v_t_lower < subset.v_t) & (subset.v_t < sat.v_t_upper)]

    if len(subset) == 0 or len(subset2) == 0:
        continue

    plt.hist(WMAP7.lookback_time(1/subset.a_acc - 1).value)
    plt.hist(WMAP7.lookback_time(1/subset2.a_acc - 1).value)
    plt.title(name)
    plt.xlabel('Infall Time [Gyr]')
    plt.savefig('figures/infall_pdfs/'+name+'.png',bbox_inches='tight')
    plt.close()
