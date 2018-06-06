import numpy as np
import pandas as pd
from utils import *

# present day properties
subs_full = load_vl2(scale=1.0)
subs, halo = subs_full.drop(subs_full.iloc[0].name), subs_full.iloc[0]
subs_full.drop(subs_full.iloc[0].name, inplace=True)

# convert coords to spherical
subs.x, subs.y, subs.z = subs.x*kpc2km, subs.y*kpc2km, subs.z*kpc2km
subs = compute_spherical_hostcentric_sameunits(df=subs)
subs.x, subs.y, subs.z = subs.x*km2kpc, subs.y*km2kpc, subs.z*km2kpc
subs.r = subs.r*km2kpc

# check that each subhalo is currently within the halo's Rvir
assert (subs.r < halo.Rvir).all()

newvals = ['a_acc', 'M_acc', 'V_acc', 'a_peri', 'd_peri']
subs = subs.reindex(columns=subs.columns.tolist() + newvals)
subs['acc_found'] = False
subs['peri_found'] = False
subs['d_peri'] = np.inf

# accretion: lookback time to r >= host's Rvir
scale_list = list_of_scales('vl2')
for a in scale_list[scale_list > 0]:
    # get z=a properties
    subs_a = load_vl2(scale=a)
    subs_a, halo_a = subs_a.drop(subs_a.iloc[0].name), subs_a.iloc[0]

    # if accretion condition satisfied, stop updating accretion values
    subs['acc_found'] = (subs['acc_found']) | (subs_a.r > halo_a.Rvir)
    print(a, np.sum(subs_a.r > halo_a.Rvir))
    subs.loc[~subs.acc_found, 'a_acc'] = a
    subs.loc[~subs.acc_found, 'M_acc'] = subs_a.loc[~subs.acc_found]['Mvir']
    subs.loc[~subs.acc_found, 'V_acc'] = subs_a.loc[~subs.acc_found]['Vmax']

    if subs['acc_found'].all():
        subs.drop('acc_found', axis=1, inplace=True)
        break

# pericenter: step forward from accretion until a local minimum in r
min_scale = np.min(subs.a_acc)
for a in scale_list[scale_list > min_scale][::-1]:
    # get z=a properties
    subs_a = load_vl2(scale=a)
    subs_a, halo_a = subs_a.drop(subs_a.iloc[0].name), subs_a.iloc[0]

    # find local minimum after accretion
    subs['acc'] = a >= subs['a_acc']
    subs['peri_found'] = (subs['peri_found']) | ((subs_a.r > subs.d_peri) & (subs['acc']))
    subs.loc[~subs.peri_found&subs.acc, 'a_peri'] = a
    subs.loc[~subs.peri_found&subs.acc, 'd_peri'] = subs_a.loc[~subs.peri_found&subs.acc]['r']

    if subs['peri_found'].all() or a==1.0:
        subs.drop(labels=['acc', 'peri_found'], axis=1, inplace=True)
        break

# save diagnostics
q = len(subs[(subs.a_acc == subs.a_peri) & (subs.d_peri > 200) & (subs.a_peri != 1.0)])/len(subs)

subs.to_pickle('derived_props/vl2')

# output diagnostics to file
with open("infall_times_diagnostics_vl2.txt", "w") as f:
    f.write('Naughty fraction (a_acc = a_peri AND d_peri > 200 AND a_peri != 1.0)\n')
    f.write(str(q) + '\n')
    f.write('\nEarliest accretion (min(a_acc))\n')
    f.write(str(np.min(subs.a_acc)) + '\n')
