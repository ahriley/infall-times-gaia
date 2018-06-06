import numpy as np
import pandas as pd
from utils import *

naughty_frac = []
earliest_acc = []

for sim in ['iHall_HiRes', 'iScylla_HiRes', 'iKauket_HiRes']:
    # present day properties
    subs_full = load_elvis(sim)
    nhosts = 2 if '&' in sim else 1
    haloIDs = list(subs_full.index.values[0:nhosts])
    subs, halos = subs_full.drop(haloIDs), subs_full.loc[haloIDs]
    subs_full.drop(haloIDs, inplace=True)
    subs = subs[np.isin(subs.hostID, halos.index)]
    subs = subs[subs.Vmax > 5]
    subs = center_on_hosts(hosts=halos, subs=subs)
    subs.x, subs.y, subs.z = subs.x*Mpc2km, subs.y*Mpc2km, subs.z*Mpc2km
    subs = compute_spherical_hostcentric_sameunits(df=subs)
    subs.x, subs.y, subs.z = subs.x*km2kpc, subs.y*km2kpc, subs.z*km2kpc
    subs.r = subs.r*km2kpc

    # check that each subhalo is currently within the halo's Rvir
    totalsubs = len(subs)
    subs = subs[subs.r < halos.loc[subs.hostID].Rvir.values]
    if totalsubs != len(subs):
        print(sim+" had "+str(totalsubs-len(subs))+" subs removed (r > Rvir)")

    newvals = ['a_acc', 'M_acc', 'V_acc', 'a_peri', 'd_peri']
    subs = subs.reindex(columns=subs.columns.tolist() + newvals)
    subs['acc_found'] = False
    subs['peri_found'] = False
    subs['d_peri'] = np.inf

    print(sim+" "+str(len(subs)))

    # accretion: lookback time to r >= host's Rvir
    scale_list = np.array(list_of_scales('elvis', sim))
    for a in scale_list[scale_list > 0]:
        # get z=a properties
        subs_a = get_halos_at_scale_elvis(sim, a)
        subs_a, halos_a = subs_a.drop(haloIDs), subs_a.loc[haloIDs]

        # restrict to subhalos of main halos at z=0
        subs_a['hostID'] = subs_full['hostID']
        subs_a = subs_a.loc[subs.index]

        # center, convert to spherical
        subs_a = center_on_hosts(hosts=halos_a, subs=subs_a)
        subs_a.x, subs_a.y, subs_a.z = subs_a.x*Mpc2km, subs_a.y*Mpc2km, subs_a.z*Mpc2km
        subs_a = compute_spherical_hostcentric_sameunits(df=subs_a)
        subs_a.x, subs_a.y, subs_a.z = subs_a.x*km2kpc, subs_a.y*km2kpc, subs_a.z*km2kpc
        subs_a.r = subs_a.r*km2kpc

        # if accretion condition satisfied, stop updating accretion values
        subs['acc_found'] = (subs['acc_found']) | (subs_a.r > halos_a.loc[subs_a['hostID']].Rvir.values)
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
        subs_a = get_halos_at_scale_elvis(sim, a)
        subs_a, halos_a = subs_a.drop(haloIDs), subs_a.loc[haloIDs]

        # restrict to subhalos of main halos at z=0
        subs_a['hostID'] = subs_full['hostID']
        subs_a = subs_a.loc[subs.index]

        # center, convert to spherical
        subs_a = center_on_hosts(hosts=halos_a, subs=subs_a)
        subs_a.x, subs_a.y, subs_a.z = subs_a.x*Mpc2km, subs_a.y*Mpc2km, subs_a.z*Mpc2km
        subs_a = compute_spherical_hostcentric_sameunits(df=subs_a)
        subs_a.x, subs_a.y, subs_a.z = subs_a.x*km2kpc, subs_a.y*km2kpc, subs_a.z*km2kpc
        subs_a.r = subs_a.r*km2kpc

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
    naughty_frac.append(q)
    earliest_acc.append(np.min(subs.a_acc))
    subs.to_pickle('derived_props/'+sim)

# output diagnostics to file
with open("infall_times_diagnostics_hires.txt", "w") as f:
    f.write('Naughty fractions (a_acc = a_peri AND d_peri > 200 AND a_peri != 1.0)\n')
    for sim in list_of_sims('elvis'):
        f.write(sim + ": " + str(q) + '\n')
    f.write('\nEarliest accretion (min(a_acc))\n')
    for sim in list_of_sims('elvis'):
        f.write(sim + ": " + str(np.min(subs.a_acc)) + '\n')
