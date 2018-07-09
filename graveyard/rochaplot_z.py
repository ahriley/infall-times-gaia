import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from astropy.cosmology import WMAP7
from scipy.stats import linregress

simlist = ['iOates', 'iKauket', 'iHera', 'iZeus', 'iScylla', 'iSerena']

for sim in simlist[:1]:
    print(sim)
    slopes, concentrations, mvirs = [], [], []

    # get z=0 properties (to limit computation)
    subs_0 = pd.read_pickle('derived_props/'+sim)

    scale_list = list_of_scales('elvis',sim)
    scale_list = scale_list[scale_list>1/(1+0.4)]
    for a_start in scale_list[:1]:
        # grab z=a_start properties
        # print("Computing for "+str(a_start))
        subs_z_full = get_halos_at_scale_elvis(sim,a_start)
        nhosts = 2 if '&' in sim else 1
        haloIDs = list(subs_z_full.index.values[0:nhosts])
        subs_z, halos_z = subs_z_full.drop(haloIDs), subs_z_full.loc[haloIDs]
        subs_z = subs_z[np.isin(subs_z.index, subs_0.index)]
        assert len(subs_z) == len(subs_0)
        subs_z['hostID'] = subs_0['hostID']

        # only include subhalos of main halo at z=a_start
        subs_z = subs_z[subs_z.pID == halos_z.loc[subs_z['hostID']].zID.values]

        # center, convert to spherical
        subs_z = center_on_hosts(hosts=halos_z, subs=subs_z)
        subs_z.x, subs_z.y, subs_z.z = subs_z.x*Mpc2km, subs_z.y*Mpc2km, subs_z.z*Mpc2km
        subs_z = compute_spherical_hostcentric_sameunits(df=subs_z)
        subs_z.x, subs_z.y, subs_z.z = subs_z.x*km2kpc, subs_z.y*km2kpc, subs_z.z*km2kpc
        subs_z.r = subs_z.r*km2kpc
        assert (subs_z.r < halos_z.loc[subs_z['hostID']].Rvir.values).all()

        # set up for finding new accretion time if necessary
        subs_z['a_acc'] = subs_0['a_acc']
        subs_z['acc_found'] = subs_z.a_acc < a_start
        subs_z.loc[~subs_z.acc_found,'a_acc'] = a_start
        print(len(subs_0), len(subs_z))

        # accretion: lookback time to r >= host's Rvir
        for a in scale_list[(scale_list > 0) & (scale_list < a_start)]:
            # get z=a properties
            subs_a = get_halos_at_scale_elvis(sim, a)
            subs_a, halos_a = subs_a.drop(haloIDs), subs_a.loc[haloIDs]

            # restrict to subhalos of main halos at z=0
            subs_a = subs_a[np.isin(subs_a.index, subs_z.index)]
            subs_a['hostID'] = subs_0['hostID']

            # center, convert to spherical
            subs_a = center_on_hosts(hosts=halos_a, subs=subs_a)
            subs_a.x, subs_a.y, subs_a.z = subs_a.x*Mpc2km, subs_a.y*Mpc2km, subs_a.z*Mpc2km
            subs_a = compute_spherical_hostcentric_sameunits(df=subs_a)
            subs_a.x, subs_a.y, subs_a.z = subs_a.x*km2kpc, subs_a.y*km2kpc, subs_a.z*km2kpc
            subs_a.r = subs_a.r*km2kpc

            # if accretion condition satisfied, stop updating accretion values
            subs_z['acc_found'] = (subs_z['acc_found']) | (subs_a.r > halos_a.loc[subs_a['hostID']].Rvir.values)
            subs_z.loc[~subs_z.acc_found, 'a_acc'] = a

            if subs_z['acc_found'].all():
                subs_z.drop('acc_found', axis=1, inplace=True)
                break

        assert (a_start >= subs_z.a_acc).all()

        # convert comoving -> physical for potential calculation
        subs_z['r'] *= a_start
        halos_z[['Rvir', 'Rs']] *= a_start

        # approx potential as NFW
        halos_z['c'] = halos_z.Rvir/halos_z.Rs
        halos_z['rho0'] = halos_z.Mvir/(4*np.pi*halos_z.Rs**3)
        halos_z['rho0'] /= np.log(1+halos_z.c) - (halos_z.c/(1+halos_z.c))
        subs_z['pot_NFW'] = potentialNFW(subs_z, halos_z)
        subs_z['pot_NFW2'] = potentialNFW_Rocha_approx(subs_z, halos_z)

        # plot the Rocha plot for z=a_start
        bind = -subs_z.pot_NFW2 - 0.5*(subs_z.v_r**2 + subs_z.v_t**2)
        infall = WMAP7.lookback_time(1/subs_z.a_acc - 1).value
        bindcut = bind > 0
        infallcut = infall > 2
        cuts = infallcut & bindcut

        slope, intercept = linregress(infall[cuts], np.log10(bind[cuts]))[:2]
        pred = intercept + slope*infall[bindcut]
        slopes.append(slope)
        concentrations.append(halos_z.iloc[0].c)
        mvirs.append(halos_z.iloc[0].Mvir)

    slopes = np.array(slopes)
    concentrations = np.array(concentrations)
    mvirs = np.array(mvirs)

    result = np.stack((scale_list,slopes,concentrations,mvirs))
    np.save('derived_props/slope_vs_a/'+sim, result, allow_pickle=False)

    plt.scatter(result[2], result[1], c=1/result[0]-1, cmap='plasma_r')
    plt.colorbar().set_label(r'z')
    plt.xlabel(r'Concentration')
    plt.ylabel(r'Slope');
    plt.savefig('figures/slope_vs_a/'+sim+'_z.png', bbox_inches='tight')
    plt.close()

    plt.scatter(result[2], result[1], c=np.log10(result[3]), cmap='plasma')
    plt.colorbar().set_label(r'log(M_vir) [$M_\odot$]')
    plt.xlabel(r'Concentration')
    plt.ylabel(r'Slope');
    plt.savefig('figures/slope_vs_a/'+sim+'_mvir.png', bbox_inches='tight')
    plt.close()
