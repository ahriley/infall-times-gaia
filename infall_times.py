import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import argrelmin

naughty_frac = []
earliest_acc = []

for sim in list_of_sims('elvis'):
    print(sim)
    halos, subs = load_elvis(sim, processed=True)
    assert (subs.r < halos.loc[subs.hostID].Rvir.values).all()

    # read in the evolutionary tracks
    x = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/X.txt')
    y = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/Y.txt')
    z = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/Z.txt')
    pos = np.stack((x,y,z), axis=2) * 1000
    IDs = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/ID.txt').astype(int)[:,0]
    a = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/scale.txt')
    rvir = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/Rvir.txt')
    Mvir = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/Mvir.txt')
    Vmax = np.loadtxt(ELVIS_DIR+'/tracks/'+sim+'/Vmax.txt')

    # get properties for main halo(s)
    nHalos = len(halos)
    halos_pos = pos[0:nHalos]

    # select out subhalos of interest
    indices = []
    for ID in subs.index:
        ii = np.where(IDs == ID)[0][0]
        indices.append(ii)
    pos = pos[indices]; IDs = IDs[indices]

    # map ID of main halos to index for halo_pos array
    mapper = {}
    for ID in halos.index:
        mapper[ID] = halos.index.get_loc(ID)

    # center the subhalos, compute r(a)
    for ii in range(len(IDs)):
        halo_ii = mapper[int(subs.iloc[ii].hostID)]
        halo_pos_ii = halos_pos[halo_ii]
        pos[ii] -= halo_pos_ii
    r = np.sqrt(pos[:,:,0]**2 + pos[:,:,1]**2 + pos[:,:,2]**2)

    # interpolate Rvir(a) for the main halos, map to halo IDs
    mapper = {}
    AMIN = -np.infty
    for ID in halos.index:
        halo_index = halos.index.get_loc(ID)
        a_halo = a[halo_index]
        rvir_halo = rvir[halo_index][a_halo > 0]
        a_halo = a_halo[a_halo > 0]
        AMIN = np.max([np.min(a_halo), AMIN])
        mapper[ID] = interp1d(a_halo, rvir_halo, kind='cubic')

    # find a_acc, a_peri for subhalos of interest
    a = a[indices]; rvir = rvir[indices]
    Mvir = Mvir[indices]; Vmax = Vmax[indices]
    accs, peris, dperis, Maccs, Vaccs = [], [], [], [], []
    bad = 0
    for ii in range(len(indices)):
        # a_acc: most recent crossing of Rvir
        inc = a[ii] > AMIN
        r_a = interp1d(a[ii][inc], r[ii][inc], kind='cubic')
        host_rvir = mapper[int(subs.iloc[ii].hostID)]
        low = a[ii][inc][np.argmax(r[ii][inc] - host_rvir(a[ii][inc]) > 0)]
        high = 1.0
        if low == high:
            bad += 1
            a_acc = np.min(a[ii][inc])
        else:
            a_acc = brentq(lambda a : r_a(a) - host_rvir(a), low, high)

        # a_peri: first local minimum after accretion
        try:
            # try to find first well-defined local minimum after accretion
            argrelmins = argrelmin(r[ii][inc], order=5)[0]
            relmins = a[ii][inc][argrelmins]
            argrelmins = argrelmins[relmins > a_acc]
            relmins = relmins[relmins > a_acc]
            idx = np.where(a[ii][inc] == relmins[-1])[0][0]
            h = a[ii][inc][idx-2:][0]
            assert h > a_acc
        except (IndexError, AssertionError):
            # if that doesn't work, just use the high=1.0
            # NOTE: in known cases, this means a_peri is close to a_acc or 1.0
            h = 1.0
        a_peri = minimize_scalar(r_a, bounds=(a_acc, h), method='bounded')['x']

        # get other desired properties
        Mvir_a = interp1d(a[ii][inc], Mvir[ii][inc], kind='cubic')
        Vmax_a = interp1d(a[ii][inc], Vmax[ii][inc], kind='cubic')
        accs.append(a_acc)
        peris.append(a_peri)
        dperis.append(float(r_a(a_peri)))
        Maccs.append(float(Mvir_a(a_acc)))
        Vaccs.append(float(Vmax_a(a_acc)))

    # print(sim+": "+str(bad)+" ("'{:.2f}'.format(bad/ii * 100)+"%)")
    # diagnostics
    naughty_frac.append(bad/ii * 100)
    earliest_acc.append(np.min(accs))

    # save the acc/peri values
    df = {'a_peri': peris, 'd_peri': dperis,
            'a_acc': accs, 'M_acc': Maccs, 'V_acc': Vaccs}
    newdata = pd.DataFrame(df, index=subs.index.values)
    newdata.to_pickle('derived_props/'+sim)

# output diagnostics to file
with open("diagnostics/infall_times_diagnostics_elvis.txt", "w") as f:
    f.write("Subhalos that never cross out of Rvir\n")
    for sim,q in zip(list_of_sims('elvis'), naughty_frac):
        f.write(sim + ": " + str(q) + '%\n')
    f.write('\nEarliest accretion (min(a_acc))\n')
    for sim,a in zip(list_of_sims('elvis'), earliest_acc):
        f.write(sim + ": " + str(a) + '\n')
