import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import argrelmin

# present day properties
halo, subs = load_vl2(scale=1.0, processed=True)
halo = halo.iloc[0]

# check that each subhalo is currently within the halo's Rvir
assert (subs.r < halo.Rvir).all()

# read in the evolutionary tracks
a = np.loadtxt(VL2_DIR+'/tracks/stepToTimeVL2.txt')[:,1][::-1]
x = np.loadtxt(VL2_DIR+'/tracks/progX.txt')*a*1000*40
y = np.loadtxt(VL2_DIR+'/tracks/progY.txt')*a*1000*40
z = np.loadtxt(VL2_DIR+'/tracks/progZ.txt')*a*1000*40
pos = np.stack((x,y,z), axis=2)
IDs = np.loadtxt(VL2_DIR+'vltwosubs.txt', skiprows=1)[:,0].astype(int)
rvir = np.loadtxt(VL2_DIR+'/tracks/progRtidal.txt')
Mvir = np.loadtxt(VL2_DIR+'/tracks/progMtidal.txt')
Vmax = np.loadtxt(VL2_DIR+'/tracks/progVmax.txt')

# get positional indices of wanted subhalos
indices = []
for ID in subs.index:
    ii = np.where(IDs == ID)[0][0]
    indices.append(ii)
indices = np.array(indices)

# center on host
halo_index = np.where(IDs == halo.name)[0][0]
pos -= pos[halo_index]
r = np.sqrt(pos[:,:,0]**2 + pos[:,:,1]**2 + pos[:,:,2]**2)
assert (r[:,0][indices] < rvir[halo_index][0]).all()

# interpolate Rvir for the halo
rvir_halo = rvir[halo_index]
host_rvir = interp1d(a, rvir_halo, kind='cubic')

# find a_acc, a_peri for subhalos of interest
accs, peris, dperis, Maccs, Vaccs = [], [], [], [], []
for ii in indices:
    # a_acc: most recent crossing of Rvir
    r_a = interp1d(a, r[ii], kind='cubic')
    low = a[np.argmax(r[ii] - host_rvir(a) > 0)]
    high = 1.0
    assert low < high
    a_acc = brentq(lambda a : r_a(a) - host_rvir(a), low, high)

    # a_peri: first local minimum after accretion
    try:
        argrelmins = argrelmin(r[ii], order=1)[0]
        relmins = a[argrelmins]
        argrelmins = argrelmins[relmins > a_acc]
        relmins = relmins[relmins > a_acc]
        idx = np.where(a == relmins[-1])[0][0]
        h = a[idx-2:][0]
        assert h > a_acc
    except (IndexError, AssertionError):
        # if that doesn't work, just use the high=1.0
        # NOTE: in known cases, this means a_peri is close to a_acc or 1.0
        h = 1.0
    a_peri = minimize_scalar(r_a, bounds=(a_acc, h), method='bounded')['x']

    Mvir_a = interp1d(a, Mvir[ii], kind='cubic')
    Vmax_a = interp1d(a, Vmax[ii], kind='cubic')
    accs.append(a_acc)
    peris.append(a_peri)
    dperis.append(float(r_a(a_peri)))
    Maccs.append(float(Mvir_a(a_acc)))
    Vaccs.append(float(Vmax_a(a_acc)))

# save the acc/peri values
df = {'a_peri': peris, 'd_peri': dperis,
        'a_acc': accs, 'M_acc': Maccs, 'V_acc': Vaccs}
newdata = pd.DataFrame(df, index=subs.index.values)
newdata.to_pickle('derived_props/vl2')
