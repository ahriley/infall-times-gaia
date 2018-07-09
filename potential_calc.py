import numpy as np
from utils import *
from scipy.integrate import quad
import pickle
from scipy.interpolate import interp1d

# get radii used by Shea
# NOTE: first midpoint is equal step from bin 1 as from bin 2 in log
r = radii_shea()

def potential_mltr(sim, subs, halos):
    halo_names = sim.split('&')

    # get mltr profile of hosts
    mapper = {}
    for i in range(len(halo_names)):
        with open(ELVIS_DIR+'profiles/'+halo_names[i]+'.pkl', 'rb') as f:
            mltr = pickle.load(f)['M.lt.r']

        # interpolate the function
        mltr_interp = interp1d(r, mltr, kind='cubic')
        assert np.allclose(mltr, mltr_interp(r))
        mapper[halos.iloc[i].name] = mltr_interp

    # potential for each subhalo (matched to the correct host)
    potential = []
    for index, sub in subs.iterrows():
        func = lambda r: (G/kpc2km)*mapper[sub.hostID](r)/r**2
        potential.append(-quad(func, sub.r, r[-1])[0])
    return potential

for sim in list_of_sims('elvis'):
    print(sim)
    halos, subs = load_elvis(sim=sim, processed=True)
    props = pd.read_pickle('derived_props/'+sim)

    # approx potential as NFW, concentrations from ELVIS paper Table 2
    halos['c'] = halo_concentrations(sim)
    halos['Rs'] = halos['Rvir']/halos['c']
    halos['rho0'] = halos.Mvir/(4*np.pi*halos.Rs**3)
    halos['rho0'] /= np.log(1+halos.c) - (halos.c/(1+halos.c))

    # calculate potential multiple ways, save result
    props['pot_NFW'] = potentialNFW(subs, halos)
    props['pot_NFW_1000'] = potentialNFW_R0(subs, halos, R0=1000)
    props['pot_NFW_approx_mltr'] = potentialNFW_R0(subs, halos, R0=r[-1])
    if 'HiRes' not in sim:
        # only have mass profiles for fiducial res
        props['pot_mltr'] = potential_mltr(sim, subs, halos)
    props.to_pickle('derived_props/'+sim)
