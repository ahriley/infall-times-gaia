import numpy as np
import pandas as pd
import glob

Mpc2km = 3.086*10**19
km2kpc = 10**3/Mpc2km
kpc2km = 1/km2kpc

ELVIS_DIR = '/Users/alexanderriley/Desktop/elvis/'
VL2_DIR = '/Users/alexanderriley/Desktop/vl2/'

def center_on_hosts(hosts, subs):
    centered = subs.copy()
    centered['x'] = subs.x.values - hosts.loc[subs['hostID']].x.values
    centered['y'] = subs.y.values - hosts.loc[subs['hostID']].y.values
    centered['z'] = subs.z.values - hosts.loc[subs['hostID']].z.values
    centered['vx'] = subs.vx.values - hosts.loc[subs['hostID']].vx.values
    centered['vy'] = subs.vy.values - hosts.loc[subs['hostID']].vy.values
    centered['vz'] = subs.vz.values - hosts.loc[subs['hostID']].vz.values

    return centered

def compute_spherical_hostcentric_sameunits(df):
    x, y, z = df.x, df.y, df.z
    vx, vy, vz = df.vx, df.vy, df.vz

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    v_r = (x*vx + y*vy + z*vz) / r
    v_theta = r*((z*(x*vx+y*vy)-vz*(x**2+y**2)) / (r**2*np.sqrt(x**2+y**2)))
    v_phi = r*np.sin(theta)*((x*vy - y*vx) / (x**2 + y**2))
    v_t = np.sqrt(v_theta**2 + v_phi**2)

    # check that coordinate transformation worked
    cart = np.sqrt(vx**2 + vy**2 + vz**2)
    sphere = np.sqrt(v_r**2 + v_theta**2 + v_phi**2)
    sphere2 = np.sqrt(v_r**2 + v_t**2)
    assert np.isclose(cart, sphere).all()
    assert np.isclose(cart, sphere2).all()

    df2 = df.copy()
    df2['r'], df2['theta'], df2['phi'] = r, theta, phi
    df2['v_r'], df2['v_theta'], df2['v_phi'] = v_r, v_theta, v_phi
    df2['v_t'] = v_t

    return df2

def get_halos_at_scale_elvis(sim, a):
    sim_dir = ELVIS_DIR+'tracks/'+sim+'/'
    index = np.argmin(np.abs(list_of_scales('elvis', sim) - a))

    # get halo properties at that redshift
    props = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Vmax', 'Mvir', 'Rvir', 'Rs', 'pID', 'ID']
    keys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Vmax', 'Mvir', 'Rvir', 'Rs', 'pID', 'zID']
    df = {}
    for key,prop in zip(keys,props):
        prop_list = []
        with open(sim_dir+prop+'.txt') as f:
            lines = f.readlines()[1:]
            for line in lines:
                split = np.array(line.split()).astype(float)
                prop_list.append(split[index])
        df[key] = prop_list

    # IDs will be for redshift 0
    IDs = []
    with open(sim_dir+'ID.txt') as f:
        lines = f.readlines()[1:]
        for line in lines:
            split = np.array(line.split()).astype(int)
            IDs.append(split[0])

    df = pd.DataFrame(df, index=IDs)
    df.index.name = 'ID'
    # use comoving distances for computing pericenter, infall
    # df.x *= a
    # df.y *= a
    # df.z *= a
    # df.Rvir *= a
    return df

def halo_concentrations(sim):
    halos = sim.split('&')
    map = {'Hera': 7.9, 'Zeus': 5.6, 'Scylla': 6.4, 'Charybdis': 7.6,
            'Romulus': 9.6, 'Remus': 12.3, 'Orion': 5.3, 'Taurus': 10.9,
            'Kek': 13.7, 'Kauket': 9.6, 'Hamilton': 9.9, 'Burr': 10.6,
            'Lincoln': 8.4, 'Douglas': 9.6, 'Serena': 14.4, 'Venus': 1.8,
            'Sonny': 2.4, 'Cher': 11.0, 'Hall': 10.3, 'Oates': 8.4,
            'Thelma': 7.1, 'Louise': 17.0, 'Siegfried': 6.5, 'Roy': 11.1,
            'iHera': 7.9, 'iZeus': 5.5, 'iScylla': 9.9, 'iCharybdis': 13.7,
            'iRomulus': 11.3, 'iRemus': 8.0, 'iOrion': 4.9, 'iTaurus': 10.4,
            'iKek': 5.5, 'iKauket': 11.1, 'iHamilton': 14.2, 'iBurr': 13.6,
            'iLincoln': 13.8, 'iDouglas': 16.1, 'iSerena': 11.4,
            'iVenus': 14.3, 'iSonny': 4.5, 'iCher': 6.4, 'iHall': 6.0,
            'iOates': 8.4, 'iThelma': 9.6, 'iLouise': 8.4, 'iSiegfried': 11.1,
            'iRoy': 3.9, 'iScylla_HiRes': 9.5, 'iKauket_HiRes': 11.8,
            'iHall_HiRes': 5.8}
    return [map[halo] for halo in halos]

def list_of_sims(sim):
    if sim != 'elvis':
        raise NotImplementedErorr("Simulation should be 'elvis' for now")
    files = glob.glob(ELVIS_DIR+'*.txt')
    files.remove(ELVIS_DIR+'README.txt')
    return [f[len(ELVIS_DIR):-4] for f in files]

def list_of_scales(suite, sim=None):
    if suite == 'elvis':
        sim_dir = ELVIS_DIR+'tracks/'+sim+'/'
        with open(sim_dir+'scale.txt') as f:
            scale_list = np.array(f.readlines()[1].split()).astype(float)
    elif suite == 'vl2':
        with open(VL2_DIR+'tracks/stepToTimeVL2.txt') as f:
            lines = f.readlines()[3:]
        scale_list = np.array([float(line.split()[1]) for line in lines])[::-1]
    return scale_list

def load_elvis(sim):
    filename = ELVIS_DIR+sim+'.txt'

    # read in the data
    with open(filename) as f:
        id, x, y, z, vx, vy, vz, vmax, vpeak, mvir = [[] for i in range(10)]
        mpeak, rvir, rmax, apeak, mstar, mstar_b = [[] for i in range(6)]
        npart, pid, upid = [], [], []
        for line in f:
            if line[0] == '#':
                continue
            items = line.split()
            id.append(int(items[0]))
            x.append(float(items[1]))
            y.append(float(items[2]))
            z.append(float(items[3]))
            vx.append(float(items[4]))
            vy.append(float(items[5]))
            vz.append(float(items[6]))
            vmax.append(float(items[7]))
            vpeak.append(float(items[8]))
            mvir.append(float(items[9]))
            mpeak.append(float(items[10]))
            rvir.append(float(items[11]))
            rmax.append(float(items[12]))
            apeak.append(float(items[13]))
            mstar.append(float(items[14]))
            mstar_b.append(float(items[15]))
            npart.append(int(items[16]))
            pid.append(int(items[17]))
            upid.append(int(items[18]))

    # convert to pandas format
    df = {'PID': pid, 'hostID': upid, 'npart': npart, 'apeak': apeak,
            'M_dm': mvir, 'M_star': mstar, 'Mpeak': mpeak, 'Mstar_b': mstar_b,
            'Vmax': vmax, 'Vpeak': vpeak, 'Rvir': rvir, 'Rmax': rmax,
            'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
    df = pd.DataFrame(df, index=id)
    df.index.name = 'ID'
    return df

def load_vl2(scale):
    if scale == 1.0:
        df = pd.read_table(VL2_DIR+'vltwosubs.txt',sep=' ',header=0,index_col='id')
        df = df.drop(columns=['rVmax[kpc]', 'M<300pc[Msun]', 'M<600pc[Msun]'])
        map = {'GCdistance[kpc]': 'r', 'peakVmax[km/s]': 'Vpeak', 'Vmax[km/s]': 'Vmax',
                'Mtidal[Msun]': 'Mvir', 'rtidal[kpc]': 'Rvir',
                'x_rel[kpc]': 'x', 'y_rel[kpc]': 'y', 'z_rel[kpc]': 'z',
                'vx_rel[kpc]': 'vx', 'vy_rel[kpc]': 'vy', 'vz_rel[kpc]': 'vz'}
        df.rename(columns=map, inplace=True)
        df.sort_values('Mvir', ascending=False, inplace=True)
        return df[(df.r < df.iloc[0].Rvir) & (df.Vmax > 5)]

    sim_dir = VL2_DIR+'tracks/'
    index = np.argmin(np.abs(list_of_scales('vl2') - scale))

    # get halo properties at that redshift
    props = ['X','Y','Z','VX','VY','VZ','Vmax','Mtidal','Rtidal','GCdistance']
    df = {}
    for prop in props:
        prop_list = []
        with open(sim_dir+'prog'+prop+'.txt') as f:
            lines = f.readlines()
            for line in lines:
                split = np.array(line.split()).astype(float)
                prop_list.append(split[index])
        df[prop.lower()] = prop_list

    # IDs will be for redshift 0
    with open(VL2_DIR+'vltwosubs.txt') as f:
        lines = f.readlines()[1:]
        IDs = np.array([int(line.split()[0]) for line in lines])

    df = pd.DataFrame(df, index=IDs)
    map = {'gcdistance': 'r','vmax': 'Vmax','mtidal': 'Mvir','rtidal': 'Rvir'}
    df.rename(columns=map, inplace=True)
    df.index.name = 'ID'
    df.x *= scale
    df.y *= scale
    df.z *= scale
    df.Rvir *= scale
    return df.loc[load_vl2(1.0).index]

def potentialNFW(subhalos, hosts):
    r = subhalos.r*kpc2km
    Rs = hosts.loc[subs['hostID']].Rs.values*kpc2km
    rho0 = hosts.loc[subs['hostID']].rho0.values/(kpc2km**3)
    return -4*np.pi*G*rho0*Rs**3*np.log(1+r/Rs)/r
