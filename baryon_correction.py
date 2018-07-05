import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils import *

# load the Nadler+ 2018 RF classifier
with open('data/nadler2018_rf.sav', 'rb') as f:
    nadler_rf = pickle.load(f)

for sim in list_of_sims('elvis'):
    subs = pd.read_pickle('derived_props/'+sim)
    assert (subs.a_peri >= subs.a_acc).all()

    # run classifier, save likelihood of survival
    features = subs[['d_peri', 'a_acc', 'V_acc', 'M_acc', 'a_peri']]
    subs['nadler2018'] = nadler_rf.predict_proba(features)[:,0]

    print(sim+": "+'{:0.2f}'.format(np.sum(subs.nadler2018>0.5)/len(subs)))
    subs.to_pickle('derived_props/'+sim)
