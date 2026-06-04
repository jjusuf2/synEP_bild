import sys, os
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~'))
file_path = Path("~/mpl_theme.py").expanduser()
if file_path.is_file():
    import mpl_theme; mpl_theme.apply();
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import bild
import noctiluca as nl
import bayesmsd
from copy import deepcopy
import pickle

from utils import *

def get_MSD_plateau(dataset):
    return 2*np.nanmean(np.concatenate([np.sum(np.square(np.array(traj)), 1) for traj in dataset]))

print(f'Loading data')

all_data = {}

for condition in ['340kb_Ce_Cp_None', '340kb_None', '340kb_Ce_Cp_IAA']:
    trajs_30s_nl, trajs_5s_nl = get_30s_5s_trajs(condition, min_traj_len=100)
    all_data[condition] = {}
    all_data[condition]['30s'] = trajs_30s_nl
    all_data[condition]['5s'] = trajs_5s_nl

fig, ax = plt.subplots()

for condition, color in zip(['340kb_Ce_Cp_IAA', '340kb_None', '340kb_Ce_Cp_None'], ['C4', 'C3', 'C0']):
    msd_30s = nl.analysis.MSD(all_data[condition]['30s'])
    msd_5s = nl.analysis.MSD(all_data[condition]['5s'])
    ax.plot(30*np.arange(1,len(msd_30s)), msd_30s[1:], color=color, label=condition)
    ax.plot(5*np.arange(1,len(msd_5s)), msd_5s[1:], color=color, linestyle='--')
    ax.axhline(get_MSD_plateau(all_data[condition]['30s']), 0, 1, color=color)
    ax.axhline(get_MSD_plateau(all_data[condition]['5s']), 0, 1, color=color, linestyle='--')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylabel('MSD [µm$^2$]')
ax.set_xlabel('Lag time ∆t [s]')

ax.legend(fontsize=8)

plt.savefig('bild_outputs/figures/MSDs_raw.png');
plt.savefig('bild_outputs/figures/MSDs_raw.svg');

for condition in ['340kb_Ce_Cp_None', '340kb_None', '340kb_Ce_Cp_IAA']:

    print(f'Fitting MSD for condition {condition}')

    fit_30s = bayesmsd.lib.TwoLocusRouseFit(all_data[condition]['30s'])
    fit_5s  = bayesmsd.lib.TwoLocusRouseFit(all_data[condition]['5s'])

    joint_fit = bayesmsd.FitGroup({
        '30s'  : fit_30s,
        '5s' : fit_5s,
    })

    joint_fit.parameters['log(Γ)'] = deepcopy(fit_30s.parameters['log(Γ) (dim 0)'])
    joint_fit.parameters['log(J)'] = deepcopy(fit_5s.parameters['log(J) (dim 0)'])

    # hacky…
    def patch_initial_params(self=joint_fit):
        params = type(self).initial_params(self)
        logG = [val for key, val in params.items() if 'log(Γ)' in key][0]
        logJ = [val for key, val in params.items() if 'log(J)' in key][0]
        params['log(Γ)'] = logG
        params['log(J)'] = logJ
        return params
    joint_fit.initial_params = patch_initial_params

    for name in joint_fit.fits_dict:
        joint_fit.parameters[f'{name} log(Γ) (dim 0)'].fix_to = 'log(Γ)'
        joint_fit.parameters[f'{name} log(J) (dim 0)'].fix_to = 'log(J)'

    result = joint_fit.run(show_progress=True)

    print()
    print(f'Parameters for {condition}:')

    print(f"Γ (µm^2 s^(-0.5)) : {np.exp(result['params']['log(Γ)']):.3}")
    print(f"J (µm^2)          : {np.exp(result['params']['log(J)']):.3}")

    print("30s    σ (nm): ", end='')
    [print(f"{np.sqrt(np.exp(result['params'][f'30s log(σ²) (dim {d})']))*1000/np.sqrt(2):.2f}", end=' ') for d in (0, 1, 2)];
    print()

    print("5s σ (nm): ", end='')
    [print(f"{np.sqrt(np.exp(result['params'][f'5s log(σ²) (dim {d})']))*1000/np.sqrt(2):.2f}", end=' ') for d in (0, 1, 2)];
    print()

    with open(f'bild_outputs/fit_params/{condition}_fit_params.pkl', 'wb') as f:
        pickle.dump(result['params'], f)
