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

def get_trajs(dt, condition):
    all_trajs = np.load(f'data/{dt}s_{condition}.npy')
    trajs = []
    for track_idx in range(len(all_trajs)):
        traj = all_trajs[track_idx,:,:]
        row_all_nan = np.all(np.isnan(traj), 1)
        non_nan_rows = np.where(~row_all_nan)[0]
        endpoint = non_nan_rows[-1] + 1 if len(non_nan_rows) > 0 else 0
        traj = traj[:endpoint]
        traj = traj / 1000  # convert to µm
        trajs.append(traj)
    return trajs

def downsample_trajs(trajs, sample_every):
    trajs_downsampled = []
    for i in range(len(trajs)):
        traj = trajs[i]
        trajs_downsampled.append(traj[::sample_every,:])
    return trajs_downsampled

def get_original_downsampled_trajs(condition, min_traj_len=50):

    print(f'Condition name: {condition}')

    original_trajs = get_trajs(30, condition)
    print(f'Loaded {len(original_trajs)} trajectories with ∆t=30s')
    original_trajs = [traj for traj in original_trajs if len(traj)>min_traj_len]
    print(f'  ↳ filtered to {len(original_trajs)} trajectories with ≥{min_traj_len} frames')
    original_trajs_hrs = np.sum([len(traj) for traj in original_trajs])*30/60/60
    print(f'    ({original_trajs_hrs:.0f} hours total)')

    downsampled_trajs = downsample_trajs(get_trajs(5, condition), 6)
    print(f'Loaded {len(downsampled_trajs)} trajectories with ∆t=5s and downsampled')
    downsampled_trajs = [traj for traj in downsampled_trajs if len(traj)>min_traj_len]
    print(f'  ↳ filtered to {len(downsampled_trajs)} trajectories with ≥{min_traj_len} frames')
    downsampled_trajs_hours = np.sum([len(traj) for traj in downsampled_trajs])*30/60/60
    print(f'    ({downsampled_trajs_hours:.0f} hours total)')
    print()

    # make noctiluca TaggedSets

    original_trajs_nl = nl.TaggedSet()
    for traj_arr in original_trajs:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = 30
        original_trajs_nl.add(traj)

    downsampled_trajs_nl = nl.TaggedSet()
    for traj_arr in downsampled_trajs:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = 30
        downsampled_trajs_nl.add(traj)

    return original_trajs_nl, downsampled_trajs_nl

def get_MSD_plateau(dataset):
    return 2*np.nanmean(np.concatenate([np.sum(np.square(np.array(traj)), 1) for traj in dataset]))


print(f'Loading data')

all_data = {}

for condition in ['340kb_Ce_Cp_None', '340kb_None', '340kb_Ce_Cp_IAA']:
    original_trajs_nl, downsampled_trajs_nl = get_original_downsampled_trajs(condition, min_traj_len=100)
    all_data[condition] = {}
    all_data[condition]['original'] = original_trajs_nl
    all_data[condition]['downsampled'] = downsampled_trajs_nl

fig, ax = plt.subplots()

for condition, color in zip(['340kb_Ce_Cp_IAA', '340kb_None', '340kb_Ce_Cp_None'], ['C4', 'C3', 'C0']):
    original_msd = nl.analysis.MSD(all_data[condition]['original'])
    downsampled_msd = nl.analysis.MSD(all_data[condition]['downsampled'])
    ax.plot(30*np.arange(len(original_msd)), original_msd, color=color, label=condition)
    ax.plot(30*np.arange(len(downsampled_msd)), downsampled_msd, color=color, linestyle='--')
    ax.axhline(get_MSD_plateau(all_data[condition]['original']), 0, 1, color=color)
    ax.axhline(get_MSD_plateau(all_data[condition]['downsampled']), 0, 1, color=color, linestyle='--')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylabel('MSD [µm$^2$]')
ax.set_xlabel('Lag time ∆t [s]')

ax.legend(fontsize=8)

plt.savefig('output/figures/MSDs_raw.png');
plt.savefig('output/figures/MSDs_raw.svg');


for condition in ['340kb_Ce_Cp_None', '340kb_None', '340kb_Ce_Cp_IAA']:

    print(f'Fitting MSD for condition {condition}')

    fit_original = bayesmsd.lib.TwoLocusRouseFit(all_data[condition]['original'])
    fit_downsampled  = bayesmsd.lib.TwoLocusRouseFit(all_data[condition]['downsampled'])

    joint_fit = bayesmsd.FitGroup({
        'original'  : fit_original,
        'downsampled' : fit_downsampled,
    })

    joint_fit.parameters['log(Γ)'] = deepcopy(fit_original.parameters['log(Γ) (dim 0)'])
    joint_fit.parameters['log(J)'] = deepcopy(fit_downsampled.parameters['log(J) (dim 0)'])

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

    print("original    σ (nm): ", end='')
    [print(f"{np.sqrt(np.exp(result['params'][f'original log(σ²) (dim {d})']))*1000/np.sqrt(2):.2f}", end=' ') for d in (0, 1, 2)];
    print()

    print("downsampled σ (nm): ", end='')
    [print(f"{np.sqrt(np.exp(result['params'][f'downsampled log(σ²) (dim {d})']))*1000/np.sqrt(2):.2f}", end=' ') for d in (0, 1, 2)];
    print()

    with open(f'output/fit_params/{condition}_fit_params.pkl', 'wb') as f:
        pickle.dump(result['params'], f)
