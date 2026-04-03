import numpy as np
import pandas as pd 

import noctiluca as nl
import bayesmsd

import sys
from copy import deepcopy

condition_name = sys.argv[1]

## load the tracks
all_tracks = pd.read_csv('/mnt/md0/jjusuf/bild/synEP_bild/all_tracks.csv')

## write a function to grab the data for a given condition and ∆t (in the proper format for BILD)
def generate_data_list(condition, delta_t):
    filenames = all_tracks.loc[(all_tracks['condition']==condition) & (all_tracks['delta_t']==delta_t), 'path']

    data_list = []

    for filename in filenames:
        table_full = pd.read_csv(filename)
        table_xyz = np.array([table_full['pro_x (nm)']-table_full['enh_x (nm)'],
                            table_full['pro_y (nm)']-table_full['enh_y (nm)'],
                            table_full['pro_z (nm)']-table_full['enh_z (nm)']]).T / 1000  # convert to µm
        data_list.append(table_xyz)

    return data_list

datasets_nl = {}

## get the data

data_30s = nl.TaggedSet()
traj_arrays = generate_data_list(condition_name, 30)
for arr in traj_arrays:
    traj = nl.Trajectory(arr)
    traj.meta['Δt'] = 30
    data_30s.add(traj)
    
data_5s = nl.TaggedSet()
traj_arrays = generate_data_list(condition_name, 5)
for arr in traj_arrays:
    traj = nl.Trajectory(arr)
    traj.meta['Δt'] = 5
    data_5s.add(traj)

## set up the individual fits
# the following code is emulated from Simon's code from the MINFLUX paper:
#   https://github.com/ahansenlab/chromatin_dynamics/blob/main/03_fitting/01_01_fit_MEF.ipynb

fit_30s = bayesmsd.lib.TwoLocusRouseFit(data_30s)
fit_30s.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'

fit_5s = bayesmsd.lib.TwoLocusRouseFit(data_5s)
fit_5s.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'

## now set up the joint fit

joint_fit = bayesmsd.FitGroup({
    '5s'  : fit_5s,
    '30s' : fit_30s,
})

joint_fit.parameters['log(Γ)'] = deepcopy(fit_5s.parameters['log(Γ) (dim 0)'])
joint_fit.parameters['log(J)'] = deepcopy(fit_5s.parameters['log(J) (dim 0)'])

# hacky...
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

## run joint fit

result = joint_fit.run(show_progress=True)

print()
print(f'Results of joint fit for {condition_name}:')
for key in ['5s log(σ²) (dim 0)','5s log(σ²) (dim 2)',
            '30s log(σ²) (dim 0)','30s log(σ²) (dim 2)',
            'log(Γ)','log(J)']:
    key_without_log = key.replace('log(σ²)','σ²')
    key_without_log = key_without_log.replace('log(J)','J')
    key_without_log = key_without_log.replace('log(Γ)','Γ')
    print(f'{key_without_log:<14} = {np.exp(result["params"][key]):.3}')

pd.Series(result['params']).to_csv(f'params_joint_fit_{condition_name}.csv')
