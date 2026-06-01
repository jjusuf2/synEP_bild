import numpy as np
import pandas as pd
import os

import noctiluca as nl
import bayesmsd

import sys
from copy import deepcopy

DATA_DIR = '/mnt/md0/jjusuf/bild/20260519_Henrik_filtered_data'

# ── parse argv ──────────────────────────────────────────────────────────────
# The user passes the full condition name including ∆t, e.g. "30s_340kb_None".
# We strip the leading "<Ns>_" token to recover the base condition, then load
# both the 30 s and 5 s .npy files (if they exist).

if len(sys.argv) < 2:
    print('Usage: python joint_MSD_fit.py <condition>  (e.g. 340kb_None)')
    sys.exit(1)

base_condition = sys.argv[1]

# ── helpers ─────────────────────────────────────────────────────────────────

def load_tagged_set(dt_label: str, dt_seconds: float):
    """
    Load trajectories from  <DATA_DIR>/<dt_label>_<base_condition>.npy
    and return a TaggedSet, or None if the file does not exist.

    The .npy file has shape (n_trajs, n_frames, 3) in nm;
    values are divided by 1000 to yield µm.
    NaN entries (missing frames) are preserved for noctiluca.
    """
    path = os.path.join(DATA_DIR, f'{dt_label}_{base_condition}.npy')
    if not os.path.isfile(path):
        return None

    arr = np.load(path)          # (n_trajs, n_frames, 3)  [nm]
    arr = arr / 1000.0           # → µm

    ts = nl.TaggedSet()
    for traj_arr in arr:
        traj = nl.Trajectory(traj_arr)
        traj.meta['Δt'] = dt_seconds
        ts.add(traj)

    print(f'  loaded {len(arr)} trajectories from {path}')
    return ts


# ── load data ────────────────────────────────────────────────────────────────

print(f'Base condition: {base_condition}')

data_30s = load_tagged_set('30s', 30)
data_5s  = load_tagged_set('5s',  5)

if data_30s is None and data_5s is None:
    raise FileNotFoundError(
        f'No data files found for base condition "{base_condition}" in {DATA_DIR}')

# ── set up fits ──────────────────────────────────────────────────────────────
# The following approach is adapted from Simon's code from the MINFLUX paper:
#   https://github.com/ahansenlab/chromatin_dynamics/blob/main/03_fitting/01_01_fit_MEF.ipynb

if data_30s is not None and data_5s is not None:
    # ── joint fit (preferred when both ∆t variants are available) ────────────
    print('Running joint fit (30 s + 5 s) …')

    fit_30s = bayesmsd.lib.TwoLocusRouseFit(data_30s)
    fit_5s  = bayesmsd.lib.TwoLocusRouseFit(data_5s)

    # for lattice, don't fix sigma_x = sigma_y
    # fit_30s.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'
    # fit_5s.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'

    joint_fit = bayesmsd.FitGroup({
        '5s'  : fit_5s,
        '30s' : fit_30s,
    })

    joint_fit.parameters['log(Γ)'] = deepcopy(fit_5s.parameters['log(Γ) (dim 0)'])
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
    print(f'Results of joint fit for {base_condition}:')
    for key in ['5s log(σ²) (dim 0)', '5s log(σ²) (dim 1)', '5s log(σ²) (dim 2)',
                '30s log(σ²) (dim 0)', '30s log(σ²) (dim 1)', '30s log(σ²) (dim 2)',
                'log(Γ)', 'log(J)']:
        key_human = (key
                     .replace('log(σ²)', 'σ²')
                     .replace('log(J)',  'J')
                     .replace('log(Γ)', 'Γ'))
        print(f'{key_human:<20} = {np.exp(result["params"][key]):.3}')

    pd.Series(result['params']).to_csv(f'params_joint_fit_{base_condition}.csv')

else:
    # ── single-∆t fit (fallback) ─────────────────────────────────────────────
    data, dt_label = (data_30s, '30s') if data_30s is not None else (data_5s, '5s')
    print(f'Only {dt_label} data found — running single-dataset fit …')

    fit = bayesmsd.lib.TwoLocusRouseFit(data)
    result = fit.run(show_progress=True)

    print()
    print(f'Results of single fit ({dt_label}) for {base_condition}:')
    for key in [f'log(σ²) (dim 0)', 'log(σ²) (dim 1)', 'log(σ²) (dim 2)',
                'log(Γ) (dim 0)', 'log(J) (dim 0)']:
        key_human = (key
                     .replace('log(σ²)', 'σ²')
                     .replace('log(J)',  'J')
                     .replace('log(Γ)', 'Γ'))
        print(f'{key_human:<20} = {np.exp(result["params"][key]):.3}')

    pd.Series(result['params']).to_csv(f'params_single_fit_{dt_label}_{base_condition}.csv')
