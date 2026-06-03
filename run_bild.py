#!/usr/bin/env python3
"""Run BILD on overlapping fixed-length sub-trajectories on Henrik's filtered data."""

import os
# Pin BLAS to 1 thread per process so multiprocessing.Pool(nproc) actually
# uses nproc threads total — otherwise OpenBLAS spawns many threads per worker
# and oversubscribes the machine. Must be set before numpy/scipy/bild imports.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import time
import pickle
import argparse
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import numpy as np

import noctiluca as nl
import bild

from utils import *

def _run_one(args):
    chunk_array, save_path, params, dE = args
    save_path = Path(save_path)
    if save_path.exists():
        return save_path.name, 'skipped (exists)'

    np.random.seed((int(time.time() * 1e6) ^ os.getpid()) % (2 ** 32))
    rng_id = np.random.get_state()[1][0]

    try:
        _L = params['L']
        w = np.zeros(3 * _L + 1)
        w[_L] = -1
        w[2 * _L] = 1
        model = bild.models.MultiStateRouse(
            3 * _L + 1, params['D'], params['k'],
            looppositions=[None, (_L, 2 * _L, 1.0 / params['L_looped'])],
            measurement=w,
            localization_error=np.asarray(params['loc_error']) * np.sqrt(2),
        )
        traj = nl.Trajectory(chunk_array)
        result = bild.sample(traj, model, dE=dE, show_progress=False)
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        return save_path.name, f'done (RNG {rng_id})'
    except Exception as e:
        return save_path.name, f'failed: {type(e).__name__}: {e}'


def main():
    parser = argparse.ArgumentParser(
        description='Run BILD on overlapping sub-trajectories.'
    )
    parser.add_argument('--condition', type=str, required=True,
                        help='Condition name, e.g., 340kb_Ce_Cp_None')
    parser.add_argument('--traj_len', type=int, required=True,
                        help='Sub-trajectory length in frames')
    parser.add_argument('--nproc', type=int, default=1,
                        help='Number of worker processes')
    parser.add_argument('--dE', type=float, default=0,
                        help='Initial ∆E value to sample at')
    args = parser.parse_args()

    condition = args.condition
    traj_len = args.traj_len
    nproc = args.nproc
    dE = args.dE

    # load fit parameters and calculate BILD model parameters
    all_params = {}
    conditions = ['340kb_Ce_Cp_None','340kb_Ce_Cp_IAA','340kb_None']
    for condition_temp in conditions:
        with open(f'output/fit_params/{condition_temp}_fit_params.pkl', 'rb') as f:
            all_params[condition_temp] = pickle.load(f)

    G = np.exp(all_params['340kb_None']['log(Γ)'])
    J = np.exp(all_params['340kb_None']['log(J)'])

    # Following eq. (5.22) in Simon's PhD thesis
    L = np.ceil(np.sqrt(4/np.pi)*J/G/np.sqrt(30)).astype(int)
    D = np.pi*L*G**2 / (4*J)
    k = np.pi/4*(L*G/J)**2  # effectively k/gamma (lowercase gamma, that is)

    D_frames = D*30
    k_frames = k*30

    # put in synEP parameters
    tether_length_looped_kb = 1.8
    dist_btwn_CTCF_sites_kb = 335

    # Looped state (from ΔRad21, with genomic rescaling)
    J_dRAD21 = np.exp(all_params['340kb_Ce_Cp_IAA']['log(J)'])
    J_looped = tether_length_looped_kb/dist_btwn_CTCF_sites_kb * J_dRAD21
    L_looped = J_looped * k / D  # from Eq. 5.24 in Simon's thesis

    bild_params_original = {
            'L': L,
            'L_looped': L_looped,
            'D': D_frames,
            'k': k_frames,
        }
    bild_params_downsampled = bild_params_original.copy()
    bild_params_original['loc_error'] = [np.sqrt(np.exp(all_params[condition][f'30s log(σ²) (dim {k})'])/2) for k in (0,1,2)]
    bild_params_downsampled['loc_error'] = [np.sqrt(np.exp(all_params[condition][f'5s log(σ²) (dim {k})'])/2) for k in (0,1,2)]

    original_trajs_nl, downsampled_trajs_nl = get_original_downsampled_trajs(condition, min_traj_len=traj_len, print_output=True)
    data_list_original = [np.array(traj) for traj in original_trajs_nl]
    data_list_downsampled = [np.array(traj) for traj in downsampled_trajs_nl]

    track_names_original = [f'original_track_{n}' for n in range(len(data_list_original))]
    track_names_downsampled = [f'downsampled_track_{n}' for n in range(len(data_list_downsampled))]
    
    save_dir = Path(f'output/bild_profiles/{condition}_initial_dE_{dE}')
    save_dir.mkdir(parents=True, exist_ok=True)

    step = traj_len // 2
    work_items = []

    for traj_array, name in zip(data_list_original, track_names_original):
        if len(traj_array) < traj_len:
            continue
        for start in range(0, len(traj_array) - traj_len + 1, step):
            end = start + traj_len
            chunk = traj_array[start:end]
            save_path = save_dir / f'{name}_bild_result_frame_{start}_to_{end}.pkl'
            work_items.append((chunk, str(save_path), bild_params_original, dE))

    for traj_array, name in zip(data_list_downsampled, track_names_downsampled):
        if len(traj_array) < traj_len:
            continue
        for start in range(0, len(traj_array) - traj_len + 1, step):
            end = start + traj_len
            chunk = traj_array[start:end]
            save_path = save_dir / f'{name}_bild_result_frame_{start}_to_{end}.pkl'
            work_items.append((chunk, str(save_path), bild_params_downsampled, dE))

    print(f'Running BILD on {len(work_items)} sub-trajectories\n' +
          f'Model parameters:\nL={L}, L_looped={L_looped:.3}, D={D_frames:.3}, k={k_frames:.3}\n' +
          f'(traj_len={traj_len}, step={step}, nproc={nproc}) ...')

    failures = 0
    with Pool(nproc) as pool, tqdm(total=len(work_items), unit='chunk') as bar:
        for name, status in pool.imap_unordered(_run_one, work_items):
            if status.startswith('failed'):
                failures += 1
                tqdm.write(f'{name}: {status}')
                bar.set_postfix(failed=failures)
            bar.update(1)

    print(f'Done. {failures} failures out of {len(work_items)} sub-trajectories.')

if __name__ == '__main__':
    main()
