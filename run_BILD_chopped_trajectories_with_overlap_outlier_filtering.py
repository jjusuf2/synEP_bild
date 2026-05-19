#!/usr/bin/env python3
"""Run BILD on overlapping fixed-length sub-trajectories from G7B8G2_GSK data."""

import os
import time
import pickle
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import noctiluca as nl
import bild
from tqdm import tqdm

CONDITION_NAME = 'G7B8G2_GSK'
SAVE_BASE = Path('/mnt/md0/jjusuf/bild/final_profiles_save_all_overlap_outlier_filtering_20260519')

L = 16
L_LOOPED = 0.334

def _run_one(args):
    chunk_array, save_path, params = args
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
        result = bild.sample(traj, model, show_progress=False)
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        return save_path.name, f'done (RNG {rng_id})'
    except Exception as e:
        return save_path.name, f'failed: {type(e).__name__}: {e}'


def main():
    parser = argparse.ArgumentParser(
        description='Run BILD on overlapping sub-trajectories.'
    )
    parser.add_argument('--delta_t', type=int, required=True, choices=[5, 30],
                        help='Time step in seconds (5 or 30)')
    parser.add_argument('--traj_len', type=int, required=True,
                        help='Sub-trajectory length in frames')
    parser.add_argument('--nproc', type=int, default=1,
                        help='Number of worker processes')
    args = parser.parse_args()

    delta_t = args.delta_t
    traj_len = args.traj_len
    nproc = args.nproc

    if delta_t == 30:
        k = 8.04
        D = 0.01194
        loc_error = np.array([0.0443, 0.0443, 0.0444])
    elif delta_t == 5:
        k = 1.34
        D = 0.00199
        loc_error = np.array([0.0425, 0.0425, 0.0449])
    else:
        raise ValueError("∆t should be 5s or 30s!")

    params = {
        'L': L,
        'L_looped': L_LOOPED,
        'D': D,
        'k': k,
        'loc_error': loc_error.tolist(),
    }

    print(f'Loading tracks for condition={CONDITION_NAME}, delta_t={delta_t}s ...')

    data = np.load(f'/mnt/md0/jjusuf/bild/20260519_Henrik_filtered_data/{delta_t}s_340kb_Ce_Cp_None.npy')
    data_list = list(data)
    for i in range(len(data_list)):
        traj = data_list[i]

        timepoint_all_nans = np.all(np.isnan(traj), axis=1)

        cutoff = len(traj)

        for j in range(len(traj)):
            if np.all(timepoint_all_nans[j:]):
                cutoff = j
                break

        traj = traj[:cutoff, :]
        data_list[i] = traj

    track_names = [f'track_{n}' for n in range(len(data_list))]
    print(f'Loaded {len(data_list)} trajectories.')

    save_dir = SAVE_BASE / f'{CONDITION_NAME}_{delta_t}s'
    save_dir.mkdir(parents=True, exist_ok=True)

    step = traj_len // 2
    work_items = []

    for traj_array, name in zip(data_list, track_names):
        if len(traj_array) < traj_len:
            continue
        for start in range(0, len(traj_array) - traj_len + 1, step):
            end = start + traj_len
            chunk = traj_array[start:end]
            save_path = save_dir / f'{name}_bild_result_frame_{start}_to_{end}.pkl'
            work_items.append((chunk, str(save_path), params))

    print(f'Running BILD on {len(work_items)} sub-trajectories '
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
