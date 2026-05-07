"""Run BILD on chopped trajectories, with parallelism and restart support.

The output of each BILD run -- the full ``SimulationResult`` -- is pickled to
its own file, so a re-invocation of the pipeline only re-runs chunks whose
result file is missing.
"""
import os
import time
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import bild
import noctiluca as nl
from tqdm import tqdm


def chop(arr, chunk_length):
    """Split ``arr`` into non-overlapping windows of ``chunk_length`` frames.

    Works for 1D (profile) or 2D (trajectory) input; the trailing partial
    window is dropped.
    """
    n = len(arr) // chunk_length
    return [arr[i * chunk_length:(i + 1) * chunk_length] for i in range(n)]


def _build_model(params):
    """Construct a fresh MultiStateRouse model from a parameter dict."""
    L = params['L']
    w = np.zeros(3 * L + 1)
    w[L] = -1
    w[2 * L] = 1
    return bild.models.MultiStateRouse(
        3 * L + 1, params['D'], params['k'],
        looppositions=[None, (L, 2 * L, 1.0 / params['L_looped'])],
        measurement=w,
        # The factor of sqrt(2) compensates for taking a difference of two
        # independently-noisy positions when forming the L--2L vector.
        localization_error=np.asarray(params['localization_error']) * np.sqrt(2),
    )


def _run_one(args):
    chunk_array, save_path, params = args
    save_path = Path(save_path)
    if save_path.exists():
        return save_path.name, 'skipped (exists)'

    # Re-seed inside the worker so multiprocessed runs don't share RNG state.
    np.random.seed((int(time.time() * 1e6) ^ os.getpid()) % (2 ** 32))
    rng_id = np.random.get_state()[1][0]

    try:
        model = _build_model(params)
        traj = nl.Trajectory(chunk_array)
        result = bild.sample(traj, model, show_progress=False)
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        return save_path.name, f'done (RNG {rng_id})'
    except Exception as e:
        return save_path.name, f'failed: {type(e).__name__}: {e}'


def run_bild_on_chunks(chunks, params, save_dir, nproc, label=''):
    """Run BILD on each chunk and pickle the SimulationResult per chunk.

    Already-completed chunks (whose pickle file exists) are skipped, so the
    function is safe to re-invoke after a crash or interruption.

    Parameters
    ----------
    chunks : list of np.ndarray
        Each chunk is a (chunk_length, 3) trajectory.
    params : dict
        Output of :func:`parameters.get_params`.
    save_dir : path-like
        Directory in which to write ``bild_result_<i>.pkl`` files.
    nproc : int
        Number of worker processes.
    label : str
        Tag prefixed to each progress line (helps when running many configs).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    for j, chunk in enumerate(chunks):
        save_path = save_dir / f'bild_result_{j}.pkl'
        if save_path.exists():
            continue
        inputs.append((chunk, save_path, params))

    n_total = len(chunks)
    n_pending = len(inputs)
    if n_pending == 0:
        print(f'[{label}] all {n_total} chunks already done', flush=True)
        return

    failures = 0
    with Pool(nproc) as pool, tqdm(
        total=n_total, initial=n_total - n_pending, desc=label, unit='chunk',
    ) as bar:
        for name, status in pool.imap_unordered(_run_one, inputs):
            if status.startswith('failed'):
                failures += 1
                tqdm.write(f'[{label}] {name}: {status}')
                bar.set_postfix(failed=failures)
            bar.update(1)
