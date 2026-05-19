"""Run the full simulation -> BILD pipeline for the configured conditions.

For each (delta_t, p, mu) condition:
  1. Simulate a Rouse polymer with a stochastic L--2L bond, save the
     ground-truth profile and the noisy 3D L--2L trajectory.
  2. Chop the trajectory into windows of various lengths and run BILD on
     each chunk. The full SimulationResult is pickled per chunk.

Both stages are restartable: if simulation outputs already exist they are
loaded from disk, and BILD chunks whose pickle file exists are skipped.

Layout under DATA_ROOT:
    trajectories/<label>/trajectory.npy        # (n_frames+1, 3)
    trajectories/<label>/profile.npy           # (n_frames+1,) int8
    trajectories/<label>/metadata.txt
    bild_results/<label>/chunk_<L>fr/bild_result_<i>.pkl
"""
# Pin BLAS backends to a single thread per process so the multiprocessing
# Pool below doesn't explode into nproc * BLAS-threads workers contending
# for cores. Must be set before numpy / scipy / bild are imported.
# `setdefault` lets the user override from the shell if they want.
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from parameters import get_params
from simulate import generate_profile, run_simulation, get_traj, add_noise, profile_statistics
from run_bild import chop, run_bild_on_chunks


# Outputs go here; covered by synEP_bild/.gitignore.
DATA_ROOT = Path(__file__).resolve().parent.parent / 'simulation_data'

CHUNK_LENGTHS = [50, 100, 200, 400, 800]
T_INIT_FRAMES = 1000

DELTA_TS = [5, 30]

# The user-requested grid: (looped_prob, mean_lifetime_min, n_frames).
CONFIGS = [
    {'p': 0.13, 'mu_min': 40,  'n_frames': 64000},
    {'p': 0.13, 'mu_min': 60, 'n_frames': 64000},
    {'p': 0.06, 'mu_min': 40,  'n_frames': 64000},
    {'p': 0.06, 'mu_min': 60, 'n_frames': 64000},
]


def condition_label(delta_t, p, mu_min):
    """Filesystem-friendly label for a (delta_t, p, mu) condition."""
    return f'dt{delta_t}s_p{p:g}_mu{mu_min:g}min'


def simulate_and_save(delta_t, p, mu_min, n_frames, traj_dir, seed):
    """Simulate (or load if cached) and save trajectory + profile.

    Returns ``(profile, trajectory)``, both indexed in frames with length
    ``n_frames + 1``.
    """
    traj_dir = Path(traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)
    profile_path = traj_dir / 'profile.npy'
    traj_path = traj_dir / 'trajectory.npy'

    if profile_path.exists() and traj_path.exists():
        print(f'[{traj_dir.name}] cached -- loading', flush=True)
        return np.load(profile_path), np.load(traj_path)

    params = get_params(delta_t)
    L = params['L']

    # Seed both default_rng and the legacy global state -- rouse.Model uses
    # np.random internally, and so does generate_profile.
    np.random.seed(seed)

    t_max_min = n_frames * delta_t / 60.0
    profile, t_add, t_remove = generate_profile(p, mu_min, t_max_min, delta_t)
    profile = profile[:n_frames + 1]

    # Quick sanity check: the empirical profile statistics should be close
    # to the requested (p, mu) when the trajectory is long enough.
    stats = profile_statistics(profile, delta_t)
    print(f'[{traj_dir.name}] ground-truth profile stats:', flush=True)
    print(f'    looped_prob       = {stats["looped_prob"]:.4f}  (target {p})', flush=True)
    print(f'    mean lifetime     = {stats["mean_lifetime_min"]:.3f} min  (target {mu_min})', flush=True)
    print(f'    median lifetime   = {stats["median_lifetime_min"]:.3f} min', flush=True)
    print(f'    n_complete_loops  = {stats["n_complete_loops"]}', flush=True)

    print(f'[{traj_dir.name}] simulating {n_frames} frames '
          f'(delta_t={delta_t}s, p={p}, mu={mu_min}min)', flush=True)
    confs = run_simulation(
        L=L, k=params['k'], D=params['D'], L_looped=params['L_looped'],
        n_frames=n_frames, t_init=T_INIT_FRAMES,
        t_add_bond=t_add, t_remove_bond=t_remove,
        show_progress=True,
    )

    confs_noisy = add_noise(confs, params['localization_error'])
    traj = get_traj(confs_noisy, L)

    # Atomic-ish writes: save to a tmp path, then rename. Prevents a partial
    # file from masquerading as a cached result on the next restart. We open
    # the file ourselves so np.save doesn't silently append ".npy" to the
    # tmp path.
    tmp_profile = profile_path.with_name(profile_path.name + '.tmp')
    tmp_traj = traj_path.with_name(traj_path.name + '.tmp')
    with open(tmp_profile, 'wb') as f:
        np.save(f, profile.astype(np.int8))
    with open(tmp_traj, 'wb') as f:
        np.save(f, traj.astype(np.float64))
    tmp_profile.replace(profile_path)
    tmp_traj.replace(traj_path)

    with open(traj_dir / 'metadata.txt', 'w') as f:
        f.write(f'delta_t = {delta_t} s\n')
        f.write(f'p = {p}\n')
        f.write(f'mu = {mu_min} min\n')
        f.write(f'n_frames = {n_frames}\n')
        f.write(f't_init = {T_INIT_FRAMES}\n')
        f.write(f'seed = {seed}\n')
        f.write(f'parameters = {params}\n')
        f.write(f'simulated_at = {datetime.now().isoformat(timespec="seconds")}\n')
        f.write('\n# ground-truth profile statistics (full trajectory)\n')
        f.write(f'looped_prob = {stats["looped_prob"]:.6f}  (target {p})\n')
        f.write(f'mean_lifetime_min = {stats["mean_lifetime_min"]:.6f}  (target {mu_min})\n')
        f.write(f'median_lifetime_min = {stats["median_lifetime_min"]:.6f}\n')
        f.write(f'n_complete_loops = {stats["n_complete_loops"]}\n')

    return profile, traj


def save_chunked_profiles(profile, traj_dir):
    """Save chopped versions of the ground-truth profile alongside trajectories."""
    for chunk_len in CHUNK_LENGTHS:
        out_path = Path(traj_dir) / f'profiles_{chunk_len}fr.npz'
        if out_path.exists():
            continue
        np.savez(out_path, *chop(profile, chunk_len))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nproc', type=int, default=8,
                        help='worker processes for BILD')
    parser.add_argument('--seed', type=int, default=7,
                        help='base RNG seed; combined with the condition label')
    parser.add_argument('--only_delta_t', type=int, default=None, choices=DELTA_TS,
                        help='restrict to a single frame interval')
    args = parser.parse_args()

    delta_ts = [args.only_delta_t] if args.only_delta_t else DELTA_TS

    print(f'data root: {DATA_ROOT}', flush=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        for delta_t in delta_ts:
            label = condition_label(delta_t, cfg['p'], cfg['mu_min'])
            print(f'\n===== {label} =====', flush=True)

            traj_dir = DATA_ROOT / 'trajectories' / label
            # Per-condition seed: stable across restarts, distinct per condition.
            cond_seed = (args.seed + abs(hash(label))) % (2 ** 31)

            profile, traj = simulate_and_save(
                delta_t, cfg['p'], cfg['mu_min'], cfg['n_frames'],
                traj_dir, seed=cond_seed,
            )
            save_chunked_profiles(profile, traj_dir)

            for chunk_len in CHUNK_LENGTHS:
                chunks = chop(traj, chunk_len)
                bild_dir = DATA_ROOT / 'bild_results' / label / f'chunk_{chunk_len}fr'
                run_bild_on_chunks(
                    chunks, get_params(delta_t), bild_dir,
                    nproc=args.nproc, label=f'{label} {chunk_len}fr',
                )


if __name__ == '__main__':
    main()
