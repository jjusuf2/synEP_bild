"""Rouse polymer simulation with a stochastically switching internal bond.

The chain has ``N = 3*L + 1`` monomers; an additional bond between monomers
``L`` and ``2L`` switches on and off at random intervals to match a target
mean lifetime and looped probability. All time arguments are physical
(minutes, seconds); conversion to frames happens internally and the outputs
are indexed in frames.
"""
import numpy as np
from tqdm import tqdm
import rouse


def generate_profile(looped_prob, looped_lifetime_min, t_max_min, delta_t_s):
    """Generate a stochastic on/off profile for the switching bond.

    On- and off-times are drawn from independent exponential distributions
    whose means satisfy ``mean_on / (mean_on + mean_off) == looped_prob``.

    Parameters
    ----------
    looped_prob : float in (0, 1)
        Mean probability of being in the looped state.
    looped_lifetime_min : float
        Mean lifetime of the looped state, in minutes.
    t_max_min : float
        Total simulation time, in minutes.
    delta_t_s : float
        Time per frame, in seconds.

    Returns
    -------
    profile : np.ndarray of int, shape (n_frames + 1,)
        Binary array; 1 = looped, 0 = unlooped.
    t_add_bond : list of int
        Frame indices at which the bond switches on.
    t_remove_bond : list of int
        Frame indices at which the bond switches off.
    """
    n_frames = int(round(t_max_min * 60.0 / delta_t_s))
    mean_on = looped_lifetime_min * 60.0 / delta_t_s
    mean_off = mean_on * (1.0 - looped_prob) / looped_prob

    def sample_interval(mean):
        return max(1, int(round(np.random.exponential(mean))))

    start_with_on = np.random.random() < looped_prob

    curr_t = 0
    t_add_bond = []
    t_remove_bond = []
    chunks = []

    if start_with_on:
        t_add_bond.append(curr_t)
        on_time = sample_interval(mean_on)
        chunks.append(np.ones(on_time, dtype=np.int8))
        curr_t += on_time
        t_remove_bond.append(curr_t)

    while curr_t <= n_frames:
        off_time = sample_interval(mean_off)
        chunks.append(np.zeros(off_time, dtype=np.int8))
        curr_t += off_time
        t_add_bond.append(curr_t)
        on_time = sample_interval(mean_on)
        chunks.append(np.ones(on_time, dtype=np.int8))
        curr_t += on_time
        t_remove_bond.append(curr_t)

    profile = np.concatenate(chunks)[:n_frames + 1]
    t_add_bond = [t for t in t_add_bond if t <= n_frames]
    t_remove_bond = [t for t in t_remove_bond if t <= n_frames]

    return profile, t_add_bond, t_remove_bond


def run_simulation(L, k, D, L_looped, n_frames,
                   t_init=1000, t_add_bond=(), t_remove_bond=(),
                   show_progress=True):
    """Simulate ``n_frames`` of a Rouse polymer with a switching L--2L bond.

    Returns ``n_frames + 1`` conformations: the equilibrated initial
    configuration plus one for each evolved frame.

    Parameters
    ----------
    L : int
        Half-spacing of the bonded monomers; chain length ``N = 3*L + 1``.
    k, D : float
        Spring constant and diffusion coefficient.
    L_looped : float
        Equilibrium length of the switching bond; the bond strength added
        when it switches on is ``1 / L_looped``.
    n_frames : int
        Number of frames to evolve after equilibration.
    t_init : int
        Number of equilibration frames before recording starts.
    t_add_bond, t_remove_bond : iterable of int
        Frame indices at which to add or remove the switching bond. The bond
        is removed by adding ``-1 / L_looped`` to its strength, which exactly
        cancels a prior add (rouse.Model.add_bonds is additive).
    show_progress : bool
        Show tqdm progress bars for equilibration and main simulation.

    Returns
    -------
    list of np.ndarray of shape (N, 3)
    """
    N = 3 * L + 1
    model = rouse.Model(N=N, D=D, k=k, d=3)

    # Diagonally stretched initial conformation; sqrt(D/k) is the typical
    # bond length, so this initialises near equilibrium.
    init = np.tile(np.arange(N)[:, None], (1, 3)) * np.sqrt(D / k)

    last_conf = init
    init_iter = range(t_init)
    if show_progress:
        init_iter = tqdm(init_iter, desc='equilibrating')
    for _ in init_iter:
        last_conf = model.evolve(last_conf, dt=1)

    add_set = set(t_add_bond)
    remove_set = set(t_remove_bond)

    conformations = [last_conf]
    main_iter = range(n_frames)
    if show_progress:
        main_iter = tqdm(main_iter, desc='simulating')
    for t in main_iter:
        if t in add_set:
            model.add_bonds([(L, 2 * L, 1.0 / L_looped)])
        if t in remove_set:
            model.add_bonds([(L, 2 * L, -1.0 / L_looped)])
        conformations.append(model.evolve(conformations[-1], dt=1))

    return conformations


def profile_statistics(profile, delta_t_s):
    """Summary statistics of an on/off profile.

    On-runs that touch either array boundary are excluded from the lifetime
    statistics (they are right- or left-censored). The looping probability
    is computed over the full profile.

    Parameters
    ----------
    profile : 1D array of 0/1
    delta_t_s : float
        Frame interval, used to report lifetimes in minutes.

    Returns
    -------
    dict with keys ``looped_prob``, ``mean_lifetime_min``,
    ``median_lifetime_min``, ``n_complete_loops``.
    """
    profile = np.asarray(profile).astype(int)
    n = len(profile)

    # Pad with zeros so runs touching the edges are detected.
    diff = np.diff(np.concatenate([[0], profile, [0]]))
    starts = np.where(diff == 1)[0]   # inclusive start, in profile coords
    ends = np.where(diff == -1)[0]    # exclusive end, in profile coords
    durations = ends - starts

    uncensored = (starts > 0) & (ends < n)
    durations_uncensored = durations[uncensored]

    if len(durations_uncensored) > 0:
        mean_min = float(np.mean(durations_uncensored)) * delta_t_s / 60.0
        median_min = float(np.median(durations_uncensored)) * delta_t_s / 60.0
    else:
        mean_min = float('nan')
        median_min = float('nan')

    return {
        'looped_prob': float(profile.mean()),
        'mean_lifetime_min': mean_min,
        'median_lifetime_min': median_min,
        'n_complete_loops': int(uncensored.sum()),
    }


def get_traj(conformations, L):
    """Extract the L-to-2L displacement vector from each conformation."""
    return np.array([conf[L, :] - conf[2 * L, :] for conf in conformations])


def add_noise(conformations, sigma):
    """Add independent zero-mean Gaussian noise to monomer positions.

    Returns new arrays; the inputs are not modified.

    Parameters
    ----------
    conformations : iterable of np.ndarray of shape (N, 3)
    sigma : array-like of shape (3,)
        Per-axis Gaussian standard deviation.
    """
    sigma = np.asarray(sigma)
    return [conf + np.random.normal(0.0, sigma, conf.shape) for conf in conformations]
