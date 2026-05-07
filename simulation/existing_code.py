import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import rouse
import pandas as pd
from tqdm import tqdm

### below is some code to run simulations

def e2e_vector(conf, t):
    '''Get the end-to-end vector of a list of conformations at a specified time'''
    return conf[t][0,:]-conf[t][-1,:]

def plot_conformation(conf, t):
    '''Plot the conformation of the polymer at a given time'''
    fig, axs = plt.subplots(1, 2, figsize=(6,3))

    axs[0].plot(conf[t][:,0], conf[t][:,1], marker='o', color='#00aaee', ms=3)
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$y$')
    axs[0].axis('square')

    axs[1].plot(conf[t][:,1], conf[t][:,2], marker='o', color='#00aaee', ms=3)
    axs[1].set_xlabel('$y$')
    axs[1].set_ylabel('$z$')
    axs[1].axis('square')

    fig.suptitle(f'$t={t}$')

    plt.tight_layout()

    plt.show()

def test_run_initialization(L, k, D, L_looped, t_max_test):

    # set up the model
    N = 3*L+1  # number of monomers
    model = rouse.Model(N=N, D=D, k=k, d=3)
    print(f'Successfully loaded {model}')

    # initialize chain as diagonally stretched linear conformation
    init = np.array([np.arange(N), np.arange(N), np.arange(N)]).T * np.sqrt(D/k)

    print(f'Running initialization test simulation of {t_max_test}')
    conformations = [init]
    last_conf = init
    for i in tqdm(range(t_max_test)):
        last_conf = model.evolve(last_conf, dt=1)
        conformations.append(last_conf)
        
    t_arr = []
    e2e_autocorr = []
    for t in range(t_max_test):
        t_arr.append(t)
        e2e_autocorr.append(np.dot(e2e_vector(conformations, 0), e2e_vector(conformations, t)))
        
    plt.plot(t_arr, e2e_autocorr)
    plt.xlabel('$t$ [frames]')
    plt.ylabel('end-to-end autocorrelation\n'+r'$\langle \vec{R}(0)\cdot \vec{R}(t) \rangle$');
    plt.show()

def run_simulation(L, k, D, L_looped, t_init, t_max, t_add_bond=(), t_remove_bond=()):

    # set up the model
    N = 3*L+1  # number of monomers
    model = rouse.Model(N=N, D=D, k=k, d=3)
    print(f'Successfully loaded {model}')

    # initialize chain as diagonally stretched linear conformation
    init = np.array([np.arange(N), np.arange(N), np.arange(N)]).T * np.sqrt(D/k)

    print(f'Running initialization of {t_init} steps')
    last_conf = init
    for i in tqdm(range(t_init)):
        last_conf = model.evolve(last_conf, dt=1)

    t_add_bond_set = set(t_add_bond)
    t_remove_bond_set = set(t_remove_bond)

    t_arr = np.arange(0, t_max+1)
    conformations = []
    conformations.append(last_conf)

    print(f'Running main simulation and saving conformations for {t_max} steps')
    for i in tqdm(range(len(t_arr) - 1)):
        t = t_arr[i]
        if t in t_add_bond_set:
            model.add_bonds([(L, 2*L, 1/L_looped)])
        if t in t_remove_bond_set:
            model.add_bonds([(L, 2*L, -1/L_looped)])
        last_conf = conformations[-1]
        conformations.append(model.evolve(last_conf, dt=1))

    return conformations

def generate_profile(dt, looped_prob, looped_lifetime, t_max):

    # calculate ON times and OFF times in terms of frames
    t_on = looped_lifetime/dt
    t_off = t_on / looped_prob * (1-looped_prob)

    def get_one_int_time_interval(t_mean):
        t = np.random.exponential(t_mean)
        if t == 0:  # round up if it was 0
            t = 1
        else:
            t = int(np.round(t))
        return t

    start_with_on = (np.random.random() < looped_prob)  # start with ON state with a prob equal to the looped prob

    curr_t = 0
    t_add_bond = []
    t_remove_bond = []
    profile = []

    if start_with_on:
        t_add_bond.append(curr_t)
        on_time = get_one_int_time_interval(t_on)
        profile.append(np.ones(on_time))
        curr_t += on_time
        t_remove_bond.append(curr_t)

    while curr_t < t_max:
        off_time = get_one_int_time_interval(t_off)
        profile.append(np.zeros(off_time))
        curr_t += off_time
        t_add_bond.append(curr_t)
        on_time = get_one_int_time_interval(t_on)
        profile.append(np.ones(on_time))
        curr_t += on_time
        t_remove_bond.append(curr_t)

    profile = np.concatenate(profile).astype('int')

    # remove any events after t_max
    profile = profile[:t_max+1]
    t_add_bond = [t for t in t_add_bond if t<=t_max]
    t_remove_bond = [t for t in t_remove_bond if t<=t_max]

    return profile, t_add_bond, t_remove_bond
   
def get_traj(conformations):
    return np.array([conf[L,:]-conf[2*L,:] for conf in conformations])

def add_noise(conformations, sigma_x_um, sigma_y_um, sigma_z_um):
    conformations_with_noise = []
    for conf in conformations:
        conf = conf
        conf[:,0] += np.random.normal(0, sigma_x_um, conf.shape[0])
        conf[:,1] += np.random.normal(0, sigma_y_um, conf.shape[0])
        conf[:,2] += np.random.normal(0, sigma_z_um, conf.shape[0])
        conformations_with_noise.append(conf)
    return conformations_with_noise

np.random.seed(7)

# # load parameters for ∆t = 30 s (joint calibration from Harvey's S+V control)
# L = 16
# k = 8.04
# D = 0.01194
# L_looped = 0.334
# localization_error = np.array([0.0443,0.0443,0.0444])

# load parameters for ∆t = 5 s (joint calibration from Harvey's S+V control)
L = 16
k = 1.34
D = 0.00199
L_looped = 0.334
localization_error = np.array([0.0425,0.0425,0.0449])

t_max = 16000
dt = 5  # seconds
looped_lifetime = 20*60  # seconds
looped_prob = 0.06

profile, t_add_bond, t_remove_bond = generate_profile(dt, looped_prob, looped_lifetime, t_max)
conformations = run_simulation(L, k, D, L_looped, t_init=1000, t_max=t_max, t_add_bond=t_add_bond, t_remove_bond=t_remove_bond)

conformations_with_noise = add_noise(conformations, localization_error[0], localization_error[1], localization_error[2])

track_length = 800

profiles = []
trajectories = []

for i in range(t_max // track_length):
    track_start = i     * track_length
    track_end   = (i+1) * track_length

    profiles.append(profile[track_start:track_end])
    trajectories.append(get_traj(conformations_with_noise[track_start:track_end]))

for track_length in [50, 100, 200, 400, 800]:

    profiles = []
    trajectories = []

    for i in range(t_max // track_length):
        track_start = i     * track_length
        track_end   = (i+1) * track_length

        profiles.append(profile[track_start:track_end])
        trajectories.append(get_traj(conformations_with_noise[track_start:track_end]))

    np.savez(f'20260504_simulation_benchmarking/sim_traj_{dt}s/profiles_{track_length}fr.npz', *profiles)
    np.savez(f'20260504_simulation_benchmarking/sim_traj_{dt}s/trajectories_{track_length}fr.npz', *trajectories)

## below is the code to run BILD on the simulated trajectories

import numpy as np
import noctiluca as nl
import bild
from multiprocessing import Pool
from datetime import datetime
import argparse
from pathlib import Path
import time
import os
import pickle

## Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument(
    "--nproc",
    type=int,
    required=True,
    help="Number of threads for multiprocessing"
)
parser.add_argument(
    "--track_len",
    type=int,
    required=True,
    help="Track length in frames of simulated trajectories"
)

args = parser.parse_args()
nproc = args.nproc
track_len = args.track_len

## Parameters

delta_t = 5
L = 16
k = 1.34
D = 0.00199
L_looped = 0.334
localization_error = np.array([0.0425,0.0425,0.0449])

## load the data for a given track length
data_list = list(np.load(f'/mnt/md0/jjusuf/bild/20260504_simulation_benchmarking/sim_traj_5s/trajectories_{track_len}fr.npz').values())

## build the model

w = np.zeros(3*L+1)
w[L] = -1
w[2*L] = 1

save_folder = Path(f'/mnt/md0/jjusuf/bild/20260504_simulation_benchmarking/bild_results_5s')

## run BILD (with multiprocessing)

def calculate_and_save_BILD_result(args):
    # initiate model every time
    model = bild.models.MultiStateRouse(3*L+1, D, k,
                                        looppositions = [None, (L, 2*L, 1/L_looped)],
                                        measurement = w,
                                        localization_error = localization_error*np.sqrt(2),
                                       )

    traj_array, j = args
    traj = nl.Trajectory(traj_array)

    try:
        np.random.seed((int(time.time()*1e6) ^ os.getpid()) % (2**32))  # use a different random seed every time
        rng = np.random.get_state()[1][0]  # get RNG state to print later (for diagnostic purposes)

        # now run BILD
        result = bild.sample(traj, model, show_progress=False)
        print(f'   [RNG {rng}] {j} done')
    except:
        profile_str = ""
        print('   trajectory failed')
    with open(f"{save_folder}/bild_result_{track_len}fr_{j}.pkl", "wb") as f:
        pickle.dump(result, f)

chunk_size = nproc
num_chunks = int(np.ceil(len(data_list) / chunk_size))

with Pool(nproc) as p:
    for i in range(num_chunks):
        print(f'starting chunk {i+1} of {num_chunks} [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
        chunk_start = i*chunk_size
        chunk_end = min((i+1)*chunk_size, len(data_list))
        indices_in_chunk = np.arange(chunk_start, chunk_end)
        inputs = [(data_list[j], j) for j in indices_in_chunk]

        chunk_profiles = p.map(calculate_and_save_BILD_result, inputs)

### below is some code for miscellaneous analyses

def extract_lifetimes(tracks):
    """
    tracks: list of 1D lists/arrays of 0/1
    
    Returns:
        durations: list of lifetimes
        events: list of booleans (True = observed, False = censored)
    """
    durations = []
    events = []
    
    for track in tracks:
        track = np.asarray(track)
        n = len(track)
        
        i = 0
        while i < n:
            if track[i] == 1:
                start = i
                while i < n and track[i] == 1:
                    i += 1
                end = i  # exclusive
                
                lifetime = end - start
                
                # censored if touching either boundary
                censored = (start == 0) or (end == n)
                
                durations.append(lifetime)
                events.append(not censored)
            else:
                i += 1
    
    return np.array(durations), np.array(events)

def kaplan_meier(durations, events):
    """
    Compute Kaplan-Meier survival curve.
    
    Returns:
        times: unique event times
        survival: survival probability at each time
    """
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    
    unique_times = np.unique(durations)
    
    n = len(durations)
    at_risk = n
    survival = 1.0
    
    times = []
    surv_probs = []
    
    for t in unique_times:
        mask = durations == t
        d_i = np.sum(events[mask])      # observed events
        c_i = np.sum(~events[mask])     # censored
        
        if at_risk > 0:
            if d_i > 0:
                survival *= (1 - d_i / at_risk)
                times.append(t)
                surv_probs.append(survival)
        
        at_risk -= (d_i + c_i)
    
    return np.array(times), np.array(surv_probs)

def median_survival(times, survival):
    """
    Median survival = first time S(t) <= 0.5
    """
    if len(survival) == 0:
        return np.nan
    
    below = np.where(survival <= 0.5)[0]
    if len(below) == 0:
        return np.inf  # never drops below 0.5
    
    return times[below[0]]

def km_median_lifetime(tracks):
    durations, events = extract_lifetimes(tracks)
    times, survival = kaplan_meier(durations, events)
    return median_survival(times, survival), times, survival