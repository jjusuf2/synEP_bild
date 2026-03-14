import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import noctiluca as nl
import bild
import bayesmsd

from datetime import datetime

from multiprocessing import Pool

import os

## load the tracks
all_tracks = pd.read_csv('all_tracks.csv', index_col=0)

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

data_list_G2_30s = generate_data_list('G7B8G2_GSK', 30)
data_nl = nl.util.userinput.make_TaggedSet(data_list_G2_30s)

## enter parameters and build model
L = 7
k = 1.137144617300876
D = 0.0038661826451376474
L_looped = 0.14278424112948787

w = np.zeros(3*L+1)
w[L] = -1
w[2*L] = 1

model = bild.models.MultiStateRouse(3*L+1, D, k,
                                    looppositions = [None, (L, 2*L, 1/L_looped)],
                                    measurement = w,
                                    localization_error = np.sqrt(np.array([0.004356220414642604, 0.004178863299165964, 0.0042861628536375535])),
                                   )

## run BILD (with multiprocessing)

all_tracks_with_profiles = all_tracks.copy()
all_tracks_with_profiles['profile'] = [None] * len(all_tracks_with_profiles)

def calculate_and_save_BILD_result(traj_num):
    traj = data_nl[traj_num]
    traj = nl.Trajectory(traj)
    try:
        result = bild.sample(traj, model, show_progress=False)
        best_profile = result.best_profile()
        profile_str = "".join(map(str, result.best_profile()))
        print(f'  tracjectory {traj_num} mean = {np.mean(best_profile)}')
    except:
        pass
        print(f'  tracjectory {traj_num} failed, skipped')
    return profile_str

chunk_size = 12
num_chunks = len(data_nl)//chunk_size

for i in range(num_chunks):
    print(f'starting chunk {i+1} of {num_chunks} [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
    chunk_start = i*chunk_size
    chunk_end = min((i+1)*chunk_size, len(data_nl))
    indices_in_chunk = np.arange(chunk_start, chunk_end)
    with Pool(chunk_size) as p:
        chunk_profiles = p.map(calculate_and_save_BILD_result, indices_in_chunk)
    all_tracks_with_profiles.loc[indices_in_chunk, 'profile'] = chunk_profiles
    all_tracks_with_profiles.to_csv('all_tracks_with_profiles.csv')  # save results

    s = ''
    for profile in all_tracks_with_profiles.iloc[:chunk_end]['profile']:
        s += profile
        print(f'mean so far: {np.mean(np.array(list(s)).astype("int")):.4f}')
