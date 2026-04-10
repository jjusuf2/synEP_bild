import numpy as np
import pandas as pd
import noctiluca as nl
import bild
from multiprocessing import Pool
from datetime import datetime
import argparse
from pathlib import Path
import sys
import time
import os

## Parse arguments

parser = argparse.ArgumentParser(description="Run BILD on specified tracks with specified parameters.")

# string argument
parser.add_argument(
    "--condition_name",
    type=str,
    required=True,
    help="Name of the condition"
)

# numeric arguments
parser.add_argument(
    "--delta_t",
    type=int,
    required=True,
    help="Time interval between frames in seconds"
)

parser.add_argument(
    "--L",
    type=int,
    required=True,
    help="L parameter for BILD model"
)

parser.add_argument(
    "--k",
    type=float,
    required=True,
    help="k parameter (k/gamma) for BILD model"
)

parser.add_argument(
    "--D",
    type=float,
    required=True,
    help="D parameter for BILD model"
)

parser.add_argument(
    "--L_looped",
    type=float,
    required=True,
    help="L_looped parameter for BILD model"
)

parser.add_argument(
    "--loc_error",
    type=str,
    required=True,
    help="localization error (spot) in x,y,z separated by commas"
)

parser.add_argument(
    "--nproc",
    type=int,
    required=True,
    help="Number of threads for multiprocessing"
)

parser.add_argument(
    "--traj_length",
    type=int,
    required=True,
    help="Length of segments to run BILD on, in frames"
)

parser.add_argument(
    "--run_number",
    type=int,
    required=False,
    help="Option to include a run number in the name, if have multiple BILD runs"
)

parser.add_argument(
    "--min_track_length",
    type=int,
    default=0,
    required=False,
    help="Minimum track length to analyze, in frames"
)

args = parser.parse_args()

# Access values
condition_name = args.condition_name
delta_t = args.delta_t
L = args.L
k = args.k
D = args.D
L_looped = args.L_looped
loc_error = args.loc_error
localization_error = np.array(loc_error.split(',')).astype('float')  # turn into a numpy array
traj_length_frames = args.traj_length
nproc = args.nproc
min_track_length = args.min_track_length
run_number = args.run_number

## load the tracks
all_tracks = pd.read_csv('/mnt/md0/jjusuf/bild/synEP_bild/all_tracks.csv')

## write a function to grab the data for a given condition and ∆t (in the proper format for BILD)
def generate_data_list(condition, delta_t):
    rows = all_tracks.loc[
        (all_tracks['condition']==condition) &
        (all_tracks['delta_t']==delta_t) &
        (all_tracks['track_len']>min_track_length)
    ]

    data_list = []
    row_indices = []

    for idx, row in rows.iterrows():
        table_full = pd.read_csv(row['path'])
        table_xyz = np.array([
            table_full['pro_x (nm)']-table_full['enh_x (nm)'],
            table_full['pro_y (nm)']-table_full['enh_y (nm)'],
            table_full['pro_z (nm)']-table_full['enh_z (nm)']
        ]).T / 1000

        if np.all(np.isnan(table_xyz)):  # don't add trajectory if it's all just nan's
            continue

        data_list.append(table_xyz)
        row_indices.append(idx)

    return data_list, row_indices

condition_name = 'G7B8G2_GSK'
delta_t = 30
data_list, row_indices = generate_data_list(condition_name, delta_t)

data_list_chopped = []
names_chopped = []

for i in range(len(data_list)):
    trajectory_data = data_list[i]
    name = all_tracks.loc[row_indices[i], 'name']
    num_segments_to_split_into = int(np.floor(len(trajectory_data) / traj_length_frames))
    if num_segments_to_split_into > 1:
        for n in range(num_segments_to_split_into):
            segment_start_index = n*traj_length_frames
            segment_end_index = (n+1)*traj_length_frames
            segment_trajectory_data = trajectory_data[segment_start_index:segment_end_index,:]
            segment_name = name + f'_frames_{str(segment_start_index).zfill(3)}_to_{str(segment_end_index).zfill(3)}'
            
            data_list_chopped.append(segment_trajectory_data)
            names_chopped.append(segment_name)

save_folder = Path(f'/mnt/md0/jjusuf/bild/profiles_20260408_chopped/{condition_name}_{delta_t}s')
save_folder.mkdir(parents=True, exist_ok=True)

# save metadata file
cmd = " ".join(sys.argv)
with open(f'{save_folder}/bild_run_info.txt', 'w') as f:
    f.write(f'{cmd}\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

## build the model

w = np.zeros(3*L+1)
w[L] = -1
w[2*L] = 1

## run BILD (with multiprocessing)

dE_arr = [0, 1, 2, 5, 10]

def calculate_and_save_BILD_result(args):
    # initiate model every time
    model = bild.models.MultiStateRouse(3*L+1, D, k,
                                        looppositions = [None, (L, 2*L, 1/L_looped)],
                                        measurement = w,
                                        localization_error = localization_error*np.sqrt(2),
                                       )

    traj_array, name = args
    traj = nl.Trajectory(traj_array)

    try:
        np.random.seed((int(time.time()*1e6) ^ os.getpid()) % (2**32))  # use a different random seed every time
        rng = np.random.get_state()[1][0]  # get RNG state to print later (for diagnostic purposes)

        # now run BILD
        result = bild.sample(traj, model, show_progress=False)
        for dE in dE_arr:
            best_profile = result.best_profile(dE=dE)
            profile_str = "".join(map(str, best_profile))
            with open(f'{save_folder}/{name}_dE_{dE}_profile.txt', 'w') as f:
                f.write(profile_str)
        #print(f'   [RNG {rng}] {name} {str(np.sum(best_profile)):>3s} /{str(len(best_profile)):>3s} timepoints looped ({np.mean(best_profile)*100:.1f}%)')
    except:
        profile_str = ""
        print('   trajectory failed')

chunk_size = nproc
num_chunks = int(np.ceil(len(data_list_chopped) / chunk_size))

with Pool(nproc) as p:
    for i in range(num_chunks):
        print(f'starting chunk {i+1} of {num_chunks} [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
        chunk_start = i*chunk_size
        chunk_end = min((i+1)*chunk_size, len(data_list_chopped))
        indices_in_chunk = np.arange(chunk_start, chunk_end)
        inputs = [(data_list_chopped[j], names_chopped[j]) for j in indices_in_chunk]

        chunk_profiles = p.map(calculate_and_save_BILD_result, inputs)
