import numpy as np
import pandas as pd
import noctiluca as nl
import bild
from multiprocessing import Pool
from datetime import datetime
import argparse
from pathlib import Path
import sys

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
nproc = args.nproc

## load the tracks
all_tracks = pd.read_csv('/mnt/md0/jjusuf/bild/synEP_bild/all_tracks.csv')

## write a function to grab the data for a given condition and ∆t (in the proper format for BILD)
def generate_data_list(condition, delta_t):
    rows = all_tracks.loc[
        (all_tracks['condition']==condition) &
        (all_tracks['delta_t']==delta_t)
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

data_list, row_indices = generate_data_list(condition_name, delta_t)

save_folder = Path(f'/mnt/md0/jjusuf/bild/final_profiles_20250310/{condition_name}_{delta_t}s')
save_folder.mkdir(parents=True, exist_ok=True)

# save metadata file
cmd = " ".join(sys.argv)
with open(f'{save_folder}/bild_run_info.txt', 'w') as f:
    f.write(f'{cmd}\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

## build the model

w = np.zeros(3*L+1)
w[L] = -1
w[2*L] = 1

model = bild.models.MultiStateRouse(3*L+1, D, k,
                                    looppositions = [None, (L, 2*L, 1/L_looped)],
                                    measurement = w,
                                    localization_error = localization_error*np.sqrt(2),
                                   )

## run BILD (with multiprocessing)

def calculate_and_save_BILD_result(args):
    traj_array, name, model = args
    traj = nl.Trajectory(traj_array)

    try:
        result = bild.sample(traj, model, show_progress=False)
        best_profile = result.best_profile()
        profile_str = "".join(map(str, best_profile))
        print(f'   {name}  {str(np.sum(best_profile)):>3s} /{str(len(best_profile)):>3s} timepoints looped (mean {np.mean(best_profile)*100:.1f}%)')
    except:
        profile_str = ""
        print('   trajectory failed')
    with open(f'{save_folder}/{name}_track.txt', 'w') as f:
        f.write(profile_str)

chunk_size = nproc
num_chunks = len(data_list)//chunk_size

with Pool(nproc) as p:
    for i in range(num_chunks):
        print(f'starting chunk {i+1} of {num_chunks} [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
        chunk_start = i*chunk_size
        chunk_end = min((i+1)*chunk_size, len(data_list))
        indices_in_chunk = np.arange(chunk_start, chunk_end)
        inputs = [(data_list[j], all_tracks.loc[row_indices[j], 'name'], model) for j in indices_in_chunk]

        chunk_profiles = p.map(calculate_and_save_BILD_result, inputs)
