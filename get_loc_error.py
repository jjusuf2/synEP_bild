import numpy as np
import pandas as pd
import bayesmsd
import argparse

## parse arguments

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

# boolean argument
parser.add_argument(
    "--round",
    action='store_true',
    help="Whether to round to 3 decimal places."
)

args = parser.parse_args()

# Access values
condition_name = args.condition_name
delta_t = args.delta_t
round = args.round

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

data_list, _ = generate_data_list(condition_name, delta_t)

fit = bayesmsd.lib.TwoLocusRouseFit(data_list)
fitres = fit.run(show_progress=True)
sigma_x = np.sqrt(np.exp(fitres['params']['log(σ²) (dim 0)'])/2)
sigma_y = np.sqrt(np.exp(fitres['params']['log(σ²) (dim 1)'])/2)
sigma_z = np.sqrt(np.exp(fitres['params']['log(σ²) (dim 2)'])/2)

if round:
    print(f"{condition_name} dt={delta_t}s sigma_spot (x,y,z): {sigma_x:.3f},{sigma_y:.3f},{sigma_z:.3f}")
else:
    print(f"{condition_name} dt={delta_t}s sigma_spot (x,y,z): {sigma_x},{sigma_y},{sigma_z}")
