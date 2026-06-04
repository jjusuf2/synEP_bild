# Running BILD on synEP data

## Environment
* python 3.9.23
* bild 0.0.5
* noctiluca 0.1.4
* bayesmsd 0.1.7

## Data processing
The folder `data_processing` contains scripts to process and filter the data.

**Raw tracks:** The tracks are stored in `export_qc_filtered_5s_WithCorrectedMS2_20260407` and `export_qc_filtered_30s_WithCorrectedMS2_20260407`. In each of these, there is a separate folder for each movie, named according to the date, condition, and other important info, and each track is a separate `.csv` file.

**Unfiltered tracks:** The first part of the script `load_and_filter_tracks.py` takes the raw tracks and consolidates them into one `.npz` file per condition, stored in `data_consolidated_npz/unfiltered_data`.

**Filtered tracks:** The second part of the script `load_and_filter_tracks.py` takes the unfiltered tracks, performs outlier filtering, and saves the results in the same `.npz` format, stored in `data_consolidated_npz/filtered_data`.

## Calibrate BILD
The script `run_calibration.py` performs MSD fitting to calibrate the underlying Rouse model parameters (this differs from other MSD fits in the manuscript, as $\alpha$ is fixed at 0.5 for BILD). We use the S+V-A6B8 (∆CTCFsites) condition as the unlooped state and rescale the parameters of the G2 ∆RAD21 condition to obtain the looped state. This script also plots the raw MSDs.

The script `view_calibration_results.py` calculates the BILD model parameters ($L$, $L_\text{looped}$, $k$, and $D$) from the Rouse model parameters ($\Gamma$, $J$, and localization error for each condition). It also plots the localization error-corrected experimental MSDs overlaid with the MSDs of the model's looped and unloooped states.

## Run BILD
The script `run_bild.py` runs BILD on the trajectory data.

```python run_bild.py --condition 340kb_Ce_Cp_None --nproc 12 --dE 2 --traj_len 100```
```python run_bild.py --condition 340kb_Ce_Cp_IAA --nproc 12 --dE 2 --traj_len 100```