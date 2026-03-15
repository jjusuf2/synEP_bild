# Running BILD on synEP data

## Environment
* python 3.9.23
* bild 0.0.5
* noctiluca 0.1.4
* bayesmsd 0.1.7

## Data
**Input data:** The final tracks are stored in a directory `final_tracks_20250310` (not part of this repository), which contains `export_qc_filtered_5sTracks` and `export_qc_filtered_30sTracks`. In each of these, there is a separate folder for each movie, named according to the date, condition, and other important info.

**Output data:** The inferred looping trajectories ("profiles") are written to a directory `final_profiles_20250310` (not part of this trajectory), in a subfolder with the condition and frame rate (e.g., `G7B8G2_GSK_5s`). Each individual profile is stored as a txt of 1's and 0's, with 1 indicating looped and 0 indicating not looped at each timepoint.

## Compile list of tracks
It is helpful to have a table containing all the available tracks and their file locations. The script `compile_list_of_tracks.py` generates this table and saves it in `all_tracks.csv`.

## Calibrate BILD
We calibrate the BILD model on the 340kb synEP loop using the scripts `calibration/run_BILD_calibration_5s.py` and `calibration/run_BILD_calibration_30s.py` for ∆t = 5 and 30 seconds respectively. These scripts use S+V-A6B8 (actually ∆CTCF-sites) as the ∆CTCF condition and G2 ∆RAD21 as the ∆RAD21 condition.

## Run BILD
Use the script `run_BILD.py` to run BILD. Input the parameters from the calibration as arguments. For `--loc_error`, make sure to input the single-spot localization error for the tracks you are running the inference on (get from MSD fitting).

Example usage:

```python run_BILD.py --condition_name G7B8G2_GSK --delta_t 30 --L 16 --k 5.94 --D 0.00884 --L_looped 0.348 --loc_error 0.047,0.046,0.046 --nproc 4```

```python run_BILD.py --condition_name G7B8G2_GSK --delta_t 5 --L 16 --k 1.67 --D 0.00239 --L_looped 0.297 --loc_error 0.044,0.040,0.044 --nproc 4```
