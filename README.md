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
We calibrate the BILD model on the 340kb synEP loop using the scripts `calibration/run_BILD_calibration_5s.py` and `calibration/run_BILD_calibration_30s.py` for ∆t = 5 and 30 seconds respectively. These scripts use S+V-A6B8 (actually ∆CTCF-sites) as the ∆CTCF condition and G2 ∆RAD21 as the ∆RAD21 condition.

## Run BILD
Use the script `run_BILD.py` to run BILD. Input the parameters from the calibration as arguments. For `--loc_error`, make sure to input the single-spot localization error for the tracks you are running the inference on (get from MSD fitting).

Example usage:

```python run_BILD.py --condition_name G7B8G2_GSK --delta_t 30 --L 16 --k 5.94 --D 0.00884 --L_looped 0.348 --loc_error 0.047,0.046,0.046 --nproc 4```

```python run_BILD.py --condition_name G7B8G2_GSK --delta_t 5 --L 16 --k 1.67 --D 0.00239 --L_looped 0.297 --loc_error 0.044,0.040,0.044 --nproc 4```

## Localization error
To get the localization error needed to run BILD, use `get_loc_error.py`. This will report the single-spot localization error in x, y, and z.

Example usage:

```python get_loc_error.py --condition_name 14A-A11E6_GSK --round --delta_t 30```
