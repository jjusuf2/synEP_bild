#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import chi2

### -------------------------------------------------------
### PART 1: Make a table of all the avaiable imaging tracks
### from Harvey's code

# define dictionaries to unify naming scheme; also specify any conditions to remove

condition_mapping = {}
conditions_to_remove = {}

condition_mapping[30] = {
    'G7B8G2_GSK_500uMdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_500nMdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_500nmdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_100uM_IAA_added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_100uM_IAAadded2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    '15A_A6_G9_GSK': '15A-A6G9_GSK',
    '15A-A6G9_GSK_IAA(100uM)added2hrBefore': '15A-A6G9_GSK_RAD21depletion',
    'S+V-A6B8_GSK_IAA(100uM)added2hrBefore': 'S+V-A6B8_GSK_RAD21depletion',
    '15B-18G9_GSK_IAA(100uM)added2hrBefore': '15B-18G9_GSK_RAD21depletion',
    'S-G5H7_GSK': 'SCre-G5-H7_GSK',
    'SCreG5H7': 'SCre-G5-H7_GSK',
    '15B-18G9': '15B-18G9_GSK',
    '15B18G9_GSK': '15B-18G9_GSK',
    'VCreC2b_GSK': 'VCre-C2b_GSK',
    'S+V-A6B8': 'S+V-A6B8_GSK',
    'G7B8G2_GSK_IAA(100uM)added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    '14A-A11E6_GSK': '14A-A11E6_GSK',
    '14A-A11_GSK': '14A-A11_GSK',
    '14B5-C9': '14B5-C9_GSK',
    '14B5-F10': '14B5-F10_GSK',
    '14B5-F8': '14B5-F8_GSK',
    '14B5-F12': '14B5-F12_GSK',
    'E11G8_GSK': 'E11G8_GSK',
    '15C-A2_GSK':'15C-A2_GSK',
    'Vika-H11_GSK': 'Vika-H11_GSK',
    'G7B8G2_GSK': 'G7B8G2_GSK',
    'G7B8G2_GSK_dTAG13(500uM)andIAA(100uM)added2hrBefore': 'G7B8G2_GSK_CTCF_RAD21depletion',
    '15C-A2_GSK_IAA(100uM)added2hrBefore': '15C-A2_GSK_RAD21depletion',
    '14B5-F10_GSK_IAA(100uM)added2hrBefore': '14B5-F10_GSK_RAD21depletion',
}

conditions_to_remove[30] = ['14B5-F12_GSK','14B5-F8_GSK','14B5-C9_GSK']

condition_mapping[5] = {
    'G7B8G2_GSK_500uMdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_500nmdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added3hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added4hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added5hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added6hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added7hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_100uM_IAA_added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added5hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added4hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added6hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added3hrBefore': 'G7B8G2_GSK_RAD21depletion',
    '15A_A6_G9_GSK': '15A-A6G9_GSK',
    '15A-A6G9': '15A-A6G9_GSK',
    'S-G5H7_GSK': 'SCre-G5-H7_GSK',
    'SCre-G5H7_GSK': 'SCre-G5-H7_GSK',
    'SCreG5H7': 'SCre-G5-H7_GSK',
    '15B-18G9': '15B-18G9_GSK',
    '15B18G9_GSK': '15B-18G9_GSK',
    '15B18B9': '15B-18G9_GSK',
    '15B18G9': '15B-18G9_GSK',
    'VCreC2b_GSK': 'VCre-C2b_GSK',
    'S+V-A6B8': 'S+V-A6B8_GSK',
    'G7B8G2_GSK_IAA(100uM)added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'VCre-C2b_GSK': 'VCre-C2b_GSK',
    'E11G8_GSK': 'E11G8_GSK',
    'Vika-H11_GSK': 'Vika-H11_GSK',
    'G7B8G2_GSK': 'G7B8G2_GSK',
    '15C-A2_GSK':'15C-A2_GSK',
    '15A-A6G9_GSK':'15A-A6G9_GSK',
}

conditions_to_remove[5] = []

folders_path_dict = {5: '/mnt/md0/jjusuf/bild/final_tracks_20260407/export_qc_filtered_5s_WithCorrectedMS2_20260407',
                    30: '/mnt/md0/jjusuf/bild/final_tracks_20260407/export_qc_filtered_30s_WithCorrectedMS2_20260407'}

# now generate the table

all_tracks = []

for delta_t in [5, 30]:

    folders_path = folders_path_dict[delta_t]
    folders_path_obj = Path(folders_path)
    for folder in folders_path_obj.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        date = folder_name[:8]
        condition_name_raw = folder_name[9:].split('_30ms')[0]

        if condition_name_raw in condition_mapping[delta_t].values():
            condition_name = condition_name_raw
        else:
            condition_name = condition_mapping[delta_t][condition_name_raw]
        
        if condition_name in conditions_to_remove[delta_t]:
            continue
        
        for file_path in (folders_path_obj / folder_name).iterdir():
            file_path_str = str(file_path)
            file_path_list = file_path_str.split('/')
            name = f'{file_path_list[-2]}_{file_path_list[-1][:-4]}'
            track_len = len(pd.read_csv(file_path_str))
            all_tracks.append([date, delta_t, condition_name, file_path_str, name, track_len])

all_tracks = pd.DataFrame(all_tracks, columns=['date','delta_t','condition','path','name','track_len'])


### -----------------------------------------------------
### PART 2: Define functions to perform outlier filtering
### from Henrik's code

def filter_out_extreme_pos(data, mu=None, sig=None, significance_level=0.01):
    """Fits a gaussian to the data and filters the output by z-score under the fitted gaussian

    Parameters
    ----------
    data : array-like, shape (n_samples, n_timepoints,ndim)
        The data to fit and score.
    mu : array-like, shape (ndim,), optional
        The mean of the gaussian to fit. If None, the mean of the data will be used. Default is None.
    sig : array-like, shape (ndim,), optional
        The standard deviation of the gaussian to fit. If None, the standard deviation of the data will be used. Default is None.
    significance_level : float, optional
        The significance level to use for scoring. Default is 0.01.

    Returns
    -------
    filtered: array-like, shape (n_samples, n_timepoints, ndim)
        The filtered data.
    mus: array-like, shape (n_samples, n_timepoints-1)
        The means of the fitted gaussians.
    stds: array-like, shape (n_samples, n_timepoints-1)
        The standard deviations of the fitted gaussians.
    pvals: array-like, shape (n_samples, n_timepoints-1)
        The p-values of the data under the fitted gaussians.
    """
    pval = significance_level / len(
        data[0]
    )  # Bonferroni correction for multiple testing
    # displacements = np.diff(data, axis=1) # (n_samples, n_timepoints-1, ndim)
    if mu is not None:
        mus_pos = mu
    else:
        mus_pos = np.nanmean(data, axis=(0, 1))  # (ndim)
    if sig is not None:
        std_pos = sig
    else:
        std_pos = np.nanstd(data, axis=(0, 1))  # (ndim)

    # mus_disp,std_disp = np.nanmean(displacements,axis=(0,1)), np.nanstd(displacements,axis=(0,1)) # (ndim)
    summand_pos = (data - mus_pos) ** 2 / std_pos**2  # (n_samples, n_timepoints, ndim)
    # summand_disp = (displacements - mus_disp)**2 / std_disp**2 # (n_samples, n_timepoints-1, ndim)
    chi_pos = np.nansum(summand_pos, axis=-1)  # (n_samples, n_timepoints)
    # chi_disp = np.nansum(summand_disp, axis=-1) # (n_samples, n_timepoints-1)
    pvals_pos = 1 - chi2.cdf(
        chi_pos, df=np.sum(~np.isnan(data), axis=-1)
    )  # (n_samples, n_timepoints)
    # pvals_disp = 1 - chi2.cdf(chi_disp, df=np.sum(~np.isnan(displacements), axis=-1)) # (n_samples, n_timepoints-1)

    significant_pos = pvals_pos < pval
    filtered = np.where((significant_pos)[..., None], np.nan, data)
    return filtered, significant_pos

def filter_out_1frame_jumps(data, significance_level=0.01):

    resid = data[:, 1:-1] - 0.5 * (data[:, :-2] + data[:, 2:])
    ms = np.nanmean(resid, axis=(0, 1))
    ss = np.nanstd(resid, axis=(0, 1))
    z = np.abs(np.nansum((resid - ms) ** 2 / ss**2, axis=-1))

    is_local_max = (
        z >= np.pad(z[:, :-1], ((0, 0), (1, 0)), constant_values=-np.inf)
    ) & (z >= np.pad(z[:, 1:], ((0, 0), (0, 1)), constant_values=-np.inf))
    pval = 1 - chi2.cdf(z, df=3)
    nobs = np.sum(~np.isnan(data[..., 0]), axis=-1)[:, None]
    significant = pval < significance_level / nobs
    single_frame_jump = significant & is_local_max
    padded_single_frame_jump = np.zeros(
        (single_frame_jump.shape[0], single_frame_jump.shape[1] + 2), dtype=bool
    )
    padded_single_frame_jump[:, 1:-1] = single_frame_jump
    filtered = np.where(padded_single_frame_jump[..., None], np.nan, data)
    return filtered, single_frame_jump

def filter_inconsistent_trajectories(ep_dat, mu, sig, significance_level=0.01):
    pval = (
        significance_level / ep_dat.shape[1]
    )  # Bonferroni correction for multiple testing
    filtered, significant_pos = filter_out_extreme_pos(
        ep_dat, mu=mu, sig=sig, significance_level=significance_level
    )
    filtered, single_frame_jump = filter_out_1frame_jumps(
        filtered, significance_level=significance_level
    )

    rv = np.linalg.norm(ep_dat, axis=-1)  # (n_samples, n_timepoints   )
    rv_filtered = np.linalg.norm(filtered, axis=-1)  # (n_samples, n_timepoints   )
    padded_single_frame_jump = np.zeros(
        (single_frame_jump.shape[0], single_frame_jump.shape[1] + 2), dtype=bool
    )
    padded_single_frame_jump[:, 1:-1] = single_frame_jump
    removed = (significant_pos | padded_single_frame_jump) & (~np.isnan(rv))

    return filtered, rv, rv_filtered, removed


### -------------------------------------------------------------
### PART 3: Finally, load imaging tracks, and save in .npy format

def generate_data_list_npy_format(condition, delta_t):
    """Return E-P separation trajectories for a given condition and ∆t: a list of (T, 3) arrays, NaN-padded to equal length."""
    matches = all_tracks.loc[
        (all_tracks['condition'] == condition) &
        (all_tracks['delta_t'] == delta_t)
    ]

    trajectories = []
    for path in matches['path']:
        track = pd.read_csv(path)
        separation = np.stack([
            track['pro_x (nm)'] - track['enh_x (nm)'],
            track['pro_y (nm)'] - track['enh_y (nm)'],
            track['pro_z (nm)'] - track['enh_z (nm)'],
        ], axis=1)

        if np.all(np.isnan(separation)):  # skip all-NaN trajectories
            continue

        trajectories.append(separation)

    max_len = max(len(traj) for traj in trajectories)
    data_list = [
        np.pad(traj, ((0, max_len - len(traj)), (0, 0)), constant_values=np.nan)
        for traj in trajectories
    ]

    data_npy = np.stack(data_list)

    return data_npy

# now define the main function

def main():
    conditions_HY = ['G7B8G2_GSK', 'G7B8G2_GSK_RAD21depletion', 'S+V-A6B8_GSK']
    conditions_HDP = ['340kb_Ce_Cp_None', '340kb_Ce_Cp_IAA', '340kb_None']

    for condition_HY, condition_HDP in zip(conditions_HY, conditions_HDP):
        for dt in [5, 30]:
            ep_res = generate_data_list_npy_format(condition_HY, dt)
            significance_level=0.00001

            # filter out positions inconsistent with ensemble distribution
            m, s = np.nanmean(ep_res, axis=(0, 1)), np.nanstd(ep_res, axis=(0, 1))
            filtered, rv, rv_filtered, removed = filter_inconsistent_trajectories(
                ep_res, mu=m, sig=s, significance_level=significance_level
            )

            # save filtered data
            np.save(f"data/{dt}s_{condition_HDP}.npy", filtered)

if __name__ == "__main__":
    main()
