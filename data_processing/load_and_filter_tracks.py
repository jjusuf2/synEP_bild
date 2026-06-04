import os
import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import chi2



# PART 0. INPUT PARAMETERS
# ------------------------

tracks_csv_folder_5s = "/mnt/md1/jjusuf/synEP/export_qc_filtered_5s_WithCorrectedMS2_20260407"
tracks_csv_folder_30s = "/mnt/md1/jjusuf/synEP/export_qc_filtered_30s_WithCorrectedMS2_20260407"

tracks_npz_folder = "/mnt/md0/jjusuf/synEP/data_consolidated_numpy"
filtered_tracks_npy_folder = "/mnt/md0/jjusuf/synEP/data_consolidated_numpy/filtered_data"

significance_level = 1e-7  # for filtering (in last cell)



# PART 1: PROCESS DATA
# --------------------

data_folders = {
    "5s": tracks_csv_folder_5s,
    "30s": tracks_csv_folder_30s,
}

# make output folders if they don't exist yet
Path(tracks_npz_folder).mkdir(parents=True, exist_ok=True)
Path(filtered_tracks_npy_folder).mkdir(parents=True, exist_ok=True)

output_dir = (tracks_npz_folder)

cell_type_synonyms = {
    "G7B8G2": ["G7B8G2_"],
    "S+V-A6B8": ["S+V-A6B8_"],
    "15A-A6G9": ["15A-A6G9_", "15A_A6_G9_"],
    "15B18G9": ["15B18G9", "15B-18G9_", "15B18B9_"],
    "15C-A2": ["15C-A2_"],
    "SCreG5H7": ["SCreG5H7_", "SCre-G5H7_", "SCre-G5-H7_", "S-G5H7_GSK"],
    "VCreC2b": ["VCreC2b_", "VCre-C2b_"],
    "14A-A11": ["14A-A11_"],
    "14A-A11E6": ["14A-A11E6_"],
    "E11G8": ["E11G8_"],
    "Vika-H11": ["Vika-H11_"],
    "14B5-C9": ["14B5-C9_"],
    "14B5-F10": ["14B5-F10_"],
    "14B5-F8": ["14B5-F8_"],
    "14B5-F12": ["14B5-F12_"],
}

common_names = {
    "14B5-C9": "1.5kb",
    "14B5-F10": "1.5kb",
    "14B5-F8": "1.5kb",
    "14B5-F12": "1.5kb",
    "15C-A2": "85kb",
    "15B18G9": "170kb",
    "15A-A6G9": "255kb",
    "S+V-A6B8": "340kb",
    "VCreC2b": "340kb_Ce",
    "SCreG5H7": "340kb_Cp",
    "G7B8G2": "340kb_Ce_Cp",
    "14A-A11": "2.362kb_noE_noP",
    "14A-A11E6": "0.395kb_noE_noP",
    "E11G8": "340kb_Ce_Cp_noE_noP",
    "Vika-H11": "340kb_Ce_Cp_noE",
}

def load_csvs_in_folders(folders):
    """Eagerly load all track CSVs in ``folders`` as a pandas dataframe
    """
    # Table of columns to read
    colnames_dtypes = {
        "frame": "Int32",
        "enh_z (nm)": "float32",
        "enh_y (nm)": "float32",
        "enh_x (nm)": "float32",
        "pro_z (nm)": "float32",
        "pro_y (nm)": "float32",
        "pro_x (nm)": "float32",
        "background (au)": "float32",
        "intensity (au)": "float32",
    }

    frames = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            d = pd.read_csv(path, usecols=colnames_dtypes.keys(), dtype=colnames_dtypes, na_values=["nan"])
            d["file"] = path
            frames.append(d)

    df = pd.concat(frames, ignore_index=True)

    # ``cell``, ``allele``, ``replicate``, ``date``, and ``folder`` are parsed from the path, not the CSV body.
    file = df["file"]
    df["cell"] = file.str.extract(r"[/\\](\d+)_\d+\.csv$")[0].astype("Int32")
    df["allele"] = file.str.extract(r"_(\d+)\.csv$")[0].astype("Int32")
    df["replicate"] = file.str.extract(r"FrameRate-(\d+)[/\\]")[0].astype("Int32")
    df["date"] = file.str.extract(r"[/\\](\d{8})_")[0]
    # ``folder`` is the acquisition directory name (one CSV == one track). Including it
    # in TRACK_KEYS keeps tracks from different source folders separate even when
    # (cell, allele, replicate, date) coincide -- e.g. dTAG/IAA incubation-time sweeps
    # acquired on the same day all reuse FrameRate-01.
    df["folder"] = file.str.extract(r"[/\\]([^/\\]+)[/\\][^/\\]+\.csv$")[0]

    # Keep only the last two path components of the file column.
    df["file"] = file.str.extract(r"([^/\\]+[/\\][^/\\]+)$")[0]
    
    return df

# Group folders according to (framerate, cell line, treatment).
folder_groups = defaultdict(list)

print(f"Loading all localizations into a merged DataFrame:")
for framerate, root in data_folders.items():
    for folder in sorted(os.listdir(root)):
        cell_type = next(
            (ct for ct, syns in cell_type_synonyms.items()
             if any(syn in folder for syn in syns)),
            None,
        )
        if cell_type is None:
            raise ValueError(f"{folder!r} did not match any cell type ({framerate})")

        if "IAA" in folder and "dTAG" in folder:
            treatment = "IAAdTAG"
        elif "IAA" in folder:
            treatment = "IAA"
        elif "dTAG" in folder:
            treatment = "dTAG"
        else:
            treatment = "None"

        key = (framerate, common_names[cell_type], treatment)
        folder_groups[key].append(os.path.join(root, folder))
        
frames = []
for (framerate, cell_type, treatment), folders in tqdm(folder_groups.items()):
    frame = load_csvs_in_folders(folders)
    frame["framerate"] = int(framerate.rstrip("s"))
    frame["cell type"] = cell_type
    frame["treatment"] = treatment
    frames.append(frame)

merged_df = pd.concat(frames, ignore_index=True)
print(f"Successfully loaded {len(merged_df)} rows across {len(frames)} conditions\n")

def nanpad(times, series):
    """Place each track's ragged values at their integer time index.

    ``times`` is a sequence of per-track time arrays; ``series`` is a list of
    matching per-track value sequences. Returns the padded time array and a
    list with one ``(n_tracks, max_len)`` padded array per entry in ``series``.
    """
    n_tracks = len(times)
    max_len = max(max(t) + 1 for t in times)
    pad_t = np.full((n_tracks, max_len), np.nan)
    pads = [np.full((n_tracks, max_len), np.nan) for _ in series]
    for i, track_t in enumerate(times):
        for t in track_t:
            pad_t[i, t] = t
        for pad, values in zip(pads, series):
            for t, value in zip(track_t, values[i]):
                pad[i, t] = value
    return pad_t, pads

# ``folder`` is part of the key so that one track == one source CSV; without it,
# acquisitions sharing (cell, allele, replicate, date) -- e.g. dTAG/IAA
# incubation-time sweeps reusing FrameRate-01 -- would be merged into one track.
TRACK_KEYS = ["cell", "allele", "replicate", "date", "folder"]
os.makedirs(output_dir, exist_ok=True)

print(f"Saving results as .npz files in {tracks_npz_folder}")

for (framerate, cell_type, treatment), group in merged_df.groupby(
    ["framerate", "cell type", "treatment"], sort=False
):
    
    group = group.copy()
    # Derived per-row columns (computed before the per-track aggregation).
    group["t"] = group["frame"] - group.groupby(TRACK_KEYS)["frame"].transform("min")
    group["dx"] = group["enh_x (nm)"] - group["pro_x (nm)"]
    group["dy"] = group["enh_y (nm)"] - group["pro_y (nm)"]
    group["dz"] = group["enh_z (nm)"] - group["pro_z (nm)"]

    tracks = group.groupby(TRACK_KEYS, sort=False).agg(
        t=("t", list),
        intensity=("intensity (au)", list),
        background=("background (au)", list),
        dx=("dx", list),
        dy=("dy", list),
        dz=("dz", list),
        file=("file", list),
    ).reset_index()

    times = [[int(v) for v in row] for row in tracks["t"]]
    ts_pad, (ints_pad, bgs_pad) = nanpad(
        times, [tracks["intensity"].tolist(), tracks["background"].tolist()]
    )
    _, (xs_pad, ys_pad, zs_pad) = nanpad(
        times,
        [tracks["dx"].tolist(), tracks["dy"].tolist(), tracks["dz"].tolist()],
    )
    dataset = np.stack([xs_pad, ys_pad, zs_pad], axis=-1)  # (n_tracks, n_t, 3)

    identifiers = np.array([
        f"{int(cell)}_{int(allele)}_{int(replicate)}_{date}_{files[0]}"
        for cell, allele, replicate, date, files in zip(
            tracks["cell"].tolist(),
            tracks["allele"].tolist(),
            tracks["replicate"].tolist(),
            tracks["date"].tolist(),
            tracks["file"].tolist(),
        )
    ])

    filename = f"{framerate}s_{cell_type}_{treatment}.npz"
    print(f"Saving {filename:<35}  {dataset.shape}")
    np.savez_compressed(
        os.path.join(output_dir, f"{framerate}s_{cell_type}_{treatment}.npz"),
        dataset=dataset,
        intensity=ints_pad,
        background=bgs_pad,
        ts_MS2=ts_pad,
        ts_pad_position=ts_pad,
        identifiers=identifiers,
    )
print()


# PART 2: FILTERING
# -----------------

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
    # A track can be entirely NaN (nobs == 0) when the enhancer-promoter distance
    # is undefined at every frame -- e.g. one of the two spots is missing throughout,
    # or filter_out_extreme_pos removed every localization in the track. There is
    # nothing to flag in such a track, so skip the per-track test (and the
    # significance_level / nobs division, which would otherwise be a divide-by-zero).
    with np.errstate(divide="ignore"):
        threshold = significance_level / nobs
    significant = (nobs > 0) & (pval < threshold)
    single_frame_jump = significant & is_local_max
    padded_single_frame_jump = np.zeros(
        (single_frame_jump.shape[0], single_frame_jump.shape[1] + 2), dtype=bool
    )
    padded_single_frame_jump[:, 1:-1] = single_frame_jump
    filtered = np.where(padded_single_frame_jump[..., None], np.nan, data)
    return filtered, single_frame_jump

def filter_inconsistent_trajectories(ep_dat, mu, sig, significance_level=0.01):
    """Main function to remove inconsistent localizations from trajectories
    """

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

datapaths = [d for d in os.listdir(tracks_npz_folder) if d.endswith(".npz")]

print(f"Performing filtering with significance level {significance_level}")
print(f"Saving results as .npy files in {filtered_tracks_npy_folder}")

num_localizations_removed_cumulative = 0
num_localizations_total_cumulative = 0

for datapath in datapaths:
    p = os.path.join(tracks_npz_folder, datapath)
    d = np.load(p, allow_pickle=True)
    ep_res = d["dataset"]

    # filter out positions inconsistent with ensemble distribution
    m, s = np.nanmean(ep_res, axis=(0, 1)), np.nanstd(ep_res, axis=(0, 1))
    filtered, rv, rv_filtered, removed = filter_inconsistent_trajectories(
        ep_res, mu=m, sig=s, significance_level=significance_level
    )

    num_localizations_removed = np.sum(removed)
    num_localizations_total = np.sum(~np.isnan(ep_res[:,:,0]))
    pct = num_localizations_removed / num_localizations_total 
    print(f"{datapath:<40} {num_localizations_removed} of {num_localizations_total} ({pct:.2%}) locs removed")

    num_localizations_removed_cumulative += num_localizations_removed
    num_localizations_total_cumulative += num_localizations_total

    #store as filtered_data
    np.save(f"{filtered_tracks_npy_folder}/{datapath}".strip(".npz"), filtered)

pct = num_localizations_removed_cumulative / num_localizations_total_cumulative 
print(f"In total, {num_localizations_removed_cumulative} of {num_localizations_total_cumulative} ({pct:.2%}) locs were removed.")
