import pandas as pd 
import numpy as np

import noctiluca as nl
import bild
import bayesmsd

from pathlib import Path

## load the tracks
all_tracks = pd.read_csv('../all_tracks.csv', index_col=0)

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

## perform the fits

print('performing noCTCF fit (∆t=5s)')
fit = bayesmsd.lib.TwoLocusRouseFit(generate_data_list('S+V-A6B8_GSK', 5))
fitres_noCTCF = fit.run(show_progress=True)

print('performing dRAD21 fit (∆t=5s)')
fit = bayesmsd.lib.TwoLocusRouseFit(generate_data_list('G7B8G2_GSK_RAD21depletion', 5))
fitres_dRAD21 = fit.run(show_progress=True)

print('performing G7B8G2 fit (∆t=5s)')
fit = bayesmsd.lib.TwoLocusRouseFit(generate_data_list('G7B8G2_GSK', 5))
fitres_G7B8G2 = fit.run(show_progress=True)


## fit results

delta_t = 5  # seconds

# Unlooped state (ΔCTCF)
G = np.exp(fitres_noCTCF['params']['log(Γ) (dim 0)'])
J = np.exp(fitres_noCTCF['params']['log(J) (dim 0)'])

# Following eq. (5.22) in Simon's PhD thesis
L_min = np.ceil(np.sqrt(4/np.pi)*J/G).astype(int)
L = 16
D = np.pi*L*G**2 / (4*J)
k = np.pi/4*(L*G/J)**2  # effectively k/gamma (lowercase gamma, that is)

# put in synEP parameters
tether_length_looped_kb = 1.8
dist_btwn_CTCF_sites_kb = 335

# Looped state (from ΔRad21, with genomic rescaling)
J_dRAD21 = np.exp(fitres_dRAD21['params']['log(J) (dim 0)'])
J_looped = tether_length_looped_kb/dist_btwn_CTCF_sites_kb * J_dRAD21
L_looped = J_looped * k / D  # from Eq. 5.24 in Simon's thesis

## If L_looped > 1.6, apply decomposition by golden ratio, as described
## in Simon's PhD thesis, paragraph below eq. (5.25).
## Since here L_looped < 1 we spare the effort

print(f"{'∆t':>10s} = {delta_t}")
print(f"{'G':>10s} = {G}")
print(f"{'J':>10s} = {J}")
print(f"{'L_min':>10s} = {L_min}")
print(f"{'L':>10s} = {L}")
print(f"{'k':>10s} = {k}")
print(f"{'D':>10s} = {D}")
print(f"{'J_dRAD21':>10s} = {J_dRAD21}")
print(f"{'J_looped':>10s} = {J_looped}")
print(f"{'L_looped':>10s} = {L_looped}")
print()
print(f"sigma_spot for S+V-A6B8_GSK:      {[np.sqrt(np.exp(fitres_noCTCF['params']['log(σ²) (dim 0)'])/2), np.sqrt(np.exp(fitres_noCTCF['params']['log(σ²) (dim 1)'])/2), np.sqrt(np.exp(fitres_noCTCF['params']['log(σ²) (dim 2)'])/2)]}")
print(f"sigma_spot for G7B8G2_GSK_∆RAD21: {[np.sqrt(np.exp(fitres_dRAD21['params']['log(σ²) (dim 0)'])/2), np.sqrt(np.exp(fitres_dRAD21['params']['log(σ²) (dim 1)'])/2), np.sqrt(np.exp(fitres_dRAD21['params']['log(σ²) (dim 2)'])/2)]}")
print(f"sigma_spot for G7B8G2_GSK:        {[np.sqrt(np.exp(fitres_G7B8G2['params']['log(σ²) (dim 0)'])/2), np.sqrt(np.exp(fitres_G7B8G2['params']['log(σ²) (dim 1)'])/2), np.sqrt(np.exp(fitres_G7B8G2['params']['log(σ²) (dim 2)'])/2)]}")

# calculate in timescale of seconds
G_sec = G / np.sqrt(delta_t)
k_sec = k / delta_t
D_sec = D / delta_t
print("Using a timescale of seconds:")
print(f"{'G':>10s} = {G_sec}")
print(f"{'k':>10s} = {k_sec}")
print(f"{'D':>10s} = {D_sec}")
