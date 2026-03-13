import pandas as pd 
import numpy as np

import noctiluca as nl
import bild
import bayesmsd

from pathlib import Path

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

## perform the fits

print('performing noCTCF fit (∆t=30s)')
fit = bayesmsd.lib.TwoLocusRouseFit(generate_data_list('S+V-A6B8_GSK', 30))
fitres_noCTCF = fit.run(show_progress=True)

print('performing dRAD21 fit (∆t=30s)')
fit = bayesmsd.lib.TwoLocusRouseFit(generate_data_list('G7B8G2_GSK_RAD21depletion', 30))
fitres_dRAD21 = fit.run(show_progress=True)

print('performing G7B8G2 fit (∆t=30s)')
fit = bayesmsd.lib.TwoLocusRouseFit(generate_data_list('G7B8G2_GSK', 30))
fitres_G7B8G2 = fit.run(show_progress=True)


## fit results

# Unlooped state (ΔCTCF)
G = np.exp(fitres_noCTCF['params']['log(Γ) (dim 0)'])
J = np.exp(fitres_noCTCF['params']['log(J) (dim 0)'])

print(f"{'gamma':>10s} = {G}")
print(f"{'J':>10s} = {J}")

# Following eq. (5.22) in Simon's PhD thesis
L = np.ceil(np.sqrt(4/np.pi)*J/G).astype(int)
D = np.pi*L*G**2 / (4*J)
k = np.pi/4*(L*G/J)**2

# put in synEP parameters
tether_length_looped_kb = 1.8
dist_btwn_CTCF_sites_kb = 335

# Looped state (from ΔRad21, with genomic rescaling)
J_dRAD21 = np.exp(fitres_dRAD21['params']['log(J) (dim 0)'])
J_looped = tether_length_looped_kb/dist_btwn_CTCF_sites_kb * J_dRAD21
L_looped = np.sqrt(4/np.pi)*J_looped/G

## If L_looped > 1.6, apply decomposition by golden ratio, as described
## in Simon's PhD thesis, paragraph below eq. (5.25).
## Since here L_looped < 1 we spare the effort

print( "Rouse model parameters from MSD fits")
print(f"{'G':>10s} = {G}")
print(f"{'J':>10s} = {J}")
print(f"{'L':>10s} = {L}")
print(f"{'k':>10s} = {k}")
print(f"{'D':>10s} = {D}")
print(f"{'J_dRAD21':>10s} = {J_dRAD21}")
print(f"{'J_looped':>10s} = {J_looped}")
print(f"{'L_looped':>10s} = {L_looped}")
print()
print(f"sigma^2 for S+V-A6B8_GSK:      {[np.exp(fitres_noCTCF['params']['log(σ²) (dim 0)']), np.exp(fitres_noCTCF['params']['log(σ²) (dim 1)']), np.exp(fitres_noCTCF['params']['log(σ²) (dim 2)'])]}")
print(f"sigma^2 for G7B8G2_GSK_∆RAD21: {[np.exp(fitres_dRAD21['params']['log(σ²) (dim 0)']), np.exp(fitres_dRAD21['params']['log(σ²) (dim 1)']), np.exp(fitres_dRAD21['params']['log(σ²) (dim 2)'])]}")
print(f"sigma^2 for G7B8G2_GSK:        {[np.exp(fitres_G7B8G2['params']['log(σ²) (dim 0)']), np.exp(fitres_G7B8G2['params']['log(σ²) (dim 1)']), np.exp(fitres_G7B8G2['params']['log(σ²) (dim 2)'])]}")
