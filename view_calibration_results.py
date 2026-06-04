import sys, os
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~'))
file_path = Path("~/mpl_theme.py").expanduser()
if file_path.is_file():
    import mpl_theme; mpl_theme.apply();
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import bild
import noctiluca as nl
import bayesmsd
from copy import deepcopy
import pickle

from utils import *

def get_MSD_plateau(dataset):
    return 2*np.nanmean(np.concatenate([np.sum(np.square(np.array(traj)), 1) for traj in dataset]))

all_params = {}
conditions = ['340kb_Ce_Cp_None','340kb_Ce_Cp_IAA','340kb_None']
for condition in conditions:
    with open(f'bild_outputs/fit_params/{condition}_fit_params.pkl', 'rb') as f:
        all_params[condition] = pickle.load(f)

G = np.exp(all_params['340kb_None']['log(Γ)'])
J = np.exp(all_params['340kb_None']['log(J)'])

# Following eq. (5.22) in Simon's PhD thesis
L = np.ceil(np.sqrt(4/np.pi)*J/G/np.sqrt(30)).astype(int)
D = np.pi*L*G**2 / (4*J)
k = np.pi/4*(L*G/J)**2  # effectively k/gamma (lowercase gamma, that is)

D_frames = D*30
k_frames = k*30

# put in synEP parameters
tether_length_looped_kb = 1.8
dist_btwn_CTCF_sites_kb = 335

# Looped state (from ΔRad21, with genomic rescaling)
J_dRAD21 = J = np.exp(all_params['340kb_Ce_Cp_IAA']['log(J)'])
J_looped = tether_length_looped_kb/dist_btwn_CTCF_sites_kb * J_dRAD21
L_looped = J_looped * k / D  # from Eq. 5.24 in Simon's thesis

## If L_looped > 1.6, apply decomposition by golden ratio, as described
## in Simon's PhD thesis, paragraph below eq. (5.25).
## Since here L_looped < 1 we spare the effort

print()
print(f"{'Γ':>10s} = {G:.3} µm^2 s^(-0.5)")
print(f"{'J':>10s} = {J:.3} µm^2")
print(f"{'J_∆RAD21':>10s} = {J_dRAD21:.3} µm^2")
print()
print(f"{'L':>10s} = {L}")
print(f"{'D':>10s} = {D:.3} µm^2/s = {D_frames:.3} µm^2/fr")
print(f"{'k':>10s} = {k:.3} s^-1 = {k_frames:.3} fr^-1")
print(f"{'L_looped':>10s} = {L_looped:.3}")
print()

conditions = ['340kb_Ce_Cp_None', '340kb_None', '340kb_Ce_Cp_IAA']
colors = ['C0', 'C3', 'C4']

# load the data
all_data = {}

for condition in ['340kb_Ce_Cp_None', '340kb_None', '340kb_Ce_Cp_IAA']:
    trajs_30s_nl, trajs_5s_nl = get_30s_5s_trajs(condition, min_traj_len=100, print_output=False)
    all_data[condition] = {}
    all_data[condition]['30s'] = trajs_30s_nl
    all_data[condition]['5s'] = trajs_5s_nl

sigma2_dict = {}
for condition in conditions:
    sigma2_dict[condition] = {}
    for data_source, dt in zip(['30s', '5s'], [30, 5]):
        sigma2_dict[condition][data_source] = np.sum([np.exp(all_params[condition][f'{dt}s log(σ²) (dim {k})']) for k in (0,1,2)])

# plot error-corrected MSDs
fig, ax = plt.subplots()

w = np.zeros(3*L+1) # total number of monomers: 3*L + 1
w[L] = -1           # designate "measurement vector" w:
w[2*L] = 1          # the observable generated from a conformation x is the scalar product w.x
                    # So w = (0, ..., 0, -1, 0, ..., 0, 1, 0, ...) means we measure
                    # the vector monomer[j] - monomer[i]

for condition, color in zip(conditions, colors):
    msd_30s = nl.analysis.MSD(all_data[condition]['30s']) - 2*sigma2_dict[condition]['30s']
    msd_5s = nl.analysis.MSD(all_data[condition]['5s']) - 2*sigma2_dict[condition]['5s']
    ax.plot(30*np.arange(1,len(msd_30s)), msd_30s[1:], color=color, label=condition)
    ax.plot(5*np.arange(1,len(msd_5s)), msd_5s[1:], color=color, linestyle='--')
    ax.axhline(get_MSD_plateau(all_data[condition]['30s']) - 2*sigma2_dict[condition]['30s'], 0, 1, color=color)
    ax.axhline(get_MSD_plateau(all_data[condition]['5s']) - 2*sigma2_dict[condition]['5s'], 0, 1, color=color, linestyle='--')

model_30s = bild.models.MultiStateRouse(3*L+1, D_frames, k_frames,
                                            looppositions = [None, (L, 2*L, 1/L_looped)],       # define "states" (here: 0=unlooped, 1=looped)
                                            measurement = w,
                                            localization_error = np.sqrt(2)*np.array([43.05,41.00,44.32])/1e3, # convert error back to distance
                                        )

model_5s = bild.models.MultiStateRouse(3*L+1, D_frames, k_frames,
                                            looppositions = [None, (L, 2*L, 1/L_looped)],       # define "states" (here: 0=unlooped, 1=looped)
                                            measurement = w,
                                            localization_error = np.sqrt(2)*np.array([44.46,40.41,43.52])/1e3, # convert error back to distance
                                        )

t_range_frames = np.arange(1, 500)
state_model = model_30s.models[0]
msd = state_model.MSD(t_range_frames, w=model_30s.measurement)
ax.plot(30*t_range_frames, msd, color='#444444', label='model state 0')
ax.axhline(state_model.MSD(np.inf, w=model_30s.measurement), 0, 1, color='#444444')

state_model = model_30s.models[1]
msd = state_model.MSD(t_range_frames, w=model_30s.measurement)
ax.plot(30*t_range_frames, msd, color='#bbbbbb', label='model state 1')
ax.axhline(state_model.MSD(np.inf, w=model_30s.measurement), 0, 1, color='#bbbbbb')

# state_model = model_5s.models[0]
# msd = state_model.MSD(t_range_frames, w=model_5s.measurement)
# ax.plot(30*t_range_frames, msd, color='#444444', label='model state 0', linestyle='--')
# ax.axhline(state_model.MSD(np.inf, w=model_5s.measurement), 0, 1, color='#444444', linestyle='--')

# state_model = model_5s.models[1]
# msd = state_model.MSD(t_range_frames, w=model_5s.measurement)
# ax.plot(30*t_range_frames, msd, color='#bbbbbb', label='model state 1', linestyle='--')
# ax.axhline(state_model.MSD(np.inf, w=model_5s.measurement), 0, 1, color='#bbbbbb', linestyle='--')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylabel('MSD [µm$^2$]')
ax.set_xlabel('Lag time ∆t [s]')

ax.legend(fontsize=8, loc='lower right')

ax.set_title('')

plt.savefig('bild_outputs/figures/MSDs_loc_error_corrected_with_model.png');
plt.savefig('bild_outputs/figures/MSDs_loc_error_corrected_with_model.svg');
