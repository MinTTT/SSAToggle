# -*- coding: utf-8 -*-

"""


 Statistics of SSA single cell data. ( SSAToggle\SSA_single_cell_tracing_generator.py )
 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys

import matplotlib.pyplot as plt
from joblib import load, dump
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
from tqdm import tqdm

# […]

# Own modules


# import simulation data

sim_data_dict = r'Y:\Data_Raid\Colony_Project\SSA_Toggle_results\20230721'

pkl_files_path = [file.name for file in os.scandir(sim_data_dict) if file.name.split('.')[-1] == 'pkl']

# get all tua
all_tua = []
all_init_G_R = []
for file_name in tqdm(pkl_files_path, postfix='Loading: '):
    pkl_data = load(os.path.join(sim_data_dict, file_name))
    tua_data = pkl_data['tua']
    all_tua.append(tua_data)
    all_init_G_R.append(pkl_data['cell_stats_init'])
parameter_dict = pkl_data['init_parameters']

# %% concatenate all data and calculate the mean tau
red_init = np.linspace(10, 100, num=10).astype(int)
green_init = np.linspace(0, 10, num=10).astype(int)
vx, vy = np.meshgrid(red_init, green_init)

parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
                                                     vx.flatten())]

concatenate_tau_all = []

for init_cds in tqdm(parameter_dict):
    # for cds_i, cds in enumerate(all_init_G_R):
    concatenate_tau = []
    for GR_i, init_GR in enumerate(all_init_G_R):
        mask = np.all(init_GR == init_cds, axis=1)
        concatenate_tau.append(all_tua[GR_i][mask])
        # concatenate_tau = np. ([tau[init_cds_i] for tau in all_tua])
    concatenate_tau_all.append(np.hstack(concatenate_tau).flatten())

# mean tau
tua_data = np.array(concatenate_tau_all)
mean_tua = np.mean(tua_data, axis=1)
# data_for_origin = np.hstack([np.array(all_init_G_R[0]), mean_tau])
# mean_tua = mean_tua.reshape(vx.shape)
# mean_tua = mean_tua[::-1, :]

# binned statistics
# from scipy.stats import binned_statistic
tau_distribution = []
for i in range(len(parameter_dict)):
    tau_distribution.append(np.histogram(tua_data[i], density=True, range=(0, 100), bins=500))
distribution_x = tau_distribution[0][1][:-1] + .5 * (tau_distribution[0][1][1] - tau_distribution[0][1][0])
tau_distribution_for_graph_pad = np.hstack(
    [distribution_x.reshape(-1, 1), ] + [hist[0].reshape(-1, 1) for hist in tau_distribution])

tua_dist_title = np.array([f'({init[1]}; {init[0]})' for init in parameter_dict])

# %% load batch culture data

data_path = r'D:\python_code\SSAToggle\Data\SSA_pop_evolve_results.joblib'
batch_data = load(data_path)
red_ratio = batch_data['results']
red_ration_mean = np.mean(red_ratio, axis=0)
red_ration_std = np.std(red_ratio, axis=0)
red_ration_cv = red_ration_std / red_ration_mean
fig_tau_vs_ratio_stat, ax1 = plt.subplots(1, 1)
ax1.scatter(mean_tua, red_ration_std)
# ax1.set_xscale('log')
fig_tau_vs_ratio_stat.show()

tau_vs_ratio = np.empty((1000 * 100, 2))
for taui, tau in enumerate(mean_tua):
    red_ratio_temp = red_ratio[:, taui]
    for j, ratio in enumerate(red_ratio_temp):
        tau_vs_ratio[j + taui * 1000, 0] = tau
        tau_vs_ratio[j + taui * 1000, 1] = ratio

fig_tau_vs_ratio, ax = plt.subplots(1, 1)
# ax.scatter(tau_vs_ratio[:, 0], tau_vs_ratio[:, 1], s=1)
ax.boxplot(red_ratio, positions=mean_tua)
ax.set_ylim(0, 1)
ax.set_xscale('linear')
fig_tau_vs_ratio.show()
