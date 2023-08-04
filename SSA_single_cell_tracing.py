# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
import time

# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
import pylab as p
from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from tqdm import tqdm
import matplotlib.pyplot as plt
# […]
# from joblib import Parallel, delayed
# Own modules
import sciplot as splt
# from kde_scatter_plot import kde_plot
# from scipy.stats import binned_statistic, gamma
from sklearn.mixture import GaussianMixture
from joblib import dump, load
from tqdm import tqdm
import threading

splt.whitegrid()

import datetime


def RG2Predict(green_signal, red_signal):
    return np.log((green_signal + 1) / (red_signal + 1))


RedColor = np.array((241, 148, 138)) / 255
GreenColor = np.array((130, 224, 170)) / 255
# %%

results_directory = r'/media/fulab/raid0/single_cell_transition_rate'
if not os.path.isdir(results_directory):
    os.makedirs(results_directory)

size = 200000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 50
growth_rate = 1.6
time_length = np.log(2) / growth_rate * 50
time_step = .1
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=24)

ratio_stat = RG2Predict(test_ret[:, 1, :], test_ret[:, 2, :])
gmm = GaussianMixture(n_components=2, random_state=1, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))
gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

class_mean = gmm.means_

green_label = 0 if class_mean[0] > class_mean[1] else 1

test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

# ======== show the predict rets =============== #
fig3, ax3 = plt.subplots(1, 1, figsize=(20, 20))
green_cells_mask = test_cls == green_label

ax3.scatter(test_ret[-2, 2][green_cells_mask], test_ret[-2, 1][green_cells_mask],
            color=GreenColor, alpha=.1)
ax3.scatter(test_ret[-2, 2][~green_cells_mask], test_ret[-2, 1][~green_cells_mask],
            color=RedColor, alpha=.1)
ax3.set_xlim(1, 100)
ax3.set_ylim(1, 200)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Red')
ax3.set_ylabel('Green')
splt.aspect_ratio(1)
fig3.show()
# fig3.savefig(os.path.join(results_directory, 'cell_states_prediction.png'))

# export cell discrimination model
model = dict(gmm=gmm, RG2Predict=RG2Predict, size=size,
             green=green,
             red=red,
             growth_rate=growth_rate,
             time_length=time_length,
             time_step=time_step)


dump(model, r'./discrimination_model/Gaussian_discrimination_gr-1.6.joblib')
# %% generate data
red_init = np.linspace(10, 100, num=10).astype(int)
green_init = np.linspace(0, 10, num=10).astype(int)
vx, vy = np.meshgrid(red_init, green_init)

parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
                                                     vx.flatten())]

loop_repeat_num = 1
batch_repeat_num = 400
growth_rate = 1.6
time_step = .05  # .1

batch_number = len(parameter_dict)
size = batch_number * batch_repeat_num
init_array = np.array(parameter_dict)  # [batch sample ..., green / red]
green = np.hstack([init_array[:, 0].flatten()] * batch_repeat_num)
red = np.hstack([init_array[:, 1].flatten()] * batch_repeat_num)
time_length = np.log(2) / growth_rate * 1000  # 5000 generation

# print(f"Start Sim: {par}. \n")
ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=12)
# Attention! Don't save simulation raw data, it's too big!
# print(f"End Sim: {par}. \n")
# time_now = datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%S')
# dump(ret, os.path.join(results_directory, f'cells_stats_G{par[0]}_R{par[1]}_lambda{growth_rate}_{time_now}.pkl'))
ratio_stat = RG2Predict(ret[:, 1, :], ret[:, 2, :])
class_pred = gmm.predict(ratio_stat.reshape(-1, 1))
green_mask = class_pred == green_label
green_mask = green_mask.reshape(ratio_stat.shape)  # [Time, Cell]

stateChange = np.vstack([np.zeros((1, green_mask.shape[1])), green_mask])
stateChange_diff = np.diff(stateChange, axis=0)

TransTime = np.zeros(size) * np.nan
for i in range(size):
    TransTime[i] = np.argmax(stateChange_diff[:, i] == 1) * time_step  # tau R->G

# transTimeList.append(TransTime.reshape(-1, 1))
init_states = np.hstack([green.reshape(-1, 1), red.reshape(-1, 1)])
summary_data = dict(cell_stats_init=init_states,
                    tua=TransTime)

# %%  mean of tau
concatenate_tau = []
for init_cds in tqdm(parameter_dict):
    # for cds_i, cds in enumerate(all_init_G_R):
    mask = np.all(init_states == init_cds, axis=1)
    concatenate_tau.append(TransTime[mask])

concatenate_tau = np.array(concatenate_tau)
mean_tua = np.mean(concatenate_tau, axis=1)
mean_tua = mean_tua.reshape(vy.shape)

mean_tua = mean_tua[::-1, :]  # mean tua matrix

# %% tau distribution
dist_of_tau = []
for tua in concatenate_tau:
    dist_of_tau.append(np.histogram(tua, range=(0, 500), bins=100, density=True))

fig_hist, ax4 = plt.subplots(1, 1)
ax4.hist(tua, range=(0, 500), bins=100, density=True)
ax4.set_xscale('log')
fig_hist.show()
