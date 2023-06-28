# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
# […]
from joblib import Parallel, delayed

import sciplot
# Own modules
import sciplot as splt
from kde_scatter_plot import kde_plot

splt.whitegrid()


# for i in tqdm(range(10)):
# cells = pyrunBatchSim(22, 1.2, 1, 1, 10, .1, 1e3)
# # cells = pyrunBatchSim(22, 1.2, 1, 1, 10, 1, 1e5)
def CalculateGreenRatio(stats: np.ndarray):
    return np.sum(stats) / len(stats)


def CalcuRatio(greenNum, redNum):
    return np.log((greenNum + .1) / (redNum + .1))


from scipy.stats import binned_statistic

# %%  Training a model for stat discrimination


size = 10000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 15
growth_rate = 1.6
time_length = np.log(2) / growth_rate * 20
time_step = .1
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=20)
ratio_stat = (test_ret[:, 1, :] + .1) / (test_ret[:, 2, :] + .1)
ratio_stat = np.log(ratio_stat)
gmm = GaussianMixture(n_components=2, random_state=0, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))
gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

class_mean = gmm.means_

green_label = 0 if class_mean[0] > class_mean[1] else 1

test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

# ======== show the predict rets =============== #
fig3, ax3 = plt.subplots(1, 1)
ax3.scatter(test_ret[-2, 2], test_ret[-2, 1], c=test_cls, cmap='coolwarm', alpha=.1)
ax3.set_xlim(1, 1e3)
ax3.set_ylim(1, 1e3)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Red')
ax3.set_ylabel('Green')
splt.aspect_ratio(1)
fig3.show()

# ======== end =============== #


# %% Statistics for Population final States


# parameter_dict = [[10, 10]]  # [Green, Red]  [10, 10], [1, 80]
red_init = np.linspace(10, 80, num=8).astype(int)
green_init = np.linspace(1, 10, num=8).astype(int)
vx, vy = np.meshgrid(red_init, green_init)
# parameter_dict = [[15, 55],  # green, red
#                   [12, 55],
#                   [9, 55],
#                   [6, 55],
#                   [3, 55],
#                   [1, 55]]
parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
                                                     vx.flatten())]

results = []
for par in parameter_dict:

    size = 50000
    growth_rate = 1.6
    time_length = np.log(2) / growth_rate * 40
    time_step = .1  # .1

    repeats_num = 100
    final_rets = []

    for i in tqdm(range(repeats_num)):
        cells = pyrunBatchSim(22, growth_rate, par[0], par[1], time_length, time_step, size)

        cells_num = len(cells)
        final_state = np.zeros((cells_num, 2))
        for j, cell in enumerate(cells):
            final_state[j, 0] = cell.green[-1]
            final_state[j, 1] = cell.red[-1]

        ratio_stat = (final_state[:, 0] + .1) / (final_state[:, 1] + .1)
        ratio_stat = np.log(ratio_stat)
        state = gmm.predict(ratio_stat.reshape(-1, 1)) == green_label

        green_ration = np.sum(state) / size
        # print(green_ration)
        final_rets.append(green_ration)
    results.append(final_rets)
    # results.append(cells)
results = np.array(results).T

results_std = np.std(results, axis=0)
results_std = results_std.reshape(vx.shape)
results_std = results_std[::-1, :]
# distb = np.histogram(results[5], 10, range=(0, 1))

results = np.loadtxt(r'./SSA_pop_evolve_finalState_results.csv', delimiter=',')

fig2, ax2 = plt.subplots(8, 8, figsize=(40, 30))
ax2f = ax2[::-1, :]
ax2f = ax2f.flatten()
for i in range(64):
    ax2f[i].hist(results[:, i], density=True, bins=50)
    # counts, bins = np.histogram(results[:, i])
    # ax2f[i].stairs(counts, bins)
    ax2f[i].set_xlim(0, 1)
    ax2f[i].set_ylim(.05, 40)
    ax2f[i].set_yscale('log')
    ax2f[i].set_title(f'({parameter_dict[i][1]}, {parameter_dict[i][0]})')
    # splt.aspect_ratio(1, ax2f[i])
fig2.tight_layout(pad=-1.0)
fig2.savefig(r'./SSA_pop_evolve_finalState_results.svg')


# %% Monitor the population changes

red_init = np.linspace(10, 80, num=8).astype(int)
green_init = np.linspace(1, 10, num=8).astype(int)
vx, vy = np.meshgrid(red_init, green_init)
# parameter_dict = [[15, 55],  # green, red
#                   [12, 55],
#                   [9, 55],
#                   [6, 55],
#                   [3, 55],
#                   [1, 55]]
parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
                                                     vx.flatten())]

size = 10000
growth_rate = 1.6
time_length = np.log(2) / growth_rate * 40
time_step = .1  # .1

repeats_num = 100
final_rets = []

par = parameter_dict[0]
for i in tqdm(range(repeats_num)):
    cells = pyrunBatchSim(22, growth_rate, par[0], par[1], time_length, time_step, size)

    cells_num = len(cells)
    # final_state = np.zeros((cells_num, 2))

    allTimes = []
    allStates = []
    for j, cell in enumerate(cells):
        cellStates = CalcuRatio(cell.green, cell.red)
        cellStates = gmm.predict(cellStates.reshape(-1, 1)) == green_label
        allStates.append(cellStates)
        allTimes.append(cell.times)

    allTimes = np.concatenate(allTimes)
    allStates = np.concatenate(allStates)

    stats = binned_statistic(allTimes, allStates, bins=100)

    timesPoints = stats.bin_edges[:-1] + 0.5 * (stats.bin_edges[1] - stats.bin_edges[0])

    fig1, ax1 = plt.subplots(1, 1)
    ax1.plot(timesPoints, stats[0])
    ax1.set_ylim(0, 1)
    fig1.show()
