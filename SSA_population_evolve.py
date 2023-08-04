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
from joblib import Parallel, delayed, load, dump

import sciplot
# Own modules
import sciplot as splt
from kde_scatter_plot import kde_plot
from scipy.stats import binned_statistic

splt.whitegrid()


# for i in tqdm(range(10)):
# cells = pyrunBatchSim(22, 1.2, 1, 1, 10, .1, 1e3)
# # cells = pyrunBatchSim(22, 1.2, 1, 1, 10, 1, 1e5)
def CalculateGreenRatio(stats: np.ndarray):
    return np.sum(stats) / len(stats)


def CalcuRatio(greenNum, redNum):
    return np.log((greenNum + .1) / (redNum + .1))


def RG2Predict(green_signal, red_signal):
    return np.log((green_signal + 1) / (red_signal + 1))


RedColor = np.array((241, 148, 138)) / 255
GreenColor = np.array((130, 224, 170)) / 255
# %%  Training a model for stat discrimination


size = 20000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 50
growth_rate = 1.6
time_length = np.log(2) / growth_rate * 50
time_step = .1
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=20)
ratio_stat = CalcuRatio(test_ret[:, 1, :], test_ret[:, 2, :])
# gmm = GaussianMixture(n_components=2, random_state=1, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

gmm = load(r'./discrimination_model/Gaussian_discrimination_gr-1.6.joblib')['gmm']

class_mean = gmm.means_

green_label = 0 if class_mean[0] > class_mean[1] else 1

test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

# ======== show the predict rets =============== #
fig3, ax3 = plt.subplots(1, 1)
green_cells_mask = test_cls == green_label

ax3.scatter(test_ret[-2, 2][green_cells_mask], test_ret[-2, 1][green_cells_mask],
            color=GreenColor, alpha=.1)
ax3.scatter(test_ret[-2, 2][~green_cells_mask], test_ret[-2, 1][~green_cells_mask],
            color=RedColor, alpha=.1)
# ax3.scatter(test_ret[-2, 2], test_ret[-2, 1], c=test_cls, cmap='coolwarm', alpha=.1)
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
red_init = np.linspace(10, 100, num=10).astype(int)
green_init = np.linspace(0, 10, num=10).astype(int)
vx, vy = np.meshgrid(red_init, green_init)
# parameter_dict = [[15, 55],  # green, red
#                   [12, 55],
#                   [9, 55],
#                   [6, 55],
#                   [3, 55],
#                   [1, 55]]
parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
                                                     vx.flatten())]

growth_rate = 1.6
time_length = 1.6 * 6 / growth_rate  # np.log(2) / growth_rate * 40
size = int(np.exp(growth_rate * time_length) * .9)
time_step = .1  # .1

repeats_num = 1000
results = []


def simulationAndDis(args):
    growth_rate, par, time_length, time_step, size, gmm = args
    cells = pyrunBatchSim(60, growth_rate, par[0], par[1], time_length, time_step, size)

    cells_num = len(cells)
    final_state = np.zeros((cells_num, 2))
    for j, cell in enumerate(cells):
        final_state[j, 0] = cell.green[-1]
        final_state[j, 1] = cell.red[-1]

    ratio_stat = (final_state[:, 0] + .1) / (final_state[:, 1] + .1)
    ratio_stat = np.log(ratio_stat)
    state = gmm.predict(ratio_stat.reshape(-1, 1)) == green_label

    green_ration = np.sum(state) / size
    return 1 - green_ration


for par in parameter_dict:
    # loop_results = []
    # for i in tqdm(range(repeats_num)):
    #     cells = pyrunBatchSim(64, growth_rate, par[0], par[1], time_length, time_step, size)
    #
    #     cells_num = len(cells)
    #     final_state = np.zeros((cells_num, 2))
    #     for j, cell in enumerate(cells):
    #         final_state[j, 0] = cell.green[-1]
    #         final_state[j, 1] = cell.red[-1]
    #
    #     ratio_stat = (final_state[:, 0] + .1) / (final_state[:, 1] + .1)
    #     ratio_stat = np.log(ratio_stat)
    #     state = gmm.predict(ratio_stat.reshape(-1, 1)) == green_label
    #
    #     green_ration = np.sum(state) / size
    #     # print(green_ration)
    #     loop_results.append(green_ration)
    # results.append(loop_results)
    results.append(Parallel(n_jobs=8, require='sharedmem')(delayed(simulationAndDis)
                                                            ((growth_rate, par, time_length, time_step, size, gmm), )
                                                            for _ in tqdm(range(repeats_num))))
    # results.append(cells)
results = np.array(results).T  # [Sample,  different init]
# results_std = np.std(results, axis=0)
# results_std = results_std.reshape(vx.shape)
# results_std = results_std[::-1, :]
# distb = np.histogram(results[5], 10, range=(0, 1))

# results = np.loadtxt(r'./Data/SSA_pop_evolve_finalState_results.csv', delimiter=',')

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
fig2.savefig(r'./Data/SSA_pop_evolve_finalState_results.svg')

dump_data = dict(parameter_dict=parameter_dict, results=results,
                 growth_rate=growth_rate,
                 time_length=time_length,
                 size=size,
                 time_step=time_step,
                 repeats_num=repeats_num)
dump(dump_data, r'./Data/SSA_pop_evolve_results.joblib')

# %% Monitor the population changes
#
# red_init = np.linspace(10, 80, num=8).astype(int)
# green_init = np.linspace(1, 10, num=8).astype(int)
# vx, vy = np.meshgrid(red_init, green_init)
# # parameter_dict = [[15, 55],  # green, red
# #                   [12, 55],
# #                   [9, 55],
# #                   [6, 55],
# #                   [3, 55],
# #                   [1, 55]]
# parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
#                                                      vx.flatten())]
#
# growth_rate = 1.6
# time_length = 1.6 * 6 / growth_rate  # np.log(2) / growth_rate * 40
# size = int(np.exp(growth_rate * time_length))
# time_step = .1  # .1
#
# repeats_num = 100
# loop_results = []
#
# par = parameter_dict[0]
# for i in tqdm(range(repeats_num)):
#     cells = pyrunBatchSim(22, growth_rate, par[0], par[1], time_length, time_step, size)
#
#     cells_num = len(cells)
#     # final_state = np.zeros((cells_num, 2))
#
#     allTimes = []
#     allStates = []
#     for j, cell in enumerate(cells):
#         cellStates = CalcuRatio(cell.green, cell.red)
#         cellStates = gmm.predict(cellStates.reshape(-1, 1)) == green_label
#         allStates.append(cellStates)
#         allTimes.append(cell.times)
#
#     allTimes = np.concatenate(allTimes)
#     allStates = np.concatenate(allStates)
#
#     stats = binned_statistic(allTimes, allStates, bins=100)
#
#     timesPoints = stats.bin_edges[:-1] + 0.5 * (stats.bin_edges[1] - stats.bin_edges[0])
#
#     fig1, ax1 = plt.subplots(1, 1)
#     ax1.plot(timesPoints, stats[0])
#     ax1.set_ylim(0, 1)
#     fig1.show()
