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
import matplotlib.pyplot as plt
# […]
from joblib import Parallel, delayed
# Own modules
import sciplot as splt
from kde_scatter_plot import kde_plot

splt.whitegrid()

# def ssa_thread_tagt(gr, green, red, time, step):
#     times, g, r = pyrunSim(gr, green, red, time, step)
#     rets = np.hstack([times.reshape((-1, 1)), g.reshape((-1, 1)), r.reshape((-1, 1))])
#     # buffer[index] = rets
#     return rets


# %%
if __name__ == '__main__':
    # %%
    from sklearn.mixture import GaussianMixture

    size = 1000000
    green = np.ones(size, dtype=int) * 20
    red = np.ones(size, dtype=int) * 15
    growth_rate = 1.6
    time_length = np.log(2) / growth_rate * 20
    time_step = .1
    test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=20)
    ratio_stat = (test_ret[:, 1, :] + .1) / (test_ret[:, 2, :] + .1)  # (Green + 1) / (Red +1 )
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

    # %%
    size = 100000
    green = np.ones(size, dtype=int) * 15
    red = np.ones(size, dtype=int) * 40
    growth_rate = 1.6
    time_length = np.log(2) / growth_rate * 20
    time_step = .1  # .1

    ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=20)
    ratio_stat = (ret[:, 1, :] + .1) / (ret[:, 2, :] + .1)
    ratio_stat = np.log(ratio_stat)
    class_pred = gmm.predict(ratio_stat.reshape(-1, 1))
    green_mask = class_pred == green_label

    green_mask = green_mask.reshape((ratio_stat.shape[0], ratio_stat.shape[-1]))

    green_number = np.sum(green_mask, axis=1)

    delta_green_ratio = np.diff(green_number) / (size - green_number[:-1]) / time_step

    time_list = np.arange(len(green_number)) * time_step
    time_diff = time_list[1:]
    green_ratio = green_number / size

    fig4, ax4 = plt.subplots(1, 1, figsize=(14, 12))
    ax4_tw = plt.twinx(ax4)
    ax4.plot(time_diff, delta_green_ratio)
    ax4_tw.plot(time_list, green_ratio, c='#a0ffa0')
    # ax4.set_yscale('log')
    # ax4.set_xlim(1, 20)
    # ax4.set_xlim(3, 10)
    # ax4.set_ylim(-0.01, 0.01)
    ax4.set_xlabel('Time, h')
    ax4.set_ylabel('Trans. rate')
    ax4_tw.set_ylabel('G ratio')
    fig4.show()

    graphPad_data = np.zeros((len(time_list) + len(time_diff), 3)) * np.nan
    graphPad_data[:len(time_list), 0] = time_list
    graphPad_data[:len(time_list), 1] = green_ratio
    graphPad_data[len(time_list):, 0] = time_diff
    graphPad_data[len(time_list):, 2] = delta_green_ratio

    # fig4, ax4 = plt.subplots(1, 1, figsize=(12, 12))
    # ax4.plot(green_tjtr[:, 0, -1], green_tjtr[:, 2, -1], '-r')
    # ax4.plot(green_tjtr[:, 0, -1], green_tjtr[:, 1, -1], '--g')
    # fig4.show()
#%%
    # KED plot
    sst_state = test_ret[-1, ...]
    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    kde_plot(sst_state[1:, :].T[:, ::-1], s=100)
    ax3.set_xlim(1, 1e3)
    ax3.set_ylim(1, 1e3)
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    splt.aspect_ratio(1)
    fig3.show()
    # %%
    cells = pyrunBatchSim(16, 1.3, 1, 1, 100, .1, 1e6)
