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


#%%
from sklearn.mixture import GaussianMixture

size = 1000000
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

# %%

parameter_dict = [[15, 35],
                  [12, 35],
                  [9, 35],
                  [6, 35],
                  [3, 35],
                  [1, 35]]
results_dict = []
fig4, ax4 = plt.subplots(1, 1, figsize=(14, 12))
ax4_tw = plt.twinx(ax4)
for par in parameter_dict:
    size = 100000
    green = np.ones(size, dtype=int) * par[0]
    red = np.ones(size, dtype=int) * par[1]
    growth_rate = 1.3
    time_length = np.log(2) / growth_rate * 30
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


    ax4.plot(time_diff, delta_green_ratio)
    ax4_tw.plot(time_list, green_ratio, c='#a0ffa0')

    ax4.set_xlabel('Time, h')
    ax4.set_ylabel('Trans. rate')
    ax4_tw.set_ylabel('G ratio')
    # fig4.show()

    graphPad_data = np.zeros((len(time_list) + len(time_diff), 3)) * np.nan
    graphPad_data[:len(time_list), 0] = time_list
    graphPad_data[:len(time_list), 1] = green_ratio
    graphPad_data[len(time_list):, 0] = time_diff
    graphPad_data[len(time_list):, 2] = delta_green_ratio
    # graphPad_pd = pd.DataFrame(data=graphPad_data, columns=['Time', 'Trans rate', 'G ratio'])
    results_dict.append(graphPad_data.copy())
fig4.show()
alldata = np.zeros((len(time_list) + len(time_diff), 1 + 2*6)) * np.nan

for i, data in enumerate(results_dict):

    alldata[:, 1+i*2:3+i*2] = data[:, 1:]

