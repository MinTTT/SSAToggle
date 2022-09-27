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
# Own modules
import sciplot as splt
from kde_scatter_plot import kde_plot

splt.whitegrid()


# for i in tqdm(range(10)):
cells = pyrunBatchSim(22, 1.2, 1, 1, 10, .1, 1e3)
# # cells = pyrunBatchSim(22, 1.2, 1, 1, 10, 1, 1e5)
# %%  Training a model for stat discrimination


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
# fig3, ax3 = plt.subplots(1, 1)
# ax3.scatter(test_ret[-2, 2], test_ret[-2, 1], c=test_cls, cmap='coolwarm', alpha=.1)
# ax3.set_xlim(1, 1e3)
# ax3.set_ylim(1, 1e3)
# ax3.set_xscale('log')
# ax3.set_yscale('log')
# ax3.set_xlabel('Red')
# ax3.set_ylabel('Green')
# splt.aspect_ratio(1)
# fig3.show()
# ======== end =============== #
# %%

parameter_dict = [[15, 35],
                  [12, 35],
                  [9, 35],
                  [6, 35],
                  [3, 35],
                  [1, 35]]
results = []
for par in parameter_dict:

    size = 100000
    growth_rate = 1.3
    time_length = np.log(2) / growth_rate * 30
    time_step = .1  # .1

    repeats_num = 5000
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

# distb = np.histogram(results[5], 10, range=(0, 1))
