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

import numpy as np  # Or any other
from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sciplot as splt
from sklearn.mixture import GaussianMixture
from joblib import dump, load, Parallel, delayed
from tqdm import tqdm
import threading
from typing import List, Optional

splt.whitegrid()

import datetime


def RG2Predict(green_signal, red_signal):
    return np.log((green_signal + 1) / (red_signal + 1))


RedColor = np.array((241, 148, 138)) / 255
GreenColor = np.array((130, 224, 170)) / 255
# %%

results_directory = r'/media/fulab/Data_Raid/Colony_Project/SSA_Toggle_results/20230723_tau_distribution'

if not os.path.isdir(results_directory):
    os.makedirs(results_directory)

size = 200000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 50
growth_rate = 1.6
time_length = np.log(2) / growth_rate * 50
time_step = .1
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=24)
#
ratio_stat = RG2Predict(test_ret[:, 1, :], test_ret[:, 2, :])
# gmm = GaussianMixture(n_components=2, random_state=1, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

gmm = load(r'./discrimination_model/Gaussian_discrimination_gr-1.6.joblib')['gmm']
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
fig3.savefig(os.path.join(results_directory, 'cell_states_prediction.png'))

# %%  This part used to generate results
# red_init = np.linspace(10, 100, num=10).astype(int)
# green_init = np.linspace(0, 10, num=10).astype(int)
# vx, vy = np.meshgrid(red_init, green_init)
#
# parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
#                                                      vx.flatten())]

# =========== I want to specify the parameters manually.
parameter_dict = [[60, 0],
                  [5, 20],
                  [5, 100],
                  [10, 60]]
# =====================================================

batch_repeat_num = 2500
growth_rate = 1.6
time_step = .05  # .1


def simulation_thread(args):
    parameter_dict, batch_repeat_num, growth_rate, time_step, results_directory, gmm_, green_label = args
    batch_number = len(parameter_dict)
    size = batch_number * batch_repeat_num
    init_array = np.array(parameter_dict)  # [batch sample ..., green / red]
    green = np.hstack([init_array[:, 0].flatten()] * batch_repeat_num)
    red = np.hstack([init_array[:, 1].flatten()] * batch_repeat_num)
    time_length = np.log(2) / growth_rate * 1000  # 5000 generation

    # print(f"Start Sim: {par}. \n")
    ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=60)
    # Attention! Don't save simulation raw data, it's too big!
    # print(f"End Sim: {par}. \n")
    # time_now = datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%S')
    # dump(ret, os.path.join(results_directory, f'cells_stats_G{par[0]}_R{par[1]}_lambda{growth_rate}_{time_now}.pkl'))
    ratio_stat = RG2Predict(ret[:, 1, :], ret[:, 2, :])
    class_pred = gmm_.predict(ratio_stat.reshape(-1, 1))
    green_mask = class_pred == green_label
    green_mask = green_mask.reshape(ratio_stat.shape)  # [Time, Cell]

    stateChange = np.vstack([np.zeros((1, green_mask.shape[1])), green_mask])
    stateChange = np.diff(stateChange, axis=0)

    TransTime = np.zeros(size) * np.nan
    for i in range(size):
        TransTime[i] = np.argmax(stateChange[:, i] == 1) * time_step  # tau R->G

    init_states = np.hstack([green.reshape(-1, 1), red.reshape(-1, 1)])
    summary_data = dict(cell_stats_init=init_states,
                        init_parameters=init_array,
                        tua=TransTime,
                        growth_rate=growth_rate,
                        time_step=time_step)
    time_now = datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%S')
    dump(summary_data, os.path.join(results_directory, f'cells_tau_lambda{growth_rate}_{time_now}.pkl'))
    return None


active_threads_num = 12
repeats_number = 500

# workers_list = Parallel(n_jobs=32, verbose=100)(delayed(simulation_thread) ((parameter_dict, batch_repeat_num,
# growth_rate, time_step, results_directory, gmm, green_label), ) for _ in range(active_threads_num * repeats_number)) #
worker_list = [None] * active_threads_num  # type: List[Optional[threading.Thread]]
for worker_i in range(active_threads_num):
    worker_list[worker_i] = threading.Thread(target=simulation_thread,
                                             args=((parameter_dict, batch_repeat_num, growth_rate, time_step,
                                                    results_directory, gmm, green_label),))
# start all
for worker in worker_list:
    worker.start()

worker_number = active_threads_num * repeats_number
finished_number = 0
pbar = tqdm(total=worker_number)
while finished_number <= worker_number:
    dead_worker_index = []
    for worker_i, worker in enumerate(worker_list):
        if not worker.is_alive():
            dead_worker_index.append(worker_i)
    if dead_worker_index:
        finished_number += 1
        pbar.update(1)
        for dead_index in dead_worker_index:
            worker_list[dead_index] = threading.Thread(target=simulation_thread,
                                                       args=((parameter_dict, batch_repeat_num, growth_rate, time_step,
                                                              results_directory, gmm, green_label),))
            worker_list[dead_index].start()
    time.sleep(1)

for worker in worker_list:
    if worker.is_alive():
        worker.join()
pbar.close()

# # for par in tqdm(parameter_dict):
# batch_number = len(parameter_dict)
# size = batch_number * batch_repeat_num
# init_array = np.array(parameter_dict)  # [batch sample ..., green / red]
# green = np.hstack([init_array[:, 0].flatten()] * batch_repeat_num)
# red = np.hstack([init_array[:, 1].flatten()] * batch_repeat_num)
# time_length = np.log(2) / growth_rate * 5000  # 5000 generation
# time_step = .05  # .1
# # print(f"Start Sim: {par}. \n")
# ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=2)
# # Attention! Don't save simulation raw data, it's too big!
# # print(f"End Sim: {par}. \n")
# # time_now = datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%S')
# # dump(ret, os.path.join(results_directory, f'cells_stats_G{par[0]}_R{par[1]}_lambda{growth_rate}_{time_now}.pkl'))
# ratio_stat = RG2Predict(ret[:, 1, :], ret[:, 2, :])
# class_pred = gmm.predict(ratio_stat.reshape(-1, 1))
# green_mask = class_pred == green_label
# green_mask = green_mask.reshape(ratio_stat.shape)  # [Time, Cell]
#
# stateChange = np.vstack([np.zeros((1, green_mask.shape[1])), green_mask])
# stateChange = np.diff(stateChange, axis=0)
#
# TransTime = np.zeros(size) * np.nan
# for i in range(size):
#     TransTime[i] = np.argmax(stateChange[:, i] == 1) * time_step  # tau R->G
#
# # transTimeList.append(TransTime.reshape(-1, 1))
#
# summary_data = dict(cell_stats_init=np.hstack([green.reshape(-1, 1), red.reshape(-1, 1)]),
#                     tua=TransTime)
# time_now = datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%S')
# dump(summary_data, os.path.join(results_directory, f'cells_tau_lambda{growth_rate}_{time_now}.pkl'))
