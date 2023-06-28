# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys

sys.path.extend(['D:\\python_code\\exp_data_explore\\CellMD3D_visulization'])
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from tqdm import tqdm
# Libs
import pandas as pd
import numpy as np  # Or any other

# Own modules
from CellMD3D_Visual import loadAllCells
import sciplot as splt
from colony_process import cartesian2polar, binned_along_radius
from joblib import dump, load

splt.whitegrid()


def logisticTarget(pars, y, x):
    a, k, n = pars

    return y - a / (1. + (x / k) ** n)


def fitGrowthTime(y, x):
    pars0 = np.array([1., 5., 1.])
    parsPrime = leastsq(logisticTarget, pars0, args=(y, x))[0]
    return parsPrime


# %%
ps = r'\\172.16.7.254\fulabshare\fulab_zc_1\ssa_in20_Colony_V0.2.11\Cells'
allCells = loadAllCells(ps)
cellsNum = 0
for i in allCells:
    cellsNum += len(i)

# t, ID, Length, GrowthRate, Ancestor, G, R, center_x, center_y, center_z
cells_array = np.zeros((cellsNum, 10))

index = 0
for cells in allCells:
    for cell in cells:
        t, ID, Length, GrowthRate, Ancestor, G, R, center_x, center_y, center_z = cell.t, cell.ID, cell.Length, cell.GrowthRate, \
                                                                                  cell.Ancestor, cell.G, cell.R, \
                                                                                  cell.center[0], \
                                                                                  cell.center[1], cell.center[2]
        cells_array[index, ...] = t, ID, Length, GrowthRate, Ancestor, G, R, center_x, center_y, center_z
        index += 1

cells_pd = pd.DataFrame(data=cells_array, columns=['t', 'ID', 'Length', 'GrowthRate', 'Ancestor', 'G', 'R',
                                                   'center_x', 'center_y', 'center_z'])

time_list = list(set(cells_pd['t'].tolist()))
time_list.sort()
colony_center = np.array([cells_pd[cells_pd['t'] == time][['center_x', 'center_y', 'center_z']].mean(axis=0).tolist()
                          for time in time_list])

for i, time in enumerate(tqdm(time_list)):
    mask = cells_pd['t'] == time
    temp_pd = cells_pd[mask][['center_x', 'center_y', 'center_z']]
    rho, _ = cartesian2polar(temp_pd['center_y'], temp_pd['center_x'], colony_center[i][::-1][1:])
    cells_pd.loc[mask, 'rho'] = rho

edgesForColonies = []
for i, time in enumerate(tqdm(time_list)):
    mask = cells_pd['t'] == time
    temp_pd = cells_pd[mask][['center_z', 'rho']]
    z_range = np.arange(0, temp_pd['center_z'].max()+.5, step=.5)
    edges_points = []
    for index in range(len(z_range) - 1):
        bin_pd = temp_pd[np.logical_and(temp_pd['center_z'] <= z_range[index + 1], temp_pd['center_z'] > z_range[index])]
        if not bin_pd.empty:
            edge_index = bin_pd['rho'].argmax()
            edges_points.append(bin_pd.iloc[edge_index][['rho', 'center_z']].to_list())
    edgesForColonies.append(np.array(edges_points))


fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
for i, time in enumerate(tqdm(time_list[::10])):
    if len(edgesForColonies[::10][i]) > 4:
        edge = edgesForColonies[::10][i]
        ax2.plot(edge[:, 0], edge[:, 1], '--', label='%.1f h' % time)
ax2.legend()
ax2.set_xlim(0)
ax2.set_ylim(0)

fig2.show()

cells_pd.to_csv(os.path.join(ps, 'allCells.csv'))

# %% Sample growth trajectory
# IDlist = np.arange(0, cells_pd['ID'].max() + 1, dtype=int)
IDlist = np.arange(0, cells_pd['ID'].max() + 1, dtype=int)

np.random.shuffle(IDlist)
sampleSize = 500

# pars = [None] * sampleSize
# for i, id in enumerate(IDlist[:sampleSize]):
#     cell1 = cells_pd[cells_pd['ID'] == id]
#     # pars.extend(fitGrowthTime(cell1['GrowthRate'], cell1['t']))
#     pars[i] = fitGrowthTime(cell1['GrowthRate'], cell1['t'])
#
# pars = np.array(pars)

fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
for i, id in enumerate(IDlist[:sampleSize]):
    cell1 = cells_pd[cells_pd['ID'] == id]
    # ax1.scatter(cell1['t'] - pars[i, 1], cell1['GrowthRate'])
    ax1.scatter(cell1['t'], cell1['GrowthRate'])

    # ax1.text(2, 1.5, ' <n> = %.2f' % np.median(pars[:, 2]), size=20)
# ax1.set_xlim(-5, 5)
ax1.set_xlabel('Time, h')
ax1.set_ylim(0, 1.25)
ax1.set_ylabel('Elongation rate, $h^{-1}$')
ax1.set_title('Growth rate trajectory')

# splt.aspect_ratio(1, ax1)
fig1.show()
