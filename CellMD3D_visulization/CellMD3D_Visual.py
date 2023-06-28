# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
# […]
import time

from tqdm import tqdm
# Libs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Or any other
from scipy.stats import binned_statistic_dd, binned_statistic
import sciplot as splt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mpl_color
# from colony_process import cartesian2polar, binned_along_radius
import threading
from typing import List, Optional
from kde_scatter_plot import kde_plot
from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim

red = np.array([231, 76, 60]) / 255
white = np.array([1, 1, 1])
green = np.array([93, 173, 226]) / 255
nodes = [0.0, .5, 1.0]

RedGreen_cmap = LinearSegmentedColormap.from_list('RedGreen', list(zip(nodes, [red, white, green])))

splt.whitegrid()
# […]

# Own modules

paras_location = [0, 1, 2, slice(3, 6), slice(6, 9), 9, slice(10, 13), slice(13, 16), 16, slice(17, 20),
                  slice(20, 23),
                  23, 24, 25, 26, 27, 28, 29]


# |t 0|ID|Type|p|q|Length|Tensor|Velocity|GrowthRate|DynFric|StaFric|time_p|time_q|age_p|age_q|Ancestor|G|R|

def read_cells_rets(CellFilePath, pbar=True):
    lines = []
    if pbar:
        p_bar = tqdm()
    with open(CellFilePath) as ret_file:
        while True:
            line = ret_file.readline()
            if line == '':
                break
            cell_pars = [float(par) for par in line.replace('\n', '').split(' ')]

            cell_parameters = [np.array(cell_pars[index]) if isinstance(index, slice) else cell_pars[index]
                               for index in paras_location]
            lines.append(Cell(*cell_parameters))
            if pbar:
                p_bar.update()

    return lines


class Cell:

    def __init__(self, t, ID, Type,  # 0, 1, 2
                 p: np.array,  # 3:6
                 q: np.array,  # 6:9
                 Length,  # 9
                 T: np.array,  # 10:13
                 Velocity: np.array,  # 13:16
                 GrowthRate,  # 16
                 DynFric, StaFric,  # slice(17, 20), slice(20, 23),
                 time_p, time_q,
                 age_p, age_q,
                 Ancestor, G, R):
        self.t = t
        self.ID = ID
        self.Type = Type
        self.p = p
        self.q = q
        self.Length = Length
        self.T = T
        self.Velocity = Velocity
        self.GrowthRate = GrowthRate
        self.DynFric = DynFric
        self.StaFric = StaFric
        self.time_p = time_p
        self.time_q = time_q
        self.age_p = age_p
        self.age_q = age_q
        self.Ancestor = Ancestor
        self.G = G
        self.R = R
        self.state = None

        self.center = self.p + self.q / 2


def loadAllCells(CellFilesPath):
    """Load all cells data from directory.

    Parameters:
        CellFilesPath (str): the PATH for cells data.

    Returns:
        data (List[List[Cell]]): list(list(Cell obj)
    """

    def readCellTarget(CellFilePath, buff, index):
        buff[index] = read_cells_rets(CellFilePath, False)

        return None

    filesDir = [file.name for file in os.scandir(CellFilesPath) if file.is_file()]
    file_number = len(filesDir)

    data = [None] * file_number
    read_threads = [threading.Thread(target=readCellTarget, args=(os.path.join(CellFilesPath, path), data, i)) for
                    i, path in enumerate(filesDir)]
    for read_thread in read_threads:
        read_thread.start()

    pbar = tqdm(total=len(filesDir))
    start_count = 0
    while True:
        count = 0
        for i in data:
            if i is not None:
                count += 1
        if start_count < count:
            pbar.update(count - start_count)
            start_count = count
        if start_count == file_number:
            break

    return data


def value2color(data, color):
    color_value = np.zeros(data.shape + (3,))
    rgb_color = np.array(color) / 255
    norm_data = data / np.ptp(data)
    for i in range(3):
        color_value[..., i] = norm_data * rgb_color[i]
    return color_value


def value2color_cmap(data, cmap):
    # color_value = np.zeros(data.shape + (3,))
    data_mask = ~np.isnan(data)
    data_norm = (data - data[data_mask].min()) / np.ptp(data[data_mask])
    color_value = cmap(data_norm)
    return color_value


def data_norm(data):
    data_mask = ~np.isnan(data)
    data_norm = (data - data[data_mask].min()) / np.ptp(data[data_mask])
    return data_norm


def cells2cc1(filePath, save_dir=None):
    """Convert the Cell data to cc1 file for rendering in PyMol.

    .cc1 file format:
    row1: atom total number;
    col1: atom type; col2: atom index; col3: x, col4: y; col5: z; col6 force; col7 bond.

    Parameters
    --------
    filePath: string
        the exported simulation data.

    save_dir: string or None, default None
        the directory specified for saving the .cc1 file.

    """
    cells = read_cells_rets(filePath)
    fileName = os.path.basename(filePath)
    fileName = '.'.join([fileName.split('.')[0], 'cc1'])
    total_atoms = 2 * len(cells)

    if save_dir is None:
        save_dir = os.path.dirname(filePath)
    else:
        try:
            os.mkdir(save_dir)
            print(f"Make dir: {save_dir}")
        except FileExistsError:
            pass
    with open(os.path.join(save_dir, fileName), 'w') as file:
        file.write(f'{total_atoms}\n')
        for i, cell in enumerate(cells):
            str1 = f"C {i * 2 + 1} {' '.join(cell.p.astype(str))} 1 {i * 2 + 2}\n"
            str2 = f"C {i * 2 + 2} {' '.join(cell.q.astype(str))} 1 {i * 2 + 1}\n"
            file.write(str1)
            file.write(str2)
    return None


def cartesian2polar(yv, xv, center):
    """
    the center should have axis0, axis1 order (y, x).

    Args:
        yv (np.ndarray): y indexes;
        xv (np.ndarray): x indexes;
        center (array-like): (y, x);

    Returns:
        tuple: pho, phi
    """
    dyv = yv - center[0]
    dxv = xv - center[1]
    pho = np.sqrt(dyv ** 2 + dxv ** 2)
    phi = np.arccos(dyv / pho)
    phi[dxv < 0] = 2 * np.pi - phi[dxv < 0]
    return pho, phi


def binned_along_radius(pho, data, bins=np.linspace(0, 1100, num=100)):
    """_summary_

    Args:
        pho (_type_): _description_
        data (_type_): _description_
        bins (_type_, optional): _description_. Defaults to np.linspace(0, 1100, num=100).

    Returns:
        tuple: the average central, mean values of bins, standard deviation of bins.
    """
    bin_mean_rets = binned_statistic(pho.flatten(), data.flatten(), statistic='mean',
                                     bins=bins)
    bin_std_rets = binned_statistic(pho.flatten(), data.flatten(), statistic='std',
                                    bins=bins)
    radius_points = (bins[:-1] + bins[1:]) / 2
    return radius_points, bin_mean_rets[0], bin_std_rets[0]


# %% Discriminate cell state

from sklearn.mixture import GaussianMixture

growth_rate = 1.6
size = 100000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 50
time_length = np.log(2) / growth_rate * 10
time_step = .1
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=22)
ratio_stat = (test_ret[:, 1, :] + .1) / (test_ret[:, 2, :] + .1)  # (Green + 1) / (Red +1 )
ratio_stat = np.log(ratio_stat)
gmm = GaussianMixture(n_components=2, random_state=0, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))
gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

class_mean = gmm.means_

green_label = 0 if class_mean[0] > class_mean[1] else 1

test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

# ======== show the predict quality =============== #
fig3, ax3 = plt.subplots(1, 1)
ax3.scatter(test_ret[-2, 2], test_ret[-2, 1], c=test_cls, cmap='coolwarm', alpha=.1)
ax3.set_xlim(1, 100)
ax3.set_ylim(1, 200)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Red')
ax3.set_ylabel('Green')
splt.aspect_ratio(1)
fig3.show()

# %%
if __name__ == '__main__':

    # %%  Import Data
    file_ps = r'Y:\fulab_zc_1\sunhui_code_ret\ssa_in25_colony_agent_based_compiled.2\Cells\27.txt'

    cells = read_cells_rets(file_ps)

    cells_all_location = np.array([cell.center for cell in cells])
    cells_all_R = np.array([cell.R for cell in cells])
    cells_all_G = np.array([cell.G for cell in cells])
    cells_all_lambda = np.array([cell.GrowthRate for cell in cells])

    cells_all_states = gmm.predict(np.log((cells_all_G + 1.) / (cells_all_R + 1.)).reshape(-1, 1))
    # %%
    # ============== Parameters for Draw =============================#
    z_top = 6
    z_bottom = -4
    location_mask = np.logical_and(cells_all_location[..., -1] > z_bottom, cells_all_location[..., -1] < z_top)
    range_factor = 1.1
    x_bin_length = 5
    y_bin_length = 5
    z_bin_length = 2
    co_axis = True

    cells_location = cells_all_location[location_mask, ...]
    cells_R = cells_all_R[location_mask]
    cells_G = cells_all_G[location_mask]
    cells_states = cells_all_states[location_mask]
    locations_min = cells_location.min(axis=0)
    locations_max = cells_location.max(axis=0)
    colony_center = np.median(cells_location, axis=0)

    # plot cell G/R states
    from kde_scatter_plot import kde_plot

    fig1, ax1 = plt.subplots(1, 1)
    kde_plot(np.hstack([cells_R.reshape(-1, 1), cells_G.reshape(-1, 1)]), axes=ax1)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(1, 200)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    splt.aspect_ratio(1)
    fig1.show()
    # %

    cells_polar_axis = cartesian2polar(cells_location[:, 1], cells_location[:, 0], colony_center[0:2][::-1])
    cells_R_binned_stats = binned_along_radius(cells_polar_axis[0], cells_R / (cells_R + cells_G),
                                               np.linspace(0, cells_polar_axis[0].max(), num=50))
    cells_G_binned_stats = binned_along_radius(cells_polar_axis[0], cells_G / (cells_R + cells_G),
                                               np.linspace(0, cells_polar_axis[0].max(), num=50))
    cells_states_binned_stats = binned_along_radius(cells_polar_axis[0], cells_states,
                                                    np.linspace(0, cells_polar_axis[0].max(), num=50))
    # cells_GoR_binned_stats = binned_along_radius(cells_polar_axis[0], cells_G / (cells_R + cells_G),
    #                                              np.linspace(0, cells_polar_axis[0].max(), num=50))

    x_range = np.arange(locations_min[0] * range_factor, locations_max[0] * range_factor, step=x_bin_length)
    y_range = np.arange(locations_min[1] * range_factor, locations_max[1] * range_factor, step=y_bin_length)
    z_range = np.arange(locations_min[2] * range_factor, locations_max[2] * range_factor, step=z_bin_length)

    R_binned = binned_statistic_dd(cells_location, cells_R, statistic='mean', bins=[x_range, y_range, z_range])
    G_binned = binned_statistic_dd(cells_location, cells_G, statistic='mean', bins=[x_range, y_range, z_range])
    states_binned = binned_statistic_dd(cells_location, cells_states, statistic='mean',
                                        bins=[x_range, y_range, z_range])
    binned_centers = [(edge[:-1] + edge[1:]) / 2 for edge in R_binned[-2]]
    binned_edges = R_binned[-2]
    xyz_e = np.meshgrid(*binned_edges, indexing='ij')
    volex_mask = ~np.isnan(R_binned[0])

    # R_color = np.zeros(R_binned[0].shape + (3,))
    # R_color_scaler = (R_binned[0] / R_binned[0][volex_mask].max())
    # R_color_max = np.array([241, 148, 138]) / 255

    R_color = value2color(R_binned[0] / (R_binned[0] + G_binned[0]), (241, 148, 138))
    G_color = value2color(G_binned[0] / (R_binned[0] + G_binned[0]), (130, 224, 170))
    G2R_color = value2color_cmap(G_binned[0] / R_binned[0], RedGreen_cmap)
    states_color = np.ones(states_binned[0].shape + (3,)) * np.nan
    states_color[states_binned[0] > 0.5] = (130, 224, 170)
    states_color[states_binned[0] <= 0.5] = (241, 148, 138)
    states_color = states_color / 255
    #% Plot Colony Bottom

    fig2, ax2 = plt.subplots(1, 1)
    ax2.scatter(*(cells_location[:, :2][cells_states == green_label]).T,
                color=np.array((130, 224, 170))/255, s=4, alpha=.5)
    ax2.scatter(*(cells_location[:, :2][cells_states != green_label]).T,
                color=np.array((241, 148, 138))/255, s=4, alpha=.5)

    ax2.set_xlim((-250, 250))
    ax2.set_ylim((-250, 250))

    fig2.show()

    # %%  Plot 3D colony
    fig1, ax1 = plt.subplots(1, 3, subplot_kw=dict(projection='3d'), figsize=(16 * 3 * 1.2, 16))
    ax1[0].voxels(*xyz_e, volex_mask, facecolors=R_color, alpha=.8)
    ax1[1].voxels(*xyz_e, volex_mask, facecolors=G_color, alpha=.8)
    ax1[2].voxels(*xyz_e, volex_mask, facecolors=states_color, alpha=.8)
    for ax in ax1:
        if co_axis:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            left_min = np.min([xlim[0], ylim[0], zlim[0]])
            right_max = np.max([xlim[1], ylim[1], zlim[1]])
            ax.set_xlim(left_min, right_max)
            ax.set_ylim(left_min, right_max)
            ax.set_zlim(0, right_max - left_min)
        ax.set_xlabel('X axis, $\mu m$', labelpad=30)
        ax.set_ylabel('y axis, $\mu m$', labelpad=30)
        ax.set_zlabel('y axis, $\mu m$', labelpad=30)

    cmp_ax = fig1.add_axes([.97, .1, .98, .8])
    fig1.colorbar(cm.ScalarMappable(norm=mpl_color.Normalize(0, 1), cmap=RedGreen_cmap), cmp_ax)
    fig1.suptitle(file_ps)
    fig1.show()

    # %% Profile
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 12))
    ax22_twin = plt.twinx(ax2)
    ax2.plot(cells_R_binned_stats[0], (cells_R_binned_stats[1]), 'r')
    ax22_twin.plot(cells_G_binned_stats[0], (cells_G_binned_stats[1]), 'g')
    # ax22_twin.plot(cells_GoR_binned_stats[0], (cells_GoR_binned_stats[1]), 'y--')
    splt.aspect_ratio(1.2)
    fig2.show()

    # %% All cell state plot
    fig3, ax3 = plt.subplots(1, 1)
    # ax3.scatter(cells_all_R, cells_all_G)
    kde_plot(np.array([cells_all_R, cells_all_G]).T)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(1, 1e3)
    ax3.set_ylim(1, 1e3)
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    splt.aspect_ratio(1)
    fig3.show()
