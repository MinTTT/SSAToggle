"""
@author: CHU Pan
@mail: pan_chu@outlook.com
@date: 2023/6/29
"""

# %%
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import sciplot as splt
import numpy as np

splt.whitegrid()
from CellMD3D_visulization.CellMD3D_utility import read_cells_rets

from typing import Union, Tuple, List


def cell_vertices(radius: float, p1: Union[Tuple, List], p2: Union[Tuple, List], num: int = 40):
    """

    Parameters
    ----------
    radius : float
    p1 : tuple or list
    p2 : tuple or list
    num : int

    Returns
    -------

    """
    if num % 2 != 0:
        num += 1
    vertices_num = num
    direction = (np.array(p2) - np.array(p1))
    norm_direction = direction / np.sqrt(np.sum(direction ** 2))

    vertical = np.array([1, - norm_direction[0] / norm_direction[1]])
    norm_vertical = vertical / np.sqrt(np.sum(vertical ** 2))
    norm_vertical_2 = -norm_vertical
    mid_point = (np.array(p2) + np.array(p1)) / 2
    vertical_point_1 = norm_vertical * radius + mid_point
    vertical_point_2 = norm_vertical_2 * radius + mid_point
    start_phi = np.arctan(norm_vertical[1] / norm_vertical[0])
    rho = np.linspace(start_phi, start_phi + np.pi * 2, vertices_num, endpoint=True)
    half_index = int(vertices_num / 2)
    vertices = np.ones((vertices_num, 2))
    vertices[:, 0] = np.cos(rho) * radius
    # vertices[half_index+1:-1, 0] = np.cos(rho[half_index-1:]) * radius

    vertices[:, 1] = np.sin(rho) * radius
    # vertices[half_index+1:-1, 1] = np.sin(rho[half_index-1:]) * radius

    dot_vertices = np.dot(vertices, direction)
    p1_mask = dot_vertices <= 0
    p2_mask = dot_vertices > 0
    vertices[p1_mask] = vertices[p1_mask] + np.array(p1)
    vertices[p2_mask] = vertices[p2_mask] + np.array(p2)

    vertices_path = np.ones((vertices_num + 3, 2))
    vertices_path[0, :] = vertical_point_1
    vertices_path[-1, :] = vertical_point_1
    vertices_path[half_index + 1, :] = vertical_point_2

    vertices_path[1:half_index + 1, :] = vertices[:half_index, :]
    vertices_path[half_index + 2:-1, :] = vertices[half_index:, :]
    return vertices_path


def cell_patches(radius: float, p1: Union[Tuple, List], p2: Union[Tuple, List], face_color: str, num=40):
    """

    Parameters
    ----------
    face_color :
    radius: float
        cell radius
    p1: tuple or list
        first sphere (x, y)
    p2: tuple or list
        second sphere (x, y)
    facecolor: string or color code
        cell color
    num

    Returns
    -------

    """
    return mpatches.PathPatch(cell_path(radius, p1, p2), facecolor=face_color, edgecolor='k', alpha=.6)


def cell_path(radius, p1, p2):
    return mpath.Path(cell_vertices(radius, p1, p2))


# %%
if __name__ == '__main__':
    cells_path = r'.\CellMD3D_visulization\example_cell_data.txt'
    cells = read_cells_rets(cells_path)

    cells_patches = [cell_patches(0.7, cell.p[:-1], cell.q[:-1], 'g') for cell in cells]

    # radius = .1
    #
    # p1 = (5.1, 2.1)
    # p2 = (5.2, 2.2)
    #
    # vertices_path = cell_vertices(radius, p1, p2)

    # cell_path = mpath.Path(vertices_path)
    # cell_patch = mpatches.PathPatch(cell_path, facecolor='g', edgecolor='k')
    fig1, ax1 = plt.subplots(1)
    # ax1.scatter(vertices[:, 0], vertices[:, 1])
    for cell_p in cells_patches:
        ax1.add_patch(cell_p)
    # ax1.set_xlim(0, 10)
    # ax1.set_ylim(0, 10)
    ax1.set_xlim(-150, 150)
    ax1.set_ylim(-150, 150)

    fig1.show()
