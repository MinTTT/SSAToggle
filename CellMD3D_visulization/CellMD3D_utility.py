from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sciplot as splt
import numpy as np
from tqdm import tqdm

splt.whitegrid()
import os
import threading

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
        CellFilesPath (str): the directory for cells data.

    Returns:
        data (List[List[Cell]]): list(list(Cell obj)
    """

    def readCellTarget(CellFilePath, buff, index):
        buff[index] = read_cells_rets(CellFilePath, False)

        return None

    filesDir = [file.name for file in os.scandir(CellFilesPath) if file.is_file()]
    filesDir.sort(key=lambda name: int(name.split('.')[0]))
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
