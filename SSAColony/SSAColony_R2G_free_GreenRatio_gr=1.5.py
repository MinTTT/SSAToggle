# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import sys
import time

sys.path.extend(['./SSAColony'])
# […]
from SSAColony_R2G_BatchRun import *

from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from CellMD3D_visulization.CellMD3D_bottom_visual import cell_path
import matplotlib.patches as mpatches

import threading
from typing import Optional, List
# Own modules
RedColor = np.array((241, 148, 138)) / 255
GreenColor = np.array((130, 224, 170)) / 255


def RG2Predict(green_signal, red_signal):
    return np.log((green_signal + 2) / (red_signal + 2))


# %%
if __name__ == '__main__':
    # %% Disriminate cell state, using Gaussian-Mixture model
    growth_rate = 1.5
    size = 100000
    green = np.ones(size, dtype=int) * 20
    red = np.ones(size, dtype=int) * 50
    time_length = np.log(2) / growth_rate * 10
    time_step = .1
    plot_flag = True
    test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=22)
    ratio_stat = RG2Predict(test_ret[:, 1, :], test_ret[:, 2, :])  # (Green + 1) / (Red +1 )

    gmm = GaussianMixture(n_components=2, random_state=6, verbose=1)
    # gmm.fit(ratio_stat[-1, :].reshape(-1, 1))
    gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

    class_mean = gmm.means_
    green_label = 0 if class_mean[0] > class_mean[1] else 1
    test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

    fig3, ax3 = plt.subplots(1, 1, figsize=(15, 15))
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
    # fig3.show()
    # %%
    # ==================== Parameters for Simulation
    compiled_path = r'/home/fulab/colony_agent_based_compiled/colony_RunBatch'
    input_path = r'/home/fulab/PycharmProjects/colony_agent_based_SSA/model_paras/ssa_in27'  # in26 have different
    # initial state of cell
    output_dir = r'/media/fulab/Data_Raid/sunhui_code_ret/lambda1.5_G3_R30_T6.7'
    source_dir = r'/home/fulab/PycharmProjects/colony_agent_based_SSA'
    field_file_flag = False
    core_number = 60
    # green_ratio_threshold = 0.9
    batchSize = 50
    # ==================== Parameters for Simulation

    # compile
    compiled_path = assign_version(compiled_path)
    input_filename = os.path.basename(input_path)
    compiled_file_name = os.path.basename(compiled_path)
    task_name = f'{input_filename}_{compiled_file_name}'  # task name
    output_dir_no_version = os.path.join(output_dir, task_name)
    output_dir = assign_version(output_dir_no_version)

    logfile = open(os.path.join(source_dir, 'logs', f'CellsMD3D_{task_name}.log'), 'a')
    write_log(f'[CellsMD3D {task_name}] -> Compiling files', logfile)
    compile_command = f'''g++ {os.path.join(source_dir, '*.cpp')} -fopenmp -O3 -o {compiled_path}'''
    write_log(compile_command, logfile)
    command = Popen(args=compile_command,
                    stdout=PIPE, universal_newlines=True, shell=True)
    write_log(f'[CellsMD3D {task_name}] -> Compiling, PID: {command.pid}', logfile)

    while True:
        output = command.stdout.readline()
        write_log(output, logfile)
        ret_code = command.poll()
        if ret_code is not None:
            for output in command.stdout.readlines():
                write_log(output, logfile)
            if ret_code == 0:
                write_log(f'[CellsMD3D {task_name}] -> Compiled finished.', logfile)
            else:
                write_log(f'[CellsMD3D {task_name}] -> Compiled failed. Exit code {ret_code}', logfile)
            break
        time.sleep(1)
    logfile.close()

    # start
    # for batch_i in range(batchSize):
    #     output_dir = assign_version(output_dir_no_version, file_mode=False)
    #
    #     if os.path.isdir(output_dir) is False:
    #         os.mkdir(output_dir)
    #
    #     logfile = open(os.path.join(output_dir, f'CellsMD3D_{task_name}.log'), 'a')
    #     command_copy = f'''cp -f {input_path} {os.path.join(output_dir, task_name + '.txt')}'''
    #     write_log(f'[CellsMD3D {task_name}] -> Copy parameter file.', logfile)
    #     command = Popen(args=command_copy,
    #                     stdout=PIPE, universal_newlines=True, shell=True)
    #     cell_files_dir = os.path.join(output_dir, 'Cells')
    #
    #     # Simulate the Colony model
    #     write_log(f'[CellsMD3D {task_name}] -> Start simulation.', logfile)
    #     if field_file_flag is False:
    #         field_file = '0'
    #     else:
    #         field_file = ''
    #     commands3 = f'''{compiled_path} {input_path} {core_number} {output_dir} {field_file}'''
    #     write_log(commands3, logfile)
    #     command = Popen(args=commands3,
    #                     stdout=PIPE, universal_newlines=True, shell=True)
    #     # morning cell states
    #     cells_file_num = 0
    #     ratio_list = []
    #
    #     # ======== show the predict quality =============== #
    #
    #     fig3.savefig(os.path.join(output_dir, 'Predict_standard.png'))
    #     plt.close(fig3)
    #
    #     while True:
    #         output = command.stdout.readline()
    #         write_log(output, logfile)
    #         wait_cells_data = True
    #
    #         ret_code = command.poll()
    #         if ret_code is not None:
    #             if ret_code == 0:
    #                 write_log(f'[CellsMD3D {task_name}] -> Simulation finished.', logfile)
    #             else:
    #                 write_log(f'[CellsMD3D {task_name}] -> Simulation failed. Exit code {ret_code}', logfile)
    #             break  # Break this simulation loop and start the next loop.
    #
    #         while wait_cells_data:  # start the simulation, and wait the first cell file.
    #             try:
    #                 cell_files = os.listdir(cell_files_dir)
    #                 wait_cells_data = False
    #                 cell_files.sort(key=lambda file_name: int(file_name.split('.')[0]))
    #             except FileNotFoundError:
    #                 write_log(f'[CellsMD3D {task_name}] -> Waiting cells data.', logfile)
    #                 cell_files = []
    #                 time.sleep(5)
    #
    #         if cells_file_num < len(cell_files):
    #             cells_file_num = len(cell_files)
    #             print(f'Loading file: {cell_files[-1]}')
    #             cells = read_cells_rets(os.path.join(cell_files_dir, cell_files[-1]))
    #             cells_all_location = np.array([cell.center for cell in cells])
    #             cells_all_R = np.array([cell.R for cell in cells])
    #             cells_all_G = np.array([cell.G for cell in cells])
    #             cells_all_lambda = np.array([cell.GrowthRate for cell in cells])
    #             # cells_all_states = gmm.predict(np.log((cells_all_G + 1.) / (cells_all_R + 1.)).reshape(-1, 1))
    #             cells_all_states = gmm.predict(RG2Predict(cells_all_G, cells_all_R).reshape(-1, 1))
    #
    #             greenRatio = np.sum(cells_all_states == green_label) / len(cells)
    #             write_log(f'Green Ratio: {greenRatio}', logfile)
    #             ratio_list.append(greenRatio)
    #             fig4, ax4 = plt.subplots(1, 1, figsize=(15, 15))
    #             ax4.scatter(cells_all_R, cells_all_G, c=cells_all_states, cmap='coolwarm', alpha=.1)
    #
    #             green_cells_mask = cells_all_states == green_label
    #
    #             ax4.scatter(test_ret[-2, 2][green_cells_mask], test_ret[-2, 1][green_cells_mask],
    #                         color=GreenColor, alpha=.1)
    #             ax4.scatter(test_ret[-2, 2][~green_cells_mask], test_ret[-2, 1][~green_cells_mask],
    #                         color=RedColor, alpha=.1)
    #             ax4.set_xlim(1, 100)
    #             ax4.set_ylim(1, 200)
    #             ax4.set_xscale('log')
    #             ax4.set_yscale('log')
    #             ax4.set_xlabel('Red')
    #             ax4.set_ylabel('Green')
    #             splt.aspect_ratio(1)
    #             fig4.savefig(os.path.join(output_dir, f'{cell_files[-1]}_population_predict.png'))
    #             plt.close(fig4)
    #             z_top = 4
    #             z_bottom = -4
    #             location_mask = np.logical_and(cells_all_location[..., -1] > z_bottom,
    #                                            cells_all_location[..., -1] < z_top)
    #
    #             cells_location = cells_all_location[location_mask, ...]
    #
    #             location_index = np.where(location_mask == True)
    #             cells_R = cells_all_R[location_mask]
    #             cells_G = cells_all_G[location_mask]
    #             cells_states = cells_all_states[location_mask]
    #             cells_p = [cell_path(0.7, cells[cell_i].p[:-1], cells[cell_i].q[:-1]) for cell_i in location_index[0]]
    #
    #             fig1_colony_bottom, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    #             for cell_i, cell_p in enumerate(cells_p):
    #                 if cells_states[cell_i] == green_label:
    #                     cell_color = tuple(GreenColor)
    #                 else:
    #                     cell_color = tuple(RedColor)
    #                 ax1.add_patch(mpatches.PathPatch(cell_p, facecolor=cell_color, edgecolor='k', alpha=0.6))
    #             ax1.set_xlim(-150, 150)
    #             ax1.set_ylim(-150, 150)
    #
    #             fig1_colony_bottom.savefig(os.path.join(output_dir, f'{cell_files[-1]}_Colony_bottom.svg'))
    #             plt.close(fig1_colony_bottom)
    #             plt.close(fig4)
    #
    #         time.sleep(5)  # every 5 seconds scann the cells files for predicting cells states and output figures
    #     write_log(f'[CellsMD3D {task_name}] -> Simulation Finish.', logfile)
    #     logfile.close()


    def thread_simulation():
        output_dir = assign_version(output_dir_no_version, file_mode=False)

        if os.path.isdir(output_dir) is False:
            os.mkdir(output_dir)

        logfile = open(os.path.join(output_dir, f'CellsMD3D_{task_name}.log'), 'a')
        command_copy = f'''cp -f {input_path} {os.path.join(output_dir, task_name + '.txt')}'''
        write_log(f'[CellsMD3D {task_name}] -> Copy parameter file.', logfile)
        command_cp = Popen(args=command_copy,
                           stdout=PIPE, universal_newlines=True, shell=True)
        cell_files_dir = os.path.join(output_dir, 'Cells')

        # Simulate the Colony model
        write_log(f'[CellsMD3D {task_name}] -> Start simulation.', logfile)
        if field_file_flag is False:
            field_file = '0'
        else:
            field_file = ''
        commands3 = f'''{compiled_path} {input_path} {core_number} {output_dir} {field_file}'''
        write_log(commands3, logfile)
        command = Popen(args=commands3,
                        stdout=PIPE, universal_newlines=True, shell=True)
        # morning cell states
        cells_file_num = 0
        ratio_list = []

        # ======== show the predict quality =============== #

        fig3.savefig(os.path.join(output_dir, 'Predict_standard.png'))
        plt.close(fig3)

        while True:
            output = command.stdout.readline()
            write_log(output, logfile)
            wait_cells_data = True

            ret_code = command.poll()
            if ret_code is not None:
                if ret_code == 0:
                    write_log(f'[CellsMD3D {task_name}] -> Simulation finished.', logfile)
                else:
                    write_log(f'[CellsMD3D {task_name}] -> Simulation failed. Exit code {ret_code}', logfile)
                break  # Break this simulation loop and start the next loop.

            while wait_cells_data:  # start the simulation, and wait the first cell file.
                try:
                    cell_files = os.listdir(cell_files_dir)
                    wait_cells_data = False
                    cell_files.sort(key=lambda file_name: int(file_name.split('.')[0]))
                except FileNotFoundError:
                    write_log(f'[CellsMD3D {task_name}] -> Waiting cells data.', logfile)
                    cell_files = []
                    time.sleep(5)

            if cells_file_num < len(cell_files):
                cells_file_num = len(cell_files)
                print(f'Loading file: {cell_files[-1]}')
                cells = read_cells_rets(os.path.join(cell_files_dir, cell_files[-1]))
                cells_all_location = np.array([cell.center for cell in cells])
                cells_all_R = np.array([cell.R for cell in cells])
                cells_all_G = np.array([cell.G for cell in cells])
                cells_all_states = gmm.predict(RG2Predict(cells_all_G, cells_all_R).reshape(-1, 1))

                greenRatio = np.sum(cells_all_states == green_label) / len(cells)
                write_log(f'Green Ratio: {greenRatio}', logfile)
                ratio_list.append(greenRatio)
                fig4, ax4 = plt.subplots(1, 1, figsize=(15, 15))
                green_cells_mask = cells_all_states == green_label

                ax4.scatter(cells_all_R[green_cells_mask], cells_all_G[green_cells_mask],
                            color=GreenColor, alpha=.1)
                ax4.scatter(cells_all_R[~green_cells_mask], cells_all_G[~green_cells_mask],
                            color=RedColor, alpha=.1)
                ax4.set_xlim(1, 100)
                ax4.set_ylim(1, 200)
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.set_xlabel('Red')
                ax4.set_ylabel('Green')
                splt.aspect_ratio(1)
                fig4.savefig(os.path.join(output_dir, f'{cell_files[-1]}_population_predict.png'))
                plt.close(fig4)

                z_top = 4
                z_bottom = -4
                location_mask = np.logical_and(cells_all_location[..., -1] > z_bottom,
                                               cells_all_location[..., -1] < z_top)

                # cells_location = cells_all_location[location_mask, ...]

                location_index = np.where(location_mask == True)
                # cells_R = cells_all_R[location_mask]
                # cells_G = cells_all_G[location_mask]
                cells_states = cells_all_states[location_mask]
                cells_p = [cell_path(0.7, cells[cell_i].p[:-1], cells[cell_i].q[:-1]) for cell_i in location_index[0]]

                fig1_colony_bottom, ax1 = plt.subplots(1, 1, figsize=(15, 15))
                for cell_i, cell_p in enumerate(cells_p):
                    if cells_states[cell_i] == green_label:
                        cell_color = tuple(GreenColor)
                    else:
                        cell_color = tuple(RedColor)
                    ax1.add_patch(mpatches.PathPatch(cell_p, facecolor=cell_color, edgecolor='k', alpha=0.6))
                ax1.set_xlim(-150, 150)
                ax1.set_ylim(-150, 150)

                fig1_colony_bottom.savefig(os.path.join(output_dir, f'{cell_files[-1]}_Colony_bottom.svg'))
                plt.close(fig1_colony_bottom)
                plt.close(fig4)

            time.sleep(5)  # every 5 seconds scann the cells files for predicting cells states and output figures
        write_log(f'[CellsMD3D {task_name}] -> Simulation Finish.', logfile)
        logfile.close()
        return None

    thread_number = 4
    task_list = [None] * thread_number  # type: List[Optional[threading.Thread]]
    empty_pool = []
    start_thread_num = 0
    while thread_number < batchSize:
        # get empty slot
        for i in range(thread_number):
            if task_list[i] is None:
                empty_pool.append(i)
            else:
                if not task_list[i].is_alive():
                    empty_pool.append(i)

        # start the simulation threads.
        if len(empty_pool) >= 1:
            # for slot_i in empty_pool:
            while empty_pool:
                slot_i = empty_pool.pop()
                task_list[slot_i] = threading.Thread(target=thread_simulation)
                task_list[slot_i].start()
                start_thread_num += 1
                time.sleep(5)  # start a thread every 5 s.

    for task in task_list:
        if task.is_alive():
            task.join()



