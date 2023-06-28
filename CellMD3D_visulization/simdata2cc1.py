# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""
#%%
# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
# […]

# Own modules
from CellMD3D_visulization.CellMD3D_Visual import cells2cc1

file_dir = r'\\172.16.7.254\fulabshare\fulab_zc_1\sunhui_code_ret\ssa_in23_Colony_V0.2.4.11\Cells'
files_scan = os.scandir(file_dir)
files = [file.name for file in files_scan if file.is_file()]

for file in files:
    cells2cc1(os.path.join(file_dir, file),
          save_dir=os.path.join(file_dir,
                                'cc1'))
