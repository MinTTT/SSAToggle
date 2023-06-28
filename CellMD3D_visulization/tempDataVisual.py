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
# […]
import matplotlib.pyplot as plt
# Own modules
import sciplot as splt
splt.whitegrid()


#%% visualize Height

heightPs = r"\\172.16.7.254\fulabshare\fulab_zc_1\sunhui_code_ret\ssa_in23_Colony_V0.3.6\Height\43.txt"


heightData = pd.read_csv(heightPs, delimiter='\t', index_col=None, header=None).iloc[:, :-1]


fig1, ax1 = plt.subplots(1, 1)
ax1.imshow(heightData, cmap='coolwarm')
fig1.show()


columnI = list(heightData.columns)
rowI = list(heightData.index)

Ixx, Iyy = np.meshgrid(columnI, rowI)
heightDataXYZ = np.hstack([Ixx.reshape(-1, 1), Iyy.reshape(-1, 1), heightData.to_numpy().reshape(-1, 1)])