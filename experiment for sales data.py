#%%
'''
改用保險業業務實證 dmp
    三年資料，假設沒有 tech. change
    先算整體的，有能力再用網路
最後有時間再來 scope property
'''
#%%
import os
import dmp
import pandas as pd
import numpy as np
import solver
import solver_r
from load_sales_data import LIFE, LIFE2019, denoise_nonpositive, ATTRIBUTES, LIFE2018, LIFE2020
from itertools import combinations
import matplotlib.pyplot as plt
#%%
