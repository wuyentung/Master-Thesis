#%%
# 分析合併公司:
# - 2016 CTBC Life 與 Taiwan Life 合併成 Taiwan Life
import fiscal_analyzing_utils as utils
from matplotlib.axes import Axes
import os
import pandas as pd
import numpy as np
import constant as const
from load_data import denoise_nonpositive, LIFE_DUMMY141516, LIFE181920
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('jet')
import seaborn as sns
sns.set_theme(style="darkgrid")
#%%
dmu_ks=[
        "CTBC Life 14", "CTBC Life 15", 'Taiwan Life 16',
        "CTBC Life 15", "DUMMY Taiwan 16", 
        'Taiwan Life 14', 'Taiwan Life 15', 'Taiwan Life 16', 
        'Taiwan Life 15', "DUMMY Taiwan 16", 
            ]
INSURANCE_EXP = [10.8484, 10.632, 17.6594, 10.632, 12.9615, 2.54, 2.3295, 17.6594, 2.3295, 12.9615, ]
OPERATION_EXP = [2.3518, 2.4684, 4.9022, 2.4684, 6.1227, 3.739, 3.6542, 4.9022, 3.6542, 6.1227, ]
UNDERWRITING_PROFIT = [153.9283, 150.8721, 125.3854, 150.8721, 133.7383, 148.6853, 147.3106, 125.3854, 147.3106, 133.7383, ]
INVESTMENT_PROFIT = [12.6541, 16.4587, 44.7054, 16.4587, 32.206, 18.7208, 16.5044, 44.7054, 16.5044, 32.206, ]
SCALE = [13.2001, 13.1005, 22.5615, 13.1005, 19.0842, 6.279, 5.9837, 22.5615, 5.9837, 19.0842, ]
PROFIT = [166.5824, 167.3307, 170.0908, 167.3307, 165.9443, 167.4061, 163.815, 170.0908, 163.815, 165.9443, ]

EFFICIENCY = [1.0242, 1.0255, 1.0514, 1.0255, 1.0785, 1.0136, 1.0337, 1.0514, 1.0337, 1.0785, ]
#%%
def _out_dir(start_idx, end_idx):
    return [((UNDERWRITING_PROFIT[end_idx]-UNDERWRITING_PROFIT[start_idx])/2)/np.abs(np.abs((UNDERWRITING_PROFIT[end_idx]-UNDERWRITING_PROFIT[start_idx])/2) + np.abs((INVESTMENT_PROFIT[end_idx]-INVESTMENT_PROFIT[start_idx])/2)), ((INVESTMENT_PROFIT[end_idx]-INVESTMENT_PROFIT[start_idx])/2)/np.abs(np.abs((UNDERWRITING_PROFIT[end_idx]-UNDERWRITING_PROFIT[start_idx])/2) + np.abs((INVESTMENT_PROFIT[end_idx]-INVESTMENT_PROFIT[start_idx])/2))]

out_dirs = [_out_dir(i, i+1) for i in range(len(dmu_ks)-1)]
out_dirs.append([np.nan, np.nan])
#%%
max_dirs = [[0.495, 0.005], [0.005, 0.495], [0.495, 0.005], [0.005, 0.495], [0., 0.], [0.495, 0.005], [0.495, 0.005], [0.495, 0.005], [0.495, 0.005], [0., 0.], ]
CONSISTENCY = [utils._cal_cosine_similarity(out_dirs[i], max_dirs[i]) for i in range(len(dmu_ks))]
EC = [EFFICIENCY[i]/EFFICIENCY[i+1] for i in range(len(dmu_ks)-1)]
EC.append(np.nan)
#%%
merged_df = pd.DataFrame(
    {
        const.INSURANCE_EXP: INSURANCE_EXP,
        const.OPERATION_EXP: OPERATION_EXP, 
        const.UNDERWRITING_PROFIT: UNDERWRITING_PROFIT, 
        const.INVESTMENT_PROFIT: INVESTMENT_PROFIT, 
        const.SCALE: SCALE, 
        const.PROFIT: PROFIT, 
        const.OUT_DIR: out_dirs, 
        const.MAX_DIR_MP: max_dirs, 
        const.CONSISTENCY: CONSISTENCY,
        const.EFFICIENCY: EFFICIENCY, 
        const.EC: EC,
    }, index=dmu_ks
)
#%%