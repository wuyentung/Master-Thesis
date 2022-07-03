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
        "CTBC Life 14", "CTBC Life 15 to real", 'Taiwan Life 16',
        "CTBC Life 15 to dummy", "DUMMY Taiwan 16", 
        'Taiwan Life 14', 'Taiwan Life 15 to real', 'Taiwan Life 16', 
        'Taiwan Life 15 to dummy', "DUMMY Taiwan 16", 
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
DMU = "Insurer"
ctbc="CTBC Life"
taiwan="Taiwan Life"
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
        DMU: [ctbc, ctbc, ctbc, ctbc, ctbc, taiwan, taiwan, taiwan, taiwan, taiwan]
    }, index=dmu_ks
)
no16 = merged_df.loc[["16" not in idx for idx in merged_df.index.tolist()]]
#%%
def analyze_plot(ax:Axes, df:pd.DataFrame, x_col = const.EC, y_col = const.CONSISTENCY, according_col=const.EFFICIENCY, fontsize=5):
    ax.hlines(y=0, xmin=.95, xmax=1.002, colors="gray", lw=1)
    ax.vlines(x=1 , ymin=-1, ymax=1, colors="gray", lw=1)
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, style=according_col, hue=according_col, s=100, zorder=10)
    utils.label_data(zip_x=df[x_col], zip_y=df[y_col], labels=df.index, fontsize=fontsize, xytext=(-10, 10), ha="left")

for col in [DMU, ]:
    for y_col in [const.CONSISTENCY,]:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
        analyze_plot(ax, no16, y_col=y_col, according_col=col, fontsize=10)
        stitle = f"2014-2016 merged DMUs {y_col} with {col}"
        ax.set_title(stitle)
        # plt.savefig(f"{stitle}.png")
        plt.show()
#%%