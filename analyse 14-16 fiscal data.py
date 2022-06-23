# %%
'''
改用保險業實證 dmp
    三年資料，假設沒有 tech. change
    先算整體的，有能力再用網路
最後有時間再來 scope property
'''
# %%
from matplotlib.axes import Axes
import fiscal_analyzing_utils as utils
import os
import pandas as pd
import numpy as np
import constant as const
import dmp
import solver
import solver_r
from load_data import denoise_nonpositive, FISCAL_ATTRIBUTES,  LIFE_DUMMY141516, ENG_NAMES_16
from exp_fiscal_data import EXPANSION_OPERATION_SMRTS_DUMMY141516, EXPANSION_INSURANCE_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516, CONTRACTION_INSURANCE_SMRTS_DUMMY141516
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('jet')
import seaborn as sns
sns.set_theme(style="darkgrid")
#%%
all_analysis = utils.get_analyze_df(
    dmu_ks=[
        'AIA Taiwan 14', 'AIA Taiwan 15', 'AIA Taiwan 16', 
        'Allianz Taiwan Life 14', 'Allianz Taiwan Life 15', 'Allianz Taiwan Life 16', 
        'Bank Taiwan Life 14', 'Bank Taiwan Life 15', 'Bank Taiwan Life 16', 
        'BNP Paribas Cardif TCB 14', 'BNP Paribas Cardif TCB 15', 'BNP Paribas Cardif TCB 16', 
        'Cardif 14', 'Cardif 15', 'Cardif 16', 
        'Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16', "DUMMY Cathay 15", 
        'Chaoyang Life 14', 'Chaoyang Life 15', 'Chaoyang Life 16', 
        'China Life 14', 'China Life 15', 'China Life 16', 
        "ACE Tempest Life 14", "ACE Tempest Life 15", 'Chubb Tempest Life 16', 
        'Chunghwa Post 14', 'Chunghwa Post 15', 'Chunghwa Post 16', 
        'CIGNA 14', 'CIGNA 15', 'CIGNA 16', 
        "CTBC Life 14", "CTBC Life 15", 
        'Farglory Life 14', 'Farglory Life 15', 'Farglory Life 16', 
        'First-Aviva Life 14', 'First-Aviva Life 15', 'First-Aviva Life 16', 
        'Fubon Life 14', 'Fubon Life 15', 'Fubon Life 16',
        "Global Life 14", 
        'Hontai Life 14', 'Hontai Life 15', 'Hontai Life 16', 
        'Mercuries Life 14', 'Mercuries Life 15', 'Mercuries Life 16', 
        'Nan Shan Life 14', 'Nan Shan Life 15', 'Nan Shan Life 16', 
        'PCA Life 14', 'PCA Life 15', 'PCA Life 16', 
        'Prudential of Taiwan 14', 'Prudential of Taiwan 15', 'Prudential of Taiwan 16', 
        "Singfor Life 14", 
        'Shin Kong Life 14', 'Shin Kong Life 15', 'Shin Kong Life 16', 
        'Taiwan Life 14', 'Taiwan Life 15', 'Taiwan Life 16', "DUMMY Taiwan 16", 
        'TransGlobe Life 14', 'TransGlobe Life 15', 'TransGlobe Life 16', 
        'Yuanta Life 14', 'Yuanta Life 15', 'Yuanta Life 16', 
        'Zurich 14', 'Zurich 15', 'Zurich 16', 
            ], df=denoise_nonpositive(LIFE_DUMMY141516),)
utils.round_analyze_df(all_analysis, round_to=4)#.to_excel("14-16 all_dmu analysis.xlsx")
#%%
no16 = all_analysis.loc[["16" not in idx for idx in all_analysis.index.tolist()]].drop(["DUMMY Cathay 15", "Singfor Life 14", "CTBC Life 15", "Global Life 14"])
#%%
for col in [const.SCALE, const.PROFIT, const.EFFICIENCY,]:
    for y_col in [const.EXPANSION_CONSISTENCY,]:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=800)
        utils.analyze_plot(ax, no16, y_col=y_col, according_col=col)
        stitle = f"2014-2016 all DMU {y_col} with {col}"
        ax.set_title(stitle)
        # plt.savefig(f"{stitle}.png")
        plt.show()
#%%
