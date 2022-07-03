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
from load_data import denoise_nonpositive, FISCAL_ATTRIBUTES,  LIFE181920
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('jet')
import seaborn as sns
sns.set_theme(style="darkgrid")
#%%
all_analysis_18 = utils.get_analyze_df(
    dmu_ks=[
        'AIA Taiwan 18', 'AIA Taiwan 19', 'AIA Taiwan 20', 
        'Allianz Taiwan Life 18', 'Allianz Taiwan Life 19', 'Allianz Taiwan Life 20', 
        'Bank Taiwan Life 18', 'Bank Taiwan Life 19', 'Bank Taiwan Life 20', 
        'BNP Paribas Cardif TCB 18', 'BNP Paribas Cardif TCB 19', 'BNP Paribas Cardif TCB 20', 
        'Cardif 18', 'Cardif 19', 'Cardif 20', 
        'Cathay Life 18', 'Cathay Life 19', 'Cathay Life 20', 
        'China Life 18', 'China Life 19', 'China Life 20', 
        "Chubb Tempest Life 18", "Chubb Tempest Life 19", 'Chubb Tempest Life 20', 
        'Chunghwa Post 18', 'Chunghwa Post 19', 'Chunghwa Post 20', 
        'CIGNA 18', 'CIGNA 19', 'CIGNA 20', 
        'Farglory Life 18', 'Farglory Life 19', 'Farglory Life 20', 
        'First-Aviva Life 18', 'First-Aviva Life 19', 'First-Aviva Life 20', 
        'Fubon Life 18', 'Fubon Life 19', 'Fubon Life 20',
        'Hontai Life 18', 'Hontai Life 19', 'Hontai Life 20', 
        'Mercuries Life 18', 'Mercuries Life 19', 'Mercuries Life 20', 
        'Nan Shan Life 18', 'Nan Shan Life 19', 'Nan Shan Life 20', 
        'PCA Life 18', 'PCA Life 19', 'PCA Life 20', 
        'Prudential of Taiwan 18', 'Prudential of Taiwan 19', 'Prudential of Taiwan 20', 
        'Shin Kong Life 18', 'Shin Kong Life 19', 'Shin Kong Life 20', 
        'Taiwan Life 18', 'Taiwan Life 19', 'Taiwan Life 20', 
        'TransGlobe Life 18', 'TransGlobe Life 19', 'TransGlobe Life 20', 
        'Yuanta Life 18', 'Yuanta Life 19', 'Yuanta Life 20', 
            ], df=denoise_nonpositive(LIFE181920), year=18)
utils.round_analyze_df(all_analysis_18, round_to=4)#.to_excel("18-20 all_dmu analysis.xlsx")
#%%
no20 = all_analysis_18.loc[["20" not in idx for idx in all_analysis_18.index.tolist()]]
#%%
for col in [const.SCALE, const.PROFIT, const.EFFICIENCY,]:
    for y_col in [const.EXPANSION_CONSISTENCY,]:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=800)
        utils.analyze_plot(ax, no20, y_col=y_col, according_col=col)
        stitle = f"2018-2020 all DMU {y_col} with {col}"
        ax.set_title(stitle)
        # plt.savefig(f"{stitle}.png")
        plt.show()
#%%