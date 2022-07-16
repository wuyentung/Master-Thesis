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
from load_data import denoise_nonpositive, LIFE_DUMMY141516, LIFE181920
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('jet')
import seaborn as sns
import plotting_utils as plotting
sns.set_theme(style="darkgrid")
#%%
all_analysis_14 = utils.get_analyze_df(
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
            ], df=denoise_nonpositive(LIFE_DUMMY141516), year=14)
utils.round_analyze_df(all_analysis_14, round_to=4)#.to_excel("14-16 all_dmu analysis.xlsx")
#%%
no16 = all_analysis_14.loc[["16" not in idx for idx in all_analysis_14.index.tolist()]].drop(["DUMMY Cathay 15", "Singfor Life 14", "CTBC Life 15", "Global Life 14"])
wanted_dmus = no16.loc[[idx in ["Fubon Life 14", "Fubon Life 15", "Shin Kong Life 14", "Shin Kong Life 15", "TransGlobe Life 14", "TransGlobe Life 15", "AIA Taiwan 14", "AIA Taiwan 15", "Yuanta Life 14", "Yuanta Life 15"] for idx in no16.index.tolist()]]
wanted_dmus.at["Yuanta Life 15", const.CONSISTENCY] = 1.02
#%%
for col in [const.SCALE,]:
    for y_col in [const.CONSISTENCY,]:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
        plotting.analyze_plot(ax, no16, y_col=y_col, according_col=col, fontsize=6, label=False)
        plotting.label_data(zip_x=wanted_dmus[const.EC], zip_y=wanted_dmus[y_col], labels=wanted_dmus.index, fontsize=6)
        stitle = f"2014-2016 all DMUs {y_col} with {col}"
        ax.set_title(stitle)
        # plt.savefig(f"{stitle} label specific.png")
        plt.show()
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
large_scale = no20.loc[[scale > 17 for scale in no20[const.SCALE].tolist()]]
bank_taiwan = no20.loc[["Bank" in idx for idx in no20.index.tolist()]]
transglobe = no20.loc[["Globe" in idx for idx in no20.index.tolist()]]
#%%
for col in [const.SCALE, ]:
    for y_col in [const.CONSISTENCY,]:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
        plotting.analyze_plot(ax, no20, y_col=y_col, according_col=col, fontsize=6, label=False)
        stitle = f"2018-2020 all DMUs {y_col} with {col}"
        plotting.label_data(zip_x=large_scale[const.EC], zip_y=large_scale[y_col], labels=large_scale.index, fontsize=6)
        plotting.label_data(zip_x=bank_taiwan[const.EC], zip_y=bank_taiwan[y_col], labels=bank_taiwan.index, fontsize=6)
        plotting.label_data(zip_x=transglobe[const.EC], zip_y=transglobe[y_col], labels=transglobe.index, fontsize=6)
        ax.set_title(stitle)
        # plt.savefig(f"{stitle} label specific.png")
        plt.show()
#%%
