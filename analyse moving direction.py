# %%
'''
改用保險業實證 dmp
    三年資料，假設沒有 tech. change
    先算整體的，有能力再用網路
最後有時間再來 scope property
'''
# %%
from turtle import shape
from matplotlib.axes import Axes
from sklearn.preprocessing import scale
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
sns.set_theme(style="darkgrid")
#%%
def analyze_plot(ax:Axes, df:pd.DataFrame, x_col = const.EC, y_col = const.CONSISTENCY, according_col=const.EFFICIENCY, fontsize=5):
    ax.hlines(y=df[y_col].median(), xmin=df[x_col].min(), xmax=df[x_col].max(), colors="gray", lw=1)
    ax.vlines(x=1 if x_col == const.EC else df[x_col].median(), ymin=df[y_col].min(), ymax=df[y_col].max(), colors="gray", lw=1)
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, hue=according_col, palette=CMAP, )
    ax.annotate("", xy=(df[x_col]['AIA Taiwan 15'], df[y_col]['AIA Taiwan 15']), xytext=(df[x_col]['AIA Taiwan 14'], df[y_col]['AIA Taiwan 14']), arrowprops=dict(arrowstyle="->", color="black"))
    utils.label_data(zip_x=df[x_col], zip_y=df[y_col], labels=df.index, fontsize=fontsize)
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
# df = no16.drop(["CTBC Life 14"])
def cal_moving_df(df:pd.DataFrame):
    scale_mean = [np.mean([df[const.SCALE][df.index[i]], df[const.SCALE][df.index[i+1]]]) for i in range(0, df.shape[0], 2)]
    consistency_abs = [np.abs(df[const.CONSISTENCY][df.index[i]] - df[const.CONSISTENCY][df.index[i+1]]) for i in range(0, df.shape[0], 2)]
    ec_abs = [np.abs(df[const.EC][df.index[i]] - df[const.EC][df.index[i+1]]) for i in range(0, df.shape[0], 2)]
    moving_df = pd.DataFrame({
        const.SCALE: scale_mean, 
        const.CONSISTENCY: consistency_abs,
        const.EC: ec_abs,
    })
    return moving_df
#%%
moving_df_14 = cal_moving_df(no16.drop(["CTBC Life 14"]))
moving_df_18 = cal_moving_df(no20)
#%%
period = "Period"
moving_df_14[period] = "2014-2016"
moving_df_18[period] = "2018-2020"
#%%
moving_df = pd.concat([moving_df_14, moving_df_18])
#%%
for y_col in [const.CONSISTENCY, const.EC]:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=800)
    sns.scatterplot(data=moving_df, x=const.SCALE, y=y_col, style=period, hue=period, s=100)
    stitle = f"moving directions two exp DMUs {y_col}"
    # ax.set_title(stitle)
    plt.savefig(f"{stitle}.png")
    plt.show()
#%%
