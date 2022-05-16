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
from exp_fiscal_data import OPERATION_SMRTS_DUMMY141516, INSURANCE_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('jet')
import seaborn as sns
sns.set_theme(style="darkgrid")
# %%
### 14~16
utils.plot_3D(dmu=['Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16'], stitle="Cathay Life", target_input=const.OPERATION_EXP, smrts_dict=OPERATION_SMRTS_DUMMY141516, df=denoise_nonpositive(LIFE_DUMMY141516), dummy_dmu=["Global Life 14", "Singfor Life 14", "DUMMY Cathay 15"])
plt.show()
#%%
utils.plot_3D(dmu=['Taiwan Life 14', 'Taiwan Life 15', 'Taiwan Life 16'], stitle="DUMMY Taiwan", target_input=const.OPERATION_EXP, smrts_dict=OPERATION_SMRTS_DUMMY141516, df=denoise_nonpositive(LIFE_DUMMY141516), dummy_dmu=["CTBC Life 14", "CTBC Life 15", "DUMMY Taiwan 16"])
plt.show()
# %%
# visualize_progress
for k in ENG_NAMES_16:
    if "Chubb Tempest Life" == k:
        continue
    for target_input in [const.INSURANCE_EXP, const.OPERATION_EXP]:
        
        if const.INSURANCE_EXP == target_input:
            smrts_dict = INSURANCE_SMRTS_DUMMY141516
        else:
            smrts_dict = OPERATION_SMRTS_DUMMY141516
            
        utils.plot_3D(dmu=[k+n for n in [' 14', ' 15', ' 16']], stitle=k, target_input=target_input, smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE_DUMMY141516))
        plt.savefig("%s %s.png" %(k, target_input), dpi=400)
        plt.show()
#%%
dmus = ["ACE Tempest Life 14", "ACE Tempest Life 15", "Chubb Tempest Life 16"]
for target_input in [const.INSURANCE_EXP, const.OPERATION_EXP]:
    
    if const.INSURANCE_EXP == target_input:
        smrts_dict = INSURANCE_SMRTS_DUMMY141516
    else:
        smrts_dict = OPERATION_SMRTS_DUMMY141516
        
    utils.plot_3D(dmu=dmus, stitle="Chubb Tempest Life", target_input=target_input, smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE_DUMMY141516))
    plt.savefig("%s %s.png" %("Chubb Tempest Life", target_input), dpi=400)
    plt.show()
#%%
dmus = ["CTBC Life 14", "CTBC Life 15",]
for target_input in [const.INSURANCE_EXP, const.OPERATION_EXP]:
    
    if const.INSURANCE_EXP == target_input:
        smrts_dict = INSURANCE_SMRTS_DUMMY141516
    else:
        smrts_dict = OPERATION_SMRTS_DUMMY141516
        
    utils.plot_3D(dmu=dmus, stitle="CTBC Life", target_input=target_input, smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE_DUMMY141516))
    plt.savefig("%s %s.png" %("CTBC Life", target_input), dpi=400)
    plt.show()
#%%
dmus = ["ACE Tempest Life 14", "ACE Tempest Life 15", "Chubb Tempest Life 16"]
utils.get_analyze_df(dmu_ks=dmus, df=denoise_nonpositive(LIFE_DUMMY141516))
#%%
utils.get_analyze_df(dmu_ks=['Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16', 'Chunghwa Post 14', 'Chunghwa Post 15', 'Chunghwa Post 16', 'Shin Kong Life 14', 'Shin Kong Life 15', 'Chaoyang Life 14', 'Chaoyang Life 15', 'China Life 15', 'China Life 16', 'Fubon Life 15', 'Fubon Life 16', 'Hontai Life 15', 'Hontai Life 16', 'CIGNA 15', 'CIGNA 16'], df=denoise_nonpositive(LIFE_DUMMY141516), round_to=4).to_excel("14-16 EFF_dmu analysis.xlsx")
# utils.get_analyze_df(dmu_ks=["ACE Tempest Life 14", "ACE Tempest Life 15", "Chubb Tempest Life 16"], df=denoise_nonpositive(LIFE_DUMMY141516))
#%%
utils.get_analyze_df(dmu_ks=["Global Life 14", "Singfor Life 14", 'Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16', "DUMMY Cathay 15", 'CTBC Life 14', 'CTBC Life 15', 'Taiwan Life 14','Taiwan Life 15', 'Taiwan Life 16', 'DUMMY Taiwan 16', ], df=denoise_nonpositive(LIFE_DUMMY141516), round_to=4).to_excel("14-16 merged_dmu analysis.xlsx")
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
no16[const.SCALE] = no16[const.INSURANCE_EXP] + no16[const.OPERATION_EXP]
no16[const.PROFIT] = no16[const.UNDERWRITING_PROFIT] + no16[const.INVESTMENT_PROFIT]
#%%
for col in [const.SCALE, const.PROFIT,]:
    fig, ax = plt.subplots(figsize=(12, 9), dpi=400)
    utils.analyze_plot(ax, no16.loc[[1 == idx for idx in no16[const.EFFICIENCY].tolist()]], according_col=col)
    stitle = f"{const.EFFICIENCY}=1 with {col}"
    ax.set_title(stitle)
    # plt.savefig(f"{stitle}.png")
    plt.show()
#%%
no16no1 = no16.loc[[1 != idx for idx in no16[const.EFFICIENCY].tolist()]]
for col in [const.SCALE, const.PROFIT, const.EFFICIENCY,]:
    fig, ax = plt.subplots(figsize=(12, 9), dpi=400)
    utils.analyze_plot(ax, no16no1, according_col=col)
    stitle = f"{const.EFFICIENCY}>1 with {col}"
    ax.set_title(stitle)
    # plt.savefig(f"{stitle}.png")
    plt.show()
#%%
no16no1no0 = no16no1.loc[[0 != idx for idx in no16no1[const.COS_SIM].tolist()]]
for col in [const.SCALE, const.PROFIT, const.EFFICIENCY,]:
    fig, ax = plt.subplots(figsize=(12, 9), dpi=400)
    utils.analyze_plot(ax, no16no1no0, according_col=col)
    stitle = f"{const.EFFICIENCY}>1, {const.COS_SIM}!=0 with {col}"
    ax.set_title(stitle)
    # plt.savefig(f"{stitle}.png")
    plt.show()
#%%
no16no1w0 = no16no1.loc[[0 == idx for idx in no16no1[const.COS_SIM].tolist()]]
fig, ax = plt.subplots(figsize=(12, 9), dpi=400)
sns.boxenplot(y=const.EC, data=no16no1w0, orient="v", ax=ax, palette="Set3")
sns.swarmplot(y=const.EC, data=no16no1w0, orient="v", ax=ax, color=".25")
plt.annotate(
    no16no1w0[const.EC].idxmax(), # this is the text
    (0, no16no1w0[const.EC].max()), # these are the coordinates to position the label
    textcoords="offset points", # how to position the text
    xytext=(-10, 0), # distance from text to points (x,y)
    ha='right', # horizontal alignment can be left, right or center
    fontsize=10, 
    ) 
stitle = f"{const.EFFICIENCY}>1, {const.COS_SIM}=0"
ax.set_title(stitle)
# plt.savefig(f"{stitle}.png")
plt.show()
#%%
for col in [const.SCALE, const.PROFIT, const.EFFICIENCY,]:
    fig, ax = plt.subplots(figsize=(12, 9), dpi=400)
    utils.analyze_plot(ax, no16, according_col=col)
    ax.set_title(col)
    # plt.savefig(f"basic {col}.png")
    plt.show()
#%%
## input output correlation plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sns.scatterplot(x=const.INSURANCE_EXP, y=const.OPERATION_EXP, data=all_analysis, ax=ax, )
ax.set_title("inputs plot")
# plt.savefig(f"inputs plot.png")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sns.scatterplot(x=const.UNDERWRITING_PROFIT, y=const.INVESTMENT_PROFIT, data=all_analysis, ax=ax, )
ax.set_title("outputs plot")
# plt.savefig(f"outputs plot.png")
plt.show()
#%%
