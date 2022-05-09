# %%
'''
改用保險業實證 dmp
    三年資料，假設沒有 tech. change
    先算整體的，有能力再用網路
最後有時間再來 scope property
'''
# %%
import fiscal_analyzing_utils as utils
import os
import pandas as pd
import numpy as np
import dmp
import solver
import solver_r
from load_data import LIFE181920, FISCAL_LIFE2019, denoise_nonpositive, FISCAL_ATTRIBUTES, FISCAL_LIFE2018, FISCAL_LIFE2020, LIFE_DUMMY141516, ENG_NAMES_16
from exp_fiscal_data import OPERATION_SMRTS181920, INSURANCE_SMRTS181920, OPERATION_SMRTS_DUMMY141516, INSURANCE_SMRTS_DUMMY141516, EFF_DICT141516
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('plasma')
# %%
# eff_dmu = ['Hontai Life 18', 'Chunghwa Post 18', 'First-Aviva Life 18', 'Hontai Life 19', 'First-Aviva Life 19', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'Cathay Life 20', 'China Life 20', 'Nan Shan Life 20', 'Shin Kong Life 20', 'Fubon Life 20', 'Hontai Life 20', 'Chunghwa Post 20', 'First-Aviva Life 20', 'BNP Paribas Cardif TCB 20', 'CIGNA 20', 'Cardif 20']
# # %%
# df = denoise_nonpositive(LIFE181920)/1000/1000

# # %%
# utils.plot_3D(dmu=['Hontai Life 18', 'Hontai Life 19', 'Hontai Life 20'], stitle="Hontai Life",
#               target_input="insurance_exp", smrts_dict=INSURANCE_SMRTS181920, df=df)
# plt.show()
# # %%
# dmus = ['Bank Taiwan Life ', 'Taiwan Life ', 'PCA Life ', 'Cathay Life ', 'China Life ', 'Nan Shan Life ', 'Shin Kong Life ', 'Fubon Life ', 'Mercuries Life ', 'Farglory Life ', 'Hontai Life ',
#         'Allianz Taiwan Life ', 'Chunghwa Post ', 'First-Aviva Life ', 'BNP Paribas Cardif TCB ', 'Prudential of Taiwan ', 'CIGNA ', 'Yuanta Life ', 'TransGlobe Life ', 'AIA Taiwan ', 'Cardif ', 'Chubb Tempest Life ']
# # %%
# # visualize_progress
# for k in dmus:
#     for target_input in ["insurance_exp", "operation_exp"]:
#         if "insurance_exp" == target_input:
#             smrts_dict = INSURANCE_SMRTS181920
#         else:
#             smrts_dict = OPERATION_SMRTS181920
#         utils.plot_3D(dmu=[k+n for n in ['18', '19', '20']], stitle=k, target_input=target_input,
#                       smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE181920)/1000/1000)
#         # plt.savefig("%s %s.png" %(k, target_input), dpi=400)
#         plt.show()
# # %%
# df18 = denoise_nonpositive(FISCAL_LIFE2018)/1000/1000
# # plt.figure(figsize=(16, 9))
# # plt.scatter(df18["insurance_exp"], df18["operation_exp"])
# # for k, row in df18.iterrows():
# #     plt.annotate(k, (row["insurance_exp"], row["operation_exp"]), fontsize=10)
# # plt.xlabel("insurance_exp")
# # plt.ylabel("operation_exp")
# # plt.show()
# df20 = denoise_nonpositive(FISCAL_LIFE2020)/1000/1000
# # %%
# # 針對每個公司弄出一個表格來
# ana_dmu_cols = ["insurance_exp", "operation_exp", "underwriting_profit", "investment_profit", "output progress direction",
#                 "insurance_exp max direction of MP", "insurance_exp cosine similarity", "operation_exp max direction of MP", "operation_exp cosine similarity", ]
# dmu_dfs = {}
# round_to = 2
# for k in dmus:
#     dmu_ks = [k+n for n in ['18', '19', '20']]
#     dmu_df = utils.get_analyze_df(dmu_ks=dmu_ks, df=df)
#     dmu_dfs[k] = dmu_df
#     break
# # %%
# utils.plot_3D(dmu=['Bank Taiwan Life 18', 'Bank Taiwan Life 19', 'Bank Taiwan Life 20'], stitle="Bank Taiwan Life", target_input="insurance_exp", smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE181920)/1000/1000)
# plt.draw()
# plt.show()
# %%
### 14~16
utils.plot_3D(dmu=['Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16'], stitle="Cathay Life", target_input="operation_exp", smrts_dict=OPERATION_SMRTS_DUMMY141516, df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000, dummy_dmu=["Global Life 14", "Singfor Life 14", "DUMMY Cathay 15"])
plt.show()
#%%
utils.plot_3D(dmu=['Taiwan Life 14', 'Taiwan Life 15', 'Taiwan Life 16'], stitle="DUMMY Taiwan", target_input="operation_exp", smrts_dict=OPERATION_SMRTS_DUMMY141516, df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000, dummy_dmu=["CTBC Life 14", "CTBC Life 15", "DUMMY Taiwan 16"])
plt.show()
# %%
# visualize_progress
for k in ENG_NAMES_16:
    if "Chubb Tempest Life" == k:
        continue
    for target_input in ["insurance_exp", "operation_exp"]:
        
        if "insurance_exp" == target_input:
            smrts_dict = INSURANCE_SMRTS_DUMMY141516
        else:
            smrts_dict = OPERATION_SMRTS_DUMMY141516
            
        utils.plot_3D(dmu=[k+n for n in [' 14', ' 15', ' 16']], stitle=k, target_input=target_input, smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000)
        plt.savefig("%s %s.png" %(k, target_input), dpi=400)
        plt.show()
#%%
dmus = ["ACE Tempest Life 14", "ACE Tempest Life 15", "Chubb Tempest Life 16"]
for target_input in ["insurance_exp", "operation_exp"]:
    
    if "insurance_exp" == target_input:
        smrts_dict = INSURANCE_SMRTS_DUMMY141516
    else:
        smrts_dict = OPERATION_SMRTS_DUMMY141516
        
    utils.plot_3D(dmu=dmus, stitle="Chubb Tempest Life", target_input=target_input, smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000)
    plt.savefig("%s %s.png" %("Chubb Tempest Life", target_input), dpi=400)
    plt.show()
#%%
dmus = ["CTBC Life 14", "CTBC Life 15",]
for target_input in ["insurance_exp", "operation_exp"]:
    
    if "insurance_exp" == target_input:
        smrts_dict = INSURANCE_SMRTS_DUMMY141516
    else:
        smrts_dict = OPERATION_SMRTS_DUMMY141516
        
    utils.plot_3D(dmu=dmus, stitle="CTBC Life", target_input=target_input, smrts_dict=smrts_dict, df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000)
    plt.savefig("%s %s.png" %("CTBC Life", target_input), dpi=400)
    plt.show()
#%%
dmus = ["ACE Tempest Life 14", "ACE Tempest Life 15", "Chubb Tempest Life 16"]
utils.get_analyze_df(dmu_ks=dmus, df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000)
#%%
utils.get_analyze_df(dmu_ks=['Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16', 'Chunghwa Post 14', 'Chunghwa Post 15', 'Chunghwa Post 16', 'Shin Kong Life 14', 'Shin Kong Life 15', 'Chaoyang Life 14', 'Chaoyang Life 15', 'China Life 15', 'China Life 16', 'Fubon Life 15', 'Fubon Life 16', 'Hontai Life 15', 'Hontai Life 16', 'CIGNA 15', 'CIGNA 16'], df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000, round_to=4).to_excel("14-16 EFF_dmu analysis.xlsx")
# utils.get_analyze_df(dmu_ks=["ACE Tempest Life 14", "ACE Tempest Life 15", "Chubb Tempest Life 16"], df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000)
#%%
utils.get_analyze_df(dmu_ks=["Global Life 14", "Singfor Life 14", 'Cathay Life 14', 'Cathay Life 15', 'Cathay Life 16', "DUMMY Cathay 15", 'CTBC Life 14', 'CTBC Life 15', 'Taiwan Life 14','Taiwan Life 15', 'Taiwan Life 16', 'DUMMY Taiwan 16', ], df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000, round_to=4).to_excel("14-16 merged_dmu analysis.xlsx")
#%%
utils.get_analyze_df(
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
            ], df=denoise_nonpositive(LIFE_DUMMY141516)/1000/1000, round_to=4).to_excel("14-16 all_dmu analysis.xlsx")
#%%
