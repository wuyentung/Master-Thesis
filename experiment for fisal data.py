#%%
'''
改用保險業實證 dmp
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
from load_data import LIFE, FISAL_LIFE2019, denoise_nonpositive, FISAL_ATTRIBUTES, FISAL_LIFE2018, FISAL_LIFE2020
from itertools import combinations
import matplotlib.pyplot as plt
#%%
# #### 測試 #####
# life_transformed = denoise_nonpositive(LIFE)
# eff_dict, lambdas_dict = solver.dea_dual(dmu=life_transformed.index, x=np.array(life_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life_transformed[['underwriting_profit', 'investment_profit']].T))
# #%%
# eff_dict
# #%%
# eff_dmu_name = []
# for key, value in eff_dict.items():
#     if round(value, 5) == 1:
#         eff_dmu_name.append(key)
# eff_dmu_name
# #%%
# # life_transformed.T[eff_dmu_name].T
# eff_dict2, lambdas_dict2 = solver.dea_dual(dmu=eff_dmu_name, x=np.array(life_transformed.T[eff_dmu_name].T[['insurance_exp', 'operation_exp']].T), y=np.array(life_transformed.T[eff_dmu_name].T[['underwriting_profit', 'investment_profit']].T))
# #%%
# df = life_transformed.T[eff_dmu_name].T
# exp = dmp.get_smrts_dfs(dmu=[i for i in range(df.shape[0])], x=np.array(df[['insurance_exp', 'operation_exp']].T), y=np.array(df[['underwriting_profit', 'investment_profit']].T), trace=False, round_to=5, dmu_wanted=None)
# #%%
# px_19, py_19, lambdas_19 = solver.project_frontier(x=np.array(life_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life_transformed[['underwriting_profit', 'investment_profit']].T), rs="vrs", orient="IO")
# #%%
# peff_dict, plambdas_dict = solver.dea_dual(dmu=life_transformed.index, x=px_19, y=py_19)
# #%%
# peff_dict
#### 測試結束 #####
#%%
def sys_smrts(df:pd.DataFrame, project=False, i_star=0):
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)/1000/1000
            
    ## project all dmu to VRS frontier in IO
    if project:
        px, py, lambdas = solver.project_frontier(x=np.array(transformed_df[['insurance_exp', 'operation_exp']].T), y=np.array(transformed_df[['underwriting_profit', 'investment_profit']].T), rs="vrs", orient="IO")
        exp = dmp.get_smrts_dfs(dmu=[i for i in range(px.shape[1])], x=px, y=py, trace=False, round_to=5, dmu_wanted=None)
        old_keys = list(exp.keys())
        for old_key in old_keys:
            exp[df.index.tolist()[old_key]] = exp.pop(old_key)
        return exp
    
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array(transformed_df[['insurance_exp', 'operation_exp']].T), y=np.array(transformed_df[['underwriting_profit', 'investment_profit']].T), orient="OO")

    eff_dmu_name = []
    for key, value in eff_dict.items():
        if round(value, 5) == 1:
            eff_dmu_name.append(key)
    
    df = transformed_df.T[eff_dmu_name].T
    exp = dmp.get_smrts_dfs(dmu=[i for i in range(df.shape[0])], x=np.array(df[['insurance_exp', 'operation_exp']].T), y=np.array(df[['underwriting_profit', 'investment_profit']].T), trace=False, round_to=5, dmu_wanted=None, i_star=i_star)
    old_keys = list(exp.keys())
    for old_key in old_keys:
        exp[eff_dmu_name[old_key]] = exp.pop(old_key)
    return exp
# #%%
expALL = sys_smrts(df=LIFE, i_star=0)
#%%
for i in range(2):
    dmp.show_smrts(sys_smrts(df=LIFE, i_star=i), path="life million %s.txt" %i)
#%%
# exp18 = sys_smrts(df=LIFE2018)
# #%%
# exp19 = sys_smrts(df=LIFE2019)
# #%%
# exp20 = sys_smrts(df=LIFE2020)
## 好奇怪，個別年的算得出來，綜合在一起卻算不出來，不知道會不會是 transform 的問題
#%%
'''
## 儲存了
for key, value in exp18.items():
    value.to_csv("./result/s-MRTS %s.csv" %key)
for key, value in exp19.items():
    value.to_csv("./result/s-MRTS %s.csv" %key)
for key, value in exp20.items():
    value.to_csv("./result/s-MRTS %s.csv" %key)
'''
#%%
# for key, value in exp20.items():
#     print(key)
#     print(value)
#     print()
#%%
## 多年度綜合跟單年度的有效率 DMU
def find_eff_dmu(df:pd.DataFrame):
    transformed_df = denoise_nonpositive(df, .1)
    # print(transformed_df)
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array(transformed_df[['insurance_exp', 'operation_exp']].T), y=np.array(transformed_df[['underwriting_profit', 'investment_profit']].T))

    eff_dmu_name = []
    for key, value in eff_dict.items():
        if round(value, 5) == 1:
            eff_dmu_name.append(key)
    return eff_dmu_name
#%%
def comb_fun(df:pd.DataFrame, fun, comb_n=None):
    results = []
    combs = []
    if comb_n is None:
        comb_n = len(df.index.tolist())
    for comb in combinations(df.index.tolist(), comb_n):
        # print(len(list(comb)))
        # n_comb_i+=1
        try:
            result = fun(df.T[list(comb)].T)
            results.append(result)
            combs.append(list(comb))
            # print(i, comb, "\n")
            # break
        except:
            continue
    if 0 == len(combs):
        return comb_fun(df, fun, comb_n-1)
    return results, combs
#%%
# eff_dmu18 = find_eff_dmu(LIFE2018)
#%%
# eff_dmu19_combs = []
# combs = []
# for comb in combinations(LIFE2019.index.tolist(), 21):
#     # n_comb_i+=1
#     try:
#         eff_dmu19_comb = find_eff_dmu(LIFE2019.T[list(comb)].T)
#         # eff_dmu19_combs.append(eff_dmu19_comb)
#         combs.append(list(comb))
#         # print(i, comb, "\n")
#         # break
#     except:
#         continue
#%%
# eff_dmu19 = find_eff_dmu(LIFE2019.T[combs[0]].T)
# #%%
# eff_dmu20 = find_eff_dmu(LIFE2020)
# #%%
# eff_dmuALL = find_eff_dmu(LIFE)
#%%
# combs_smrts, combs_comb = comb_fun(df=LIFE.T[eff_dmuALL].T, fun=sys_smrts)
## 結果在下一行找好了
denoise_LIFE = denoise_nonpositive(LIFE)
success_smrts_dmu = ['Chunghwa Post 18', 'TransGlobe Life 18', 'Hontai Life 19', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'Cathay Life 20', 'China Life 20', 'Nan Shan Life 20', 'Shin Kong Life 20', 'Fubon Life 20', 'Hontai Life 20']
success_smrts_df = denoise_LIFE.T[success_smrts_dmu].T
success_smrts = sys_smrts(success_smrts_df)
#%%
sys_smrts(success_smrts_df, i_star=0)
## 結果還是都是 0 ，想改成從 x 或 y 的 scale 出發分群
#%%
## 列印結果
# path = 'success_smrts.txt'
# f = open(path, 'w')
# for key, value in success_smrts.items():
#     print(key, file=f)
#     print(value, file=f)
#     print("\n", file=f)
# f.close()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(denoise_LIFE["underwriting_profit"], denoise_LIFE["investment_profit"], c="blue")
# plt.scatter(LIFE["underwriting_profit"], LIFE["investment_profit"], c="green")
plt.xlabel("underwriting_profit after transform", fontsize=20)
plt.ylabel("investment_profit", fontsize=20)
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(denoise_LIFE[FISAL_ATTRIBUTES[0]], denoise_LIFE[FISAL_ATTRIBUTES[0]], c="blue")
# plt.scatter(LIFE[ATTRIBUTES[0]], LIFE[ATTRIBUTES[0]], c="green")
plt.show()
#%%
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
#%%
Z = linkage(LIFE[FISAL_ATTRIBUTES[:2]], method='ward')
plt.figure(figsize=(12, 8))
dn = dendrogram(Z, above_threshold_color='#bcbddc', orientation='right', labels=LIFE.index.to_list(),)
plt.title("HAC for input", fontsize=25)
plt.xlabel("cluster distance", fontsize=20)
plt.ylabel("Life Insurance companies 18-20", fontsize=20)
# plt.savefig("HAC for input in Life Insurance companies 18-20", dpi=1600)
plt.show()
#%%
K_cluster = 2
hac = AgglomerativeClustering(n_clusters=K_cluster, affinity='euclidean', linkage='ward').fit_predict(LIFE[FISAL_ATTRIBUTES[:2]])
#%%
hac_life = pd.concat([LIFE, pd.DataFrame(hac, columns=["HAC cluster"], index=LIFE.index)], axis=1)
#%%
combs_hac0_smrts, combs_hac0_comb = comb_fun(df=hac_life[hac_life["HAC cluster"] == 0], fun=sys_smrts)
# print(combs_hac0_comb[0])
#%%
hac0_comb = ['Cathay Life 18', 'Nan Shan Life 18', 'Fubon Life 18', 'Cathay Life 19', 'Nan Shan Life 19', 'Shin Kong Life 19', 'Fubon Life 19', 'Cathay Life 20', 'Nan Shan Life 20', 'Shin Kong Life 20', 'Fubon Life 20']
hac0_smrts0 = sys_smrts(df=hac_life.T[hac0_comb].T, i_star=0)
hac0_smrts1 = sys_smrts(df=hac_life.T[hac0_comb].T, i_star=1)
#%%
# combs_hac1_eff, combs_hac1_effcomb = comb_fun(df=hac_life[hac_life["HAC cluster"] == 1], fun=find_eff_dmu)
#%%
hac1_dmu = ['Hontai Life 18', 'Chunghwa Post 18', 'First-Aviva Life 18', 'TransGlobe Life 18', 'First-Aviva Life 19', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'China Life 20', 'Hontai Life 20', 'Chunghwa Post 20', 'First-Aviva Life 20']
combs_hac1_smrts, combs_hac1_comb = comb_fun(df=hac_life.T[hac1_dmu].T, fun=sys_smrts)
# result: ['Hontai Life 18', 'Chunghwa Post 18', 'TransGlobe Life 18', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'China Life 20', 'Hontai Life 20']
#%%
hac1_comb = ['Hontai Life 18', 'Chunghwa Post 18',  'TransGlobe Life 18', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'China Life 20', 'Hontai Life 20']
hac1_smrts0 = sys_smrts(df=hac_life.T[hac1_comb].T, i_star=0)
hac1_smrts1 = sys_smrts(df=hac_life.T[hac1_comb].T, i_star=1)
#%%
## 試試看 2003 資料踢掉富邦，再計算 s-MRTS
## 改成用兩個險種的核保收益計算 s-MRTS