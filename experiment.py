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
from load_data import LIFE, LIFE2019, denoise_nonpositive, ATTRIBUTES, LIFE2018, LIFE2020
from itertools import combinations
import matplotlib.pyplot as plt
#%%
#### 測試 #####
'''life_transformed = denoise_nonpositive(LIFE)
eff_dict, lambdas_dict = solver.dea_dual(dmu=life_transformed.index, x=np.array(life_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life_transformed[['underwriting_profit', 'investment_profit']].T))
#%%
eff_dict
#%%
eff_dmu_name = []
for key, value in eff_dict.items():
    if round(value, 5) == 1:
        eff_dmu_name.append(key)
eff_dmu_name
#%%
# life_transformed.T[eff_dmu_name].T
eff_dict2, lambdas_dict2 = solver.dea_dual(dmu=eff_dmu_name, x=np.array(life_transformed.T[eff_dmu_name].T[['insurance_exp', 'operation_exp']].T), y=np.array(life_transformed.T[eff_dmu_name].T[['underwriting_profit', 'investment_profit']].T))
#%%
df = life_transformed.T[eff_dmu_name].T
exp = dmp.get_smrts_dfs(dmu=[i for i in range(df.shape[0])], x=np.array(df[['insurance_exp', 'operation_exp']].T), y=np.array(df[['underwriting_profit', 'investment_profit']].T), trace=False, round_to=5, dmu_wanted=None)
#%%
px_19, py_19, lambdas_19 = solver.project_frontier(x=np.array(life_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life_transformed[['underwriting_profit', 'investment_profit']].T), rs="vrs", orient="IO")
#%%
peff_dict, plambdas_dict = solver.dea_dual(dmu=life_transformed.index, x=px_19, y=py_19)
#%%
peff_dict'''
#### 測試結束 #####
#%%
def sys_smrts(df:pd.DataFrame, project=False, i_star=0):
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)
            
    ## project all dmu to VRS frontier in IO
    if project:
        px, py, lambdas = solver.project_frontier(x=np.array(transformed_df[['insurance_exp', 'operation_exp']].T), y=np.array(transformed_df[['underwriting_profit', 'investment_profit']].T), rs="vrs", orient="IO")
        exp = dmp.get_smrts_dfs(dmu=[i for i in range(px.shape[1])], x=px, y=py, trace=False, round_to=5, dmu_wanted=None)
        old_keys = list(exp.keys())
        for old_key in old_keys:
            exp[df.index.tolist()[old_key]] = exp.pop(old_key)
        return exp
    
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array(transformed_df[['insurance_exp', 'operation_exp']].T), y=np.array(transformed_df[['underwriting_profit', 'investment_profit']].T))

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
# expALL = sys_smrts(df=LIFE)
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
## 結果在下一行找好了
# combs_smrts, combs_comb = comb_fun(df=LIFE.T[eff_dmuALL].T, fun=sys_smrts)
denoise_LIFE = denoise_nonpositive(LIFE)
success_smrts_dmu = ['Chunghwa Post 18', 'TransGlobe Life 18', 'Hontai Life 19', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'Cathay Life 20', 'China Life 20', 'Nan Shan Life 20', 'Shin Kong Life 20', 'Fubon Life 20', 'Hontai Life 20']
success_smrts_df = denoise_LIFE.T[success_smrts_dmu].T
success_smrts = sys_smrts(success_smrts_df)
#%%
sys_smrts(success_smrts_df, i_star=0)
## 結果還是都是 0 ，想改成從 x 或 y 的 scale 出發分群
#%%
## 列印結果
path = 'success_smrts.txt'
f = open(path, 'w')
for key, value in success_smrts.items():
    print(key, file=f)
    print(value, file=f)
    print("\n", file=f)
f.close()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(denoise_LIFE["underwriting_profit"], denoise_LIFE["investment_profit"], c="blue")
plt.scatter(LIFE["underwriting_profit"], LIFE["investment_profit"], c="green")
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(denoise_LIFE[ATTRIBUTES[0]], denoise_LIFE[ATTRIBUTES[0]], c="blue")
# plt.scatter(LIFE[ATTRIBUTES[0]], LIFE[ATTRIBUTES[0]], c="green")
plt.show()
#%%
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
#%%
# transformed18 = denoise_nonpositive(LIFE2018, min_value=.1)
# transformed19 = denoise_nonpositive(LIFE2019.T[combs[0]].T, min_value=.1)
# transformed20 = denoise_nonpositive(LIFE2020, min_value=.1)
# #%%
# # transformedALL = denoise_nonpositive(LIFE, min_value=.1)
# #%%
# transformedALL2 = denoise_nonpositive(pd.concat([transformed18, transformed19, transformed20]), min_value=.1)
# #%%
# eff_anual_combs = []
# combs_anual = []
# for comb in combinations(transformedALL2.index.tolist(), 63):
#     # n_comb_i+=1
#     try:
#         eff_anual_comb = find_eff_dmu(transformedALL2.T[list(comb)].T)
#         # sys_smrts(transformedALL2.T[list(comb)].T)
#         eff_anual_combs.append(eff_anual_comb)
#         combs_anual.append(list(comb))
#         # print(i, comb, "\n")
#         # break
#     except:
#         continue
# #%%
# eff_anual_dmu = find_eff_dmu(transformedALL2)
# #%%
# ## 轉換後找各年度有效率的公司
# eff_anual_dmu = find_eff_dmu(pd.concat([transformed18, transformed19, transformed20]).T[eff_dmu18+eff_dmu19+eff_dmu20].T)
# #%%
# ## 還是無法
# '''
# exp_anualALL = sys_smrts(pd.concat([transformed18, transformed19, transformed20]).T[eff_anual_dmu].T)
# '''
# ## 只能一個一個拿掉來看了
# #%%
# anual_dmu_df = pd.concat([transformed18, transformed19, transformed20]).T[eff_anual_dmu].T
# #%%
# # valid_comb = []
# # for i in range(17, 0, -1):
# #     valid_comb_i = []
# #     n_comb_i = 0
# comb_expALLs = []
# for comb in combinations(eff_anual_dmu, 16):
#     # n_comb_i+=1
#     try:
#         comb_expALL = sys_smrts(anual_dmu_df.T[list(comb)].T)
#         comb_expALLs.append(comb_expALL)
#         # print(i, comb, "\n")
#         # break
#     except:
#         continue
#     # if len(valid_comb_i) < n_comb_i:
#     #     print(i, len(valid_comb_i))
#     #     valid_comb.append(valid_comb_i)
#     # else:
#     #     print(f"all combination of {i} is validable")
#     #     break
# #%%
# for r in eff_anual_dmu:
#     v = []
#     for comb in comb_expALLs:
#         v.append(np.sum([1 if r in c else 0 for c in comb.keys()]))
#     print(f"{r}: {v}")
# #%%
# abanded_eff = ["First-Aviva Life 18", "First-Aviva Life 19", "Bank Taiwan Life 20", "First-Aviva Life 20"]
# abanded_df = anual_dmu_df.T[abanded_eff].T
# #%%
# for key, value in comb_expALLs[1].items():
#     print(key)
#     print(value)
#     print()
# #%%

#%%
