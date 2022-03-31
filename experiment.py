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
#%%
#### 測試 #####
life_transformed = denoise_nonpositive(LIFE)
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
peff_dict
#### 測試結束 #####
#%%
def sys_smrts(df:pd.DataFrame, project=False):
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
    exp = dmp.get_smrts_dfs(dmu=[i for i in range(df.shape[0])], x=np.array(df[['insurance_exp', 'operation_exp']].T), y=np.array(df[['underwriting_profit', 'investment_profit']].T), trace=False, round_to=5, dmu_wanted=None)
    old_keys = list(exp.keys())
    for old_key in old_keys:
        exp[eff_dmu_name[old_key]] = exp.pop(old_key)
    return exp
# #%%
# expALL = sys_smrts(df=LIFE)
# #%%
exp18 = sys_smrts(df=LIFE2018)
exp19 = sys_smrts(df=LIFE2019)
exp20 = sys_smrts(df=LIFE2020)
## 好奇怪，個別年的算得出來，綜合在一起卻算不出來，不知道會不會是 transform 的問題
#%%
for key, value in exp18.items():
    value.to_csv("./result/s-MRTS %s.csv" %key)
for key, value in exp19.items():
    value.to_csv("./result/s-MRTS %s.csv" %key)
for key, value in exp20.items():
    value.to_csv("./result/s-MRTS %s.csv" %key)
#%%
# for key, value in exp18.items():
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
eff_dmu18 = find_eff_dmu(LIFE2018)
eff_dmu19 = find_eff_dmu(LIFE2019)
eff_dmu20 = find_eff_dmu(LIFE2020)
eff_dmuALL = find_eff_dmu(LIFE)
#%%
transformed18 = denoise_nonpositive(LIFE2018, min_value=.1)
#%%
transformed19 = denoise_nonpositive(LIFE2019, min_value=.1)
#%%
transformed20 = denoise_nonpositive(LIFE2020, min_value=.1)
#%%
transformedALL = denoise_nonpositive(LIFE, min_value=.1)
#%%
transformedALL2 = denoise_nonpositive(pd.concat([transformed18, transformed19, transformed20]), min_value=.1)
#%%
eff_anual_dmu = find_eff_dmu(pd.concat([transformed18, transformed19, transformed20]).T[eff_dmu18+eff_dmu19+eff_dmu20].T)
#%%
eff_anual_dmu = find_eff_dmu(pd.concat([transformed18, transformed19, transformed20]))
#%%
find_eff_dmu(transformedALL)
#%%
find_eff_dmu(transformedALL2)
#%%
