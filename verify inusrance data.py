#%%
import os
import dmp
import pandas as pd
import numpy as np
import solver
import solver_r
from load_data import denoise_nonpositive
from itertools import combinations
import matplotlib.pyplot as plt
#%%
def sys_smrts(df:pd.DataFrame, i_star=0, xcol:list=None, ycol:list=None):
    if xcol is None:
        xcol = df.columns.tolist()[i_star]
    if ycol is None:
        ycol = df.columns.tolist()[-2:]
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)
    # print(np.array([transformed_df[xcol].T]))
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array([transformed_df[xcol].T]), y=np.array(transformed_df[ycol].T))

    eff_dmu_name = []
    for key, value in eff_dict.items():
        if round(value, 5) == 1:
            eff_dmu_name.append(key)
    
    df = transformed_df.T[eff_dmu_name].T
    exp = dmp.get_smrts_dfs(dmu=[i for i in range(df.shape[0])], x=np.array([df[xcol].T]), y=np.array(df[ycol].T), trace=False, round_to=5, dmu_wanted=None, 
                            # i_star=i_star
                            )
    old_keys = list(exp.keys())
    for old_key in old_keys:
        exp[eff_dmu_name[old_key]] = exp.pop(old_key)
    return exp
#%%
# verify_df = pd.read_csv("./verify data/24 non-life 2003.csv", index_col=0)
verify_df = pd.read_csv("./verify data/24 non-life 2003.csv", index_col=0).dropna().astype('float')
#%%
# exp001 = sys_smrts(verify_df, i_star=0)
#%%
## 老師覺得拿掉富邦會讓整個 frontier 變比較平緩
exp010 = sys_smrts(verify_df.drop(["Fubon"]), i_star=0)
#%%
## 結果也是一樣計算不出 alpha
## 列印結果
# path = '24 non-life 2003 s-MRTS.txt'
# f = open(path, 'w')
# for key, value in exp001.items():
#     print(key, file=f)
#     print(value, file=f)
#     print("\n", file=f)
# f.close()
#%%
path = '24 non-life 2003 s-MRTS without Fubon.txt'
f = open(path, 'w')
for key, value in exp010.items():
    print(key, file=f)
    print(value, file=f)
    print("\n", file=f)
f.close()

#%%

## C eff 取 3
c3 = []
for c in combinations(verify_df.index.tolist(), 3):
    c3.append(sys_smrts(verify_df.T[list(c)].T, i_star=0))
#%%
