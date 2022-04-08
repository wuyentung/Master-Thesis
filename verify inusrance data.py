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
def sys_smrts(df:pd.DataFrame, i_star=0):
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)
    
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
#%%
verify_df = pd.read_csv("./verify data/24 non-life 2003.csv", index_col=0)
#%%
