#%%
import dmp
import pandas as pd
import numpy as np
import solver
from load_data import LIFE181920, denoise_nonpositive, LIFE141516
#%%
def sys_smrts(df:pd.DataFrame, project=False, i_star=0):
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)/1000/1000
                
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
    return exp, eff_dict
#%%
INSURANCE_SMRTS181920, EFF_DICT181920 = sys_smrts(df=LIFE181920, i_star=0)
OPERATION_SMRTS181920, EFF_DICT181920 = sys_smrts(df=LIFE181920, i_star=1)
#%%
INSURANCE_SMRTS141516, EFF_DICT141516 = sys_smrts(df=LIFE141516, i_star=0)
OPERATION_SMRTS141516, EFF_DICT141516 = sys_smrts(df=LIFE141516, i_star=1)