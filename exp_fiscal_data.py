#%%
import dmp
import pandas as pd
import numpy as np
import constant as const
import solver
from load_data import denoise_nonpositive, LIFE141516, LIFE_DUMMY141516
import time
import fiscal_analyzing_utils as utils
import matplotlib.pyplot as plt
#%%
def sys_smrts(df:pd.DataFrame, project=False, i_star=0, div_norm=6, round_to=6):
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df, div_norm=div_norm, round_to=round_to)
                
    eff_dict, lambdas_dict, projected_x, projected_y = solver.dea_dual(dmu=transformed_df.index, x=np.array(transformed_df[[const.INSURANCE_EXP, const.OPERATION_EXP]].T), y=np.array(transformed_df[[const.UNDERWRITING_PROFIT, const.INVESTMENT_PROFIT]].T), orient=const.OUTPUT_ORIENT)
    
    if project:
        transformed_df[const.INSURANCE_EXP] = projected_x[0]
        transformed_df[const.OPERATION_EXP] = projected_x[1]
        transformed_df[const.UNDERWRITING_PROFIT] = projected_y[0]
        transformed_df[const.INVESTMENT_PROFIT] = projected_y[1]
        df = transformed_df
        print("project=True cannot calculate since underflow problem")
        return None, eff_dict, df
        
    else:
        ## efficienct dmu only
        eff_dmu_name = []
        for key, value in eff_dict.items():
            if round(value, 5) == 1:
                eff_dmu_name.append(key)
        
        df = transformed_df.loc[eff_dmu_name]

    smrts_dfs = dmp.get_smrts_dfs(dmu=df.index, x=np.array(df[[const.INSURANCE_EXP, const.OPERATION_EXP]].T), y=np.array(df[[const.UNDERWRITING_PROFIT, const.INVESTMENT_PROFIT]].T), trace=False, round_to=5, wanted_idxs=None, i_star=i_star)
    
    if project:
        return smrts_dfs, eff_dict, lambdas_dict, df
    return smrts_dfs, eff_dict, lambdas_dict
#%%
# INSURANCE_SMRTS181920, EFF_DICT181920, LAMBDA_DICT181920 = sys_smrts(df=LIFE181920, i_star=0)
# OPERATION_SMRTS181920, EFF_DICT181920, LAMBDA_DICT181920 = sys_smrts(df=LIFE181920, i_star=1)
#%%
INSURANCE_SMRTS141516, EFF_DICT141516, LAMBDA_DICT141516 = sys_smrts(df=LIFE141516, i_star=0)
OPERATION_SMRTS141516, EFF_DICT141516, LAMBDA_DICT141516 = sys_smrts(df=LIFE141516, i_star=1)
#%%
INSURANCE_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516 = sys_smrts(df=LIFE_DUMMY141516, i_star=0)
OPERATION_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516 = sys_smrts(df=LIFE_DUMMY141516, i_star=1)
#%%
if __name__ == "__main__":
    n, d, df = sys_smrts(df=LIFE_DUMMY141516, i_star=1, project=True)
    #%%
    for d in [4, 5, 6,]:
        for rt in [4, 5, 6,]:
            s_time = time.time()
            try:
                OPERATION_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, df1 = sys_smrts(df=LIFE_DUMMY141516, i_star=0, project=True, div_norm=d, round_to=rt)
                print(f"======success======")
                print(f"divide_norm in {d} with round to {rt} infeasible, time: {round(time.time() - s_time, 2)} sec")
            except:
                print(f"divide_norm in {d} with round to {rt} infeasible, time: {round(time.time() - s_time, 2)} sec")
                ## result: 6, 6 usually the longest time
    #%%
    temp = denoise_nonpositive(LIFE_DUMMY141516)
    temp
    #%%
    temp.round(4)
    #%%
