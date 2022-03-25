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
from load_data import LIFE, LIFE2019, denoise_nonpositive, ATTRIBUTES
#%%
life2019_transformed = denoise_nonpositive(LIFE2019)
eff_dict, lambdas_dict = solver.dea_dual(dmu=life2019_transformed.index, x=np.array(life2019_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life2019_transformed[['underwriting_profit', 'investment_profit']].T))
#%%
eff_dict
#%%
px_19, py_19, lambdas_19 = solver.project_frontier(x=np.array(life2019_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life2019_transformed[['underwriting_profit', 'investment_profit']].T), rs="vrs", orient="IO")
#%%
peff_dict, plambdas_dict = solver.dea_dual(dmu=life2019_transformed.index, x=px_19, y=py_19)
#%%
peff_dict
#%%
exp = dmp.get_smrts_dfs(dmu=[i for i in range(px_19.shape[1])], x=px_19, y=py_19, trace=False, round_to=5, dmu_wanted=None)
#%%
