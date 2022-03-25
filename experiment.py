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
# #%%
# def project_frontier(x:np.ndarray, y:np.ndarray, orient="IO", rs="vrs"):
#     projected_x = x.copy()
#     projected_y = y.copy()
#     lamdas = {}
#     K = x.shape[1]
#     for r in range(K):
#         if "IO" == orient:
#             obj, vars_dict = solver_r.io_dual(dmu=[i for i in range(K)], r=r, x=x, y=y, rs=rs)
#             projected_x[:, r] *= obj
#         elif "OO" == orient:
#             obj, vars_dict = solver_r.oo_dual(dmu=[i for i in range(K)], r=r, x=x, y=y, rs=rs)
#             projected_y[:, r] *= obj
#         else:
#             raise ValueError("only 'IO' or 'OO' can ba calculated") 
#         print(obj)
#         lamdas[r] = vars_dict
        
#     return projected_x, projected_y, lamdas
#%%
px_19, py_19, lambdas_19 = solver.project_frontier(x=np.array(life2019_transformed[['insurance_exp', 'operation_exp']].T), y=np.array(life2019_transformed[['underwriting_profit', 'investment_profit']].T), rs="vrs")
#%%
peff_dict, plambdas_dict = solver.dea_dual(dmu=life2019_transformed.index, x=px_19, y=py_19)
#%%
peff_dict