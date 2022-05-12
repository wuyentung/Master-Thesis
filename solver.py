#%%
'''
define dea solvers for all dmu
'''
import pandas as pd
import numpy as np 
import solver_r
'''data structure
x:array([
    [i1_k1, i1_k2, ..., i1_kK], 
    [i2_k1, i2_k2, ..., i2_kK], 
    ...,
    [iI_k1, iI_k2, ..., iI_kK], 
    ]) I inputs, K firms
    
y:array([
    [j1_k1, j1_k2, ..., j1_kK], 
    [j2_k1, j2_k2, ..., j2_kK], 
    ...,
    [jJ_k1, jJ_k2, ..., jJ_kK], 
    ]) J outputs, K firms
'''
#%%
def dea_dual(dmu:list, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001, orient="IO", rs="vrs"):
    ## vrs dual dea solver using solver_r.io_vrs_dual() or solver_r.oo_vrs_dual()
    eff_dict = {}
    lambdas_dict = {}
    projected_x = x.copy()
    projected_y = y.copy()
    K = len(dmu)
    if "IO" == orient:
        for r in range(K):
            eff_dict[dmu[r]], lambdas_dict[dmu[r]] = solver_r.io_dual(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD, rs=rs)
            projected_x[:, r] *= eff_dict[dmu[r]]
    elif "OO" == orient:
        for r in range(K):
            eff_dict[dmu[r]], lambdas_dict[dmu[r]] = solver_r.oo_dual(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD, rs=rs)
            projected_y[:, r] *= eff_dict[dmu[r]]
    else:
        raise ValueError("only 'IO' or 'OO' can ba calculated") 
    return eff_dict, lambdas_dict, projected_x, projected_y
#%%
def dea_vrs(dmu:list, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001, orient="OO"):
    ## vrs dual dea solver using solver_r.io_vrs_dual() or solver_r.oo_vrs_dual()
    eff_dict = {}
    v_dict = {}
    u_dict = {}
    intercept_dict = {}
    if "IO" == orient:
        raise ValueError("IO vrs is not implemented yet")
        for r in dmu:
            eff_dict[r], v_dict[r], u_dict[r], intercept_dict[r], = solver_r.io_vrs(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD)
    elif "OO" == orient:
        for r in dmu:
            eff_dict[r], v_dict[r], u_dict[r], intercept_dict[r],  = solver_r.oo_vrs(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD)
    else:
        raise ValueError("only 'IO' or 'OO' can ba calculated") 
    return eff_dict, v_dict, u_dict, intercept_dict
#%%
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
#         lamdas[r] = vars_dict
        
#     return projected_x, projected_y, lamdas