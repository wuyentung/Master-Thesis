#%%
'''
define dea solvers for all dmu
'''
import pandas as pd
import numpy as np 
import constant as const
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
def dea_dual(dmu:list, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001, orient=const.INPUT_ORIENT, rs=const.VRS):
    ## vrs dual dea solver using solver_r.io_vrs_dual() or solver_r.oo_vrs_dual()
    if const.INPUT_ORIENT == orient:
        solver_r_dual = solver_r.io_dual
    elif const.OUTPUT_ORIENT == orient:
        solver_r_dual = solver_r.oo_dual
    else:
        raise ValueError("only const.INPUT_ORIENT or const.OUTPUT_ORIENT can ba calculated") 
    
    eff_dict = {}
    lambdas_dict = {}
    projected_x = x.copy()
    projected_y = y.copy()
    K = len(dmu)
    
    for r in range(K):
        eff_dict[dmu[r]], r_lambdas_dict = solver_r_dual(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD, rs=rs)
        projected_x[:, r] *= eff_dict[dmu[r]]
        r_lambdas_df = pd.DataFrame.from_dict(r_lambdas_dict, orient="index", columns=[const.LAMBDA])
        r_lambdas_df["DMU_name"] = dmu
        r_lambdas_df = r_lambdas_df.set_index("DMU_name")
        lambdas_dict[dmu[r]] = r_lambdas_df
    '''
    lambdas_dict: {dmu_name: df(index=dmu_name, column=lambda)}
    '''
        
    return eff_dict, lambdas_dict, projected_x, projected_y
#%%
def dea_vrs(dmu:list, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001, orient=const.OUTPUT_ORIENT):
    ## vrs dual dea solver using solver_r.io_vrs_dual() or solver_r.oo_vrs_dual()
    eff_dict = {}
    v_dict = {}
    u_dict = {}
    intercept_dict = {}
    if const.INPUT_ORIENT == orient:
        raise ValueError("IO vrs is not implemented yet")
        for r in dmu:
            eff_dict[r], v_dict[r], u_dict[r], intercept_dict[r], = solver_r.io_vrs(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD)
    elif const.OUTPUT_ORIENT == orient:
        for r in dmu:
            eff_dict[r], v_dict[r], u_dict[r], intercept_dict[r],  = solver_r.oo_vrs(dmu=dmu, r=r, x=x, y=y, THRESHOLD=THRESHOLD)
    else:
        raise ValueError("only const.INPUT_ORIENT or const.OUTPUT_ORIENT can ba calculated") 
    return eff_dict, v_dict, u_dict, intercept_dict
#%%
# def project_frontier(x:np.ndarray, y:np.ndarray, orient=const.INPUT_ORIENT, rs=const.VRS):
#     projected_x = x.copy()
#     projected_y = y.copy()
#     lamdas = {}
#     K = x.shape[1]
#     for r in range(K):
#         if const.INPUT_ORIENT == orient:
#             obj, vars_dict = solver_r.io_dual(dmu=[i for i in range(K)], r=r, x=x, y=y, rs=rs)
#             projected_x[:, r] *= obj
#         elif const.OUTPUT_ORIENT == orient:
#             obj, vars_dict = solver_r.oo_dual(dmu=[i for i in range(K)], r=r, x=x, y=y, rs=rs)
#             projected_y[:, r] *= obj
#         else:
#             raise ValueError("only const.INPUT_ORIENT or const.OUTPUT_ORIENT can ba calculated") 
#         lamdas[r] = vars_dict
        
#     return projected_x, projected_y, lamdas