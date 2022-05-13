#%%
'''
define solvers for dmu r
'''
import gurobipy as gp
import pandas as pd
import numpy as np 
gp.setParam("LogToConsole", 0)
gp.setParam("LogFile", "log.txt")
import constant as const
'''DEA data structure
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
def io_dual(dmu:list, r:int, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001, rs=const.VRS):
    ## input-oriented dual dea solver
    ## if vrs: cal vrs dual, else: cal crs dual
    
    I = x.shape[0]
    J = y.shape[0]
    K = len(dmu)
    lambda_k = {}
    
    m = gp.Model(f"io_vrs_dual_{r}")

    theta_r = m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"theta_{r}", 
            ub=1
            )
    for k in range(K):
        lambda_k[k]=m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"lambda_k{k}", 
            lb=THRESHOLD
            )
    m.update()

    ## efficiency
    m.setObjective(theta_r, gp.GRB.MINIMIZE)

    ## s.t.
    for i in range(I):
        m.addConstr(gp.quicksum(lambda_k[k] * x[i, k] for k in range(K)) <= theta_r * x[i, r])
    for j in range(J):
        m.addConstr(gp.quicksum(lambda_k[k] * y[j, k] for k in range(K)) >= y[j, r])
        
    if const.VRS == rs:
        m.addConstr(gp.quicksum(lambda_k[k] for k in range(K)) == 1)
    elif const.NIRS == rs:
        m.addConstr(gp.quicksum(lambda_k[k] for k in range(K)) <= 1)
    elif const.NDRS == rs:
        m.addConstr(gp.quicksum(lambda_k[k] for k in range(K)) >= 1)

    m.optimize()
    
    return m.objVal, m.getAttr('x', lambda_k)
#%%
def oo_dual(dmu:list, r:int, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001, rs=const.VRS):
    ## output-oriented dual dea solver
    ## if vrs: cal vrs dual, else: cal crs dual
    
    I = x.shape[0]
    J = y.shape[0]
    K = len(dmu)
    lambda_k = {}
    
    m = gp.Model(f"oo_vrs_dual_{r}")

    theta_r = m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"theta_{r}", 
            lb=1
            )
    for k in range(K):
        lambda_k[k]=m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"lambda_k{k}", 
            lb=THRESHOLD
            )
    m.update()

    ## efficiency
    m.setObjective(theta_r, gp.GRB.MAXIMIZE)

    ## s.t.
    for j in range(J):
        m.addConstr(gp.quicksum(lambda_k[k] * y[j, k] for k in range(K)) >= theta_r * y[j, r])
    for i in range(I):
        m.addConstr(gp.quicksum(lambda_k[k] * x[i, k] for k in range(K)) <= x[i, r])
    if const.VRS == rs:
        m.addConstr(gp.quicksum(lambda_k[k] for k in range(K)) == 1)
    elif const.NIRS == rs:
        m.addConstr(gp.quicksum(lambda_k[k] for k in range(K)) <= 1)
    elif const.NDRS == rs:
        m.addConstr(gp.quicksum(lambda_k[k] for k in range(K)) >= 1)

    m.optimize()
    
    return m.objVal, m.getAttr('x', lambda_k)
#%%
def oo_vrs(dmu:list, r:int, x:np.ndarray, y:np.ndarray,  THRESHOLD=0.000000000001):
    ## input-oriented vrs dual dea solver
    
    I = x.shape[0]
    J = y.shape[0]
    K = len(dmu)
    
    v, u= {}, {}

    m = gp.Model("DEA_VRS_OO")

    for i in range(I):
        v[i]=m.addVar(vtype=gp.GRB.CONTINUOUS,name="v_%d"%i, 
            lb=THRESHOLD
            )

    for j in range(J):
        u[j]=m.addVar(vtype=gp.GRB.CONTINUOUS,name="u_%d"%j, 
            lb=THRESHOLD 
            )
        
    v0 = m.addVar(vtype=gp.GRB.CONTINUOUS,name="v0", lb=-1000)
    
    m.update()

    m.setObjective(gp.quicksum([v[i] * x[i, r] for i in range(I)]) + v0, gp.GRB.MINIMIZE)

    m.addConstr(gp.quicksum([u[j] * y[j, r] for j in range(J)]) == 1)
    for k in range(K):
        m.addConstr(gp.quicksum([v[i] * x[i, k] for i in range(I)]) - gp.quicksum([u[j] * y[j, k] for j in range(J)]) + v0 >= 0)
        
    m.optimize()
    
    vars_dict = {}
    for var in m.getVars():
        vars_dict[var.varName] = var.x
        
    return m.objVal, vars_dict
#%%
# def _projecting(target:np.ndarray, lambda_r:dict):
#     N = target.shape[0]
#     projected = {}
#     for n in range(N):
#         projected[n] = np.sum([target[n, dmu_i] * lambda_i for dmu_i, lambda_i in lambda_r.items()])
#     return projected
# #%%
# def project(x:np.ndarray, y:np.ndarray, lambda_r:dict):
    
#     x_project = _projecting(x, lambda_r)
#     y_project = _projecting(y, lambda_r)
#     return x_project, y_project
#%%