#%%
'''
solving directional margin productivity
'''
import gurobipy as gp
import pandas as pd
import numpy as np 
gp.setParam("LogToConsole", 0)
gp.setParam("LogFile", "log.txt")
#%%
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

gy: array([
    gy1, gy2, ..., gyJ
])
'''
def cal_alpha_r(dmu_idxs:list, x:np.ndarray, y:np.ndarray, gy:np.ndarray, i_star:int, THRESHOLD:float, I, J, obj_fun, third_rhs, r):
    ## calculate alpha for rth DMU
    v = {}
    u = {}
    u0_plus = {}
    
    m = gp.Model(f"dmp_{r}")

    for i in range(I):
        v[i]=m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"v_{i}", 
            # lb=THRESHOLD
            )

    for j in range(J):
        u[j]=m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"u_{j}", 
            # lb=THRESHOLD 
            )
    
    u0_plus = m.addVar(vtype=gp.GRB.CONTINUOUS,name="u0+", )
    u0_minus = m.addVar(vtype=gp.GRB.CONTINUOUS,name="u0-", )
    
    m.update()

    ## alpha
    m.setObjective(v[i_star] / np.max(x[i_star]), obj_fun)

    ## s.t.
    m.addConstr(gp.quicksum(v[i] * x[i, dmu_idxs.index(r)] / np.max(x[i]) for i in range(I)) - gp.quicksum(u[j] * y[j, dmu_idxs.index(r)] / np.max(y[j]) for j in range(J)) + u0_plus - u0_minus == 0)
    for k in dmu_idxs:
        m.addConstr(gp.quicksum(v[i] * x[i, dmu_idxs.index(k)] / np.max(x[i]) for i in range(I)) - gp.quicksum(u[j] * y[j, dmu_idxs.index(k)] / np.max(y[j]) for j in range(J)) + u0_plus - u0_minus >= 0)
    m.addConstr(gp.quicksum(u[j] * gy[j] for j in range(J)) == third_rhs)
        
    m.optimize()
    
    if 2 == m.status:
        return m.objVal
    # return np.nan
    multiplier = 1.0005
    # print(f"DMU index {r} using direction {gy} is infeasible or unbounded, multipling y with {multiplier} to meet the frontier,")
    # print(f"x: {x}, \ny:{y}")
    # print("=======\n\n")
    y[:, r]*=(multiplier)
    return cal_alpha_r(dmu_idxs, x, y, gy, i_star, THRESHOLD, I, J, obj_fun, third_rhs, r)
#%%
def cal_alpha(dmu_idxs:list, x:np.ndarray, y:np.ndarray, gy:np.ndarray, i_star:int, DMP_contraction:bool, THRESHOLD:float, wanted_idxs:list):
    ## i_star: index of the change of single input Xi*, which is the target we want to investigate
    ## dmu_wanted: the dmu we want to investigate
    """_summary_

    Args:
        dmu_idxs (list): _description_
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        gy (np.ndarray): _description_
        i_star (int): _description_
        DMP_contraction (bool): True means left side of DMP, False means right side of DMP
        THRESHOLD (float, optional): _description_
        wanted_idxs (list, optional): _description_

    Returns:
        _type_: _description_
    """
    I = x.shape[0]
    J = y.shape[0]
    alpha = {}
    if DMP_contraction:
        obj_fun = gp.GRB.MAXIMIZE
        # third_rhs = -1
    else:
        obj_fun = gp.GRB.MINIMIZE
    third_rhs = 1
        
    for r in wanted_idxs:
        alpha[r] = cal_alpha_r(dmu_idxs, x, y, gy, i_star, THRESHOLD, I, J, obj_fun, third_rhs, r)
    return alpha
#%%
def cal_dmp(dmu_idxs:list, alpha:dict, y:np.ndarray, gy:np.ndarray, wanted_idxs:list):
    ## dmu_wanted: the dmu we want to investigate
    J = y.shape[0]
    dmp = {}
    for r in wanted_idxs:
        dmp[r] = [None] * J
        for j in range(J):
            dmp[r][j] = alpha[r] * gy[j] * np.max(y[j])
    return dmp
#%%
def cal_smrts(dmu_idxs:list, dmp1:dict, dmp2:dict, round_to:int, wanted_idxs:list):
    """return s-MRTS dictionary between two directions (dmp1, dmp2)

    Args:
        dmu (list): [description]
        dmp1 (dict): [description]
        dmp2 (dict): [description]

    Returns:
        dict: [description]
    """
    smrts = {}
    for r in wanted_idxs:
        smrts[r] = (dmp1[r][0] - dmp2[r][0]) / (dmp1[r][1] - dmp2[r][1]) if round((dmp1[r][1] - dmp2[r][1]), round_to) else np.nan
    return smrts
#%%
class Dmu_Direction(object):
    def __init__(self, dmu, direction, alpha:float, dmp:list, smrts:float, round_to:int):
        self.dmu = dmu
        self.direction = direction
        self.name = f"{dmu}_{direction}"
        
        self.alpha = np.round(alpha, round_to)
        self.dmp = np.round(dmp, round_to)
        if smrts:
            self.smrts = np.round(smrts, round_to)
        else:
            self.smrts = smrts
    # def round_to(self, x, f):
    #     return np.round(x, f)
#%%
DIRECTIONS = [ 
    [.99, .01], 
    [.9, .1], 
    [.8, .2], 
    [.7, .3], 
    [.6, .4], 
    [.5, .5], 
    [.4, .6], 
    [.3, .7], 
    [.2, .8], 
    [.1, .9], 
    [.01, .99], 
]
NEG_DIRECTIONS = [ 
    [-0.99, -0.01], # since [-1, 0] can cause infeasible or unbounded
    [-0.9, -0.1], 
    [-0.8, -0.2], 
    [-0.7, -0.3], 
    [-0.6, -0.4], 
    [-0.5, -0.5], 
    [-0.4, -0.6], 
    [-0.3, -0.7], 
    [-0.2, -0.8], 
    [-0.1, -0.9], 
    [0, -1], 
]
#%%
def get_smrts_dfs(dmu:list, x:np.ndarray, y:np.ndarray, DMP_contraction:bool, trace=False, round_to:int=2, wanted_idxs:list=None, i_star:int=0, THRESHOLD=0.0000000001):
    dmu_idxs = [i for i in range(len(dmu))]
    ## wanted_idxs: the index of dmu we want to investigate
    if wanted_idxs is None:
        wanted_idxs = dmu_idxs

    results = []
    dmp_directions = []
    
    # if DMP_contraction:
    #     directions = NEG_DIRECTIONS
    # else:
    directions = DIRECTIONS
        
    for i in range(len(directions)):
        direction = directions[i]
        alpha = cal_alpha(dmu_idxs=dmu_idxs, x=x, y=y, gy=direction, wanted_idxs=wanted_idxs, i_star=i_star, DMP_contraction=DMP_contraction, THRESHOLD=THRESHOLD)
        dmp = cal_dmp(dmu_idxs=dmu_idxs, alpha=alpha, y=y, gy=direction, wanted_idxs=wanted_idxs)
        dmp_directions.append(dmp)
        if not i:
            smrts = {r:None for r in wanted_idxs}
        else:
            smrts = cal_smrts(dmu_idxs=dmu_idxs, dmp1=dmp_directions[i-1], dmp2=dmp_directions[i], round_to=round_to, wanted_idxs=wanted_idxs)
        if trace:
            print(direction)
            for r in wanted_idxs:
                print(r, alpha[r], dmp[r], smrts[r])
            print()
        for r in wanted_idxs:
            results.append(Dmu_Direction(dmu=dmu[r], direction=direction, alpha=alpha[r], dmp=dmp[r], smrts=smrts[r], round_to=round_to))
    col_names = ["direction", "alpha", "DMP", "s-MRTS"]
    dfs = {}
    for r in wanted_idxs:
        df = pd.DataFrame(columns=col_names)
        for res in results:
            if res.dmu == dmu[r]:
                df = df.append(pd.DataFrame([[str(res.direction), res.alpha, res.dmp, res.smrts]], columns=col_names))
                # print(res.direction, res.alpha, res.dmp)
        dfs[dmu[r]] = df.set_index("direction")
    # dfs["A"]
    return dfs
#%%
def show_smrts(smrts_dict:dict, path:str=None):
    if path is None:
        f = None
    else:
        f = open(path, "w")
    
    for key, value in smrts_dict.items():
        print(key, file=f)
        print(value, file=f)
        print("\n", file=f)
    
    if path is None:
        return
    else:
        f.close()
    return
#%%
## unit test
if __name__ == "__main__":
    dmu = ["A", "B", "C", "BC", "AC"]
    x = np.array([[2, 4, 1, 2.5, 1.5]])
    y = np.array([
        [1, 2, 4, 3, 2.5], 
        [200, 300, 100, 200, 150], 
        ])
    dfs = get_smrts_dfs(dmu, x, y, trace=False, round_to=4, wanted_idxs=[0, 1, 2, 3, 4], DMP_contraction=False)
    # print(dfs["A"])
    # print(2)
#%%
