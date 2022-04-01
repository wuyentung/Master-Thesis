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
def cal_alpha(dmu:list, x:np.ndarray, y:np.ndarray, gy:np.ndarray, i_star:int, THRESHOLD=0.000000000001, dmu_wanted:list=None):
    ## i_star: index of the change of single input Xi*, which is the target we want to investigate
    ## dmu_wanted: the dmu we want to investigate, defalt None
    
    I = x.shape[0]
    J = y.shape[0]
    alpha = {}
    for r in dmu_wanted:
        v = {}
        u = {}
        u0 = {}
        
        m = gp.Model(f"dmp_{r}")

        for i in range(I):
            v[i]=m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"v_{i}", 
                lb=THRESHOLD
                )

        for j in range(J):
            u[j]=m.addVar(vtype=gp.GRB.CONTINUOUS,name=f"u_{j}", 
                lb=THRESHOLD 
                )
        
        u0 = m.addVar(vtype=gp.GRB.CONTINUOUS,name="u0", )
        
        m.update()

        ## alpha
        m.setObjective(v[i_star] / np.max(x[i_star]), gp.GRB.MINIMIZE)

        ## s.t.
        m.addConstr(gp.quicksum(v[i] * x[i, dmu.index(r)] / np.max(x[i]) for i in range(I)) - gp.quicksum(u[j] * y[j, dmu.index(r)] / np.max(y[j]) for j in range(J)) + u0 == 0)
        for k in dmu:
            m.addConstr(gp.quicksum(v[i] * x[i, dmu.index(k)] / np.max(x[i]) for i in range(I)) - gp.quicksum(u[j] * y[j, dmu.index(k)] / np.max(y[j]) for j in range(J)) + u0 >= 0)
        m.addConstr(gp.quicksum(u[j] * gy[j] for j in range(J)) == 1)
            
        m.optimize()
        
        alpha[r] = m.objVal
    
    return alpha
#%%
def cal_dmp(dmu:list, alpha:dict, y:np.ndarray, gy:np.ndarray, dmu_wanted:list):
    ## dmu_wanted: the dmu we want to investigate
    J = y.shape[0]
    dmp = {}
    for r in dmu_wanted:
        dmp[r] = [None] * J
        for j in range(J):
            dmp[r][j] = alpha[r] * gy[j] * np.max(y[j])
    return dmp
#%%
def cal_smrts(dmu:list, dmp1:dict, dmp2:dict, round_to:int, dmu_wanted:list):
    """return s-MRTS dictionary between two directions (dmp1, dmp2)

    Args:
        dmu (list): [description]
        dmp1 (dict): [description]
        dmp2 (dict): [description]

    Returns:
        dict: [description]
    """
    smrts = {}
    for r in dmu_wanted:
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
    ## 0314 exp
    # [.55, .45], 
    # [.45, .55],  
    [1, 0], 
    [.9, .1], 
    [.8, .2], 
    [.7, .3], 
    [.6, .4], 
    [.5, .5], 
    [.4, .6], 
    [.3, .7], 
    [.2, .8], 
    [.1, .9], 
    [0, 1], 
]
#%%
def get_smrts_dfs(dmu, x, y, trace=False, round_to:int=2, dmu_wanted:list=None, i_star:int=0):
    ## dmu_wanted: the dmu we want to investigate
    if dmu_wanted is None:
        dmu_wanted = dmu

    results = []
    # alpha_directions = []
    dmp_directions = []
    # smrts_directions = []
    for i in range(len(DIRECTIONS)):
        direction = DIRECTIONS[i]
        alpha = cal_alpha(dmu=dmu, x=x, y=y, gy=direction, dmu_wanted=dmu_wanted, i_star=i_star)
        dmp = cal_dmp(dmu=dmu, alpha=alpha, y=y, gy=direction, dmu_wanted=dmu_wanted)
        # alpha_directions.append(alpha)
        dmp_directions.append(dmp)
        if not i:
            smrts = {r:None for r in dmu_wanted}
        else:
            smrts = cal_smrts(dmu=dmu, dmp1=dmp_directions[i-1], dmp2=dmp_directions[i], round_to=round_to, dmu_wanted=dmu_wanted)
        # smrts_directions.append(smrts)
        if trace:
            print(direction)
            for r in dmu_wanted:
                print(r, alpha[r], dmp[r], smrts[r])
            print()
        for r in dmu_wanted:
            results.append(Dmu_Direction(dmu=r, direction=direction, alpha=alpha[r], dmp=dmp[r], smrts=smrts[r], round_to=round_to))

    col_names = ["direction", "alpha", "DMP", "s-MRTS"]
    dfs = {}
    for r in dmu_wanted:
        df = pd.DataFrame(columns=col_names)
        for res in results:
            if res.dmu == r:
                df = df.append(pd.DataFrame([[res.direction, res.alpha, res.dmp, res.smrts]], columns=col_names))
                # print(res.direction, res.alpha, res.dmp)
        dfs[r] = df.set_index("direction")
    # dfs["A"]
    return dfs
#%%
## unit test
if __name__ == "__main__":
    dmu = ["A", "B", "C"]
    x = np.array([[2, 4, 1]])
    y = np.array([
        [1, 2, 4], 
        [200, 300, 100], 
        ])
    dfs = get_smrts_dfs(dmu, x, y, trace=False, round_to=5)
    # print(dfs["A"])
    # print(2)
#%%
