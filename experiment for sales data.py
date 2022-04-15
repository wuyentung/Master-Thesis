#%%
'''
改用保險業業務實證 dmp
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
from load_sales_data import LIFE, LIFE2019, denoise_nonpositive, ATTRIBUTES, LIFE2018, LIFE2020
from itertools import combinations
import matplotlib.pyplot as plt
import pickle
import pickle5 as p
#%%
def sys_smrts(df:pd.DataFrame, project=False, i_star=0, xcol:list=None, ycol:list=None):
    if xcol is None:
        xcol = df.columns.tolist()[i_star]
    if ycol is None:
        ycol = df.columns.tolist()[-2:]
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)
    
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array([transformed_df[xcol].T]), y=np.array(transformed_df[ycol].T))

    eff_dmu_name = []
    for key, value in eff_dict.items():
        if round(value, 5) == 1:
            eff_dmu_name.append(key)
    
    df = transformed_df.T[eff_dmu_name].T
    exp = dmp.get_smrts_dfs(dmu=[i for i in range(df.shape[0])], x=np.array([df[xcol].T]), y=np.array(df[ycol].T), trace=False, round_to=5, dmu_wanted=None, 
                            # i_star=i_star
                            )
    old_keys = list(exp.keys())
    for old_key in old_keys:
        exp[eff_dmu_name[old_key]] = exp.pop(old_key)
    return exp
#%%
## 多年度綜合跟單年度的有效率 DMU
def find_eff_dmu(df:pd.DataFrame):
    transformed_df = denoise_nonpositive(df, .1)
    # print(transformed_df)
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array(transformed_df[ATTRIBUTES[:2]].T), y=np.array(transformed_df[ATTRIBUTES[-2:]].T))

    eff_dmu_name = []
    for key, value in eff_dict.items():
        if round(value, 5) == 1:
            eff_dmu_name.append(key)
    return eff_dmu_name
#%%
def comb_fun(df:pd.DataFrame, fun, comb_n=None):
    results = []
    combs = []
    if comb_n is None:
        comb_n = len(df.index.tolist())
    for comb in combinations(df.index.tolist(), comb_n):
        # print(len(list(comb)))
        # n_comb_i+=1
        try:
            result = fun(df.T[list(comb)].T)
            results.append(result)
            combs.append(list(comb))
            # print(i, comb, "\n")
            # break
        except:
            continue
    if 0 == len(combs):
        return comb_fun(df, fun, comb_n-1)
    return results, combs
#%%
exp001 = sys_smrts(LIFE2018, i_star=1)
#%%
combs_smrts19, combs_comb19 = comb_fun(df=LIFE2019, fun=sys_smrts)
#%%
combs_smrts20, combs_comb20 = comb_fun(df=LIFE2020, fun=sys_smrts)
#%%
# combs_life_eff, combs_life_eff_comb = comb_fun(df=LIFE, fun=find_eff_dmu)
life_eff = ['Nan Shan Life 18', 'Mercuries Life 18', 'Farglory Life 18', 'First-Aviva Life 18', 'Prudential of Taiwan 18', 'Nan Shan Life 19', 'Farglory Life 19', 'First-Aviva Life 19', 'Cardif 19', 'Taiwan Life 20', 'Cathay Life 20', 'Nan Shan Life 20', 'Mercuries Life 20', 'Farglory Life 20', 'Hontai Life 20', 'First-Aviva Life 20']
#%%
# combs_smrts, combs_comb = comb_fun(df=LIFE.T[life_eff].T, fun=sys_smrts)
expLIFE = sys_smrts(df=LIFE.T[['Nan Shan Life 18', 'Mercuries Life 18', 'Farglory Life 18', 'First-Aviva Life 18', 'Prudential of Taiwan 18', 'Nan Shan Life 19', 'Farglory Life 19', 'Cardif 19', 'Taiwan Life 20', 'Cathay Life 20', 'Nan Shan Life 20', 'Mercuries Life 20', 'Farglory Life 20']].T, i_star=1)
#%%
## 取小於 30000000 的資料
# plt.scatter(LIFE[ATTRIBUTES[-1]], LIFE[ATTRIBUTES[-2]])
# combs_life_small_eff, combs_life_small_eff_comb = comb_fun(df=LIFE[LIFE["健康保險核保利潤"] < 30000000], fun=find_eff_dmu)
# combs_small_smrts, combs_small_comb = comb_fun(df=LIFE.T[['Mercuries Life 18', 'Farglory Life 18', 'First-Aviva Life 18', 'Prudential of Taiwan 18', 'Farglory Life 19', 'First-Aviva Life 19', 'BNP Paribas Cardif TCB 19', 'Cardif 19', 'Taiwan Life 20', 'Mercuries Life 20', 'Farglory Life 20', 'Hontai Life 20', 'First-Aviva Life 20']].T, fun=sys_smrts)
expLIFE_small = sys_smrts(LIFE.T[['Mercuries Life 18', 'Farglory Life 18', 'First-Aviva Life 18', 'Prudential of Taiwan 18', 'Farglory Life 19', 'BNP Paribas Cardif TCB 19', 'Cardif 19', 'Taiwan Life 20', 'Mercuries Life 20', 'Farglory Life 20']].T, i_star=1)
#%%
## C eff 取 3
c3 = []
for c in combinations(['Nan Shan Life 18', 'Mercuries Life 18', 'Farglory Life 18', 'First-Aviva Life 18', 'Prudential of Taiwan 18', 'Nan Shan Life 19', 'Farglory Life 19', 'Cardif 19', 'Taiwan Life 20', 'Cathay Life 20', 'Nan Shan Life 20', 'Mercuries Life 20', 'Farglory Life 20'], 3):
    c3.append(sys_smrts(LIFE.T[list(c)].T, i_star=0))
#%%
c3[0]
#%%
#%%
has_alpha = []
for i in range(len(c3)):
    for key, value in c3[i].items():
        # print(sum(value["alpha"]))
        if sum(value["alpha"]):
            print(i)
            print(key, file=None)
            print(value, file=None)
            has_alpha.append(i)
            break
    if i // 20:
        print(i)
        print("\n", file=None)
#%%
filename = "sales_data_smrts.pickle"
with open(filename, 'wb') as handle:
    pickle.dump(c3, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(filename, 'rb') as handle:
    b = pickle.load(handle)
#%%
filename = "sales_data_smrts.pickle"
with open(filename, 'rb') as handle:
    c3 = p.load(handle)
#%%
has_alpha = []
for i in range(len(c3)):
    for key, value in c3[i].items():
        # print(sum(value["alpha"]))
        if sum(value["alpha"]):
        # if round(sum(value["alpha"]), 5):
            print("\n", file=None)
            print(i)
            print(key, file=None)
            print(value, file=None)
            has_alpha.append(i) 
            print("\n", file=None)
            break
    if i % 30 == 0:
        print(i)
#%%

def plot_3D(dmu:list, stitle:str, i_star=0, df:pd.DataFrame=LIFE):
    label_size = 20
    title_size = 30
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 10))
    
    # ax.stem(data.y1, data.y2, data.x1) // can be implemented as follows
    # lines = []
    for k in (dmu):
        color = "blue"
        
        x = df[ATTRIBUTES[-2]][k]
        y = df[ATTRIBUTES[-1]][k]
        z = df[ATTRIBUTES[i_star]][k]
        ax.plot3D([x, x], [y, y], [z, min(df[ATTRIBUTES[i_star]][dmu])], color=color, zorder=1, linestyle="--")
        ax.scatter(x, y, z, marker="o", s=30, color=color, zorder=2)
        ax.text(x, y, z, '%s' % (k), size=20, zorder=10, color="black", horizontalalignment='center', verticalalignment='top',)
        
    # plt.legend(handles=lines, loc='lower right')
    ax.view_init(30, -80)
    ax.set_xlabel(ATTRIBUTES[-2], fontsize=label_size)
    ax.set_ylabel(ATTRIBUTES[-1], fontsize=label_size)
    ax.set_zlabel(ATTRIBUTES[i_star], fontsize=label_size)
    ax.set_title(stitle, fontsize=title_size)
    plt.tight_layout()

# plot_3D(list(c3[594].keys()))
#%%
def good_result(smrts_dict:dict, i_star=0, save=False):
    
    stitle = "2018-20 life insurance "
    dmu = list(smrts_dict.keys())
    plot_3D(dmu, stitle+"\n"+str(dmu), i_star)

    if save:
        stitle+=str(dmu)
        dmp.show_smrts(smrts_dict, path=stitle+".txt")
        plt.savefig(stitle+".png", dpi=400)
    else:
        dmp.show_smrts(smrts_dict)
    
    plt.show()
    
    return
#%%
good_result(smrts_dict=c3[28], save=True)
#%%
good_result(smrts_dict=c3[187], save=True)
#%%
good_result(smrts_dict=c3[190], save=True)
#%%
