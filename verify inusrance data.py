#%%
import os
import dmp
import pandas as pd
import numpy as np
import solver
import solver_r
from load_data import denoise_nonpositive
from itertools import combinations
import matplotlib.pyplot as plt
import pickle5 as p
#%%
def sys_smrts(df:pd.DataFrame, i_star=0, xcol:list=None, ycol:list=None):
    if xcol is None:
        xcol = df.columns.tolist()[i_star]
    if ycol is None:
        ycol = df.columns.tolist()[-2:]
    ## transform data
    ## s-MRTS for  whole system
    transformed_df = denoise_nonpositive(df)/1000/1000
    # print(np.array([transformed_df[xcol].T]))
    eff_dict, lambdas_dict = solver.dea_dual(dmu=transformed_df.index, x=np.array([transformed_df[xcol].T]), y=np.array(transformed_df[ycol].T), orient="OO")

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
# verify_df = pd.read_csv("./verify data/24 non-life 2003.csv", index_col=0)
verify_df = pd.read_csv("./verify data/24 non-life 2003.csv", index_col=0).dropna().astype('float')
#%%
exp001 = sys_smrts(verify_df, i_star=0)
#%%
for i in range(2):
    dmp.show_smrts(sys_smrts(verify_df, i_star=i), path="2003 non-life million %s.txt" %i)
#%%
## 老師覺得拿掉富邦會讓整個 frontier 變比較平緩
exp010 = sys_smrts(verify_df.drop(["Fubon"]), i_star=0)
#%%
## 結果也是一樣計算不出 alpha
## 列印結果
# path = '24 non-life 2003 s-MRTS.txt'
# f = open(path, 'w')
# for key, value in exp001.items():
#     print(key, file=f)
#     print(value, file=f)
#     print("\n", file=f)
# f.close()
#%%
path = '24 non-life 2003 s-MRTS without Fubon.txt'
f = open(path, 'w')
for key, value in exp010.items():
    print(key, file=f)
    print(value, file=f)
    print("\n", file=f)
f.close()

#%%

## C eff 取 3
c3 = []
for c in combinations(verify_df.index.tolist(), 3):
    c3.append(sys_smrts(verify_df.T[list(c)].T, i_star=0))
#%%
has_alpha = []
for i in range(len(c3)):
    for key, value in c3[i].items():
        # print(sum(value["alpha"]))
        if sum(value["alpha"]):
            print(i)
            print(key, file=None)
            print(value, file=None)
            print("\n", file=None)
            has_alpha.append(i)
            break
    if i // 20:
        print(i)

#%%
import pickle
filename = "verify_insurance.pickle"
with open(filename, 'wb') as handle:
    pickle.dump(c3, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(filename, 'rb') as handle:
    b = pickle.load(handle)
#%%
filename = "verify_insurance.pickle"
with open(filename, 'rb') as handle:
    c3 = p.load(handle)
#%%
has_alpha = []
for i in range(len(c3)):
    for key, value in c3[i].items():
        # print(sum(value["alpha"]))
        # if sum(value["alpha"]):
        if round(sum(value["alpha"]), 4):
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
def plot_3D(dmu:list, stitle:str, i_star=0, df:pd.DataFrame=verify_df):
    label_size = 20
    title_size = 30
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 10))
    # ax.stem(data.y1, data.y2, data.x1) // can be implemented as follows
    # lines = []
    for k in (dmu):
        color = "blue"
        
        x = df[df.columns.to_list()[-2]][k]
        y = df[df.columns.to_list()[-1]][k]
        z = df[df.columns.to_list()[i_star]][k]
        ax.plot3D([x, x], [y, y], [z, min(df[df.columns.to_list()[i_star]][dmu])], color=color, zorder=1, linestyle="--")
        ax.scatter(x, y, z, marker="o", s=30, color=color, zorder=2)
        ax.text(x, y, z, '%s' % (k), size=20, zorder=10, color="black", horizontalalignment='center', verticalalignment='top',)
        
    # plt.legend(handles=lines, loc='lower right')
    ax.view_init(30, -80)
    ax.set_xlabel(df.columns.to_list()[-2], fontsize=label_size)
    ax.set_ylabel(df.columns.to_list()[-1], fontsize=label_size)
    ax.set_zlabel(df.columns.to_list()[i_star], fontsize=label_size)
    ax.set_title(stitle, fontsize=title_size)
    plt.tight_layout()

# plot_3D(list(c3[594].keys()))
#%%
def good_result(smrts_dict:dict, i_star=0, save=False):
    
    stitle = "2003 nonlife insurance "
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
# good_result(c3[594], save=False)
#%%
good_result(c3[594], save=True)
#%%
good_result(c3[2011], save=True)
#%%
good_result(c3[2017], save=True)
#%%
good_result(c3[2022], save=True)
#%%
