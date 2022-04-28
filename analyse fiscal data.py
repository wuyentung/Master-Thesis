#%%
'''
改用保險業實證 dmp
    三年資料，假設沒有 tech. change
    先算整體的，有能力再用網路
最後有時間再來 scope property
'''
#%%
import os
import pandas as pd
import numpy as np
import dmp
import solver
import solver_r
from load_data import LIFE, FISCAL_LIFE2019, denoise_nonpositive, FISCAL_ATTRIBUTES, FISCAL_LIFE2018, FISCAL_LIFE2020
from exp_fiscal_data import OPERATION_SMRTS, INSURANCE_SMRTS
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('plasma')
#%%
eff_dmu = ['Hontai Life 18', 'Chunghwa Post 18', 'First-Aviva Life 18', 'Hontai Life 19', 'First-Aviva Life 19', 'Bank Taiwan Life 20', 'Taiwan Life 20', 'Cathay Life 20', 'China Life 20', 'Nan Shan Life 20', 'Shin Kong Life 20', 'Fubon Life 20', 'Hontai Life 20', 'Chunghwa Post 20', 'First-Aviva Life 20', 'BNP Paribas Cardif TCB 20', 'CIGNA 20', 'Cardif 20']
#%%
df = denoise_nonpositive(LIFE)/1000/1000
#%%
df.T[['Hontai Life 18', 'Hontai Life 19', 'Hontai Life 20']].T
#%%
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
#%%
def find_max_dir_mp(smrts_df:pd.DataFrame):
    max_dmp_dis = 0
    max_dir_mp = "[0, 0]"
    for idx, row in smrts_df.iterrows():
        dmp = row["DMP"]
        ## 相加後會是總獲利
        dmp_dis = dmp[0] + dmp[1]
        # dmp_dis = np.square(dmp[0]**2 + dmp[1]**2)
        # print(mdp_dis)
        if dmp_dis > max_dmp_dis:
            max_dmp_dis = dmp_dis
            max_dir_mp = idx
    return max_dir_mp
# find_max_dir_mp(OPERATION_SMRTS["Hontai Life 18"])
#%%
def float_direction(str_direction:str):
    for direction in dmp.DIRECTIONS:
        if str(direction) == str_direction:
            return direction
    return [0, 0]
#%%
## 成功計算出 s-MRTS 後視覺化資料
def plot_3D(dmu:list, stitle:str, target_input="insurance_exp", df:pd.DataFrame=df):
    
    df = df.T[dmu].T
    
    label_size = 20
    title_size = 20
    if "insurance_exp" == target_input:
        smrts_dict = INSURANCE_SMRTS
    else:
        smrts_dict = OPERATION_SMRTS
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 10))
    lines = []
    x_range = df["underwriting_profit"].max() - df["underwriting_profit"].min()
    y_range = df["investment_profit"].max() - df["investment_profit"].min()
    min_range = np.min([x_range, y_range])
    # ax.stem(data.y1, data.y2, data.x1) // can be implemented as follows
    for i in range(len(dmu)):
        color = CMAP(i/len(dmu))
        
        x_start = df["underwriting_profit"][dmu[i]]
        y_start = df["investment_profit"][dmu[i]]
        z_start = df[target_input][dmu[i]]
        z_min =  min(df[target_input][dmu])
        
        ax.plot3D([x_start, x_start], [y_start, y_start], [z_start, z_min], color=color, zorder=1, linestyle="--")
        ax.scatter(x_start, y_start, z_start, marker="o", s=30, color=color, zorder=2)
        ax.text(x_start, y_start, z_start, '%s' % (dmu[i]), size=15, zorder=10, color="black", horizontalalignment='center', verticalalignment='bottom',)
        
        ## max direction of MP
        if dmu[i] in smrts_dict:
            smrts_df = smrts_dict[dmu[i]]
            max_dir_mp_str = find_max_dir_mp(smrts_df)
            # print(max_dir_mp_str)
            max_dir_mp = float_direction(max_dir_mp_str)
            # print(max_dir_mp)

            a = Arrow3D([x_start, x_start+max_dir_mp[0]*min_range/3], [y_start, y_start+max_dir_mp[1]*min_range/3], [ z_min,  z_min], mutation_scale=20, lw=2, arrowstyle="->", color="red")
            ax.add_artist(a)
            ax.text(x_start+max_dir_mp[0]*min_range/3, y_start+max_dir_mp[1]*min_range/3,  z_min, '%s' % (max_dir_mp_str), size=15, zorder=10, color="red", horizontalalignment='left', verticalalignment='top',)
        
        ## 前進方向紀錄
        if len(dmu)-1 == i:
            continue
        x_end = df["underwriting_profit"][dmu[i+1]]
        y_end = df["investment_profit"][dmu[i+1]]
        z_end = df[target_input][dmu[i+1]]
        
        a = Arrow3D([x_start, x_end], [y_start, y_end], [ z_min,  z_min], mutation_scale=20, lw=2, arrowstyle="->", color="gray")
        ax.add_artist(a)
        
        ax.text(x_start, y_start,  z_min, "%.2f : %.2f" %(((x_end-x_start)/2)/(((x_end-x_start)/2) + ((y_end-y_start)/2)), ((y_end-y_start)/2)/(((x_end-x_start)/2) + ((y_end-y_start)/2))), horizontalalignment='left', verticalalignment='center', size=15,)
            
    plt.legend(handles=lines, loc='lower left', ncol=2)
    
    ax.view_init(45, -80)
    ax.set_xlabel(df.columns.to_list()[-2], fontsize=label_size)
    ax.set_ylabel(df.columns.to_list()[-1], fontsize=label_size)
    ax.set_zlabel(target_input, fontsize=label_size)
    ax.set_title("\n".join(wrap(stitle, 50)), fontsize=title_size)
    plt.tight_layout()
#%%
plot_3D(dmu=['Hontai Life 18', 'Hontai Life 19', 'Hontai Life 20'], stitle="Hontai Life", target_input="insurance_exp")
plt.draw()
plt.show()
#%%
plot_3D(dmu=['Hontai Life 18', 'Hontai Life 19', 'Hontai Life 20'], stitle="Hontai Life", target_input="operation_exp", df=df.T[['Hontai Life 18', 'Hontai Life 19', 'Hontai Life 20']].T)
plt.draw()
plt.show()
#%%
dmus = ['Bank Taiwan Life ', 'Taiwan Life ', 'PCA Life ', 'Cathay Life ', 'China Life ', 'Nan Shan Life ', 'Shin Kong Life ', 'Fubon Life ', 'Mercuries Life ', 'Farglory Life ', 'Hontai Life ', 'Allianz Taiwan Life ', 'Chunghwa Post ', 'First-Aviva Life ', 'BNP Paribas Cardif TCB ', 'Prudential of Taiwan ', 'CIGNA ', 'Yuanta Life ', 'TransGlobe Life ', 'AIA Taiwan ', 'Cardif ', 'Chubb Tempest Life ']
#%%
## visualize_progress
for k in dmus:
    for target_input in ["insurance_exp", "operation_exp"]:
        plot_3D(dmu=[k+n for n in ['18', '19', '20']], stitle=k, target_input=target_input, df=denoise_nonpositive(LIFE)/1000/1000)
        plt.draw()
        plt.savefig("%s %s.png" %(k, target_input), dpi=400)
        plt.show()
#%%
df.plot.scatter("insurance_exp", "operation_exp")
#%%
df18 = denoise_nonpositive(FISCAL_LIFE2018)/1000/1000
df18.plot.scatter("insurance_exp", "operation_exp")
#%%
plt.figure(figsize=(16, 9))
plt.scatter(df18["insurance_exp"], df18["operation_exp"])
for k, row in df18.iterrows():
    plt.annotate(k, (row["insurance_exp"], row["operation_exp"]), fontsize=10)
plt.xlabel("insurance_exp")
plt.ylabel("operation_exp")
plt.show()
#%%
plt.figure(figsize=(20, 15))
plt.scatter(df["insurance_exp"], df["operation_exp"])
for k, row in df.iterrows():
    plt.annotate(k, (row["insurance_exp"], row["operation_exp"]), fontsize=10)
plt.xlabel("insurance_exp")
plt.ylabel("operation_exp")
plt.show()
#%%
df20 = denoise_nonpositive(FISCAL_LIFE2020)/1000/1000
#%%
