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
from load_data import LIFE, FISAL_LIFE2019, denoise_nonpositive, FISAL_ATTRIBUTES, FISAL_LIFE2018, FISAL_LIFE2020
from exp_fiscal_data import OPERATION_SMRTS, INSURANCE_SMRTS
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('plasma')
#%%
eff_dmu = ['Hontai Life 18',
 'Chunghwa Post 18',
 'First-Aviva Life 18',
 'TransGlobe Life 18',
 'Hontai Life 19',
 'First-Aviva Life 19',
 'Bank Taiwan Life 20',
 'Taiwan Life 20',
 'Cathay Life 20',
 'China Life 20',
 'Nan Shan Life 20',
 'Shin Kong Life 20',
 'Fubon Life 20',
 'Hontai Life 20',
 'Chunghwa Post 20',
 'First-Aviva Life 20']
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
        dmp_dis = np.square(dmp[0]**2 + dmp[1]**2)
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
    # ax.stem(data.y1, data.y2, data.x1) // can be implemented as follows
    lines = []
    x_range = df["underwriting_profit"].max() - df["underwriting_profit"].min()
    y_range = df["investment_profit"].max() - df["investment_profit"].min()
    min_range = np.min([x_range, y_range])
    for k in (dmu):
        color = CMAP(dmu.index(k)/len(dmu))
        # color = "blue"
        
        x = df["underwriting_profit"][k]
        y = df["investment_profit"][k]
        z = df[target_input][k]
        ax.plot3D([x, x], [y, y], [z, min(df[target_input][dmu])], color=color, zorder=1, linestyle="--")
        ax.scatter(x, y, z, marker="o", s=30, color=color, zorder=2)
        ax.text(x, y, z, '%s' % (k), size=15, zorder=10, color="black", horizontalalignment='center', verticalalignment='top',)
        ## s-MRTS plot
        if k in smrts_dict:
            smrts_df = smrts_dict[k]
            max_dir_mp_str = find_max_dir_mp(smrts_df)
            # print(max_dir_mp_str)
            max_dir_mp = float_direction(max_dir_mp_str)
            # print(max_dir_mp)
            
            # line, = ax.plot3D(np.array([smrts_df["DMP"][i][0] for i in range(11)]) + x, np.array([smrts_df["DMP"][i][1] for i in range(11)]) + y, [z]*11, label='%s'%k, color=color)
            # lines.append(line)

            a = Arrow3D([x, x+max_dir_mp[0]*min_range/3], [y, y+max_dir_mp[1]*min_range/3], [z, z], mutation_scale=20, lw=2, arrowstyle="->", color="red")
            ax.add_artist(a)
            ax.text(x+max_dir_mp[0]*min_range/3, y+max_dir_mp[1]*min_range/3, z, '%s' % (max_dir_mp_str), size=15, zorder=10, color="black", horizontalalignment='left', verticalalignment='center',)
            
    plt.legend(handles=lines, loc='lower left', ncol=2)
    for i in range(len(dmu)-1):
        
        x_start = df["underwriting_profit"][dmu[i]]
        x_end = df["underwriting_profit"][dmu[i+1]]
        y_start = df["investment_profit"][dmu[i]]
        y_end = df["investment_profit"][dmu[i+1]]
        z_start = df[target_input][dmu[i]]
        z_end = df[target_input][dmu[i+1]]
        
        a = Arrow3D([x_start, x_end], [y_start, y_end], [z_start, z_end], mutation_scale=20, lw=2, arrowstyle="->", color="gray")
        ax.add_artist(a)
        
        ax.text(x_start, y_start, z_start, "%.2f : %.2f" %(((x_end-x_start)/2)/(((x_end-x_start)/2) + ((y_end-y_start)/2)), ((y_end-y_start)/2)/(((x_end-x_start)/2) + ((y_end-y_start)/2))), horizontalalignment='left', verticalalignment='center', size=15,)
        
    ax.view_init(60, -80)
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
