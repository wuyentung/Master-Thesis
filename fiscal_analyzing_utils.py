import os
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import dmp
import solver
import solver_r
from load_data import LIFE181920, FISCAL_LIFE2019, denoise_nonpositive, FISCAL_ATTRIBUTES, FISCAL_LIFE2018, FISCAL_LIFE2020
from exp_fiscal_data import OPERATION_SMRTS_DUMMY141516, INSURANCE_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516
from itertools import combinations
import matplotlib.pyplot as plt
from textwrap import wrap
CMAP = plt.get_cmap('plasma')
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
        

def _find_max_dir_mp(smrts_df:pd.DataFrame):
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

def _float_direction(str_direction:str):
    for direction in dmp.DIRECTIONS:
        if str(direction) == str_direction:
            return direction
    return [0, 0]
#%%
def _cal_cosine_similarity(vec_a, vec_b):
    # Dot and norm
    dot = sum(a*b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a*a for a in vec_a) ** 0.5
    norm_b = sum(b*b for b in vec_b) ** 0.5

    # Cosine similarity
    if norm_b==0 or norm_a==0:
        return 0
    cos_sim = dot / (norm_a*norm_b)
    return cos_sim
## 成功計算出 s-MRTS 後視覺化資料
def plot_3D(dmu:list, stitle:str, df:pd.DataFrame, smrts_dict:dict, target_input="insurance_exp", view_v=45, view_h=-80, dummy_dmu:list=None):
    
    if "insurance_exp" != target_input and "operation_exp" != target_input:
        raise ValueError("target_input should be 'insurance_exp' or 'operation_exp'.")
    if dummy_dmu is None:
        dummy_dmu = []
    all_dmu = dmu+dummy_dmu
    df = df.T[all_dmu].T
    
    label_size = 20
    title_size = 20
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 10))
    lines = []
    x_range = df["underwriting_profit"].max() - df["underwriting_profit"].min()
    y_range = df["investment_profit"].max() - df["investment_profit"].min()
    min_range = np.min([x_range, y_range])
    # ax.stem(data.y1, data.y2, data.x1) // can be implemented as follows
    for i in range(len(all_dmu)):
        color = CMAP(i/len(all_dmu))
        
        x_start = df["underwriting_profit"][all_dmu[i]]
        y_start = df["investment_profit"][all_dmu[i]]
        z_start = df[target_input][all_dmu[i]]
        z_min =  min(df[target_input])
        
        ax.plot3D([x_start, x_start], [y_start, y_start], [z_start, z_min], color=color, zorder=1, linestyle="--", alpha=.9)
        ax.scatter(x_start, y_start, z_start, marker="o", s=30, color=color, zorder=2)
        ax.text(x_start, y_start, z_start, '%s' % (all_dmu[i]), size=15, zorder=10, color="black", horizontalalignment='center', verticalalignment='bottom',)
        
        ## max direction of MP
        if all_dmu[i] in smrts_dict:
            smrts_df = smrts_dict[all_dmu[i]]
            max_dir_mp_str = _find_max_dir_mp(smrts_df)
            # print(max_dir_mp_str)
            smrts_color = "red"
        else:
            max_dir_mp_str = "[0.5, 0.5]"
            smrts_color = "orangered"
            
        max_dir_mp = _float_direction(max_dir_mp_str)
        # print(max_dir_mp)
            
        a = Arrow3D([x_start, x_start+max_dir_mp[0]*min_range/3], [y_start, y_start+max_dir_mp[1]*min_range/3], [ z_min,  z_min], mutation_scale=20, lw=2, arrowstyle="->", color=smrts_color)
        ax.add_artist(a)
        ax.text((x_start+max_dir_mp[0]*min_range/3+x_start)/2, (y_start+max_dir_mp[1]*min_range/3+y_start)/2,  z_min, '%s' % (max_dir_mp_str), size=15, zorder=10, color=smrts_color, horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle='round4', facecolor='white', alpha=0.3))
        
        ## 前進方向紀錄
        if len(dmu)-1 <= i:
            continue
        x_end = df["underwriting_profit"][dmu[i+1]]
        y_end = df["investment_profit"][dmu[i+1]]
        z_end = df[target_input][dmu[i+1]]
        
        a = Arrow3D([x_start, x_end], [y_start, y_end], [ z_min,  z_min], mutation_scale=20, lw=2, arrowstyle="->", color=color, alpha=.7)
        ax.add_artist(a)
        
        ax.text((x_end+x_start)/2, (y_end+y_start)/2,  z_min, "%.2f : %.2f" %(((x_end-x_start)/2)/np.abs(np.abs((x_end-x_start)/2) + np.abs((y_end-y_start)/2)), ((y_end-y_start)/2)/np.abs(np.abs((x_end-x_start)/2) + np.abs((y_end-y_start)/2))), horizontalalignment='left', verticalalignment='center', size=15, color=color)
            
    plt.legend(handles=lines, loc='lower left', ncol=2)
    
    ax.view_init(view_v, view_h)
    ax.set_xlabel(df.columns.to_list()[-2], fontsize=label_size)
    ax.set_ylabel(df.columns.to_list()[-1], fontsize=label_size)
    ax.set_zlabel(target_input, fontsize=label_size)
    ax.set_title("\n".join(wrap(stitle, 50)), fontsize=title_size)
    plt.tight_layout()
#%%
def round_analyze_df(analyze_df:pd.DataFrame, round_to:int=2):
    for col in analyze_df.columns:
        try:
            analyze_df[col] = np.round(analyze_df[col], round_to)
        except:
            print(col)
            for idx in analyze_df.index:
                analyze_df.at[idx, col] = np.round(analyze_df.at[idx, col], round_to)
            # "operation_exp cosine similarity": np.round(operation_cos_sims, round_to).tolist(), 
    return analyze_df
#%%
def get_analyze_df(dmu_ks:list, df:pd.DataFrame,):
    
    insurance_exps = df["insurance_exp"][dmu_ks]
    operation_exps = df["operation_exp"][dmu_ks]
    underwriting_profits = df["underwriting_profit"][dmu_ks]
    investment_profits = df["investment_profit"][dmu_ks]
    
    def _out_dir(start_idx, end_idx):
        return [((underwriting_profits[end_idx]-underwriting_profits[start_idx])/2)/np.abs(np.abs((underwriting_profits[end_idx]-underwriting_profits[start_idx])/2) + np.abs((investment_profits[end_idx]-investment_profits[start_idx])/2)), ((investment_profits[end_idx]-investment_profits[start_idx])/2)/np.abs(np.abs((underwriting_profits[end_idx]-underwriting_profits[start_idx])/2) + np.abs((investment_profits[end_idx]-investment_profits[start_idx])/2))]
    out_dirs = [_out_dir(i, i+1) for i in range(len(dmu_ks)-1)]
    out_dirs.append([np.nan, np.nan])
    ## insurance_exp max direction of MP
    ## investment_profit max direction of MP
    insurance_max_dirs = []
    insurance_cos_sims = []
    operation_max_dirs = []
    operation_cos_sims = []
    for target_input in ["insurance_exp", "operation_exp"]:
        if "insurance_exp" == target_input:
            smrts_dict = INSURANCE_SMRTS_DUMMY141516
            max_dirs = insurance_max_dirs
            cos_sims = insurance_cos_sims
        else:
            smrts_dict = OPERATION_SMRTS_DUMMY141516
            max_dirs = operation_max_dirs
            cos_sims = operation_cos_sims
        
        for n in range(len(dmu_ks)):
            ## max direction of MP
            if dmu_ks[n] in smrts_dict:
                smrts_df = smrts_dict[dmu_ks[n]]
                max_dir_mp_str = _find_max_dir_mp(smrts_df)
            else:
                max_dir_mp_str = "[0.5, 0.5]"
            max_dirs.append(_float_direction(max_dir_mp_str))
            # max_dir_mp = float_direction(max_dir_mp_str)
            cos_sims.append(_cal_cosine_similarity(out_dirs[n], max_dirs[n]))
    ## overall cosine similarity
    overall_cos_sims = [insurance_cos_sims[i] if insurance_cos_sims[i] else operation_cos_sims[i] for i in range(len(dmu_ks))]
    
    ## effiency and eff_change
    effiencies = [EFF_DICT_DUMMY141516[k] for k in dmu_ks]
    eff_changes = [(effiencies[i+1] - effiencies[i])/effiencies[i] for i in range(len(dmu_ks)-1)]
    eff_changes.append(np.nan)
    
    dmu_df = pd.DataFrame(
        {
            "insurance_exp": insurance_exps, 
            "operation_exp": operation_exps, 
            "underwriting_profit": underwriting_profits, 
            "investment_profit": investment_profits, 
            "output progress direction": out_dirs, 
            "insurance_exp max direction of MP": insurance_max_dirs, 
            "insurance_exp cosine similarity": insurance_cos_sims, 
            "operation_exp max direction of MP": operation_max_dirs, 
            "operation_exp cosine similarity": operation_cos_sims, 
            "overall cosine similarity": overall_cos_sims, 
            "efficiency": effiencies, 
            "efficiency changing": eff_changes, 
        }, index=dmu_ks
        )
    return dmu_df