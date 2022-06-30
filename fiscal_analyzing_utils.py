import os
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import dmp
import solver
import solver_r
import constant as const
from load_data import denoise_nonpositive, FISCAL_ATTRIBUTES
from smrts_fiscal_data import EXPANSION_OPERATION_SMRTS_DUMMY141516, EXPANSION_INSURANCE_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516, INSURANCE_SMRTS181920, OPERATION_SMRTS181920, EFF_DICT181920, LAMBDA_DICT181920
from itertools import combinations
import matplotlib.pyplot as plt
CMAP = plt.get_cmap('jet')
from textwrap import wrap
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
sns.set_theme(style="darkgrid")

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
def year_determin(year:int):
    if year in [14, 15, 16]:
        if 16 == year:
            print("this could be default value using 2014-2016 data")
        return EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516, EXPANSION_INSURANCE_SMRTS_DUMMY141516, EXPANSION_OPERATION_SMRTS_DUMMY141516, const.LAST_Y_14
    return EFF_DICT181920, LAMBDA_DICT181920, INSURANCE_SMRTS181920, OPERATION_SMRTS181920, const.LAST_Y_18
#%%
def _find_max_dir_mp(smrts_df:pd.DataFrame, DMP_contraction:bool):
    max_dmp_dis = -np.inf
    max_dir_mp = "[0, 0]"
    for idx, row in smrts_df.iterrows():
        dmp = row["DMP"]
        ## 相加後會是總獲利
        dmp_dis = dmp[0] + dmp[1]
        ## contraction 反而要挑最小，亦即負最大
        if DMP_contraction:
            dmp_dis*=-1
        # dmp_dis = np.square(dmp[0]**2 + dmp[1]**2)
        # print(mdp_dis)
        if dmp_dis and dmp_dis > max_dmp_dis:
            max_dmp_dis = dmp_dis
            max_dir_mp = idx
    return max_dir_mp

def _float_direction(str_direction:str):
    for direction in dmp.DIRECTIONS:
        if str(direction) == str_direction:
            return direction
    for direction in dmp.NEG_DIRECTIONS:
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
#%%
def _find_ref_dmu(lamda_df:pd.DataFrame, DMP_contraction:str, ):
    if DMP_contraction:
        bad_ks = ["Zurich 16"]
    else:
        bad_ks = ["Zurich 16", "Cardif 16"]
        
    lamda_df_copy = lamda_df.sort_values(by=const.LAMBDA, ascending=False)
    
    if lamda_df_copy[const.LAMBDA].max() > .99:
        return lamda_df_copy[const.LAMBDA].idxmax()
    
    for dmu_k in lamda_df_copy.index:
        if dmu_k in bad_ks:
            continue
        return dmu_k
## 成功計算出 s-MRTS 後視覺化資料
def plot_3D(dmu:list, stitle:str, df:pd.DataFrame, smrts_dict:dict, target_input=const.INSURANCE_EXP, view_v=45, view_h=-80, dummy_dmu:list=None, DMP_contraction:bool=False, year:int=16):
    eff_dict, lambda_dict, insurance_smrts, operation_smrts, last_Y = year_determin(year)
    
    if const.INSURANCE_EXP != target_input and const.OPERATION_EXP != target_input:
        raise ValueError("target_input should be const.INSURANCE_EXP or const.OPERATION_EXP.")
    if dummy_dmu is None:
        dummy_dmu = []
    all_dmu = dmu+dummy_dmu
    df = df.T[all_dmu].T
    
    label_size = 20
    title_size = 20
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 10))
    lines = []
    x_range = df[const.UNDERWRITING_PROFIT].max() - df[const.UNDERWRITING_PROFIT].min()
    y_range = df[const.INVESTMENT_PROFIT].max() - df[const.INVESTMENT_PROFIT].min()
    min_range = np.min([x_range, y_range])
    # ax.stem(data.y1, data.y2, data.x1) // can be implemented as follows
    for i in range(len(all_dmu)):
        color = CMAP(i/len(all_dmu))
        
        x_start = df[const.UNDERWRITING_PROFIT][all_dmu[i]]
        y_start = df[const.INVESTMENT_PROFIT][all_dmu[i]]
        z_start = df[target_input][all_dmu[i]]
        z_min =  min(df[target_input])
        
        ax.plot3D([x_start, x_start], [y_start, y_start], [z_start, z_min], color=color, zorder=1, linestyle="--", alpha=.9)
        ax.scatter(x_start, y_start, z_start, marker="o", s=30, color=color, zorder=2)
        ax.text(x_start, y_start, z_start, '%s' % (all_dmu[i]), size=15, zorder=10, color="black", horizontalalignment='center', verticalalignment='bottom',)
        
        ## max direction of MP
        if all_dmu[i] in smrts_dict:
            smrts_color = "red"
        else:
            # max_dir_mp = [0.5, 0.5]
            smrts_color = "orangered"
        reference_dmu = lambda_dict[all_dmu[i]][const.LAMBDA].idxmax()
        
        reference_dmu = _find_ref_dmu(lamda_df=lambda_dict[all_dmu[i]], DMP_contraction=DMP_contraction)
        # print(reference_dmu, LAMBDA_DICT_DUMMY141516[all_dmu[i]][const.LAMBDA])
        smrts_df = smrts_dict[reference_dmu]
        max_dir_mp = _float_direction(_find_max_dir_mp(smrts_df, DMP_contraction))
            
        # print(max_dir_mp)
            
        a = Arrow3D([x_start, x_start+max_dir_mp[0]*min_range/3], [y_start, y_start+max_dir_mp[1]*min_range/3], [ z_min,  z_min], mutation_scale=20, lw=2, arrowstyle="->", color=smrts_color)
        ax.add_artist(a)
        ax.text((x_start+max_dir_mp[0]*min_range/3+x_start)/2, (y_start+max_dir_mp[1]*min_range/3+y_start)/2,  z_min, '%s' % (str(np.round(max_dir_mp, 2))), size=15, zorder=10, color=smrts_color, horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle='round4', facecolor='white', alpha=0.3))
        
        ## 前進方向紀錄
        if len(dmu)-1 <= i:
            continue
        x_end = df[const.UNDERWRITING_PROFIT][dmu[i+1]]
        y_end = df[const.INVESTMENT_PROFIT][dmu[i+1]]
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
        if isinstance(analyze_df[col].iloc[0], str):
            continue
        try:
            analyze_df[col] = np.round(analyze_df[col], round_to)
        except:
            print(col)
            for idx in analyze_df.index:
                analyze_df.at[idx, col] = np.round(analyze_df.at[idx, col], round_to)
            # const.OPERATION_COS_SIM: np.round(operation_cos_sims, round_to).tolist(), 
    return analyze_df
#%%
def get_analyze_df(dmu_ks:list, df:pd.DataFrame, year:int=16):
    
    eff_dict, lambda_dict, insurance_smrts, operation_smrts, last_Y = year_determin(year)
    
    insurance_exps = df[const.INSURANCE_EXP][dmu_ks]
    operation_exps = df[const.OPERATION_EXP][dmu_ks]
    underwriting_profits = df[const.UNDERWRITING_PROFIT][dmu_ks]
    investment_profits = df[const.INVESTMENT_PROFIT][dmu_ks]
    
    def _out_dir(start_idx, end_idx):
        return [((underwriting_profits[end_idx]-underwriting_profits[start_idx])/2)/np.abs(np.abs((underwriting_profits[end_idx]-underwriting_profits[start_idx])/2) + np.abs((investment_profits[end_idx]-investment_profits[start_idx])/2)), ((investment_profits[end_idx]-investment_profits[start_idx])/2)/np.abs(np.abs((underwriting_profits[end_idx]-underwriting_profits[start_idx])/2) + np.abs((investment_profits[end_idx]-investment_profits[start_idx])/2))]
    
    out_dirs = [_out_dir(i, i+1) if dmu_ks[i] not in last_Y else [np.nan, np.nan] for i in range(len(dmu_ks)-1)]
    out_dirs.append([np.nan, np.nan])
    
    # reference_dmus = [_find_ref_dmu(lamda_df=LAMBDA_DICT_DUMMY141516[k], DMP_contraction=True) for k in dmu_ks]
    # reference_lambdas = [LAMBDA_DICT_DUMMY141516[dmu_ks[i]].loc[reference_dmus[i]][const.LAMBDA] for i in range(len(dmu_ks))]
    
    def _cal_cos_sim(smrts_dict, DMP_contraction):
        max_dirs = []
        cos_sims = []
        for i in range(len(dmu_ks)):
            ## max direction of MP
            smrts_df = smrts_dict[dmu_ks[i]]
            max_dir_mp_str = _float_direction(_find_max_dir_mp(smrts_df, DMP_contraction))
            max_dirs.append(max_dir_mp_str)
            if dmu_ks[i] in last_Y:
                cos_sims.append(np.nan)
            else:
                cos_sims.append(_cal_cosine_similarity(out_dirs[i], max_dirs[i]))
        return max_dirs, cos_sims
    
    expansion_insurance_max_dirs, expansion_insurance_cos_sims = _cal_cos_sim(smrts_dict=insurance_smrts, DMP_contraction=False)
    expansion_operation_max_dirs, expansion_operation_cos_sims = _cal_cos_sim(smrts_dict=operation_smrts, DMP_contraction=False)
    
    ## marginal consistency
    expansion_consistencies = [expansion_insurance_cos_sims[i] if expansion_insurance_cos_sims[i] else expansion_operation_cos_sims[i] for i in range(len(dmu_ks))]
    
    ## effiency and eff_change
    effiencies = [eff_dict[k] for k in dmu_ks]
    eff_changes = [effiencies[i]/effiencies[i+1] if dmu_ks[i] not in last_Y else np.nan for i in range(len(dmu_ks)-1)]
    eff_changes.append(np.nan)
    
    dmu_df = pd.DataFrame(
        {
            const.INSURANCE_EXP: insurance_exps, 
            const.OPERATION_EXP: operation_exps, 
            const.UNDERWRITING_PROFIT: underwriting_profits, 
            const.INVESTMENT_PROFIT: investment_profits, 
            
            const.SCALE: insurance_exps + operation_exps, 
            const.PROFIT: underwriting_profits + investment_profits, 
            const.OUT_DIR: out_dirs, 
            
            const.EXPANSION_INSURANCE_MAXDMP: expansion_insurance_max_dirs, 
            # const.EXPANSION_INSURANCE_COS_SIM: expansion_insurance_cos_sims, 
            const.EXPANSION_OPERATION_MAXDMP: expansion_operation_max_dirs, 
            # const.EXPANSION_OPERATION_COS_SIM: expansion_operation_cos_sims, 
            const.EXPANSION_CONSISTENCY: expansion_consistencies,
            
            const.EFFICIENCY: effiencies, 
            const.EC: eff_changes, 
        }, index=dmu_ks
        )
    
    return dmu_df
#%%
def label_data(zip_x, zip_y, labels, xytext=(0, 5), ha='center', fontsize=5):
    # zip joins x and y coordinates in pairs
    c = 0
    for x,y in zip(zip_x, zip_y):
        label = f"{labels[c]}"

        plt.annotate(
            label, # this is the text
            (x,y), # these are the coordinates to position the label
            textcoords="offset points", # how to position the text
            xytext=xytext, # distance from text to points (x,y)
            ha=ha, # horizontal alignment can be left, right or center
            fontsize=fontsize, 
            ) 
        c+=1
    
#%%
def analyze_plot(ax:Axes, df:pd.DataFrame, x_col = const.EC, y_col = const.EXPANSION_CONSISTENCY, according_col=const.EFFICIENCY):
    ax.hlines(y=df[y_col].median(), xmin=df[x_col].min(), xmax=df[x_col].max(), colors="gray", lw=1)
    ax.vlines(x=1 if x_col == const.EC else df[x_col].median(), ymin=df[y_col].min(), ymax=df[y_col].max(), colors="gray", lw=1)
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, hue=according_col, palette=CMAP, )
    label_data(zip_x=df[x_col], zip_y=df[y_col], labels=df.index)