from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
CMAP = plt.get_cmap('jet')
sns.set_theme(style="darkgrid")
import constant as const
import calculating_utils as cal_utils

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
def analyze_plot(ax:Axes, df:pd.DataFrame, x_col = const.EC, y_col = const.CONSISTENCY, according_col=const.EFFICIENCY, fontsize=5, label=True):
    ax.hlines(y=df[y_col].median(), xmin=df[x_col].min(), xmax=df[x_col].max(), colors="gray", lw=1)
    ax.vlines(x=1 if x_col == const.EC else df[x_col].median(), ymin=df[y_col].min(), ymax=df[y_col].max(), colors="gray", lw=1)
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, hue=according_col, palette=CMAP, )
    if label:
        label_data(zip_x=df[x_col], zip_y=df[y_col], labels=df.index, fontsize=fontsize)
#%%
## 成功計算出 s-MRTS 後視覺化資料
def plot_3D(dmu:list, stitle:str, df:pd.DataFrame, smrts_dict:dict, lambda_dict:dict, target_input=const.INSURANCE_EXP, view_v=45, view_h=-80, dummy_dmu:list=None, DMP_contraction:bool=False):
    
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
        
        reference_dmu = cal_utils.find_ref_dmu(lamda_df=lambda_dict[all_dmu[i]], DMP_contraction=DMP_contraction)
        # print(reference_dmu, LAMBDA_DICT_DUMMY141516[all_dmu[i]][const.LAMBDA])
        smrts_df = smrts_dict[reference_dmu]
        max_dir_mp = cal_utils.float_direction(cal_utils.find_max_dir_mp(smrts_df, DMP_contraction))
            
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