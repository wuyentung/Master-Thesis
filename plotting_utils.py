import os
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import constant as const
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