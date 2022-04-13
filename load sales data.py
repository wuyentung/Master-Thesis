#%%
import os
import pandas as pd
import numpy as np
#%%
ENG_NAMES = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']
life2018_raw_df = pd.read_excel("./sales data/2018.xlsx", header=0, index_col=0).replace({'－': "1"}).astype("int32")
life2019_raw_df = pd.read_excel("./sales data/2019.xlsx", header=0, index_col=0).replace({'－': "1"}).astype("int32")
life2020_raw_df = pd.read_excel("./sales data/2020.xlsx", header=0, index_col=0).replace({'－': "1"}).astype("int32")
#%%
def preprocessing(df:pd.DataFrame):
    ## 計算各險種的核保利潤
    for i in ["健康", "年金"]:
        df["%s保險核保利潤" %i] = df["%s保險保費收入" %i] - df["%s保險給付" %i]
    return df
#%%
life2018_raw_df = preprocessing(life2018_raw_df)
life2019_raw_df = preprocessing(life2019_raw_df)
life2020_raw_df = preprocessing(life2020_raw_df)
#%%
