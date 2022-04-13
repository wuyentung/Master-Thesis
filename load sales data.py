#%%
import os
import pandas as pd
import numpy as np
from load_data import LIFE, FISAL_LIFE2019, FISAL_ATTRIBUTES, FISAL_LIFE2018, FISAL_LIFE2020
#%%
ENG_NAMES = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']
life2018_raw_df = pd.read_excel("./sales data/2018.xlsx", header=0, index_col=0).replace({'－': "1"}).astype("int32")
life2019_raw_df = pd.read_excel("./sales data/2019.xlsx", header=0, index_col=0).replace({'－': "1"}).astype("int32")
life2020_raw_df = pd.read_excel("./sales data/2020.xlsx", header=0, index_col=0).replace({'－': "1"}).astype("int32")
#%%
def preprocessing(df:pd.DataFrame, year:str):
    ## 重新命名
    df["name"] = [name+" %s"%year for name in ENG_NAMES]
    df = df.set_index("name")
    
    ## 新增 input 欄位
    if "18" == year:
        fisal_df = FISAL_LIFE2018
    elif  "19" == year:
        fisal_df = FISAL_LIFE2019
    else:
        fisal_df = FISAL_LIFE2020     
    df["insurance_exp"] = fisal_df["insurance_exp"]
    df["operation_exp"] = fisal_df["operation_exp"]
    
    ## 計算各險種的核保利潤
    for i in ["健康", "年金"]:
        df["%s保險核保利潤" %i] = df["%s保險保費收入" %i] - df["%s保險給付" %i]
        ## 拿掉核保利潤等於零的 DMU_k
        for k, row in df.iterrows():
            if 0 == row["%s保險核保利潤" %i]:
                df = df.drop(k)
    return df
#%%
LIFE2018 = preprocessing(life2018_raw_df, "18")
LIFE2019 = preprocessing(life2019_raw_df, "19")
LIFE2020 = preprocessing(life2020_raw_df, "20")
LIFE = pd.concat([LIFE2018, LIFE2019, LIFE2020])
#%%
def denoise_nonpositive(df:pd.DataFrame, min_value=.1):
    ## correct df to at least .1 if there is value <= 0
    df = df.copy()
    for col in df.columns:
        if df[col].min()-min_value < 0:
            # print(col)
            # print(df[col])
            # print()
            df[col] = df[col] - df[col].min() + 1
            # print(df[col])
            # for index, value in df[col].items():
            #     if value-min_value < 0:
            #     # if value == df[col].min():
            #         df[col][index]+=min_value
            #         # print(df[col][index])
    return df
#%%
