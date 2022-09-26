import pandas as pd
import numpy as np
import dmp
import constant as const

#%%
def find_max_dir_mp(smrts_df:pd.DataFrame, DMP_contraction:bool=False):
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

def float_direction(str_direction:str):
    for direction in dmp.DIRECTIONS:
        if str(direction) == str_direction:
            return direction
    for direction in dmp.NEG_DIRECTIONS:
        if str(direction) == str_direction:
            return direction
    return [0, 0]
#%%
def cal_cosine_similarity(vec_a, vec_b):
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
def find_ref_dmu(lamda_df:pd.DataFrame, DMP_contraction:str, ):
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