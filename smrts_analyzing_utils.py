import pandas as pd
import numpy as np
import constant as const
from smrts_fiscal_data import EXPANSION_OPERATION_SMRTS_DUMMY141516, EXPANSION_INSURANCE_SMRTS_DUMMY141516, EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516, INSURANCE_SMRTS181920, OPERATION_SMRTS181920, EFF_DICT181920, LAMBDA_DICT181920
import calculating_utils as cal_utils
        
#%%
def year_determin(year:int):
    if year in [14, 15, 16]:
        if 16 == year:
            print("this could be default value using 2014-2016 data")
        return EFF_DICT_DUMMY141516, LAMBDA_DICT_DUMMY141516, EXPANSION_INSURANCE_SMRTS_DUMMY141516, EXPANSION_OPERATION_SMRTS_DUMMY141516, const.LAST_Y_14
    return EFF_DICT181920, LAMBDA_DICT181920, INSURANCE_SMRTS181920, OPERATION_SMRTS181920, const.LAST_Y_18
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
def get_analyze_df(dmu_ks:list, df:pd.DataFrame, year:int=16, remain_last=False):
    
    eff_dict, lambda_dict, insurance_smrts_dict, operation_smrts_dict, last_Y = year_determin(year)
    if remain_last:
        last_Y = []
    
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
    
    ins_max_dirs = []
    op_max_dirs = []
    consistencies = []
    for i in range(len(dmu_ks)):
        ## max direction of MP
        ins_max_dirs.append(cal_utils.float_direction(cal_utils.find_max_dir_mp(insurance_smrts_dict[dmu_ks[i]])))
        op_max_dirs.append(cal_utils.float_direction(cal_utils.find_max_dir_mp(operation_smrts_dict[dmu_ks[i]])))
        
        ## marginal consistency
        if dmu_ks[i] in last_Y:
            consistencies.append(np.nan)
        else:
            ins_cos_sim = cal_utils.cal_cosine_similarity(out_dirs[i], ins_max_dirs[i]) if np.sum(ins_max_dirs[i]) else np.nan
            op_cos_sim = cal_utils.cal_cosine_similarity(out_dirs[i], op_max_dirs[i]) if np.sum(op_max_dirs[i]) else np.nan
            if np.isnan(ins_cos_sim) and np.isnan(op_cos_sim):
                consistencies.append(0)
            else:
                consistencies.append(np.nanmean([ins_cos_sim, op_cos_sim]))
    
    
    
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
            
            # const.EXPANSION_INSURANCE_MAXDMP: expansion_insurance_max_dirs, 
            # const.EXPANSION_INSURANCE_COS_SIM: expansion_insurance_cos_sims, 
            # const.EXPANSION_OPERATION_MAXDMP: expansion_operation_max_dirs, 
            # const.EXPANSION_OPERATION_COS_SIM: expansion_operation_cos_sims, 
            # const.EXPANSION_CONSISTENCY: expansion_consistencies,
            
            const.INS_MAX_DIR_MP: ins_max_dirs,
            const.OP_MAX_DIR_MP: op_max_dirs,
            const.CONSISTENCY: consistencies,
            
            const.EFFICIENCY: effiencies, 
            const.EC: eff_changes, 
        }, index=dmu_ks
        )
    
    return dmu_df