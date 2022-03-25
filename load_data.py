#%%
import os
import pandas as pd
import numpy as np
#%%
files = os.listdir("./data")
#%%
life2018_raw_df = pd.read_excel("./data/%s" %files[0], header=3, index_col=0)
life2019_raw_df = pd.read_excel("./data/%s" %files[1], header=3, index_col=0)
life2020_raw_df = pd.read_excel("./data/%s" %files[2], header=3, index_col=0)
#%%
ENG_NAMES = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']
CHI_NAMES = ['臺銀人壽', '台灣人壽', '保誠人壽', '國泰人壽', '中國人壽', '南山人壽', '新光人壽', '富邦人壽', '三商美邦人壽' '遠雄人壽', '宏泰人壽', '安聯人壽', '中華郵政', '第一金人壽', '合作金庫人壽', '保德信國際人壽', '康健人壽', '元大人壽', '全球人壽', '友邦人壽', '法國巴黎人壽', '安達人壽',]
IDX = [
    '營業收入', 
    '\u3000簽單保費收入', 
    '\u3000再保費收入', 
    '\u3000減：再保費支出', 
    '\u3000\u3000\u3000未滿期保費準備淨變動', 
    '\u3000自留滿期保費收入', 
    '\u3000再保佣金收入', 
    '\u3000手續費收入', 
    '\u3000淨投資損益', 
    '      利息收入', 
    '      透過損益按公允價值衡量之金融資產及負債損益', 
    '      透過其他綜合損益按公允價值衡量之金融資產\n      已實現損益', 
    '      除列按攤銷後成本衡量之金融資產淨損益', 
    '      採用權益法認列之關聯企業及合資權益之份額', 
    '      兌換損益－投資', 
    '      外匯價格變動準備金淨變動', 
    '      投資性不動產損益', 
    '      投資之預期信用減損損失及迴轉利益', 
    '      其他投資減損損失及迴轉利益', 
    '      金融資產重分類損益', 
    '      其他淨投資損益', 
    '      採用覆蓋法重分類之損益', 
    '\u3000其他營業收入', 
    '\u3000分離帳戶保險商品收益', 
    '營業成本', 
    '\u3000保險賠款與給付', 
    '\u3000再保賠款與給付', 
    '\u3000壽險紅利給付', 
    '\u3000減：攤回再保賠款與給付', 
    '  自留保險賠款與給付', 
    '  保險負債淨變動', 
    '    賠款準備淨變動', 
    '    責任準備淨變動', 
    '    特別準備淨變動', 
    '    保費不足準備淨變動', 
    '    負債適足準備淨變動', 
    '    其他準備淨變動', 
    '  具金融商品性質之保險契約準備淨變動', 
    '  承保費用支出', 
    '  佣金費用', 
    '  其他營業成本', 
    '  財務成本', 
    '  分離帳戶保險商品費用', 
    '營業費用', 
    '營業外收入費用', 
    '\u3000營業外收入及利益', 
    '\u3000營業外費用及損失', 
    '繼續營業單位稅前純益（損益）', 
    '所得稅費用（利益）', 
    '繼續營業單位本期淨利（淨損）', 
    '停業單位損益', 
    '本期淨利(淨損)', 
    '本期其他綜合損益（稅後淨額）', 
    '本期綜合損益總額',
    ]
#%%
cal_insurance_exp = lambda df, col: df[col]['  承保費用支出'] + df[col]['  佣金費用']
cal_operation_exp = lambda df, col: df[col]['  其他營業成本'] + df[col]['  財務成本'] + df[col]['營業費用']

cal_insurance_income = lambda df, col: df[col]['\u3000簽單保費收入'] - df[col]['\u3000\u3000\u3000未滿期保費準備淨變動'] + df[col]['\u3000手續費收入']
cal_reinsurance_exp = lambda df, col: df[col]['\u3000減：再保費支出']
cal_reinsurance_income = lambda df, col: df[col]['\u3000再保費收入']

cal_underwriting_profit = lambda df, col: df[col]['營業收入'] - df[col]['營業成本'] - df[col]['營業費用'] - df[col]['\u3000淨投資損益']
cal_investment_profit = lambda df, col: df[col]['\u3000淨投資損益']
#%%
def single_insurer(df, name):
    try:
        df[name]
    except:
        if "First" in name:
            name = "First Life"
    return [cal_insurance_exp(df=df, col=name), cal_operation_exp(df=df, col=name), cal_insurance_income(df=df, col=name), cal_reinsurance_exp(df=df, col=name), cal_reinsurance_income(df=df, col=name), cal_underwriting_profit(df=df, col=name), cal_investment_profit(df=df, col=name), ]
#%%
ATTRIBUTES = ["insurance_exp", "operation_exp", "insurance_income", "reinsurance_exp", "reinsurance_income", "underwriting_profit", "investment_profit"]
LIFE2018 = pd.DataFrame([single_insurer(df=life2018_raw_df, name=name) for name in ENG_NAMES], index=[name+" 18" for name in ENG_NAMES], columns=["insurance_exp", "operation_exp", "insurance_income", "reinsurance_exp", "reinsurance_income", "underwriting_profit", "investment_profit"])
LIFE2019 = pd.DataFrame([single_insurer(df=life2019_raw_df, name=name) for name in ENG_NAMES], index=[name+" 19" for name in ENG_NAMES], columns=["insurance_exp", "operation_exp", "insurance_income", "reinsurance_exp", "reinsurance_income", "underwriting_profit", "investment_profit"])
LIFE2020 = pd.DataFrame([single_insurer(df=life2020_raw_df, name=name) for name in ENG_NAMES], index=[name+" 20" for name in ENG_NAMES], columns=["insurance_exp", "operation_exp", "insurance_income", "reinsurance_exp", "reinsurance_income", "underwriting_profit", "investment_profit"])
#%%
def denoise_nonpositive(df:pd.DataFrame):
    ## correct df to at least .1 if there is value <= 0
    df = df.copy()
    for col in df.columns:
        if df[col].min() <= 0:
            df[col] = df[col] - df[col].min()
            for index, value in df[col].items():
                if value == df[col].min():
                    df[col][index]+=.1
    return df
#%%
LIFE = pd.concat([LIFE2018, LIFE2019, LIFE2020])
# denoise_nonpositive(df=LIFE)["reinsurance_income"].to_list()
#%%
