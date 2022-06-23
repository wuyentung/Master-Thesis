# %%
import os
import pandas as pd
import numpy as np
import constant as const
# %%


def denoise_nonpositive(df: pd.DataFrame, div_norm=6, round_to=6):
    # correct df to at least min_value/(10**div_norm) if there is value <= 0
    df = df.copy()
    df = df/(10**div_norm)
    df = df.round(round_to)
    
    min_value = 1/(10 ** min(div_norm, round_to))
    
    for col in df.columns:
        if df[col].min()-min_value < 0:
            # print(col)
            # print(df[col])
            # print()
            df[col] = df[col] + np.abs(df[col].min()) + min_value
    return df


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
# %%


def find_index(df: pd.DataFrame, target: str):
    for idx in df.index:
        if isinstance(idx, str) and target in idx:
            return idx
    raise ValueError(f"{target} is not in df.index")

# %%


def cal_insurance_exp(df, col): return df[col][find_index(
    df, '承保費用支出')] + df[col][find_index(df, '佣金費用')]


def cal_operation_exp(df, col): return df[col][find_index(
    df, '其他營業成本')] + df[col][find_index(df, '財務成本')] + df[col][find_index(df, '營業費用')]


def cal_insurance_income(df, col): return df[col][find_index(df, '簽單保費收入')] - df[col][find_index(
    df, '\u3000\u3000\u3000未滿期保費準備淨變動')] + df[col][find_index(df, '\u3000手續費收入')]


def cal_reinsurance_exp(
    df, col): return df[col][find_index(df, '\u3000減：再保費支出')]


def cal_reinsurance_income(
    df, col): return df[col][find_index(df, '\u3000再保費收入')]


def cal_underwriting_profit(df, col): return df[col][find_index(df, '營業收入')] - df[col][find_index(
    df, '營業成本')] - df[col][find_index(df, '營業費用')] - df[col][find_index(df, '\u3000淨投資損益')]


def cal_investment_profit(
    df, col): return df[col][find_index(df, '\u3000淨投資損益')]
# %%


def single_insurer(df, name):
    try:
        df[name]
    except:
        if "First" in name:
            name = "First Life"
    return [cal_insurance_exp(df=df, col=name), cal_operation_exp(df=df, col=name), cal_insurance_income(df=df, col=name), cal_reinsurance_exp(df=df, col=name), cal_reinsurance_income(df=df, col=name), cal_underwriting_profit(df=df, col=name), cal_investment_profit(df=df, col=name), ]


# %%
files181920 = os.listdir("./fisal data 18-20")
# %%
life2018_raw_df = pd.read_excel(
    "./fisal data 18-20/%s" % files181920[0], header=3, index_col=0)
life2019_raw_df = pd.read_excel(
    "./fisal data 18-20/%s" % files181920[1], header=3, index_col=0)
life2020_raw_df = pd.read_excel(
    "./fisal data 18-20/%s" % files181920[2], header=3, index_col=0)
# %%
ENG_NAMES_18 = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']
CHI_NAMES_18 = ['臺銀人壽', '台灣人壽', '保誠人壽', '國泰人壽', '中國人壽', '南山人壽', '新光人壽', '富邦人壽', '三商美邦人壽' '遠雄人壽', '宏泰人壽',
                '安聯人壽', '中華郵政', '第一金人壽', '合作金庫人壽', '保德信國際人壽', '康健人壽', '元大人壽', '全球人壽', '友邦人壽', '法國巴黎人壽', '安達人壽', ]
# %%
FISCAL_ATTRIBUTES = [const.INSURANCE_EXP, const.OPERATION_EXP, "insurance_income",
                     "reinsurance_exp", "reinsurance_income", const.UNDERWRITING_PROFIT, const.INVESTMENT_PROFIT]
FISCAL_LIFE2018 = pd.DataFrame([single_insurer(df=life2018_raw_df, name=name) for name in ENG_NAMES_18], index=[
                               name+" 18" for name in ENG_NAMES_18], columns=FISCAL_ATTRIBUTES)
FISCAL_LIFE2019 = pd.DataFrame([single_insurer(df=life2019_raw_df, name=name) for name in ENG_NAMES_18], index=[
                               name+" 19" for name in ENG_NAMES_18], columns=FISCAL_ATTRIBUTES)
FISCAL_LIFE2020 = pd.DataFrame([single_insurer(df=life2020_raw_df, name=name) for name in ENG_NAMES_18], index=[
                               name+" 20" for name in ENG_NAMES_18], columns=FISCAL_ATTRIBUTES).astype("float")
LIFE181920 = pd.concat([FISCAL_LIFE2018, FISCAL_LIFE2019, FISCAL_LIFE2020])
# denoise_nonpositive(df=LIFE)["reinsurance_income"].to_list()
# %%
files141516 = os.listdir("./fisal data 14-16")
# %%
life2014_raw_df = pd.read_excel(
    "./fisal data 14-16/PDF2060_2014.xls", header=3, index_col=0)
life2015_raw_df = pd.read_excel(
    "./fisal data 14-16/PDF2060_2015.xls", header=3, index_col=0)
life2016_raw_df = pd.read_excel(
    "./fisal data 14-16/PDF2060_2016.xls", header=3, index_col=0)
# %%
ENG_NAMES_14 = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life', 'Global Life', 'Mercuries Life', 'Chaoyang Life', 'Singfor Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'CTBC Life', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'ACE Tempest Life', 'Zurich']

ENG_NAMES_15 = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life', 'Mercuries Life', 'Chaoyang Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'CTBC Life', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'ACE Tempest Life', 'Zurich']

ENG_NAMES_16 = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life', 'Mercuries Life', 'Chaoyang Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life', 'Zurich']
try:
    FISCAL_LIFE2014 = pd.DataFrame([single_insurer(df=life2014_raw_df, name=name) for name in ENG_NAMES_14], index=[
        name+" 14" for name in ENG_NAMES_14], columns=FISCAL_ATTRIBUTES)
    FISCAL_LIFE2015 = pd.DataFrame([single_insurer(df=life2015_raw_df, name=name) for name in ENG_NAMES_15], index=[
        name+" 15" for name in ENG_NAMES_15], columns=FISCAL_ATTRIBUTES)
    FISCAL_LIFE2016 = pd.DataFrame([single_insurer(df=life2016_raw_df, name=name) for name in ENG_NAMES_16], index=[
        name+" 16" for name in ENG_NAMES_16], columns=FISCAL_ATTRIBUTES)
except:
    print("this device should be MAC")
    FISCAL_LIFE2014 = pd.read_csv("./fisal data 14-16/2014.csv", index_col=0)
    FISCAL_LIFE2015 = pd.read_csv("./fisal data 14-16/2015.csv", index_col=0)
    FISCAL_LIFE2016 = pd.read_csv("./fisal data 14-16/2016.csv", index_col=0)
# %%
LIFE141516 = pd.concat([FISCAL_LIFE2014, FISCAL_LIFE2015, FISCAL_LIFE2016])
# %%
LIFE_DUMMY141516 = LIFE141516.copy()
LIFE_DUMMY141516.loc["DUMMY Cathay 15"] = LIFE141516.loc["Global Life 14"] + \
    LIFE141516.loc["Cathay Life 14"] + LIFE141516.loc["Singfor Life 14"]
LIFE_DUMMY141516.loc["DUMMY Taiwan 16"] = LIFE141516.loc["Taiwan Life 15"] + \
    LIFE141516.loc["CTBC Life 15"]
# %%
