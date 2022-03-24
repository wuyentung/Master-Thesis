#%%
import os
import pandas as pd
import numpy as np
#%%
files = os.listdir("./data")
#%%
life2018_pd = pd.read_excel("./data/%s" %files[0], header=3, index_col=0)
#%%
life2018_pd.shape
#%%
life2018_pd.columns
#%%
ENG_NAMES = ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']
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
for idx in life2018_pd.index.tolist():
    print(idx if isinstance(idx, str) and "收入" in idx else "")
#%%
print([idx if isinstance(idx, str) else "" for idx in life2018_pd.index.tolist()])
#%%
