## string constant use multiple times in repositories

## DEA utils
LAMBDA = "lambda"
INPUT_ORIENT = "IO"
OUTPUT_ORIENT = "OO"
VRS = "vrs"
NIRS = "nirs"
NDRS = "ndrs"
## columns of df
INSURANCE_EXP = "Insurance Expenses"
OPERATION_EXP = "Operation Expenses"
UNDERWRITING_PROFIT = "Underwriting Profit"
INVESTMENT_PROFIT = "Investment Profit"
EFFICIENCY = "Efficiency"
SCALE = "Scale"
PROFIT = "Profit"
OUT_DIR = "output progress direction"
REF_DMU = "reference DMU"
REF_LAMBDA = "reference lambda"

EXPANSION_INSURANCE_MAXDMP = "Ins. Exp. max direction of MP"
EXPANSION_INSURANCE_COS_SIM = "Ins. Exp. cosine similarity"
EXPANSION_OPERATION_MAXDMP = "Op. Exp. max direction of MP"
EXPANSION_OPERATION_COS_SIM = "Op. Exp. cosine similarity"
EXPANSION_CONSISTENCY = "Marginal Profit Consistency"

# CONTRACTION_INSURANCE_MAXDMP = "contraction insurance_exp max direction of MP"
# CONTRACTION_INSURANCE_COS_SIM = "contraction insurance_exp cosine similarity"
# CONTRACTION_OPERATION_MAXDMP = "contraction operation_exp max direction of MP"
# CONTRACTION_OPERATION_COS_SIM = "contraction operation_exp cosine similarity"
# CONTRACTION_CONSISTENCY = "Contraction Marginal Revenue Consistency"
EC = "Efficiency Change"

## DMU
LAST_Y_14 = ['AIA Taiwan 16', 'Allianz Taiwan Life 16', 'Bank Taiwan Life 16', 'BNP Paribas Cardif TCB 16', 'Cardif 16', 'Cathay Life 16', "DUMMY Cathay 15", 'Chaoyang Life 16', 'China Life 16', 'Chubb Tempest Life 16', 'Chunghwa Post 16', 'CIGNA 16', "CTBC Life 15", 'Farglory Life 16', 'First-Aviva Life 16', 'Fubon Life 16', "Global Life 14", 'Hontai Life 16', 'Mercuries Life 16', 'Nan Shan Life 16', 'PCA Life 16', 'Prudential of Taiwan 16', "Singfor Life 14", 'Shin Kong Life 16', 'Taiwan Life 16', "DUMMY Taiwan 16", 'TransGlobe Life 16', 'Yuanta Life 16', 'Zurich 16', ]
LAST_Y_18 = [dmu + " 20" for dmu in ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']]