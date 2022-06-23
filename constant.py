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

EXPANSION_INSURANCE_MAXDMP = "expansion insurance_exp max direction of MP"
EXPANSION_INSURANCE_COS_SIM = "expansion insurance_exp cosine similarity"
EXPANSION_OPERATION_MAXDMP = "expansion operation_exp max direction of MP"
EXPANSION_OPERATION_COS_SIM = "expansion operation_exp cosine similarity"
EXPANSION_CONSISTENCY = "Expansion Marginal Profit Consistency"

CONTRACTION_INSURANCE_MAXDMP = "contraction insurance_exp max direction of MP"
CONTRACTION_INSURANCE_COS_SIM = "contraction insurance_exp cosine similarity"
CONTRACTION_OPERATION_MAXDMP = "contraction operation_exp max direction of MP"
CONTRACTION_OPERATION_COS_SIM = "contraction operation_exp cosine similarity"
CONTRACTION_CONSISTENCY = "Contraction Marginal Profit Consistency"
EC = "Efficiency Change"

## DMU
LAST_Y = [dmu + " 20" for dmu in ['Bank Taiwan Life', 'Taiwan Life', 'PCA Life', 'Cathay Life', 'China Life', 'Nan Shan Life', 'Shin Kong Life', 'Fubon Life',  'Mercuries Life', 'Farglory Life', 'Hontai Life', 'Allianz Taiwan Life', 'Chunghwa Post', 'First-Aviva Life', 'BNP Paribas Cardif TCB', 'Prudential of Taiwan', 'CIGNA', 'Yuanta Life', 'TransGlobe Life', 'AIA Taiwan', 'Cardif', 'Chubb Tempest Life']]