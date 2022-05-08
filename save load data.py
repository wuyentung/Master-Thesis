# %%
import pandas as pd
from load_data import FISCAL_LIFE2014, FISCAL_LIFE2015, FISCAL_LIFE2016
# %%
FISCAL_LIFE2014.to_csv("./fisal data 14-16/2014.csv")
FISCAL_LIFE2015.to_csv("./fisal data 14-16/2015.csv")
FISCAL_LIFE2016.to_csv("./fisal data 14-16/2016.csv")
# # %%
# pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["13", "24"]
#              ).to_csv("./fisal data 14-16/temp.csv")
# # %%
# temp = pd.read_csv("./fisal data 14-16/temp.csv", index_col=0)
# %%
