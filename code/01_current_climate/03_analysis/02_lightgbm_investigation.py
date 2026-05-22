import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

ahs_climate_full = pl.read_csv("data/interim/current_climate/01_02_04_cat_collapsed_ahs_climate.csv")
ahs_climate = pl.read_csv("data/processed/current_climate/01_02_05_basic_ready_for_trees_ahs_climate.csv")

# --- Train and test splits ---
X = ahs_climate.drop(["energy_deprivation"])
y = ahs_climate["energy_deprivation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ----------- Look for leakage -----------
# ROC of 0.96 is really good (although the precision isn't great but energy poor is only 17% of the population)

# 1. Check POVLVLINC 
print(ahs_climate.select(["POVLVLINC", "energy_deprivation"]).corr())

# 2. Check PERPOVLVL vs POVLVLINC -- are these both income variables or is one derived?
print(ahs_climate.select(["PERPOVLVL", "POVLVLINC", "HINCP", "energy_deprivation"]).corr())


corr = ahs_climate.to_pandas().corr()["energy_deprivation"].abs().sort_values(ascending=False)
print(corr.head(20))

# All seems fine, no high correlations