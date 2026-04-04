import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

ahs_climate_full = pl.read_csv("data/transitory/01_02_04_cat_collapsed_ahs_climate.csv")
ahs_climate = pl.read_csv("data/processed/LightGBM_data/01_02_04_ready_for_lightgbm_ahs_climate.csv")

# --- Train and test splits ---
X = ahs_climate.drop(["energy_poverty"])
y = ahs_climate["energy_poverty"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ----------- Look for leakage -----------

# 1. Check POVLVLINC 
print(ahs_climate.select(["POVLVLINC", "energy_poverty"]).corr())

# 2. Check PERPOVLVL vs POVLVLINC -- are these both income variables or is one derived?
print(ahs_climate.select(["PERPOVLVL", "POVLVLINC", "HINCP", "energy_poverty"]).corr())


corr = ahs_climate.to_pandas().corr()["energy_poverty"].abs().sort_values(ascending=False)
print(corr.head(20))