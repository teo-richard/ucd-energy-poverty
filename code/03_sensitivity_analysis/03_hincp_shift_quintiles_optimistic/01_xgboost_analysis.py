import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap, prepare_cat_cols, filter_temp_vars, get_feature_names
import pandas as pd
import polars as pl

CAT_COLS_NO_CBSA = [  # Does not include SEWTYPE, ASECONDRY, or OMB13CBSA
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

COLS_TO_DROP = ["energy_deprivation", "year", "WEIGHT", "CONTROL"]
TEMP_RENAME = {
    "proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp",
    "proj_dtr": "dtr", "proj_HDD_approx": "HDD_approx", "proj_CDD_approx": "CDD_approx",
}

# Quintile-specific HINCP multipliers (1 + percent increase)
QUINTILE_SHIFTS = {1: 1.15, 2: 1.20, 3: 1.20, 4: 1.20, 5: 1.25}

raw_model_w = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")
cal_model_w = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights_calibrated.pkl")

# Load 2050 projected climate data (income shifts applied on top of this)
data = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv").rename(TEMP_RENAME)
raw_pd = filter_temp_vars(data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas())

cbsa = raw_pd["OMB13CBSA"].copy()  # keep for heterogeneity analysis
X_proj_base = prepare_cat_cols(raw_pd, CAT_COLS_NO_CBSA)[get_feature_names(cal_model_w)]

# Assign income quintiles based on HINCP distribution in the projected dataset
quintiles = pd.qcut(X_proj_base["HINCP"], q=5, labels=[1, 2, 3, 4, 5])

# Apply quintile-specific shifts to a copy of the projected dataset
X_proj_shifted = X_proj_base.copy()
X_proj_shifted["HINCP"]     = X_proj_shifted["HINCP"].astype(float)
X_proj_shifted["constr_PERPOVLVL"] = X_proj_shifted["constr_PERPOVLVL"].astype(float)
for q, factor in QUINTILE_SHIFTS.items():
    mask = quintiles == q
    X_proj_shifted.loc[mask, "HINCP"]          = X_proj_shifted.loc[mask, "HINCP"]          * factor
    X_proj_shifted.loc[mask, "constr_PERPOVLVL"] = X_proj_shifted.loc[mask, "constr_PERPOVLVL"] * factor

# --- Per-quintile results ---
print(f"\n\n{'=' * 60}")
print("PER-QUINTILE RESULTS")
print("=" * 60)

rows = []
for q, factor in QUINTILE_SHIFTS.items():
    mask = (quintiles == q).values
    base_prob    = cal_model_w.predict_proba(X_proj_base[mask])[:, 1].mean()
    shifted_prob = cal_model_w.predict_proba(X_proj_shifted[mask])[:, 1].mean()
    rows.append({
        "quintile":          q,
        "hincp_shift_pct":   round((factor - 1) * 100, 1),
        "n":                 int(mask.sum()),
        "baseline_ep_prob":  round(float(base_prob),    4),
        "shifted_ep_prob":   round(float(shifted_prob), 4),
    })

per_quintile = pl.DataFrame(rows)
print(per_quintile)
per_quintile.write_csv("tree_model_output/sensitivity_hincp_quintiles_optimistic_per_quintile.csv")

# --- Overall summary ---
print(f"\n\n{'=' * 60}")
print("OVERALL SUMMARY")
print("=" * 60)

baseline_overall = cal_model_w.predict_proba(X_proj_base)[:, 1].mean()
shifted_overall  = cal_model_w.predict_proba(X_proj_shifted)[:, 1].mean()
print(f"2050 projected mean EP probability:              {baseline_overall:.4f}")
print(f"2050 projected + income shifts mean EP prob:     {shifted_overall:.4f}")

baseline_ep_rate = (cal_model_w.predict_proba(X_proj_base)[:, 1] > 0.5).mean()
shifted_ep_rate  = (cal_model_w.predict_proba(X_proj_shifted)[:, 1] > 0.5).mean()
print(f"2050 projected EP rate (threshold 0.5):          {baseline_ep_rate:.4f}")
print(f"2050 projected + income shifts EP rate (t=0.5):  {shifted_ep_rate:.4f}")

# --- SHAP on shifted dataset ---
run_shap(raw_model_w, X_proj_shifted, name="sensitivity_hincp_quintiles_optimistic_xgboost",
         label="HINCP quintile shifts (optimistic) on 2050 projected climate")

print(baseline_ep_rate)
print(shifted_ep_rate)