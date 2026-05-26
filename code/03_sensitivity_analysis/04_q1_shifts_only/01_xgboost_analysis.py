import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, prepare_cat_cols, filter_temp_vars, get_feature_names
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

Q1_SHIFTS = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]

model_w = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights_calibrated.pkl")

# Load 2050 projected climate data
data = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv").rename(TEMP_RENAME)
raw_pd = filter_temp_vars(data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas())

cbsa = raw_pd["OMB13CBSA"].copy()  # keep for heterogeneity analysis
X_proj_base = prepare_cat_cols(raw_pd, CAT_COLS_NO_CBSA)[get_feature_names(model_w)]

quintiles = pd.qcut(X_proj_base["HINCP"], q=5, labels=[1, 2, 3, 4, 5])
q1_mask = (quintiles == 1).values

# Baseline (no income shifts anywhere)
base_probs     = model_w.predict_proba(X_proj_base)[:, 1]
base_q1_prob   = base_probs[q1_mask].mean()
base_overall   = base_probs.mean()
base_ep_rate   = (base_probs > 0.5).mean()

rows = []
for shift in Q1_SHIFTS:
    factor = 1 + shift
    X_shifted = X_proj_base.copy()
    X_shifted["HINCP"]     = X_shifted["HINCP"].astype(float)
    X_shifted["constr_PERPOVLVL"] = X_shifted["constr_PERPOVLVL"].astype(float)
    X_shifted.loc[q1_mask, "HINCP"]          = X_shifted.loc[q1_mask, "HINCP"]          * factor
    X_shifted.loc[q1_mask, "constr_PERPOVLVL"] = X_shifted.loc[q1_mask, "constr_PERPOVLVL"] * factor

    probs          = model_w.predict_proba(X_shifted)[:, 1]
    q1_ep_prob     = probs[q1_mask].mean()
    overall_ep_prob = probs.mean()
    ep_rate        = (probs > 0.5).mean()

    rows.append({
        "q1_shift_pct":       round(shift * 100, 0),
        "q1_ep_prob":         round(float(q1_ep_prob),      4),
        "overall_ep_prob":    round(float(overall_ep_prob), 4),
        "ep_rate_t0.5":       round(float(ep_rate),         4),
    })

print(f"\n\n{'=' * 60}")
print("BASELINE (2050 projected, no income shifts)")
print("=" * 60)
print(f"Q1 mean EP probability:      {base_q1_prob:.4f}")
print(f"Overall mean EP probability: {base_overall:.4f}")
print(f"Overall EP rate (t=0.5):     {base_ep_rate:.4f}")

print(f"\n\n{'=' * 60}")
print("SUMMARY: Q1-only income shifts on 2050 projected climate")
print("=" * 60)
summary = pl.DataFrame(rows)
print(summary)
summary.write_csv("tree_model_output/sensitivity/sensitivity_q1_shifts_only_summary.csv")
