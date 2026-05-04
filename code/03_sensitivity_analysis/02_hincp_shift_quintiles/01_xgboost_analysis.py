import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap, prepare_cat_cols
import pickle
import pandas as pd
import polars as pl

CAT_COLS_NO_CBSA = [  # Does not include SEWTYPE, ASECONDRY, or OMB13CBSA
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

# Quintile-specific HINCP multipliers (1 + percent increase)
QUINTILE_SHIFTS = {1: 1.063, 2: 1.253, 3: 1.220, 4: 1.198, 5: 1.282}

model_w = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")

with open("data/processed/current_climate/with_weights/01_03_00_no_cbsa_X_test.pkl", "rb") as f:
    X_test_base_w = prepare_cat_cols(pickle.load(f).to_pandas(), CAT_COLS_NO_CBSA)

# Assign income quintiles based on HINCP distribution in the test set
quintiles = pd.qcut(X_test_base_w["HINCP"], q=5, labels=[1, 2, 3, 4, 5])

# Apply quintile-specific shifts to a copy of the test set
X_test_shifted = X_test_base_w.copy()
X_test_shifted["HINCP"] = X_test_shifted["HINCP"].astype(float)
for q, factor in QUINTILE_SHIFTS.items():
    mask = quintiles == q
    X_test_shifted.loc[mask, "HINCP"] = X_test_shifted.loc[mask, "HINCP"] * factor

# --- Per-quintile results ---
print(f"\n\n{'=' * 60}")
print("PER-QUINTILE RESULTS")
print("=" * 60)

rows = []
for q, factor in QUINTILE_SHIFTS.items():
    mask = (quintiles == q).values
    base_prob    = model_w.predict_proba(X_test_base_w[mask])[:, 1].mean()
    shifted_prob = model_w.predict_proba(X_test_shifted[mask])[:, 1].mean()
    rows.append({
        "quintile":           q,
        "hincp_shift_pct":    round((factor - 1) * 100, 1),
        "n":                  int(mask.sum()),
        "baseline_ep_prob":   round(float(base_prob),    4),
        "shifted_ep_prob":    round(float(shifted_prob), 4),
    })

per_quintile = pl.DataFrame(rows)
print(per_quintile)

# --- Overall summary ---
print(f"\n\n{'=' * 60}")
print("OVERALL SUMMARY")
print("=" * 60)

baseline_overall = model_w.predict_proba(X_test_base_w)[:, 1].mean()
shifted_overall  = model_w.predict_proba(X_test_shifted)[:, 1].mean()
print(f"Baseline mean EP probability:   {baseline_overall:.4f}")
print(f"Post-shift mean EP probability: {shifted_overall:.4f}")

# --- SHAP on shifted dataset ---
run_shap(model_w, X_test_shifted, name="sensitivity_hincp_quintiles_xgboost",
         label="HINCP quintile shifts")


baseline_ep_rate = (model_w.predict_proba(X_test_base_w)[:, 1] > 0.5).mean()
shifted_ep_rate  = (model_w.predict_proba(X_test_shifted)[:, 1] > 0.5).mean()
print(f"Baseline EP rate (threshold 0.5): {baseline_ep_rate:.4f}")
print(f"Shifted EP rate  (threshold 0.5): {shifted_ep_rate:.4f}")