import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap, prepare_cat_cols
import polars as pl

CAT_COLS_NO_CBSA = [  # Does not include SEWTYPE, ASECONDRY, or OMB13CBSA
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

COLS_TO_DROP = ["energy_poverty", "year", "WEIGHT", "CONTROL"]
TEMP_RENAME  = {"proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp"}

FACTORS = {90: 0.90, 100: 1.00, 110: 1.10, 120: 1.20, 130: 1.30}

model_w  = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")

# Load 2050 projected climate data (income shifts applied on top of this)
data = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv").rename(TEMP_RENAME)
raw_pd = data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas()

cbsa = raw_pd["OMB13CBSA"].copy()  # keep for heterogeneity analysis
X_proj_base = prepare_cat_cols(raw_pd, CAT_COLS_NO_CBSA)[list(model_w.get_booster().feature_names)]

# --- Vary HINCP simultaneously across all households ---
results = []

for key, factor in FACTORS.items():
    change = key - 100
    print(f"\n\n{'=' * 60}")
    print(f"HINCP FACTOR: ×{factor}  ({change:+d}% of baseline)")
    print("=" * 60)

    X_scaled = X_proj_base.copy()
    X_scaled["HINCP"]     = X_scaled["HINCP"].astype(float)     * factor
    X_scaled["PERPOVLVL"] = X_scaled["PERPOVLVL"].astype(float) * factor

    run_shap(model_w, X_scaled, name=f"sensitivity_hincp_{key:03d}_xgboost",
             label=f"HINCP ×{factor} on 2050 projected climate")

    mean_pred_prob = model_w.predict_proba(X_scaled)[:, 1].mean()
    results.append({"factor": factor, "mean_predicted_ep_prob": round(float(mean_pred_prob), 4)})
    print(f"Mean predicted EP probability: {mean_pred_prob:.4f}")


print(f"\n\n{'=' * 60}")
print("SUMMARY: Mean predicted EP probability by HINCP factor")
print("=" * 60)
summary = pl.DataFrame(results)
print(summary)
