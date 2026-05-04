import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap, prepare_cat_cols
import pickle
import polars as pl

CAT_COLS_NO_CBSA = [  # Does not include SEWTYPE, ASECONDRY, or OMB13CBSA
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

FACTORS = {90: 0.90, 100: 1.00, 110: 1.10, 120: 1.20, 130: 1.30}

model_w  = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")
model_nw = load_model("data/processed/models/current_climate_xgboost_no_cbsa_without_weights.pkl")

# Load the current climate no-CBSA test splits
with open("data/processed/current_climate/with_weights/01_03_00_no_cbsa_X_test.pkl", "rb") as f:
    X_test_base_w = prepare_cat_cols(pickle.load(f).to_pandas(), CAT_COLS_NO_CBSA)

with open("data/processed/current_climate/without_weights/01_03_00_no_cbsa_X_test_no_weight.pkl", "rb") as f:
    X_test_base_nw = prepare_cat_cols(pickle.load(f).to_pandas(), CAT_COLS_NO_CBSA)

# --- Vary HINCP in test set; model weights are fixed from current climate training ---
results = []

for key, factor in FACTORS.items():
    change = key - 100
    print(f"\n\n{'=' * 60}")
    print(f"HINCP FACTOR: ×{factor}  ({change:+d}% of baseline)")
    print("=" * 60)

    X_test_scaled = X_test_base_w.copy()
    X_test_scaled["HINCP"] = X_test_scaled["HINCP"] * factor

    run_shap(model_w, X_test_scaled, name=f"sensitivity_hincp_{key:03d}_xgboost", label=f"HINCP ×{factor}")

    mean_pred_prob = model_w.predict_proba(X_test_scaled)[:, 1].mean()
    results.append({"factor": factor, "mean_predicted_ep_prob": round(float(mean_pred_prob), 4)})
    print(f"Mean predicted EP probability: {mean_pred_prob:.4f}")


print(f"\n\n{'=' * 60}")
print("SUMMARY: Mean predicted EP probability by HINCP factor")
print("=" * 60)
summary = pl.DataFrame(results)
print(summary)
