import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_splits, run_lightgbm, run_shap, save_model, TEMP_VARS


# --- With CBSA ---
# splits, splits_nw = load_splits(
#     "data/processed/current_climate",
#     "01_03_00"
# )

# model_w, X_test_w = run_lightgbm(**splits, label="WITH WEIGHTS")
# save_model(model_w, "data/processed/models/current_climate_lightgbm_with_weights.pkl")
# # run_shap(model_w, X_test_w, name = "current_climate_lightgbm", label="WITH WEIGHTS")

# model_nw, X_test_nw = run_lightgbm(
#     **{k: splits_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
#     label="WITHOUT WEIGHTS",
# )
# save_model(model_nw, "data/processed/models/current_climate_lightgbm_without_weights.pkl")

# --- Without CBSA ---
splits_nc, splits_nc_nw = load_splits(
    "data/processed/current_climate",
    "01_03_00",
    cbsa=False,
    temp_vars=TEMP_VARS
)

model_nc_w, X_test_nc_w = run_lightgbm(**splits_nc, label="WITH WEIGHTS — no CBSA", cbsa=False)
save_model(model_nc_w, "data/processed/models/current_climate_lightgbm_no_cbsa_with_weights.pkl")

# model_nc_nw, X_test_nc_nw = run_lightgbm(
#     **{k: splits_nc_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
#     label="WITHOUT WEIGHTS — no CBSA",
# )
# save_model(model_nc_nw, "data/processed/models/current_climate_lightgbm_no_cbsa_without_weights.pkl")
