import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_splits, run_xgboost, run_shap, save_model


CAT_COLS = [  # Does not include SEWTYPE and ASECONDRY because we dropped those
    "TENURE", "BLD", "INTLANG", "DIVISION", "OMB13CBSA",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

CAT_COLS_NO_CBSA = [col for col in CAT_COLS if col != "OMB13CBSA"]

# --- With CBSA ---
# splits, splits_nw = load_splits(
#     "data/processed/current_climate",
#     "01_03_00"
# )

# model_w, X_test_w = run_xgboost(**splits, cat_cols=CAT_COLS, scale_pos_weight=4.83, label="WITH WEIGHTS")
# run_shap(model_w, X_test_w, name="current_climate_xgboost", label="WITH WEIGHTS")
# save_model(model_w, "data/processed/models/current_climate_xgboost_with_weights.pkl")

# model_nw, X_test_nw = run_xgboost(
#     **{k: splits_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
#     cat_cols=CAT_COLS, scale_pos_weight=4.83, label="WITHOUT WEIGHTS",
# )
# save_model(model_nw, "data/processed/models/current_climate_xgboost_without_weights.pkl")

# --- Without CBSA ---
splits_nc, splits_nc_nw = load_splits(
    "data/processed/current_climate",
    "01_03_00",
    cbsa=False,
)

model_nc_w, X_test_nc_w = run_xgboost(**splits_nc, cat_cols=CAT_COLS, scale_pos_weight=4.83, label="WITH WEIGHTS — no CBSA", cbsa=False)
run_shap(model_nc_w, X_test_nc_w, name="current_climate_xgboost_no_cbsa", label="WITH WEIGHTS — no CBSA")
save_model(model_nc_w, "data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")

# model_nc_nw, X_test_nc_nw = run_xgboost(
#     **{k: splits_nc_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
#     cat_cols=CAT_COLS_NO_CBSA, scale_pos_weight=4.83, label="WITHOUT WEIGHTS — no CBSA",
# )
# save_model(model_nc_nw, "data/processed/models/current_climate_xgboost_no_cbsa_without_weights.pkl")
