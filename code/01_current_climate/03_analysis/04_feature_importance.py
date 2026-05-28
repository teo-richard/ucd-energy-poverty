import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, get_feature_names
import polars as pl

# XGBoost
xgb_model = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")
xgb_importance = pl.DataFrame({
    "feature": get_feature_names(xgb_model),
    "importance": xgb_model.feature_importances_,
}).sort("importance", descending=True)
xgb_importance.write_csv("output/current_climate/xgboost_no_cbsa_feature_importance.csv")
print("XGBoost feature importance saved.")

# LightGBM
lgb_model = load_model("data/processed/models/current_climate_lightgbm_no_cbsa_with_weights.pkl")
lgb_importance = pl.DataFrame({
    "feature": lgb_model.feature_name_,
    "importance": lgb_model.feature_importances_,
}).sort("importance", descending=True)
lgb_importance.write_csv("output/current_climate/lightgbm_no_cbsa_feature_importance.csv")
print("LightGBM feature importance saved.")
