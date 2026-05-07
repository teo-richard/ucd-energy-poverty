import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, prepare_cat_cols
import polars as pl

CAT_COLS_NO_CBSA = [
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]
COLS_TO_DROP = ["energy_poverty", "year", "WEIGHT", "CONTROL"]
TEMP_RENAME = {"proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp"}

model = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")
feature_names = list(model.get_booster().feature_names)

# 2050 predictions (pre-computed)
proj_clim_2050 = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")
pred_2050 = pl.read_csv("tree_model_output/xgboost_2050_with_weights.csv")

# 2023 predictions (generated here so comparison uses same model)
curr_clim = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate_no_cbsa.csv")
raw_2023 = curr_clim.drop([c for c in COLS_TO_DROP if c in curr_clim.columns]).to_pandas()
X_2023 = prepare_cat_cols(raw_2023, CAT_COLS_NO_CBSA)[feature_names]
pred_probs_2023 = model.predict_proba(X_2023)[:, 1]
pred_2023 = pl.DataFrame({"CONTROL": curr_clim["CONTROL"], "pred_prob_xgb_w_2023": pred_probs_2023})

# Join: 2050 climate data (carries OMB13CBSA) + both sets of predictions
joined = (
    proj_clim_2050
    .rename(TEMP_RENAME)
    .join(pred_2050, on="CONTROL", how="left")
    .join(pred_2023, on="CONTROL", how="left")
)

by_cbsa = (
    joined
    .group_by("OMB13CBSA")
    .agg([
        pl.col("pred_prob_xgb_w").mean().alias("mean_ep_prob_2050"),
        pl.col("pred_prob_xgb_w_2023").mean().alias("mean_ep_prob_2023"),
        (pl.col("pred_prob_xgb_w") - pl.col("pred_prob_xgb_w_2023")).mean().alias("mean_ep_delta"),
        pl.len().alias("n"),
    ])
    .sort("mean_ep_delta", descending=True)
)

print(by_cbsa.head(20))
by_cbsa.write_csv("tree_model_output/heterogeneity_by_cbsa.csv")
