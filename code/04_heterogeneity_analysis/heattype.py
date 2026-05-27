import sys
sys.path.insert(0, "code/00_shared")
sys.path.insert(0, "code/04_heterogeneity_analysis")
from analysis_functions import load_model, prepare_cat_cols, filter_temp_vars, get_feature_names
from heterogeneity_utils import run_heterogeneity
import polars as pl

CAT_COLS_NO_CBSA = [
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

HEATTYPE_LABELS = pl.DataFrame({
    "HEATTYPE": [1, 2, 3, 4, 5, 6, 7, 8, 999],
    "heattype_label": [
        "Forced air furnace",
        "Steam/hot water",
        "Heat pump",
        "Electric baseboard",
        "Pipeless furnace",
        "Portable electric (poverty signal)",
        "Cooking stove for heat (poverty signal)",
        "No heating system",
        "Other",
    ],
})

model = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights_calibrated.pkl")
feature_names = get_feature_names(model)

# 2050 predictions (pre-computed)
proj_clim_2050 = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")
pred_2050 = pl.read_csv("output/projections/xgboost_2050_with_weights.csv")

# 2023 predictions (generated here so comparison uses same model)
curr_clim = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate_no_cbsa.csv")
raw_2023 = filter_temp_vars(curr_clim.drop([c for c in COLS_TO_DROP if c in curr_clim.columns]).to_pandas())
X_2023 = prepare_cat_cols(raw_2023, CAT_COLS_NO_CBSA)[feature_names]
pred_probs_2023 = model.predict_proba(X_2023)[:, 1]
pred_2023 = pl.DataFrame({"CONTROL": curr_clim["CONTROL"], "pred_prob_xgb_w_2023": pred_probs_2023})

# Join: 2050 climate data (carries HEATTYPE) + both sets of predictions
joined = (
    proj_clim_2050
    .rename(TEMP_RENAME)
    .join(pred_2050, on="CONTROL", how="left")
    .join(pred_2023, on="CONTROL", how="left")
)

by_heattype = (
    joined
    .group_by("HEATTYPE")
    .agg([
        pl.col("pred_prob_xgb_w").mean().alias("mean_ep_prob_2050"),
        pl.col("pred_prob_xgb_w_2023").mean().alias("mean_ep_prob_2023"),
        (pl.col("pred_prob_xgb_w") - pl.col("pred_prob_xgb_w_2023")).mean().alias("mean_ep_delta"),
        pl.len().alias("n"),
    ])
    .join(HEATTYPE_LABELS, on="HEATTYPE", how="left")
    .sort("mean_ep_prob_2023", descending=True)
)

print(by_heattype)
by_heattype.write_csv("output/heterogeneity/heterogeneity_by_heattype.csv")

# Full three-step heterogeneity analysis (distribution + conditional + SHAP).
# Pass the already-loaded objects so nothing is read from disk a second time.
df_het = curr_clim.with_columns(pl.Series("pred_prob", pred_probs_2023))
run_heterogeneity("HEATTYPE", df=df_het, X_pd=X_2023, model=model)
