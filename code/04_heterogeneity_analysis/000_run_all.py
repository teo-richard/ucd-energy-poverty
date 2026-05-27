"""
run_all.py
==========
Run every heterogeneity analysis in one pass.

Data and model are loaded once and shared across all dimensions.
SHAP computation is the slow part — expect ~10-20 minutes total.

Run from the project root:
    python code/04_heterogeneity_analysis/run_all.py
"""

import sys
import polars as pl

sys.path.insert(0, "code/00_shared")
sys.path.insert(0, "code/04_heterogeneity_analysis")

from heterogeneity_utils import load_data, run_heterogeneity

TEMP_RENAME = {
    "proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp",
    "proj_dtr": "dtr", "proj_HDD_approx": "HDD_approx", "proj_CDD_approx": "CDD_approx",
}

# ---------------------------------------------------------------------------
# Load once — shared by all dimensions
# ---------------------------------------------------------------------------
df, X_pd, model = load_data()

# ---------------------------------------------------------------------------
# Simple dimensions — single call each
# ---------------------------------------------------------------------------
run_heterogeneity("HHRACE",   df=df, X_pd=X_pd, model=model)
run_heterogeneity("TENURE",   df=df, X_pd=X_pd, model=model)
run_heterogeneity("DIVISION", df=df, X_pd=X_pd, model=model)
run_heterogeneity("DISHH",    df=df, X_pd=X_pd, model=model)

# ---------------------------------------------------------------------------
# Age — bin HHAGE into cohorts first
# ---------------------------------------------------------------------------
df_age = df.with_columns(
    pl.when(pl.col("HHAGE") < 35).then(pl.lit("1_Under35"))
    .when(pl.col("HHAGE") < 50).then(pl.lit("2_35to49"))
    .when(pl.col("HHAGE") < 65).then(pl.lit("3_50to64"))
    .otherwise(pl.lit("4_65plus"))
    .alias("age_group")
)
run_heterogeneity("age_group", df=df_age, X_pd=X_pd, model=model)

# ---------------------------------------------------------------------------
# Heating type — also produce the 2023 vs 2050 mean comparison CSV
# ---------------------------------------------------------------------------
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

proj_clim_2050 = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")
pred_2050      = pl.read_csv("output/projections/xgboost_2050_with_weights.csv")
pred_probs_2023 = model.predict_proba(X_pd)[:, 1]
pred_2023 = pl.DataFrame({"CONTROL": df["CONTROL"], "pred_prob_xgb_w_2023": pred_probs_2023})

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

df_het = df.with_columns(pl.Series("pred_prob", pred_probs_2023))
run_heterogeneity("HEATTYPE", df=df_het, X_pd=X_pd, model=model)

print("\nAll heterogeneity analyses complete.")
