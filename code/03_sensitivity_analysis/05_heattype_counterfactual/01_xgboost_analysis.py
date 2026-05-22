import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, prepare_cat_cols, filter_temp_vars
import numpy as np
import pandas as pd
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

HEATTYPE_LABELS = {
    1: "Forced air furnace",
    2: "Steam/hot water",
    3: "Heat pump",
    4: "Electric baseboard",
    5: "Pipeless furnace",
    6: "Portable electric (poverty signal)",
    7: "Cooking stove for heat (poverty signal)",
    8: "No heating system",
    999: "Other",
}
HEATTYPE_VALUES = list(HEATTYPE_LABELS.keys())

model_w = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")
feature_names = list(model_w.get_booster().feature_names)

data = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv").rename(TEMP_RENAME)
raw_pd = filter_temp_vars(data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas())

X_proj_base = prepare_cat_cols(raw_pd, CAT_COLS_NO_CBSA)[feature_names]
quintiles = pd.qcut(X_proj_base["HINCP"], q=5, labels=[1, 2, 3, 4, 5])

# Baseline EP probabilities (actual HEATTYPE, per household)
base_probs = model_w.predict_proba(X_proj_base)[:, 1]

rows = []
for q in [1, 2, 3, 4, 5]:
    q_mask = (quintiles == q).values
    baseline_q = base_probs[q_mask].mean()

    for ht in HEATTYPE_VALUES:
        # Modify raw_pd (integer HEATTYPE) before re-encoding to avoid
        # pandas Categorical assignment errors with unseen category values.
        raw_cf = raw_pd.copy()
        raw_cf.loc[q_mask, "HEATTYPE"] = ht
        X_cf = prepare_cat_cols(raw_cf, CAT_COLS_NO_CBSA)[feature_names]
        cf_probs = model_w.predict_proba(X_cf)[:, 1][q_mask]

        rows.append({
            "quintile":               q,
            "heattype":               ht,
            "heattype_label":         HEATTYPE_LABELS[ht],
            "n":                      int(q_mask.sum()),
            "baseline_ep_prob":       round(float(baseline_q),              4),
            "counterfactual_ep_prob": round(float(cf_probs.mean()),         4),
            "delta":                  round(float(cf_probs.mean() - baseline_q), 4),
            "p10":                    round(float(np.percentile(cf_probs, 10)), 4),
            "p25":                    round(float(np.percentile(cf_probs, 25)), 4),
            "p50":                    round(float(np.percentile(cf_probs, 50)), 4),
            "p75":                    round(float(np.percentile(cf_probs, 75)), 4),
            "p90":                    round(float(np.percentile(cf_probs, 90)), 4),
        })

results = pl.DataFrame(rows)

for q in [1, 2, 3, 4, 5]:
    print(f"\n\n{'=' * 70}")
    print(f"QUINTILE {q} — HEATTYPE COUNTERFACTUAL SWEEP (2050 projected climate)")
    print("=" * 70)
    print(results.filter(pl.col("quintile") == q)
          .select(["heattype_label", "n", "baseline_ep_prob",
                   "counterfactual_ep_prob", "delta", "p25", "p50", "p75"]))

print(f"\n\n{'=' * 70}")
print("DELTA RANGE BY QUINTILE (max - min counterfactual_ep_prob)")
print("=" * 70)
delta_range = (
    results
    .group_by("quintile")
    .agg([
        pl.col("counterfactual_ep_prob").max().alias("max_cf_ep"),
        pl.col("counterfactual_ep_prob").min().alias("min_cf_ep"),
        (pl.col("counterfactual_ep_prob").max() - pl.col("counterfactual_ep_prob").min()).alias("delta_range"),
        pl.col("baseline_ep_prob").first().alias("baseline_ep_prob"),
    ])
    .sort("quintile")
)
print(delta_range)

results.write_csv("tree_model_output/sensitivity_heattype_counterfactual.csv")
print("\nWrote tree_model_output/sensitivity_heattype_counterfactual.csv")
