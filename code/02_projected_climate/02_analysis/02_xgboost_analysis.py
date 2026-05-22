import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap, prepare_cat_cols, filter_temp_vars
import polars as pl

CAT_COLS_NO_CBSA = [  # Does not include SEWTYPE, ASECONDRY, or OMB13CBSA
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

YEARS = [2050]
COLS_TO_DROP = ["energy_deprivation", "year", "WEIGHT", "CONTROL"]
TEMP_RENAME = {
    "proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp",
    "proj_dtr": "dtr", "proj_HDD_approx": "HDD_approx", "proj_CDD_approx": "CDD_approx",
}

model_nc_w  = load_model("data/processed/models/current_climate_xgboost_no_cbsa_with_weights.pkl")
# model_nc_nw = load_model("data/processed/models/current_climate_xgboost_no_cbsa_without_weights.pkl")

for year in YEARS:
    print(f"\n\n{'=' * 60}")
    print(f"YEAR: {year}")
    print("=" * 60)

    data = pl.read_csv(f"data/processed/projected_climate/02_02_ahs_cmip_{year}.csv").rename(TEMP_RENAME)
    raw_pd = filter_temp_vars(data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas())

    cbsa = raw_pd["OMB13CBSA"].copy()
    X = prepare_cat_cols(raw_pd, CAT_COLS_NO_CBSA)[list(model_nc_w.get_booster().feature_names)]

    pred_probs_w = model_nc_w.predict_proba(X)[:, 1]
    mean_w = pred_probs_w.mean()
    print(f"WITH WEIGHTS model    — mean predicted EP probability: {mean_w:.4f}")
    run_shap(model_nc_w, X, name=f"proj_climate_xgboost_{year}", label=f"WITH WEIGHTS — {year}")

    out = pl.DataFrame({"CONTROL": data["CONTROL"], "pred_prob_xgb_w": pred_probs_w})
    out.write_csv(f"tree_model_output/xgboost_{year}_with_weights.csv")

    # mean_nw = model_nc_nw.predict_proba(X)[:, 1].mean()
    # print(f"WITHOUT WEIGHTS model — mean predicted EP probability: {mean_nw:.4f}")
