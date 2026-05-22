import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap, filter_temp_vars
import polars as pl

YEARS = [2050]
COLS_TO_DROP = ["energy_deprivation", "year", "WEIGHT", "CONTROL"]
TEMP_RENAME = {
    "proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp",
    "proj_dtr": "dtr", "proj_HDD_approx": "HDD_approx", "proj_CDD_approx": "CDD_approx",
}

model_nc_w  = load_model("data/processed/models/current_climate_lightgbm_no_cbsa_with_weights.pkl")
# model_nc_nw = load_model("data/processed/models/current_climate_lightgbm_no_cbsa_without_weights.pkl")

for year in YEARS:
    print(f"\n\n{'=' * 60}")
    print(f"YEAR: {year}")
    print("=" * 60)

    data = pl.read_csv(f"data/processed/projected_climate/02_02_ahs_cmip_{year}.csv").rename(TEMP_RENAME)
    raw_pd = filter_temp_vars(data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas())

    cbsa = raw_pd["OMB13CBSA"].copy()
    X = raw_pd.drop(columns=["OMB13CBSA"])

    mean_w = model_nc_w.predict_proba(X)[:, 1].mean()
    print(f"WITH WEIGHTS model    — mean predicted EP probability: {mean_w:.4f}")

    # mean_nw = model_nc_nw.predict_proba(X)[:, 1].mean()
    # print(f"WITHOUT WEIGHTS model — mean predicted EP probability: {mean_nw:.4f}")
