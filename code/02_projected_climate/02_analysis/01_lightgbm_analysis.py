import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, run_shap
import polars as pl

YEARS = [2050]
COLS_TO_DROP = ["energy_poverty", "year", "WEIGHT"]
TEMP_RENAME = {"proj_tasmin": "mintemp", "proj_tasmax": "maxtemp", "proj_tas": "avgtemp"}

model_w     = load_model("data/processed/models/current_climate_lightgbm_with_weights.pkl")
model_nw    = load_model("data/processed/models/current_climate_lightgbm_without_weights.pkl")
model_nc_w  = load_model("data/processed/models/current_climate_lightgbm_no_cbsa_with_weights.pkl")
model_nc_nw = load_model("data/processed/models/current_climate_lightgbm_no_cbsa_without_weights.pkl")

# --- With CBSA ---
for year in YEARS:
    print(f"\n\n{'=' * 60}")
    print(f"YEAR: {year}")
    print("=" * 60)

    data = pl.read_csv(f"data/processed/projected_climate/02_02_ahs_cmip_{year}.csv").rename(TEMP_RENAME)
    X = data.drop([c for c in COLS_TO_DROP if c in data.columns]).to_pandas()

    mean_w = model_w.predict_proba(X)[:, 1].mean()
    print(f"WITH WEIGHTS model    — mean predicted EP probability: {mean_w:.4f}")

    mean_nw = model_nw.predict_proba(X)[:, 1].mean()
    print(f"WITHOUT WEIGHTS model — mean predicted EP probability: {mean_nw:.4f}")

# --- Without CBSA (2050 only) ---
print(f"\n\n{'=' * 60}")
print("YEAR: 2050 — no CBSA")
print("=" * 60)

data_nc = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050_no_cbsa.csv").rename(TEMP_RENAME)
X_nc = data_nc.drop([c for c in COLS_TO_DROP if c in data_nc.columns]).to_pandas()

mean_nc_w = model_nc_w.predict_proba(X_nc)[:, 1].mean()
print(f"WITH WEIGHTS model    — mean predicted EP probability: {mean_nc_w:.4f}")

mean_nc_nw = model_nc_nw.predict_proba(X_nc)[:, 1].mean()
print(f"WITHOUT WEIGHTS model — mean predicted EP probability: {mean_nc_nw:.4f}")
