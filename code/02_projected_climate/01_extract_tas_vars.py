import numpy as np
import xarray as xr
import polars as pl

NC_PATH = "data/raw/climate projection/CMIP6-LOCA2_1950-2100_County2023_monthly.nc"

# ------ LOAD ------------------------------------------------------------------

# chunks="auto" lets dask pick chunk sizes; data stays on disk until .values is called
ds = xr.open_dataset(NC_PATH, mask_and_scale=True, chunks={"time": 240, "ensemble": 36, "GEOID": 150})

exp_id_vals = ds["experiment_id"].values  # shape (72,), already decoded strings

# ------ EXTRACT tas, tasmin, tasmax ------------------------------------------

# Shape: (time=1812, ensemble=72, GEOID=3109)
# Available scenarios: "ssp245", "ssp370", "ssp585"

SCENARIO = "ssp585"

ensemble_mask = exp_id_vals == SCENARIO
ds_scen = ds[["tasmin", "tasmax", "tas"]].isel(ensemble=ensemble_mask).sel(
    time=slice("2027-01-01", "2050-12-31")
)

# Stream one time chunk at a time to avoid loading the full array into memory
TIME_CHUNK = 240  # match native NetCDF chunk size
frames = []

for t_start in range(0, ds_scen.sizes["time"], TIME_CHUNK):
    chunk = ds_scen.isel(time=slice(t_start, t_start + TIME_CHUNK))
    chunk_loaded = chunk[["tasmin", "tasmax", "tas"]].load()

    time_vals = chunk_loaded.coords["time"].values
    ensemble_vals = chunk_loaded.coords["ensemble"].values.astype(str)
    geoid_vals = chunk_loaded.coords["GEOID"].values.astype(str)

    t, e, g = np.meshgrid(time_vals, ensemble_vals, geoid_vals, indexing="ij")

    frames.append(pl.DataFrame({
        "date": pl.Series(t.ravel().astype("datetime64[ms]")).cast(pl.Date),
        "ensemble": pl.Series(e.ravel()),
        "GEOID": pl.Series(g.ravel()),
        "tasmin": pl.Series(chunk_loaded["tasmin"].values.ravel()),
        "tasmax": pl.Series(chunk_loaded["tasmax"].values.ravel()),
        "tas": pl.Series(chunk_loaded["tas"].values.ravel()),
    }))
    print(f"  loaded months {t_start}–{min(t_start + TIME_CHUNK, ds_scen.sizes['time']) - 1}")

df = pl.concat(frames)

df = df.drop("ensemble")
df.write_csv("processed/projected_climate/cmip6_locs2_tax_vars.csv")