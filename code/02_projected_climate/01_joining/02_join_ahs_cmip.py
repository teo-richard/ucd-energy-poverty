import polars as pl
import polars.selectors as cs
from polars import col, lit, when


# Important Note: this will use the data from current climate joined with ahs
#   I'll just drop the current climate vars and put in the projected climate vars
#   This is because the AHS data is already cleaned but was cleaned after joining and I don't want to go refactor it

ahs_with_noaa = pl.read_csv("data/processed/current_climate/01_02_05_basic_ready_for_trees_ahs_climate.csv")
proj_climate = pl.read_csv("data/interim/projected_climate/02_01_cmip6_loca2_tas_vars.csv")

crosswalk_raw = pl.read_csv("data/external/crosswalks/cbsa2fipsxw_2023.csv")
# Build 5-digit county FIPS (GEOID) to match CMIP6; countycountyequivalent is a name string, not a code
crosswalk = (
    crosswalk_raw
    .with_columns(
        (pl.col("fipsstatecode") * 1000 + pl.col("fipscountycode")).alias("GEOID")
    )
    .select("GEOID", "cbsacode")
)


ahs_no_noaa = ahs_with_noaa.drop("mintemp", "maxtemp", "avgtemp")

# Join county-level CMIP6 data to crosswalk to get CBSA codes
proj_climate_with_cbsa = proj_climate.join(
    crosswalk,
    on="GEOID",
    how="left"
)

# Aggregate from county level to CBSA level, keeping year dimension
proj_climate_cbsa = (
    proj_climate_with_cbsa
    .group_by("year", "cbsacode")
    .agg(
        pl.col("tasmin").mean().alias("proj_tasmin"),
        pl.col("tasmax").mean().alias("proj_tasmax"),
        pl.col("tas").mean().alias("proj_tas"),
    )
)

# --- Joining the years I want ---

YEARS = [2030, 2040, 2050]

def join_ahs_with_cmip(years):
    ahs_cmip_joined_YEARS = {}
    for year in years:
        proj_year = proj_climate_cbsa.filter(col("year") == year)
        ahs_cmip_joined = ahs_no_noaa.join(
            proj_year,
            left_on="OMB13CBSA", right_on="cbsacode",
            how="inner"
        )
        ahs_cmip_joined.write_csv(f"data/processed/projected_climate/02_02_ahs_cmip_{year}.csv")
        ahs_cmip_joined_no_cbsa = ahs_cmip_joined.drop("OMB13CBSA")
        ahs_cmip_joined_no_cbsa.write_csv(f"data/processed/projected_climate/02_02_ahs_cmip_{year}_no_cbsa.csv")

    return 


join_ahs_with_cmip(YEARS)
