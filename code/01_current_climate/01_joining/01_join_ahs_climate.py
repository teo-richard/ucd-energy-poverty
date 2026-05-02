import polars as pl
from polars import col, lit, when


# ------ COUNTY TO CBSA CROSSWALK -----------------------------------------

crosswalk_raw = pl.read_csv("data/external/crosswalks/cbsa2fipsxw_2023.csv")
crosswalk = (
    crosswalk_raw
    .select("countycountyequivalent", "statename", "cbsacode")
)

# ------ AHS SURVEY -----------------------------------------

household_raw = pl.read_csv("data/raw/AHS 2023 National PUF v1.1 CSV/household.csv")


yearly_utils_cost = household_raw["UTILAMT"] * 12
# Yearly income is HINCP

household = (
    household_raw
    .with_columns(yearly_utils_cost.alias("yearly_utils_cost"))
)

household = household.with_columns(
    when(col("yearly_utils_cost") > 0.1 * col("HINCP"))
    .then(1)
    .otherwise(0)
    .alias("energy_poverty")
)




# ------ NOAA -----------------------------------------

noaa_combined = pl.read_csv("data/interim/current_climate/01_01_01_noaa_combined.csv")

noaa_combined_with_cbsa = noaa_combined.join(
    crosswalk, 
    left_on = ["Name", "State"],
    right_on = ["countycountyequivalent", "statename"],
    how = "left"
    )


rural_counties = (
    noaa_combined_with_cbsa
    .filter(col("cbsacode").is_null())
)

noaa_combined_no_rural = (
    noaa_combined_with_cbsa
    .filter(~col("cbsacode").is_null())
)

noaa_combined_no_rural.shape
household.shape

# ------ Join climate with AHS -----------------------------------------

# Turn the CBSA code in the climate data into a string so we can merge on this
noaa_combined_no_rural = noaa_combined_no_rural.with_columns(col("cbsacode").cast(str))


# In the AHS data, strip the quotes out of the CBSA codes
household = (
    household
    .with_columns(col("OMB13CBSA").str.replace_all("'", ""))
)

# 2. Aggregate the climate by metro area (i.e. by CBSA code)
#   If you don't do this then in the join, each county will have multiple rows from AHS attached to it
noaa_combined_no_rural = (
    noaa_combined_no_rural
    .group_by("cbsacode")
    .agg(
        col("mintemp").mean().alias("mintemp"),
        col("maxtemp").mean().alias("maxtemp"),
        col("avgtemp").mean().alias("avgtemp")
    )
)


ahs_climate_joined = household.join(
    noaa_combined_no_rural,
    left_on = "OMB13CBSA", right_on = "cbsacode",
    how = "inner"
)

ahs_climate_joined.shape

ahs_climate_joined["OMB13CBSA"].n_unique() # 15 unique metro regions

# ------------------------------------------------------

print("\nScript ran successfully.\n")

csv_string = "data/interim/current_climate/01_01_02_joined_ahs_climate.csv"
ahs_climate_joined.write_csv(csv_string)

print(f"\nWrote file to \"{csv_string}\" \n")