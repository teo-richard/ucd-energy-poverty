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


# ------ NCLIMGRID -----------------------------------------

cdd_2023_raw = pl.read_csv("data/raw/NClimGrid/2023_nclimgrid_cdd.csv")
cdd_2023 = (
    cdd_2023_raw
    .select("ID", "Name", "State", "Value")
)
hdd_2023_raw = pl.read_csv("data/raw/NClimGrid/2023_nclimgrid_hdd.csv")
hdd_2023 = (
    hdd_2023_raw
    .select("ID", "Name", "State", "Value")
)

cdd_hdd_2023 = cdd_2023.join(
    hdd_2023, on = ["ID", "Name", "State"], how = "inner"
)


cdd_hdd_2023 = cdd_hdd_2023.rename({
    "Value": "cdd_value",
    "Value_right": "hdd_value"
})


cdd_hdd_2023.head
cdd_hdd_2023.shape


cdd_hdd_2023_with_cbsa = cdd_hdd_2023.join(
    crosswalk, 
    left_on = ["Name", "State"],
    right_on = ["countycountyequivalent", "statename"],
    how = "left"
    )

cdd_hdd_2023_with_cbsa.head
cdd_hdd_2023_with_cbsa.shape

rural_counties = (
    cdd_hdd_2023_with_cbsa
    .filter(col("cbsacode").is_null())
)

cdd_hdd_2023_no_rural = (
    cdd_hdd_2023_with_cbsa
    .filter(~col("cbsacode").is_null())
)

cdd_hdd_2023_no_rural.shape
household.shape

# ------ Join cdd_hdd_2023_no_rural with AHS -----------------------------------------

# Turn the CBSA code in the climate data into a string so we can merge on this
cdd_hdd_2023_no_rural = cdd_hdd_2023_no_rural.with_columns(col("cbsacode").cast(str))


# In the AHS data, strip the quotes out of the CBSA codes
household = (
    household
    .with_columns(col("OMB13CBSA").str.replace_all("'", ""))
)

# 2. Aggregate the climate by metro area (i.e. by CBSA code)
#   If you don't do this then in the join, each county will have multiple rows from AHS attached to it
cdd_hdd_2023_no_rural = (
    cdd_hdd_2023_no_rural
    .group_by("cbsacode")
    .agg(
        col("cdd_value").mean().alias("cdd_value"),
        col("hdd_value").mean().alias("hdd_value")
    )
)


ahs_climate_joined = household.join(
    cdd_hdd_2023_no_rural,
    left_on = "OMB13CBSA", right_on = "cbsacode",
    how = "inner"
)

ahs_climate_joined.shape

ahs_climate_joined["OMB13CBSA"].n_unique() # 15 unique metro regions

# ------------------------------------------------------

print("\nScript ran successfully.\n")

csv_string = "data/transitory/ahs_climate_joined.csv"
ahs_climate_joined.write_csv(csv_string)

print(f"\nWrote file to \"{csv_string}\" \n")