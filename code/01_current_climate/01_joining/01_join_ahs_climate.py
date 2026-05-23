import polars as pl
from polars import col, lit, when
import polars.selectors as cs


# ------ COUNTY TO CBSA CROSSWALK -----------------------------------------

crosswalk_raw = pl.read_csv("data/external/crosswalks/cbsa2fipsxw_2023.csv")
crosswalk = (
    crosswalk_raw
    .select("countycountyequivalent", "statename", "cbsacode")
)

# ------ AHS SURVEY -----------------------------------------

household_raw = pl.read_csv("data/raw/AHS 2023 National PUF v1.1 CSV/household.csv")


household = (
    household_raw
    .with_columns(
        when((col("HOT") == "'1'") | (col("COLD") == "'1'"))
        .then(1)
        .otherwise(0)
        .alias("energy_deprivation")
    )
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


# ------ AHS PERSON-LEVEL DEMOGRAPHICS -----------------------------------------
# Aggregate person.csv to the household level and merge onto the joined dataset.
#
# Householder-level demographics (age, sex, race, Hispanic origin, education,
# marital status) are NOT extracted here because household.csv already contains
# exact equivalents — HHAGE, HHSEX, HHRACE, HHSPAN, HHGRAD, HHMAR — for the
# reference person (confirmed r = 1.0 in the correlation-check step).
# Adding them from person.csv would create perfect duplicates.
# Similarly, n_persons duplicates NUMPEOPLE from household.csv.
#
# Only the three variables below are genuinely novel (no equivalent in household.csv):
#   • max_age      — age of the oldest household member; differs from HHAGE when
#                    an elderly parent/grandparent lives with a younger householder.
#   • has_children — flag for any member under 18 (vulnerability indicator).
#   • any_hispanic — flag for any member identifying as Hispanic; differs from
#                    HHSPAN when the householder is non-Hispanic but another
#                    member is Hispanic (r ≈ −0.91, not a perfect duplicate).
#
# Other exclusions:
#   • RACEAS / RACEPI: only filled for Asian/PI persons → >10% nulls → auto-dropped.
#   • J-flags (imputation indicators): dropped by the cleaning step anyway.
#   • Boolean flags cast to Int8 (0/1) for downstream model compatibility.
#   • CONTROL kept quoted to match ahs_climate_joined (cleaning step strips later).

person_raw = pl.read_csv(
    "data/raw/AHS 2023 National PUF v1.1 CSV/person.csv",
    columns=["CONTROL", "AGE", "SPAN"],
)

# Strip quotes from SPAN; cast AGE to Int32 and null-out negative sentinel codes.
person = (
    person_raw
    .with_columns(
        col("SPAN").str.replace_all("'", ""),
        col("AGE").cast(pl.Int32, strict=False),
    )
    .with_columns(
        when(col("AGE") < 0).then(None).otherwise(col("AGE")).alias("AGE"),
    )
)

# --- Household-level aggregates (all members) — three novel variables ---
person_features = (
    person
    .group_by("CONTROL")
    .agg(
        col("AGE").max().alias("max_age"),
        (col("AGE") < 18).any().cast(pl.Int8).alias("has_children"),
        (col("SPAN") == "1").any().cast(pl.Int8).alias("any_hispanic"),
    )
)

# Left-join onto the main dataset; CONTROL keys match because both retain AHS quoting.
ahs_climate_joined = ahs_climate_joined.join(person_features, on="CONTROL", how="left")

# Diagnostic: null rate for max_age among completed interviews should be ~0%.
_completed = ahs_climate_joined.filter(col("INTSTATUS") == "'1'")
_null_rate = _completed["max_age"].null_count() / _completed.height
print(f"  Person merge complete. max_age null rate (INTSTATUS='1' only) = {_null_rate:.1%} (expect 0%)")

# ------------------------------------------------------

print("\nScript ran successfully.\n")

csv_string = "data/interim/current_climate/01_01_02_joined_ahs_climate.csv"
ahs_climate_joined.write_csv(csv_string)

print(f"\nWrote file to \"{csv_string}\" \n")