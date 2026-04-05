"""
# This script:
# DROPS COLUMNS
    # WEIGHT|WGT
    # J* flags
    # Columns we don't want in our models
    # Control variables
    # Interview status other than occupied
    # Low variance columns
    # Too many null values
    # Columns that only exist in 2023
"""


# --- Imports and load data ---
import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate_raw = pl.read_csv("data/interim/current_climate/01_01_01_joined_ahs_climate.csv")



# --------------------------------------------------------------------------------

# --- Drop columns ---
# NOTE: KEEP CONTROL UNTIL END IN CASE WE WANT TO ADD SOME COLUMN BACK
# NOTE: KEEP 
# WEIGHT|WGT are columns telling you how to replicate the weights
# J* flags tell you how variables were recorded
# Columns we don't want in our model (leak, mold)
# ---- Filter rows ----
# Keep only occupied interview (INTSTATUS == 1)
ahs_climate = (
    ahs_climate_raw
    .with_columns(cs.string().str.replace_all("'", ""))
    .select((cs.all() - cs.matches("(?i)(WEIGHT|WGT|^J)")) | cs.matches("(?i)^WEIGHT$")) # Get the weight column we want back in
    .select(cs.all() - cs.matches("(?i)LEAK|MOLD"))
    .filter(
        col("INTSTATUS") == "1"
    )
    .drop("UTILAMT", "INTSTATUS", "ELECAMT", "GASAMT", "OILAMT", "TRASHAMT", "WATERAMT", "SPLITSAMP")
)

# --- Drop low variance columns ---

low_var_threshold = 0.95
low_var_cols = [
    c for c in ahs_climate.columns
    if ahs_climate[c].value_counts().sort("count", descending=True)["count"][0] 
    / ahs_climate.height > low_var_threshold
]

ahs_climate = (
    ahs_climate
    .drop(low_var_cols)
    .with_columns(cs.string().replace(["-6", "-7", "-8", "-9"], None)) # String columns
    .with_columns(
        when(cs.numeric().is_in([-6, -7, -8, -9]))
        .then(None)
        .otherwise(cs.numeric())
        .name.keep()
    )
)

ahs_climate.null_count()


# --- Drop columns with too many null values ---
null_threshold = 0.2
too_many_nulls = [
    c for c in ahs_climate.columns if ahs_climate[c].null_count()
    / ahs_climate.height > null_threshold
]

ahs_climate = (
    ahs_climate
    .drop(too_many_nulls)
)

# --- Drop columns that are only in 2023 year ---
only_2023 = ["SOGIRESP", "HHSOGILGBT", "HHSOGISO", "HHSOGIG", "HHGEN"]
ahs_climate = ahs_climate.drop(only_2023)


# --- Drop rows in the outcome column that are NA ---
#print(ahs_climate.height) # 26700
ahs_climate = (
    ahs_climate
    .filter(col("energy_poverty").is_not_null())
)
#print(ahs_climate.height) #26700, so no rows removed

print("Energy Poverty variable: \n")
print(ahs_climate.select(col("energy_poverty").value_counts(sort=True)))
# About 17% are in energy poverty by my definition (4581 in EP, 22119 not in EP)


# --- Turn columns to numeric to make them easier to work with in the future ---
var = "HHPRNTHOME"
ahs_climate[var].dtype == pl.String
ahs_climate[var].dtype == pl.Float64

is_i64_float64 = []
not_converted = []
colnames = ahs_climate.columns
for var in colnames:
    if ahs_climate[var].dtype == pl.String:
        try:
            ahs_climate = ahs_climate.with_columns(
                col(var).str.to_integer().alias(var)
            )
            is_i64_float64.append(var)
        except:
            print(f"\nError occurred with variable ** {var} ** \n")
    elif (ahs_climate[var].dtype == pl.Int64) | (ahs_climate[var].dtype == pl.Float64):
        is_i64_float64.append(var)
    else:
        not_converted.append(var) # None in this list!

for var in colnames:
    if (ahs_climate[var].dtype != pl.Int64) & (ahs_climate[var].dtype != pl.Float64):
        print(var)
# Does not print anything therefore all our variables are encoded as numeric
# All we did was strip everything of the string encoding since it was all numbers encoded as strings
# There were no awkward variables so didn't have to do any extra work :)


# --- Quick fix of PERPOVLVL ---
ahs_climate = ahs_climate.with_columns(
    pl.col("PERPOVLVL")
    .replace({1: 0, 501: 501})
)

# --------------------------------------------------------------------------------
print("\nRan script successfully.")
print(f"Data shape: {ahs_climate.shape}")

# --- Write the data ---
csv_string = "data/interim/current_climate/01_02_01_basic_clean_ahs_climate.csv"
print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)


print(f"\nData written to: \"{csv_string}\"\n\n")

