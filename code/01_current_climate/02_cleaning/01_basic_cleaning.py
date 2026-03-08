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


# ---- Imports and load data ----
import polars as pl
import polars.selectors as cs
from polars import col, lit, when

ahs_climate_raw = pl.read_csv("data/transitory/ahs_climate_joined.csv")



# --------------------------------------------------------------------------------

# ---- Drop columns ----
# WEIGHT|WGT are columns telling you how to replicate the weights
# J* flags tell you how variables were recorded
# Columns we don't want in our model (leak, mold)
# Control variables
# ---- Filter rows ----
# Keep only occupied interview (INTSTATUS == 1)
ahs_climate = (
    ahs_climate_raw
    .with_columns(cs.string().str.replace_all("'", ""))
    .select(cs.all() - cs.matches("(?i)(WEIGHT|WGT|^J\\w|CONTROL)"))
    .select(cs.all() - cs.matches("(?i)LEAK|MOLD"))
    .filter(
        col("INTSTATUS") == "1"
    )
    .drop("UTILAMT", "INTSTATUS", "ELECAMT", "GASAMT", "OILAMT", "TRASHAMT", "WATERAMT", "SPLITSAMP")
)

# Now drop low variance columns

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


# Drop columns with too many null values
null_threshold = 0.2
too_many_nulls = [
    c for c in ahs_climate.columns if ahs_climate[c].null_count()
    / ahs_climate.height > null_threshold
]

ahs_climate = (
    ahs_climate
    .drop(too_many_nulls)
)

# Drop columns that are only in 2023 year
only_2023 = ["SOGIRESP", "HHSOGILGBT", "HHSOGISO", "HHSOGIG", "HHGEN"]
ahs_climate = ahs_climate.drop(only_2023)


# Drop rows in the outcome column that are NA


# Write the data

ahs_climate.write_csv("data/transitory/basic_clean_ahs_climate.csv")


print("\nRan script successfully.\n")

