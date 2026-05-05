import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate_raw = pl.read_csv("data/interim/current_climate/01_02_04_cat_collapsed_ahs_climate.csv")

ahs_climate_excl = (
    ahs_climate_raw
    # utils cost flags
    .filter(~((col("TENURE") == 2) & (col("yearly_utils_cost") == 0))) # exclude renters who don't explicitely pay utilities
    .filter(~((col("TENURE") == 1) & (col("yearly_utils_cost") == 0))) # exclude homeowners with 0 utility cost
    .filter(col("CONTROL") != 11034668) # extreme outlier (yearly util cost = 18960 but income is 38k)
    # income flags
    .filter(col("HINCP") != 0) # Zero income
    .filter(col("HINCP") >= 4999) # Potential noise, exclude to be safe
    .filter(col("yearly_utils_cost") <= col("HINCP")) # Impossible
    .filter(col("HINCP") <= 848000) # implausible, high income
)



ahs_climate_excl.write_csv("data/interim/current_climate/05_ahs_excl.csv")