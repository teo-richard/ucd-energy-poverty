import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate_raw = pl.read_csv("data/interim/current_climate/01_02_01_basic_clean_ahs_climate.csv")

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
    # issues with room variables
    .filter(~(col("TOTROOMS") < (col("BEDROOMS") + col("BATHROOMS")))) # total rooms can't be less than bedrooms plus bathrooms
    .filter(col("FINROOMS") != 0)
    .filter(col("BEDROOMS") != 0)
    # other
    .filter(col("YRBUILT") < 2024) # sanity check

)



ahs_climate_excl.write_csv("data/interim/current_climate/01_02_03_basic_excl_ahs_climate.csv")