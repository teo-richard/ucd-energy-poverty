import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate_raw = pl.read_csv("data/interim/current_climate/01_02_01_basic_clean_ahs_climate.csv")

desc_df = ahs_climate_raw.describe()
desc_df.write_csv("data/interim/ahs_climate_descr.csv")