import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate = pl.read_csv("data/transitory/basic_clean_ahs_climate.csv")

