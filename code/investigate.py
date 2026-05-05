import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

current = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate.csv")
proj = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")


print("Current temps:")
print(current[["mintemp", "maxtemp", "avgtemp"]].describe())

print("Projected temps:")
print(proj[["proj_tasmin", "proj_tasmax", "proj_tas"]].describe())