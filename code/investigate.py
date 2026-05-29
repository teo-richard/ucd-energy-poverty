import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

current = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate.csv")
proj = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")


print("Current temps:")
current_temp_summary = current[["mintemp", "maxtemp", "avgtemp"]].describe()
print(current_temp_summary)
current_temp_summary.write_csv("output/current_temp_summary.csv")

print("Projected temps:")
proj_temp_summary = proj[["proj_tasmin", "proj_tasmax", "proj_tas"]].describe()
print(proj_temp_summary)
proj_temp_summary.write_csv("output/projected_temp_summary.csv")