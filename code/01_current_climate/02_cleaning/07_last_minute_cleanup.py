import polars as pl
import polars.selectors as cs
from polars import col, lit, when

ahs_climate = pl.read_csv("data/interim/current_climate/01_02_06_cat_collapsed_ahs_climate.csv")

ahs_climate = ahs_climate.drop(["TOTHCAMT", "INSURAMT"])

# Drop yearly utils cost because it will cause data leakage (mechanically related to energy poverty)
ahs_climate = ahs_climate.drop("yearly_utils_cost")


# --------------------------------------------------------------------------------
csv_string = "data/processed/current_climate/basic_ready_for_trees_ahs_climate.csv"
import polars as pl


# Getting one without the CBSA as those are correlated with temp variables

ahs_climate_no_cbsa = ahs_climate.drop("OMB13CBSA")
ahs_climate_no_cbsa.write_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate_no_cbsa.csv")


print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)
print(f"Data written to: \"{csv_string}\"\n")
