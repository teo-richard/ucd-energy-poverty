import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate = pl.read_csv("data/interim/current_climate/01_02_04_features_engineered_ahs_climate.csv")

sum(ahs_climate["energy_poverty"] == 1) / ahs_climate.height

corr = ahs_climate.to_pandas().corr()

corr = pl.from_pandas(corr.reset_index().rename(columns={"index": "var1"}))

corr = (
    corr
    .unpivot(index="var1", variable_name="var2", value_name="correlation")
    .filter(pl.col("var1") < pl.col("var2"))
    .filter(pl.col("correlation").abs() > 0.8)
    .sort("correlation", descending=True)
)

with pl.Config(tbl_rows=-1):
    print(corr)


# --- Dropping variables that are measuring the same thing ---
# See project_notes.md for information

variables_to_drop = ["NUMPEOPLE", "FINCP", "NUMYNGKIDS"] 


ahs_climate = (
    ahs_climate
    .drop(variables_to_drop)
)


# --------------------------------------------------------------------------------
print("\nRan script successfully.")
print(f"Data shape: {ahs_climate.shape}")

# --- Write the data ---
csv_string = "data/interim/current_climate/01_02_05_corr_checked_ahs_climate.csv"
print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)


print(f"\nData written to: \"{csv_string}\"\n\n")