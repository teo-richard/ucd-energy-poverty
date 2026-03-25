import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate = pl.read_csv("data/transitory/basic_clean_ahs_climate.csv")

sum(ahs_climate["energy_poverty"] == 1) / ahs_climate.height

corr = ahs_climate.select(cs.numeric()).to_pandas().corr()

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
