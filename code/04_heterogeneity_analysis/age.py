"""
age.py — Heterogeneity analysis by age of householder (HHAGE).

HHAGE is continuous, so it is binned into four cohorts before analysis.
The binned column is attached to df_pl only; it does not affect X_pd or the model.

Age cohorts:
  1_Under35  — householder under 35
  2_35to49   — 35–49
  3_50to64   — 50–64
  4_65plus   — 65 and older

Run from the project root:
    python code/04_heterogeneity_analysis/age.py
"""

import sys
import polars as pl
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import load_data, run_heterogeneity

GROUP_COL = "age_group"

df, X_pd, model = load_data()

df = df.with_columns(
    pl.when(pl.col("HHAGE") < 35).then(pl.lit("1_Under35"))
    .when(pl.col("HHAGE") < 50).then(pl.lit("2_35to49"))
    .when(pl.col("HHAGE") < 65).then(pl.lit("3_50to64"))
    .otherwise(pl.lit("4_65plus"))
    .alias(GROUP_COL)
)

run_heterogeneity(GROUP_COL, df=df, X_pd=X_pd, model=model)
