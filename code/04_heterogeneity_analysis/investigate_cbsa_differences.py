"""
investigate_cbsa_differences.py
================================
Descriptive statistics for a user-defined list of variables, stratified by CBSA.

Output: output/heterogeneity_analysis/cbsa_stats.csv
        one row per (CBSA, variable) with columns: n, min, max, mean, median, variance

Run from the project root:
    python code/04_heterogeneity_analysis/investigate_cbsa_differences.py
"""

import os
import sys
import polars as pl

sys.path.insert(0, "code/00_shared")
sys.path.insert(0, "code/04_heterogeneity_analysis")

from heterogeneity_utils import load_df

# ---------------------------------------------------------------------------
# Edit this list to add/remove variables (use exact column names from the CSV)
# ---------------------------------------------------------------------------
VARIABLES: list[str] = [
    "HHAGE",       # age
    "DISHH",
    "HEATTYPE",
    "HHRACE",
    "TENURE",
    "ACPRIMARY",
    "FUSEBLOW",
    "ROACH",
    "WALLCRACK",
]

CBSA_COL    = "OMB13CBSA"
OUTPUT_PATH = "output/heterogeneity/cbsa_stats.csv"

# ---------------------------------------------------------------------------
# Load data and validate
# ---------------------------------------------------------------------------
df = load_df()

missing = [v for v in VARIABLES if v not in df.columns]
if missing:
    raise ValueError(f"Variables not found in data: {missing}")

# ---------------------------------------------------------------------------
# Compute per-CBSA stats for each variable
# ---------------------------------------------------------------------------
numeric_vars = [v for v in VARIABLES if df[v].dtype not in (pl.Utf8, pl.String, pl.Categorical)]
skipped_vars = [v for v in VARIABLES if v not in numeric_vars]

if skipped_vars:
    print(f"Skipping non-numeric variables: {skipped_vars}")

agg_exprs = []
for var in numeric_vars:
    agg_exprs += [
        pl.col(var).count().alias(f"{var}__n"),
        pl.col(var).min().alias(f"{var}__min"),
        pl.col(var).max().alias(f"{var}__max"),
        pl.col(var).mean().alias(f"{var}__mean"),
        pl.col(var).median().alias(f"{var}__median"),
        pl.col(var).var().alias(f"{var}__variance"),
    ]

wide = df.group_by(CBSA_COL).agg(agg_exprs).sort(CBSA_COL)

# Unpivot to long format: one row per (CBSA, variable)
stat_names = ["n", "min", "max", "mean", "median", "variance"]

rows = []
for var in numeric_vars:
    chunk = wide.select(
        [pl.col(CBSA_COL)]
        + [pl.col(f"{var}__{s}").alias(s) for s in stat_names]
    ).with_columns(pl.lit(var).alias("variable"))
    rows.append(chunk)

long = (
    pl.concat(rows)
    .select([CBSA_COL, "variable"] + stat_names)
    .sort([CBSA_COL, "variable"])
)

# ---------------------------------------------------------------------------
# Pivot to wide mean table: one row per CBSA, one column per variable
# ---------------------------------------------------------------------------
mean_wide = (
    long.filter(pl.col("variable").is_in(numeric_vars))
    .select([CBSA_COL, "variable", "mean"])
    .pivot(index=CBSA_COL, on="variable", values="mean")
    .sort(CBSA_COL)
)

MEAN_TABLE_PATH = OUTPUT_PATH.replace("cbsa_stats.csv", "cbsa_mean_table.csv")

# ---------------------------------------------------------------------------
# Save and print summary
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
long.write_csv(OUTPUT_PATH)
mean_wide.write_csv(MEAN_TABLE_PATH)

print(f"Saved {len(long)} rows ({long[CBSA_COL].n_unique()} CBSAs × {len(numeric_vars)} variables) → {OUTPUT_PATH}")
print(f"Saved mean table ({mean_wide.shape[0]} CBSAs × {mean_wide.shape[1] - 1} variables) → {MEAN_TABLE_PATH}")
print()
print("\nMean by CBSA:")
print(mean_wide)
