import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate = pl.read_csv("data/transitory/01_02_02_data_checked_ahs_climate.csv")


# ---------- Creating Featues ----------

# --- Feature: PARFOREIGNCOUNTRY ---
# 1 if at least one parent was born in a foreign country (i.e. not in US aka code 057), 0 otherwise
# Parent variables: HHFNTVTY (birth country of father), HHMNTVTY (birth country of mother), and HHNATVTY (birth country of householder)

ahs_climate = (
    ahs_climate
    .with_columns(
        when(col("HHFNTVTY") == 57).then(lit(1))
        .when(col("HHMNTVTY") == 57).then(lit(1))
        .otherwise(lit(0))
        .alias("PARFOREIGNCOUNTRY")
    )
)


# --- Feature: SAMECOUNTRY ---
# 1 if householder was born in same country as at least one parent's birth country, 0 otherwise
# Parent variables: HHFNTVTY (birth country of father), HHMNTVTY (birth country of mother), and HHNATVTY (birth country of householder)

ahs_climate = (
    ahs_climate
    .with_columns(
        when((col("HHFNTVTY") == col("HHNATVTY")) | (col("HHMNTVTY") == col("HHNATVTY"))).then(lit(1))
        .otherwise(lit(0))
        .alias("HHSAMECOUNTRY")
    )
)

# --- Feature: PARSAMECOUNTRY --- 
# 1 if parents born in same country as each other, 0 otherwise
# Parent variables: HHFNTVTY (birth country of father) and HHMNTVTY (birth country of mother)

ahs_climate = (
    ahs_climate
    .with_columns(
        when(col("HHFNTVTY") == col("HHMNTVTY")).then(lit(1))
        .otherwise(lit(0))
        .alias("PARSAMECOUNTRY")
    )
)



# ---------- Drop source variables ----------
ahs_climate = ahs_climate.drop(
    ("HHFNTVTY", "HHMNTVTY", "HHNATVTY")
)

# --------------------------------------------------------------------------------
print("\nRan script successfully.")
print(f"Data shape: {ahs_climate.shape}")

# --- Write the data ---
csv_string = "data/transitory/01_02_03_features_engineered_ahs_climate.csv"
print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)



print(f"\nData written to: \"{csv_string}\"\n\n")