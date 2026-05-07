import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

# Official Census 2023 poverty thresholds
# Structure: {household_size: {num_children: threshold}}
POVERTY_THRESHOLDS_2023 = {
    1: {
        0: 15003,
    },
    2: {
        0: 19678,
        1: 20609,
    },
    3: {
        0: 23834,
        1: 24517,
        2: 24537,
    },
    4: {
        0: 30900,
        1: 31422,
        2: 30451,
        3: 30771,
    },
    5: {
        0: 37676,
        1: 38176,
        2: 36908,
        3: 35760,
        4: 34325,
    },
    6: {
        0: 43938,
        1: 44530,
        2: 43433,
        3: 42209,
        4: 40325,
        5: 37262,
    },
    7: {
        0: 50453,
        1: 51076,
        2: 49868,
        3: 48760,
        4: 46799,
        5: 43682,
        6: 40414,
    },
    8: {
        0: 56338,
        1: 57006,
        2: 55826,
        3: 54710,
        4: 52762,
        5: 49742,
        6: 46484,
        7: 46074,
    },
    9: {
        0: 67142,
        1: 67951,
        2: 66735,
        3: 65575,
        4: 63595,
        5: 60475,
        6: 57286,
        7: 56823,
        8: 54935,
    },
}

def lookup_poverty_threshold(household_size: int, num_children: int) -> float:
    size = min(int(household_size), 9)
    size_thresholds = POVERTY_THRESHOLDS_2023[size]
    max_children = max(size_thresholds.keys())
    children = min(int(num_children), max_children)
    return float(size_thresholds.get(children, size_thresholds[0]))


ahs_climate = pl.read_csv("data/interim/current_climate/01_02_03_basic_excl_ahs_climate.csv")


# ---------- Creating Features ----------

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

# --- Feature: constr_PERPOVLVL ---
# Reconstruct uncapped poverty ratio; published PERPOVLVL is top-coded at 501.
# Uses 2023 Census poverty thresholds keyed by household size and children under 18.
# NUMPEOPLE = total household size (available here, dropped later in 03_correlation_checks)
# Children under 18 = HHOLDKIDS (6-17) + HHYNGKIDS (under 6)

ahs_climate = ahs_climate.with_columns([
    pl.struct(["NUMPEOPLE", "HHOLDKIDS", "HHYNGKIDS"])
    .map_elements(
        lambda x: lookup_poverty_threshold(
            x["NUMPEOPLE"],
            x["HHOLDKIDS"] + x["HHYNGKIDS"],
        ),
        return_dtype=pl.Float64
    )
    .alias("poverty_threshold_2023"),
])

ahs_climate = ahs_climate.with_columns([
    (pl.col("HINCP") / pl.col("poverty_threshold_2023") * 100)
    .alias("constr_PERPOVLVL"),
])

# Validate against the published PERPOVLVL for non-top-coded households
validation = ahs_climate.filter(pl.col("PERPOVLVL") < 501).select([
    "HINCP",
    "NUMPEOPLE",
    "HHOLDKIDS",
    "HHYNGKIDS",
    "PERPOVLVL",
    "constr_PERPOVLVL",
    "poverty_threshold_2023",
]).with_columns([
    (pl.col("constr_PERPOVLVL") - pl.col("PERPOVLVL"))
    .abs()
    .alias("abs_difference")
])

print("\nValidation vs published PERPOVLVL (non-top-coded rows):")
print(validation.select([
    pl.col("abs_difference").mean().alias("mean_abs_error"),
    pl.col("abs_difference").median().alias("median_abs_error"),
    pl.col("abs_difference").max().alias("max_abs_error"),
]))

print("\nTop-coded households (PERPOVLVL == 501) — constr_PERPOVLVL uncapped:")
print(
    ahs_climate.filter(pl.col("PERPOVLVL") == 501)
    .select(["HINCP", "PERPOVLVL", "constr_PERPOVLVL"])
    .describe()
)


# --- Features: DTR, HDD_approx, CDD_approx ---
ahs_climate = ahs_climate.with_columns([
    (pl.col("maxtemp") - pl.col("mintemp")).alias("dtr"),
    ((65 - pl.col("avgtemp")).clip(lower_bound=0) * 365).alias("HDD_approx"),
    ((pl.col("avgtemp") - 65).clip(lower_bound=0) * 365).alias("CDD_approx"),
])

# ---------- Drop source variables ----------
ahs_climate = ahs_climate.drop(
    ("HHFNTVTY", "HHMNTVTY", "HHNATVTY", "poverty_threshold_2023", "PERPOVLVL")
)

# --------------------------------------------------------------------------------
print("\nRan script successfully.")
print(f"Data shape: {ahs_climate.shape}")

# --- Write the data ---
csv_string = "data/interim/current_climate/01_02_04_features_engineered_ahs_climate.csv"
print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)



print(f"\nData written to: \"{csv_string}\"\n\n")
