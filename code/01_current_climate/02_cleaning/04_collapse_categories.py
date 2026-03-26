import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import pandas as pd

ahs_climate = pl.read_csv("data/transitory/01_02_03_features_engineered_ahs_climate.csv")
var_summary = pl.read_csv("data/internal/nominal_variable_summary.csv")

# --- Get baseline pct energy poor ---

print(
    sum(ahs_climate["energy_poverty"] == 1) / ahs_climate.height 
)

# Baseline is 17.16 percent energy poor


# ---------- Do Not Collapse ----------
# Low enough number of distinct categories (under 10) that we will leave as-is or something like geography where we'd want to have that as distinct categories

low_num_distinct_cat = [
                "TENURE", 
                "INTLANG", 
                "INTMONTH",
                "HHMAR", 
                "HHCITSHP", 
                "HSHLDTYPE", 
                "MLHH",
                "PARTNER",
                "COOKFUEL", 
                "DRYER", 
                "HOTWATER",
                "FIREPLACE", 
                "MULTIGEN",
                "SAMEHHLD"
                ]

geography_vars = [
            "DIVISION",
            "OMB13CBSA"
            ]

do_not_collapse = low_num_distinct_cat + geography_vars



# ---------- Consider dropping ----------
# All but 1 category are under 10% of observations AND this other category with k-1 categories has about 17% energy poor
# So these would addd nothing to tree models 
# Excluding "ACPRIMARY", "ACSECNDRY", "SUPP1HEAT" from this even though they fit because these seem quite relevant to energy poverty so idk
to_drop = ["SEWTYPE", "ACPRIMARY", "ACSECNDRY", "SUPP1HEAT"]

# ---------- To Collapse ----------
to_collapse = [
            "BLD",
            "HHRACE",
            "HEATTYPE",
            "ACPRIMARY"
            "ACSECONDRY",
            "SUPP1HEAT"
            ]


# ---------- Collapsing ----------


def get_prop_table(var):
    tab = ahs_climate.group_by(var).len().sort("len", descending=True).with_columns(
        (col("len") / ahs_climate.height).alias("pct")
    )
    return tab


def get_other_categories(var, threshold):
    others = (
        ahs_climate.group_by(var).len().with_columns((col("len") / ahs_climate.height).alias("pct"))
        .filter(col("pct") >= threshold)[var]
        .to_list()
    )
    return others


def combine_to_other_category(data, var, threshold):
    categories_list = get_other_categories("BLD", threshold)
    df = (
        data
        .with_columns(
            when(col(var).is_in(categories_list))
            .then(col(var))
            .otherwise(lit(999))
            .alias(var)
        )
    )

    return df

d = combine_to_other_category(ahs_climate, "BLD", 0.05)
d.filter(col("BLD") == 999)["BLD"]


threshold_dict = {
    "BLD": 0.05,
    "HHRACE": 0.01, 
    "HEATTYPE": 0.01, 
    "ACPRIMARY": 0.01, 
    "ACSECNDRY": 0.01, 
    "SUPP1HEAT": 0.01
    }

ahs_climate_collapsed = ahs_climate
for var, threshold in threshold_dict.items():
    ahs_climate_collapsed = combine_to_other_category(ahs_climate_collapsed, var, threshold)



# --------------------------------------------------------------------------------
print("\nRan script successfully.")
print(f"Data shape: {ahs_climate.shape}")

# --- Write the data ---
csv_string = "data/transitory/01_02_04_cat_collapsed_ahs_climate.csv"
print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)


print(f"\nData written to: \"{csv_string}\"\n\n")
