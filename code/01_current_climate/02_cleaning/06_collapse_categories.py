import polars as pl
import polars.selectors as cs
from polars import col, lit, when

ahs_climate = pl.read_csv("data/interim/current_climate/01_02_05_corr_checked_ahs_climate.csv")
var_summary = pl.read_csv("data/internal/nominal_variable_summary.csv")

# --- Get baseline pct energy poor ---
print("\nBaseline Percent Energy Poor:")
print(sum(ahs_climate["energy_poverty"] == 1) / ahs_climate.height )
print("\n")

# Baseline is 17.16 percent energy poor


# ---------- Do Not Collapse ----------
# Low enough number of distinct categories (under 10) that we will leave as-is or something like geography where we'd want to have that as distinct categories

low_num_distinct_cat = [
                "TENURE", 
                "INTLANG", 
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
to_drop = ["SEWTYPE", "ACSECNDRY"]
ahs_climate = ahs_climate.drop(to_drop)

# ---------- Setting Up Collapsing ----------
to_collapse_automatically = []

BLD_map = {
    1: 1,   # mobile home — strong poverty signal, keep separate
    2: 2,   # single family detached
    3: 3,   # single family attached
    4: 4,   # small multifamily (2-4 units, codes 4-5)
    5: 4,
    6: 5,   # medium multifamily (5-19 units, codes 6-7)
    7: 5,
    8: 6,   # large multifamily (20+ units, codes 8-9)
    9: 6,
    10: 999, # boat/RV/van — collapse to other, tiny population
}



ACPRIMARY_map = {
    1: 1,   # central AC
    2: 1,
    3: 1,
    4: 1,
    5: 2,   # room AC
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 3,  # no AC
}

SUPP1HEAT_map = {
    14: 1,  # no supplemental heat
    3: 2,   # forced air / heat pump (codes 3, 4)
    4: 2,
    8: 3,   # steam/hot water (code 8)
    1: 4,   # cookstove/oven used for heat (codes 1, 5) — poverty signal
    5: 4,
    7: 5,   # portable electric (code 7) — poverty signal
    9: 6,   # wood/biomass (codes 9, 12)
    12: 6,
    6: 7,   # pipeless furnace (code 6)
    10: 8,  # vented room heater (code 10)
    11: 8,  # unvented room heater (code 11)
    2: 9,   # built-in electric
    13: 999, # something else
    -9: 999, # not reported
}

HEATTYPE_map = {
    1: 1,   # forced air furnace — dominant category
    2: 2,   # steam/hot water
    3: 3,   # heat pump
    4: 4,   # electric baseboard
    5: 5,   # pipeless furnace
    8: 6,   # portable electric heaters — poverty signal, keep separate
    14: 7,  # cooking stove used for heating — strong poverty signal, keep separate
    13: 8,  # no heating system — keep separate
    6: 999, # vented room heaters
    7: 999, # unvented room heaters
    9: 999, # wood/franklin stove
    10: 999, # fireplace with inserts
    11: 999, # fireplace without inserts
    12: 999, # other
}

HHRACE_map = {
    1: 1,    # White
    2: 2,    # Black
    3: 3,    # American Indian / Alaska Native
    4: 4,    # Asian
    5: 5,    # Hawaiian / Pacific Islander
    6: 6,    # Multiracial
    7: 6,
    8: 6,
    9: 6,
    10: 6,
    11: 6,
    12: 6,
    13: 6,
    14: 6,
    15: 6,
    16: 6,
    17: 6,
    18: 6,
    19: 6,
    20: 6,
    21: 6,
    -6: 999, # not applicable
}

# anything not in the map gets 999

var_maps = {
    "ACPRIMARY": ACPRIMARY_map,
    "SUPP1HEAT": SUPP1HEAT_map,
    "HEATTYPE": HEATTYPE_map,
    "BLD": BLD_map,
    "HHRACE": HHRACE_map
}

# ---------- Collapsing Automatically ----------
# IMPORTANT NOTE: WE DECIDED TO MAKE MANUAL MAPS THEREFORE THE BELOW CODE IS NOT USED
# KEEPING BECAUSE IT WORKS AND MAYBE WE'LL USE IT IN THE FUTURE. DOUBTFUL BUT GAF BRO

# Not used, just for visualizing if needed
def get_prop_table(var):
    tab = ahs_climate.group_by(var).len().sort("len", descending=True).with_columns(
        (col("len") / ahs_climate.height).alias("pct")
    )
    return tab


def get_other_categories(var, threshold):
    """
    Get a list values of var that are small percent of all values enough to be combined into an "other" category
    """
    categories_to_keep = (
        ahs_climate.group_by(var).len().with_columns((col("len") / ahs_climate.height).alias("pct"))
        .filter(col("pct") > threshold)[var]
        .to_list()
    )
    return categories_to_keep


def combine_to_other_category(data, var, threshold):
    """
    Combine the small categories into one category
    """
    categories_list = get_other_categories(var, threshold)
    full_list_categories_to_keep = categories_list + values_to_keep_separate.get(var, [])
    all_cats = ahs_climate[var].unique().to_list()
    collapsed_cats_tracker[var] = [c for c in all_cats if c not in full_list_categories_to_keep]
    df = (
        data
        .with_columns(
            when(col(var).is_in(full_list_categories_to_keep))
            .then(col(var))
            .otherwise(lit(999)) # Encode "other" as 999
            .alias(var)
        )
    )


    return df, collapsed_cats_tracker


threshold_dict = {
    "BLD": 0.05,
    "HHRACE": 0.01, 
    "HEATTYPE": 0.01, 
    "ACPRIMARY": 0.01, 
    "ACSECNDRY": 0.01, 
    "SUPP1HEAT": 0.01
    }

collapsed_cats_tracker = {}

# Values I think are important to not lump into an "other" category such as things with theory background or None categories
values_to_keep_separate = {
    "HEATTYPE": [8, 13, 14]
}

for var, threshold in threshold_dict.items():
    if var in to_collapse_automatically:
        ahs_climate, collapsed_cats_tracker = combine_to_other_category(ahs_climate, var, threshold)

print("\nAUTOMATIC COLLAPSES ONLY: Categories (the codes, check codebook for description) collapsed into 999:")
print(collapsed_cats_tracker)


# ---------- Collapsing Manually Mapped Vars ----------

for var_name, var_map in var_maps.items():
    ahs_climate = (
        ahs_climate
        .with_columns(
            col(var_name).replace_strict(var_map, default=999).alias(var_name)
        )
    )



# --------------------------------------------------------------------------------
print("\nRan script successfully.")
print(f"Data shape: {ahs_climate.shape}")

# --- Write the data ---
csv_string = "data/interim/current_climate/01_02_06_cat_collapsed_ahs_climate.csv"
print(f"\nWriting data to {csv_string} now...")
ahs_climate.write_csv(csv_string)
print(f"Data written to: \"{csv_string}\"\n")



