import polars as pl


# Important Note: this will use the data from current climate joined with ahs
#   I'll just drop the current climate vars and put in the projected climate vars
#   This is because the AHS data is already cleaned but was cleaned after joining and I don't want to go refactor it

ahs_cur_climate = pl.read_csv("data/processed/current_climate/01_02_05_basic_ready_for_trees_ahs_climate.csv")

ahs_no_climate = ahs_cur_climate.drop("tasmin", "taxmax") # Dropping climate variables from the NOAA data (sorry ik it's confusing my bad g)

