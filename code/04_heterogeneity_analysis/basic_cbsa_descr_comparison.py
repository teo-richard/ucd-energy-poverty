import polars as pl

curr = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate_no_cbsa.csv")
proj = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")

# curr is the no_cbsa version so grab CBSA from the 2050 file
cbsa_map = proj.select(["CONTROL", "OMB13CBSA"])

curr_by_cbsa = (
    curr.join(cbsa_map, on="CONTROL", how="left")
    .group_by("OMB13CBSA")
    .agg([
        pl.col("mintemp").mean().alias("mintemp_2023"),
        pl.col("maxtemp").mean().alias("maxtemp_2023"),
        pl.col("avgtemp").mean().alias("avgtemp_2023"),
    ])
)

proj_by_cbsa = (
    proj.group_by("OMB13CBSA")
    .agg([
        pl.col("proj_tasmin").mean().alias("mintemp_2050"),
        pl.col("proj_tasmax").mean().alias("maxtemp_2050"),
        pl.col("proj_tas").mean().alias("avgtemp_2050"),
    ])
)

out = curr_by_cbsa.join(proj_by_cbsa, on="OMB13CBSA", how="inner").sort("OMB13CBSA")

out.write_csv("code/heterogeneity_analysis/cbsa_temp_comparison.csv")
print(out)
