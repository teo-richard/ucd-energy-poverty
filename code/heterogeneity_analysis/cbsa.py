import polars as pl

proj_clim_2050 = pl.read_csv("data/processed/projected_climate/02_02_ahs_cmip_2050.csv")
pred_2050 = pl.read_csv("tree_model_output/xgboost_2050_with_weights.csv")


proj_clim_prob = proj_clim_2050.join(pred_2050, on = "CONTROL", how = "left")

# Now analyze by CBSA
proj_clim_prob_by_cbsa = proj_clim_prob.group_by("OMB13CBSA").agg([
    pl.col("pred_prob_xgb_w").mean().alias("mean_ep_prob"),
    pl.col("energy_poverty").mean().alias("actual_ep_rate"),
    pl.len().alias("n")
]).sort("mean_ep_prob", descending=True)

print(proj_clim_prob_by_cbsa.head)

proj_clim_prob_by_cbsa.write_csv("code/heterogeneity_analysis/by_cbsa.csv")