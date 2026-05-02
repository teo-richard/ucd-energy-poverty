import polars as pl
from polars import col, lit, when

years = ["2023"]

def combine_noaa(years):
    combined = pl.DataFrame()
        
    for year in years:

        mintemp_2023_raw = pl.read_csv(f"data/raw/NOAA/separated_datasets/{year}_noaa_mintemp.csv", skip_rows=3, truncate_ragged_lines=True)
        mintemp_2023 = (
            mintemp_2023_raw
            .select("ID", "Name", "State", "Value")
        )

        maxtemp_2023_raw = pl.read_csv(f"data/raw/NOAA/separated_datasets/{year}_noaa_maxtemp.csv", skip_rows=3, truncate_ragged_lines=True)
        maxtemp_2023 = (
            maxtemp_2023_raw
            .select("ID", "Name", "State", "Value")
        )

        avgtemp_2023_raw = pl.read_csv(f"data/raw/NOAA/separated_datasets/{year}_noaa_avgtemp.csv", skip_rows=3, truncate_ragged_lines=True)
        avgtemp_2023 = (
            avgtemp_2023_raw
            .select("ID", "Name", "State", "Value")
        )


        combined_iter_year = (
            mintemp_2023
            .join(maxtemp_2023, on=["ID", "Name", "State"], how = "inner")
            .rename({"Value": "mintemp", "Value_right": "maxtemp"})
            .join(avgtemp_2023, on=["ID", "Name", "State"], how = "inner")
            .rename({"Value": "avgtemp"})
        )

        combined = pl.concat([combined, combined_iter_year])

        combined.write_csv("data/interim/current_climate/01_01_01_noaa_combined.csv")

    return 

combine_noaa(years)