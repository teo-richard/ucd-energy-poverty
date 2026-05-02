import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_splits, run_lightgbm


YEARS = [2030, 2040, 2050]

for year in YEARS:
    print(f"\n\n{'=' * 60}")
    print(f"YEAR: {year}")
    print("=" * 60)

    splits, splits_nw = load_splits(
        "data/processed/projected_climate",
        f"02_02_00_{year}"
    )

    run_lightgbm(**splits, label=f"WITH WEIGHTS — {year}")
    run_lightgbm(**{k: splits_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
                 label=f"WITHOUT WEIGHTS — {year}")
