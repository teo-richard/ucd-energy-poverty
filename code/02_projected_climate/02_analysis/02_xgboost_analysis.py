import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_splits, run_xgboost


CAT_COLS = [  # Does not include SEWTYPE and ASECONDRY because we dropped those
    "TENURE", "BLD", "INTLANG", "DIVISION", "INTMONTH", "OMB13CBSA",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

YEARS = [2030, 2040, 2050]

for year in YEARS:
    print(f"\n\n{'=' * 60}")
    print(f"YEAR: {year}")
    print("=" * 60)

    splits, splits_nw = load_splits(
        "data/processed/projected_climate",
        f"02_02_00_{year}"
    )

    # Compute class imbalance ratio from training labels for this year's data
    n_neg = (splits["y_train"] == 0).sum()
    n_pos = (splits["y_train"] == 1).sum()
    scale_pos_weight = round(n_neg / n_pos, 2)

    run_xgboost(**splits, cat_cols=CAT_COLS, scale_pos_weight=scale_pos_weight,
                label=f"WITH WEIGHTS — {year}")
    run_xgboost(**{k: splits_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
                cat_cols=CAT_COLS, scale_pos_weight=scale_pos_weight,
                label=f"WITHOUT WEIGHTS — {year}")
