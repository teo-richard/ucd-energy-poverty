import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_splits, run_xgboost


CAT_COLS = [  # Does not include SEWTYPE and ASECONDRY because we dropped those
    "TENURE", "BLD", "INTLANG", "DIVISION", "INTMONTH", "OMB13CBSA",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

splits, splits_nw = load_splits(
    "data/processed/current_climate",
    "01_03_00"
)

run_xgboost(**splits, cat_cols=CAT_COLS, scale_pos_weight=4.83, label="WITH WEIGHTS")
run_xgboost(**{k: splits_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
            cat_cols=CAT_COLS, scale_pos_weight=4.83, label="WITHOUT WEIGHTS")
