import sys
sys.path.insert(0, "code/00_shared")
from analysis_functions import load_splits, run_lightgbm


splits, splits_nw = load_splits(
    "data/processed/current_climate",
    "01_03_00"
)

run_lightgbm(**splits, label="WITH WEIGHTS")
run_lightgbm(**{k: splits_nw[k] for k in ["X_train", "X_test", "y_train", "y_test"]},
             label="WITHOUT WEIGHTS")
