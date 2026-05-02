import polars as pl
from sklearn.model_selection import train_test_split
import pickle
import os


YEARS = [2030, 2040, 2050]


def get_splits(data):
    X = data.drop(["energy_poverty"])
    y = data["energy_poverty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if "WEIGHT" in data.columns:
        w_train = X_train["WEIGHT"]
        X_train = X_train.drop("WEIGHT")
        X_test  = X_test.drop("WEIGHT")
    else:
        w_train = []

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "w_train": w_train,
    }


os.makedirs("data/processed/projected_climate/with_weights",    exist_ok=True)
os.makedirs("data/processed/projected_climate/without_weights", exist_ok=True)

for year in YEARS:
    data            = pl.read_csv(f"data/processed/projected_climate/02_02_ahs_cmip_{year}.csv")
    data_no_weights = data.drop("WEIGHT")

    info_with_weights    = get_splits(data)
    info_without_weights = get_splits(data_no_weights)
    info_without_weights.pop("w_train")

    prefix = f"02_02_00_{year}"

    for name, file in info_with_weights.items():
        with open(f"data/processed/projected_climate/with_weights/{prefix}_{name}.pkl", "wb") as f:
            pickle.dump(file, f)

    for name, file in info_without_weights.items():
        with open(f"data/processed/projected_climate/without_weights/{prefix}_{name}_no_weight.pkl", "wb") as f:
            pickle.dump(file, f)
