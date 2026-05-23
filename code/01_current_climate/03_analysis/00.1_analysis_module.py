import polars as pl
from sklearn.model_selection import train_test_split
import pickle
import os


def get_splits(data):
    X = data.drop(["energy_deprivation", "CONTROL"]) # WEIGHT is dropped when creating weight splits and nw splits
    y = data["energy_deprivation"]

    # Step 1: hold out 20% as test — identical to the previous single split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 2: carve 12.5% of trainval as calibration set (≈10% of full dataset)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=42, stratify=y_trainval
    )

    if "WEIGHT" in data.columns:
        w_train = X_train["WEIGHT"]
        w_cal   = X_cal["WEIGHT"]
        X_train = X_train.drop("WEIGHT")
        X_cal   = X_cal.drop("WEIGHT")
        X_test  = X_test.drop("WEIGHT")
    else:
        w_train = []
        w_cal   = []

    return {
        "X_train": X_train,
        "X_cal":   X_cal,
        "X_test":  X_test,
        "y_train": y_train,
        "y_cal":   y_cal,
        "y_test":  y_test,
        "w_train": w_train,
        "w_cal":   w_cal,
    }


os.makedirs("data/processed/current_climate/with_weights",    exist_ok=True)
os.makedirs("data/processed/current_climate/without_weights", exist_ok=True)

# --- With CBSA ---
ahs_climate = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate.csv")
ahs_climate_no_weights = ahs_climate.drop("WEIGHT")

info_with_weights    = get_splits(ahs_climate)
info_without_weights = get_splits(ahs_climate_no_weights)
info_without_weights.pop("w_train")
info_without_weights.pop("w_cal")

for name, file in info_with_weights.items():
    with open(f"data/processed/current_climate/with_weights/01_03_00_{name}.pkl", "wb") as f:
        pickle.dump(file, f)

for name, file in info_without_weights.items():
    with open(f"data/processed/current_climate/without_weights/01_03_00_{name}_no_weight.pkl", "wb") as f:
        pickle.dump(file, f)

# --- Without CBSA ---
ahs_no_cbsa = pl.read_csv("data/processed/current_climate/basic_ready_for_trees_ahs_climate_no_cbsa.csv")
ahs_no_cbsa_no_weights = ahs_no_cbsa.drop("WEIGHT")

info_no_cbsa_with_weights    = get_splits(ahs_no_cbsa)
info_no_cbsa_without_weights = get_splits(ahs_no_cbsa_no_weights)
info_no_cbsa_without_weights.pop("w_train")
info_no_cbsa_without_weights.pop("w_cal")

for name, file in info_no_cbsa_with_weights.items():
    with open(f"data/processed/current_climate/with_weights/01_03_00_no_cbsa_{name}.pkl", "wb") as f:
        pickle.dump(file, f)

for name, file in info_no_cbsa_without_weights.items():
    with open(f"data/processed/current_climate/without_weights/01_03_00_no_cbsa_{name}_no_weight.pkl", "wb") as f:
        pickle.dump(file, f)
