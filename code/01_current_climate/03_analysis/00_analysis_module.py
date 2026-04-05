import polars as pl
from sklearn.model_selection import train_test_split
import pickle


ahs_climate = pl.read_csv("data/processed/current_climate/01_02_05_basic_ready_for_trees_ahs_climate.csv")
ahs_climate_no_weights = ahs_climate.drop("WEIGHT")

# --- Train and test splits ---
def get_splits(data):
    X = ahs_climate.drop(["energy_poverty"])
    y = ahs_climate["energy_poverty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if "WEIGHT" in data.columns:
        w_train = X_train["WEIGHT"]

        X_train = X_train.drop("WEIGHT")
        X_test = X_test.drop("WEIGHT")
    else:
        w_train = []

    info = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "w_train": w_train
    }

    return info

info_with_weights = get_splits(ahs_climate)
info_without_weights = get_splits(ahs_climate_no_weights)
info_without_weights.popitem() # Removes w_train


for filename, file in info_with_weights.items():
    with open(f"data/processed/current_climate/with_weights/01_03_00_{filename}.pkl", "wb") as f:
        pickle.dump(file, f)


for filename, file in info_without_weights.items():
    with open(f"data/processed/current_climate/without_weights/01_03_00_{filename}_no_weight.pkl", "wb") as f:
        pickle.dump(file, f)