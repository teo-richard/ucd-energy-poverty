import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import pickle


ahs_climate = pl.read_csv("data/processed/current_climate/01_02_05_basic_ready_for_trees_ahs_climate.csv")

# --- Train and test splits ---
# // Weighted \\
splits = {}

for filename in ["X_train", "X_test", "y_train", "y_test", "w_train"]:
    with open(f"data/processed/current_climate/with_weights/01_03_00_{filename}.pkl", "rb") as f:
        splits[filename] = pickle.load(f) # Polars dataframes, except for w_train

# Unpack into individual variables
X_train = splits["X_train"]
X_test  = splits["X_test"]
y_train = splits["y_train"]
y_test  = splits["y_test"]
w_train = splits["w_train"]


# // Without Weights\\
splits_without_weight = {}

for filename in ["X_train", "X_test", "y_train", "y_test"]:
    with open(f"data/processed/current_climate/without_weights/01_03_00_{filename}_no_weight.pkl", "rb") as f:
        splits_without_weight[filename] = pickle.load(f) # Polars dataframes

# Unpack into individual variables
X_train_without_weight = splits_without_weight["X_train"]
X_test_without_weight  = splits_without_weight["X_test"]
y_train_without_weight = splits_without_weight["y_train"]
y_test_without_weight  = splits_without_weight["y_test"]



# ----------------------------------------------------------------------------------------------


# --- Define model ---
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ----------------------------------------------------------------------------------------------

# ---------- With Weights ----------
print("\n\nRUNNING LIGHTGBM WITH WEIGHTS")
print("--------------------------------------------------------")

# --- Fit ---
# can't get it to work without .to_numpy I think because of some naming issues in my environment ?
model.fit(
    X_train.to_numpy(), y_train.to_numpy(),
    eval_set=[(X_test.to_numpy(), y_test.to_numpy())],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    sample_weight=w_train.to_numpy()
)

# --- Evaluate ---
y_pred_proba = model.predict_proba(X_test.to_numpy())[:, 1]
y_pred = model.predict(X_test.to_numpy())

print("AUC-ROC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print(classification_report(y_test, y_pred))

# Check feature importance to make sure PERPOVLVL isn't doing all the work
feature_names = X_train.columns
importance = pd.Series(model.feature_importances_, index=feature_names)
print(importance.sort_values(ascending=False).head(20))

# ---------- Without Weights ----------
print("\n\nRUNNING LIGHTGBM WITHOUT WEIGHTS")
print("--------------------------------------------------------")


# --- Fit ---
# can't get it to work without .to_numpy I think because of some naming issues in my environment ?
model.fit(
    X_train_without_weight.to_numpy(), y_train_without_weight.to_numpy(),
    eval_set=[(X_test_without_weight.to_numpy(), y_test_without_weight.to_numpy())],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

# --- Evaluate ---
y_pred_proba = model.predict_proba(X_test_without_weight.to_numpy())[:, 1]
y_pred = model.predict(X_test_without_weight.to_numpy())

print("AUC-ROC:", round(roc_auc_score(y_test_without_weight, y_pred_proba), 4))
print(classification_report(y_test_without_weight, y_pred))

# Check feature importance to make sure PERPOVLVL isn't doing all the work
feature_names = X_train_without_weight.columns
importance = pd.Series(model.feature_importances_, index=feature_names)
print(importance.sort_values(ascending=False).head(20))

