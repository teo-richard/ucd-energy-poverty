import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import shap
import pickle


ahs_climate = pl.read_csv("data/processed/current_climate/01_02_05_basic_ready_for_trees_ahs_climate.csv")

# --- Train and test splits ---
# // Weighted \\
splits = {}

for filename in ["X_train", "X_test", "y_train", "y_test", "w_train"]:
    with open(f"data/processed/current_climate/with_weights/01_03_00_{filename}.pkl", "rb") as f:
        splits[filename] = pickle.load(f) # Polars dataframes, except for w_train

# Unpack into individual variables
X_train = splits["X_train"].to_pandas()
X_test  = splits["X_test"].to_pandas()
y_train = splits["y_train"].to_pandas()
y_test  = splits["y_test"].to_pandas()
w_train = splits["w_train"]


# // Without Weights\\
splits_without_weight = {}

for filename in ["X_train", "X_test", "y_train", "y_test"]:
    with open(f"data/processed/current_climate/without_weights/01_03_00_{filename}_no_weight.pkl", "rb") as f:
        splits_without_weight[filename] = pickle.load(f) # Polars dataframes

# Unpack into individual variables
X_train_without_weight = splits_without_weight["X_train"].to_pandas()
X_test_without_weight  = splits_without_weight["X_test"].to_pandas()
y_train_without_weight = splits_without_weight["y_train"].to_pandas()
y_test_without_weight  = splits_without_weight["y_test"].to_pandas()


cat_cols = [ # Does not include SEWTYPE and ASECONDRY because we dropped those
    "TENURE", "BLD", "INTLANG", "DIVISION", "INTMONTH", "OMB13CBSA",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD"
]


# When i convert to Pandas Dataframe, pandas is forced to convert any column with a missing value to float because NaN has a special floating point value
# Fill NAs with a sentinel value first, then cast to int, then category
# -1 works as a sentinel since your real codes are all positive
# Then you will have all integers, not floating point dtypes which XGBoost cannot work with
for col in cat_cols:
    X_train[col] = X_train[col].fillna(-1).astype(int).astype('category')
    X_test[col] = X_test[col].fillna(-1).astype(int).astype('category')


# ----------------------------------------------------------------------------------------------


model_xgb = xgb.XGBClassifier(
    enable_categorical=True,
    tree_method="hist",
    n_estimators=1000,
    learning_rate=0.05, # Should have it same as LightGBM for fair comparison
    max_depth=6,
    scale_pos_weight=4.83, # 17695/3665 = 4.83 (num negatives / num positives)
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=50
)


# ----------------------------------------------------------------------------------------------

# ---------- With Weights ----------
print("\n\nRUNNING XGBOOST WITH WEIGHTS")
print("--------------------------------------------------------")

model_xgb.fit(
    X_train, y_train.to_numpy(),
    eval_set=[(X_test, y_test.to_numpy())],
    verbose=50,               # print progress every 50 rounds
    sample_weight=w_train.to_numpy()
)


# --- Evaluation ---
# Get predicted probabilities for the positive class (energy poor)
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# Get binary predictions using default 0.5 threshold
y_pred_xgb = model_xgb.predict(X_test)

# AUC-ROC — primary metric for imbalanced classification
print("XGBoost AUC-ROC:", round(roc_auc_score(y_test, y_pred_proba_xgb), 4))

# Full breakdown of precision, recall, f1 per class
print(classification_report(y_test, y_pred_xgb))

# Feature importance
importance_xgb = pl.DataFrame({
    "feature": X_train.columns,
    "importance": model_xgb.feature_importances_
}).sort("importance", descending=True)
print(importance_xgb.head(20))


# --- SHAP ---



# TreeExplainer is optimized specifically for tree models like XGBoost
explainer = shap.TreeExplainer(model_xgb)

# Compute SHAP values for the test set — shape is (n_samples, n_features)
shap_values = explainer.shap_values(X_test)

# Explain a single prediction — a "waterfall" showing each feature's push/pull
shap.plots.waterfall(explainer(X_test)[0])

# Global summary — shows which features matter most and in what direction
shap.summary_plot(shap_values, X_test)



# ---------- Without Weights ----------
print("\n\nRUNNING XGBOOST WITHOUT WEIGHTS")
print("--------------------------------------------------------")

model_xgb.fit(
    X_train_without_weight, y_train_without_weight.to_numpy(),
    eval_set=[(X_test_without_weight, y_test_without_weight.to_numpy())],
    verbose=50               # print progress every 50 rounds
)


# --- Evaluation ---
# Get predicted probabilities for the positive class (energy poor)
y_pred_proba_xgb = model_xgb.predict_proba(X_test_without_weight)[:, 1]

# Get binary predictions using default 0.5 threshold
y_pred_xgb = model_xgb.predict(X_test_without_weight)

# AUC-ROC — primary metric for imbalanced classification
print("XGBoost AUC-ROC:", round(roc_auc_score(y_test_without_weight, y_pred_proba_xgb), 4))

# Full breakdown of precision, recall, f1 per class
print(classification_report(y_test_without_weight, y_pred_xgb))

# Feature importance
importance_xgb = pl.DataFrame({
    "feature": X_train_without_weight.columns,
    "importance": model_xgb.feature_importances_
}).sort("importance", descending=True)
print(importance_xgb.head(20))


# --- SHAP ---

# TreeExplainer is optimized specifically for tree models like XGBoost
explainer = shap.TreeExplainer(model_xgb)

# Compute SHAP values for the test set — shape is (n_samples, n_features)
shap_values = explainer.shap_values(X_test_without_weight)

# Explain a single prediction — a "waterfall" showing each feature's push/pull
shap.plots.waterfall(explainer(X_test_without_weight)[0])

# Global summary — shows which features matter most and in what direction
shap.summary_plot(shap_values, X_test_without_weight)