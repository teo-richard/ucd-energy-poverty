import sys
sys.path.insert(0, "code/00_shared")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

from analysis_functions import (
    load_splits, _save_calibration_plot, save_model, TEMP_VARS
)


CAT_COLS_NO_CBSA = [
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

splits_nc, _ = load_splits(
    "data/processed/current_climate",
    "01_03_00",
    cbsa=False,
    temp_vars=TEMP_VARS,
)

X_train_pd = splits_nc["X_train"].to_pandas()
X_test_pd  = splits_nc["X_test"].to_pandas()
y_train    = splits_nc["y_train"].to_numpy()
y_test     = splits_nc["y_test"].to_numpy()
w_train    = splits_nc["w_train"].to_numpy()

cat_cols  = [c for c in CAT_COLS_NO_CBSA if c in X_train_pd.columns]
num_cols  = [c for c in X_train_pd.columns if c not in cat_cols]

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
])
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
preprocessor = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols),
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("lr",  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1, C=1.0)),
])

print("\n\nRUNNING LOGISTIC REGRESSION WITH WEIGHTS — no CBSA")
print("--------------------------------------------------------")

# 5-fold CV to generate calibration plot
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y_train))
fold_data = []

for fold_tr_idx, fold_val_idx in skf.split(X_train_pd, y_train):
    X_fold_tr  = X_train_pd.iloc[fold_tr_idx]
    X_fold_val = X_train_pd.iloc[fold_val_idx]
    y_fold_tr  = y_train[fold_tr_idx]
    y_fold_val = y_train[fold_val_idx]
    w_fold_tr  = w_train[fold_tr_idx]

    fold_pipeline = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ])),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1, C=1.0)),
    ])
    fold_pipeline.fit(X_fold_tr, y_fold_tr, lr__sample_weight=w_fold_tr)
    fold_probs = fold_pipeline.predict_proba(X_fold_val)[:, 1]
    oof_probs[fold_val_idx] = fold_probs
    fold_data.append((y_fold_val, fold_probs))

_save_calibration_plot(
    y_train, oof_probs,
    name="logistic_regression_no_cbsa",
    label="WITH WEIGHTS — no CBSA (5-fold CV)",
    model_name="Logistic Regression",
    fold_data=fold_data,
)
print("*Calibration plot saved.\n\n")

# Final model on full training set
pipeline.fit(X_train_pd, y_train, lr__sample_weight=w_train)

y_pred_proba = pipeline.predict_proba(X_test_pd)[:, 1]
y_pred       = pipeline.predict(X_test_pd)

print("Logistic Regression AUC-ROC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print(classification_report(y_test, y_pred))

# Top-20 features by absolute coefficient
ohe_feature_names = pipeline["pre"].named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols)
all_feature_names = list(ohe_feature_names) + num_cols
coefs = pipeline["lr"].coef_[0]
importance = (
    pd.Series(np.abs(coefs), index=all_feature_names)
    .sort_values(ascending=False)
    .head(20)
)
print("\nTop-20 features by |coefficient|:")
print(importance.to_string())

save_model(pipeline, "data/processed/models/current_climate_logistic_regression_no_cbsa_with_weights.pkl")
print("\nModel saved.")
