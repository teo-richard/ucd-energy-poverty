import polars as pl
import polars.selectors as cs
from polars import col, lit, when
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd


ahs_climate = pl.read_csv("data/processed/LightGBM_data/01_02_04_ready_for_lightgbm_ahs_climate.csv")

# --- Train and test splits ---
X = ahs_climate.drop(["energy_poverty"])
y = ahs_climate["energy_poverty"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Define model ---
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --- Fit ---
model.fit(
    X_train.to_numpy(), y_train.to_numpy(),
    eval_set=[(X_test.to_numpy(), y_test.to_numpy())],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
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