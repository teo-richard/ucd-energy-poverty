import os
import pickle
import joblib
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def prepare_cat_cols(X_pd, cat_cols):
    """Apply sentinel/int/category encoding for XGBoost categorical columns."""
    X = X_pd.copy()
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna(-1).astype(int).astype("category")
    return X


def load_splits(processed_dir, prefix, cbsa=True):
    """
    Load weighted and unweighted train/test split pickles.

    Expects:
      {processed_dir}/with_weights/{prefix}_{name}.pkl
      {processed_dir}/without_weights/{prefix}_{name}_no_weight.pkl

    cbsa=False drops OMB13CBSA from X_train and X_test after loading.

    Returns (splits_weighted, splits_unweighted) as dicts with keys
    X_train, X_test, y_train, y_test (and w_train for the weighted dict).
    """
    splits_weighted = {}
    for name in ["X_train", "X_test", "y_train", "y_test", "w_train"]:
        path = f"{processed_dir}/with_weights/{prefix}_{name}.pkl"
        with open(path, "rb") as f:
            splits_weighted[name] = pickle.load(f)

    splits_unweighted = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        path = f"{processed_dir}/without_weights/{prefix}_{name}_no_weight.pkl"
        with open(path, "rb") as f:
            splits_unweighted[name] = pickle.load(f)

    if not cbsa:
        for splits in [splits_weighted, splits_unweighted]:
            for key in ["X_train", "X_test"]:
                if key in splits and "OMB13CBSA" in splits[key].columns:
                    splits[key] = splits[key].drop("OMB13CBSA")

    return splits_weighted, splits_unweighted


def run_lightgbm(X_train, X_test, y_train, y_test, w_train=None, label="", cbsa=True):
    """
    Train and evaluate a LightGBM classifier.

    Pass w_train for a weighted run; omit (or pass None) for unweighted.
    cbsa=False drops OMB13CBSA from the feature set before training.
    Prints AUC-ROC, classification report, and top-20 feature importance.
    """
    header = f"LIGHTGBM {label}".strip() if label else "LIGHTGBM"
    print(f"\n\nRUNNING {header}")
    print("--------------------------------------------------------")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    X_train_pd = X_train.to_pandas()
    X_test_pd  = X_test.to_pandas()

    if not cbsa:
        X_train_pd = X_train_pd.drop(columns=["OMB13CBSA"], errors="ignore")
        X_test_pd  = X_test_pd.drop(columns=["OMB13CBSA"],  errors="ignore")

    fit_kwargs = dict(
        eval_set=[(X_test_pd, y_test.to_numpy())],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train.to_numpy()

    model.fit(X_train_pd, y_train.to_numpy(), **fit_kwargs)

    y_pred_proba = model.predict_proba(X_test_pd)[:, 1]
    y_pred = model.predict(X_test_pd)

    print("AUC-ROC:", round(roc_auc_score(y_test, y_pred_proba), 4))
    print(classification_report(y_test, y_pred))

    importance = pd.Series(model.feature_importances_, index=X_train_pd.columns)
    print(importance.sort_values(ascending=False).head(20))

    return model, X_test_pd


def run_xgboost(X_train, X_test, y_train, y_test, w_train=None,
                cat_cols=None, scale_pos_weight=4.83, label="", cbsa=True):
    """
    Train and evaluate an XGBoost classifier with SHAP explainability.

    Pass w_train for a weighted run; omit (or pass None) for unweighted.
    cat_cols: list of categorical column names requiring sentinel/int/category encoding.
    scale_pos_weight: ratio of negatives to positives in training data.
    cbsa=False drops OMB13CBSA from the feature set (and cat_cols) before training.
    Prints AUC-ROC, classification report, top-20 feature importance, and SHAP plots.
    """
    header = f"XGBOOST {label}".strip() if label else "XGBOOST"
    print(f"\n\nRUNNING {header}")
    print("--------------------------------------------------------")

    X_train_pd = X_train.to_pandas()
    X_test_pd  = X_test.to_pandas()

    if not cbsa:
        X_train_pd = X_train_pd.drop(columns=["OMB13CBSA"], errors="ignore")
        X_test_pd  = X_test_pd.drop(columns=["OMB13CBSA"],  errors="ignore")
        if cat_cols:
            cat_cols = [c for c in cat_cols if c != "OMB13CBSA"]

    if cat_cols:
        # Polars → Pandas converts nullable int columns with NaN to float.
        # Fill with sentinel -1, cast to int, then category so XGBoost
        # receives valid categorical dtypes (not floats).
        X_train_pd = prepare_cat_cols(X_train_pd, cat_cols)
        X_test_pd  = prepare_cat_cols(X_test_pd, cat_cols)

    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )

    fit_kwargs = dict(
        eval_set=[(X_test_pd, y_test.to_numpy())],
        verbose=50,
    )
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train.to_numpy()

    model.fit(X_train_pd, y_train.to_numpy(), **fit_kwargs)

    y_pred_proba = model.predict_proba(X_test_pd)[:, 1]
    y_pred       = model.predict(X_test_pd)

    print("XGBoost AUC-ROC:", round(roc_auc_score(y_test, y_pred_proba), 4))
    print(classification_report(y_test, y_pred))

    importance = pl.DataFrame({
        "feature": X_train_pd.columns.tolist(),
        "importance": model.feature_importances_,
    }).sort("importance", descending=True)
    print(importance.head(20))

    return model, X_test_pd


def run_shap(model, X_test_pd, name, label="", max_samples=500): # Max samples=None one if you want the full dataset passed
    """
    Run SHAP TreeExplainer on a fitted model and display a waterfall + summary plot.

    max_samples: randomly subsample rows before computing SHAP values so this
    doesn't hang on large datasets. Set to None to use all rows.
    """
    header = f" — {label}" if label else ""
    print(f"\n\nSHAP ANALYSIS{header}")
    print("--------------------------------------------------------")

    if max_samples is not None and len(X_test_pd) > max_samples:
        X_shap = X_test_pd.sample(n=max_samples, random_state=42)
    else:
        X_shap = X_test_pd

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explainer(X_shap)[0], show=False)
    plt.tight_layout()
    plt.savefig(f'shap_images/{name}_waterfall.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(f'shap_images/{name}_summary.png')
    plt.clf()


