import pickle
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, classification_report


def load_splits(processed_dir, prefix):
    """
    Load weighted and unweighted train/test split pickles.

    Expects:
      {processed_dir}/with_weights/{prefix}_{name}.pkl
      {processed_dir}/without_weights/{prefix}_{name}_no_weight.pkl

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

    return splits_weighted, splits_unweighted


def run_lightgbm(X_train, X_test, y_train, y_test, w_train=None, label=""):
    """
    Train and evaluate a LightGBM classifier.

    Pass w_train for a weighted run; omit (or pass None) for unweighted.
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

    return model


def run_xgboost(X_train, X_test, y_train, y_test, w_train=None,
                cat_cols=None, scale_pos_weight=4.83, label=""):
    """
    Train and evaluate an XGBoost classifier with SHAP explainability.

    Pass w_train for a weighted run; omit (or pass None) for unweighted.
    cat_cols: list of categorical column names requiring sentinel/int/category encoding.
    scale_pos_weight: ratio of negatives to positives in training data.
    Prints AUC-ROC, classification report, top-20 feature importance, and SHAP plots.
    """
    header = f"XGBOOST {label}".strip() if label else "XGBOOST"
    print(f"\n\nRUNNING {header}")
    print("--------------------------------------------------------")

    X_train_pd = X_train.to_pandas()
    X_test_pd  = X_test.to_pandas()

    if cat_cols:
        # Polars → Pandas converts nullable int columns with NaN to float.
        # Fill with sentinel -1, cast to int, then category so XGBoost
        # receives valid categorical dtypes (not floats).
        for col in cat_cols:
            X_train_pd[col] = X_train_pd[col].fillna(-1).astype(int).astype("category")
            X_test_pd[col]  = X_test_pd[col].fillna(-1).astype(int).astype("category")

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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_pd)
    shap.plots.waterfall(explainer(X_test_pd)[0])
    shap.summary_plot(shap_values, X_test_pd)

    return model
