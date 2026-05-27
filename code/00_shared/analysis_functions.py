import os
import pickle
import joblib
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
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


ALL_TEMP_VARS = ["maxtemp", "mintemp", "avgtemp", "dtr", "HDD_approx", "CDD_approx"]
TEMP_VARS = ["avgtemp"]  # ← edit this one list to change temp vars used across all analyses


def filter_temp_vars(df_pd, temp_vars=None):
    """Drop all ALL_TEMP_VARS columns not in temp_vars from a pandas DataFrame."""
    if temp_vars is None:
        temp_vars = TEMP_VARS
    to_drop = [v for v in ALL_TEMP_VARS if v not in temp_vars and v in df_pd.columns]
    return df_pd.drop(columns=to_drop) if to_drop else df_pd


def load_splits(processed_dir, prefix, cbsa=True, temp_vars=None):
    """
    Load weighted and unweighted train/test split pickles.

    Expects:
      {processed_dir}/with_weights/{prefix}_{name}.pkl
      {processed_dir}/without_weights/{prefix}_{name}_no_weight.pkl

    cbsa=False drops OMB13CBSA from X_train and X_test after loading.
    temp_vars: list of temperature variables to keep (subset of ALL_TEMP_VARS).
               All other temp vars are dropped. Pass None to keep all.

    Returns (splits_weighted, splits_unweighted) as dicts with keys
    X_train, X_test, y_train, y_test (and w_train for the weighted dict).
    """
    splits_weighted = {}
    for name in ["X_train", "X_cal", "X_test", "y_train", "y_cal", "y_test", "w_train", "w_cal"]:
        path = f"{processed_dir}/with_weights/{prefix}_{name}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                splits_weighted[name] = pickle.load(f)

    splits_unweighted = {}
    for name in ["X_train", "X_cal", "X_test", "y_train", "y_cal", "y_test"]:
        path = f"{processed_dir}/without_weights/{prefix}_{name}_no_weight.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                splits_unweighted[name] = pickle.load(f)

    if not cbsa:
        for splits in [splits_weighted, splits_unweighted]:
            for key in ["X_train", "X_cal", "X_test"]:
                if key in splits and "OMB13CBSA" in splits[key].columns:
                    splits[key] = splits[key].drop("OMB13CBSA")

    if temp_vars is not None:
        to_drop = [v for v in ALL_TEMP_VARS if v not in temp_vars]
        for splits in [splits_weighted, splits_unweighted]:
            for key in ["X_train", "X_cal", "X_test"]:
                if key in splits:
                    cols = list(splits[key].columns)
                    drop = [c for c in to_drop if c in cols]
                    if drop:
                        splits[key] = splits[key].drop(drop)

    return splits_weighted, splits_unweighted


class _PlattScaler:
    """Post-hoc Platt scaling wrapper around a pre-fitted classifier.

    Fits a logistic regression (the sigmoid A/B parameters) on a held-out
    calibration set without re-training the base estimator. Provides the
    same predict_proba / predict interface as the underlying model.
    """

    def __init__(self, raw_model):
        self.estimator = raw_model  # kept for get_feature_names() compatibility
        self._lr = None

    def fit(self, X_cal_pd, y_cal_np, sample_weight=None):
        raw_probs = self.estimator.predict_proba(X_cal_pd)[:, 1]
        # Clamp to avoid log(0); logistic regression on log-odds is Platt scaling
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        logits = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1e10, random_state=42, max_iter=1000)
        fit_kwargs = {}
        if sample_weight is not None and len(sample_weight) > 0:
            fit_kwargs["sample_weight"] = sample_weight
        self._lr.fit(logits, y_cal_np, **fit_kwargs)
        return self

    def predict_proba(self, X):
        raw_probs = self.estimator.predict_proba(X)[:, 1]
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        logits = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
        return self._lr.predict_proba(logits)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_model(raw_model, X_cal_pd, y_cal, w_cal=None):
    """Fit Platt scaling on a held-out calibration set.

    Fits only the sigmoid A/B logistic layer — the tree is not retrained.
    Returns a _PlattScaler whose .predict_proba() gives calibrated probabilities.
    """
    y_cal_np = y_cal.to_numpy() if hasattr(y_cal, "to_numpy") else np.array(y_cal)
    w_cal_np = w_cal.to_numpy() if w_cal is not None and hasattr(w_cal, "to_numpy") else w_cal

    scaler = _PlattScaler(raw_model)
    scaler.fit(X_cal_pd, y_cal_np, sample_weight=w_cal_np if w_cal_np is not None and len(w_cal_np) > 0 else None)
    return scaler


def get_feature_names(model):
    """Return XGBoost feature names from a raw model or a CalibratedClassifierCV wrapper."""
    if hasattr(model, "get_booster"):
        return list(model.get_booster().feature_names)
    if hasattr(model, "estimator") and hasattr(model.estimator, "get_booster"):
        return list(model.estimator.get_booster().feature_names)
    return None


def run_lightgbm(X_train, X_test, y_train, y_test, w_train=None,
                 X_cal=None, y_cal=None, w_cal=None,
                 label="", cbsa=True):
    """
    Train and evaluate a LightGBM classifier.

    Pass w_train for a weighted run; omit (or pass None) for unweighted.
    cbsa=False drops OMB13CBSA from the feature set before training.
    X_cal/y_cal/w_cal: held-out calibration set for Platt scaling. If provided,
    a CalibratedClassifierCV(method='sigmoid') is fit and returned as the second
    element of the return tuple.
    Prints AUC-ROC, classification report, and top-20 feature importance.
    Returns (raw_model, calibrated_model, X_test_pd). calibrated_model is None
    if no calibration set is supplied.
    """
    header = f"LIGHTGBM {label}".strip() if label else "LIGHTGBM"
    print(f"\n\nRUNNING {header}")
    print("--------------------------------------------------------")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
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

    calibrated_model = None
    if X_cal is not None and y_cal is not None:
        X_cal_pd = X_cal.to_pandas()
        if not cbsa:
            X_cal_pd = X_cal_pd.drop(columns=["OMB13CBSA"], errors="ignore")
        calibrated_model = calibrate_model(model, X_cal_pd, y_cal, w_cal)
        print("*Platt scaling calibrator fit on held-out calibration set.")

        y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test)
        y_prob_cal = calibrated_model.predict_proba(X_test_pd)[:, 1]
        _save_calibration_plot(
            y_test_np, y_pred_proba,
            name=f"{label.replace(' ', '_').replace('—', '').strip()}_lgbm_pre_vs_post_cal" if label else "lgbm_pre_vs_post_cal",
            label=f"{label} — test set" if label else "test set",
            y_prob_cal=y_prob_cal,
            model_name="LightGBM",
        )
        print("*Pre-vs-post calibration plot saved.\n\n")

    return model, calibrated_model, X_test_pd


# scale_pos_weight: recompute as (n_negative / n_positive) after pipeline runs with energy_deprivation
def run_xgboost(X_train, X_test, y_train, y_test, w_train=None,
                cat_cols=None, scale_pos_weight=4.713795960346256,
                X_cal=None, y_cal=None, w_cal=None,
                label="", cbsa=True, name=None):
    """
    Train and evaluate an XGBoost classifier with SHAP explainability.

    Pass w_train for a weighted run; omit (or pass None) for unweighted.
    cat_cols: list of categorical column names requiring sentinel/int/category encoding.
    scale_pos_weight: ratio of negatives to positives in training data.
    cbsa=False drops OMB13CBSA from the feature set (and cat_cols) before training.
    name: if provided, runs 5-fold CV on the training set and saves a calibration plot
          to output/calibration/{name}_calibration.png before final training.
    X_cal/y_cal/w_cal: held-out calibration set for Platt scaling. If provided,
    a CalibratedClassifierCV(method='sigmoid') is fit and returned as the second
    element of the return tuple.
    Prints AUC-ROC, classification report, and top-20 feature importance.
    Returns (raw_model, calibrated_model, X_test_pd). calibrated_model is None
    if no calibration set is supplied.
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

    if name is not None:
        y_tr_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train)
        w_np    = w_train.to_numpy() if w_train is not None and hasattr(w_train, "to_numpy") else w_train
        oof_probs = np.zeros(len(y_tr_np))
        fold_data = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_tr_idx, fold_val_idx in skf.split(X_train_pd, y_tr_np):
            X_fold_tr  = X_train_pd.iloc[fold_tr_idx]
            X_fold_val = X_train_pd.iloc[fold_val_idx]
            y_fold_tr  = y_tr_np[fold_tr_idx]
            y_fold_val = y_tr_np[fold_val_idx]
            fold_model = xgb.XGBClassifier(
                enable_categorical=True, tree_method="hist",
                n_estimators=1000, learning_rate=0.02, max_depth=6,
                scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, eval_metric="logloss", early_stopping_rounds=50, min_child_weight = 3
            )
            fold_kwargs = dict(eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            if w_np is not None:
                fold_kwargs["sample_weight"] = w_np[fold_tr_idx]
            fold_model.fit(X_fold_tr, y_fold_tr, **fold_kwargs)
            fold_probs = fold_model.predict_proba(X_fold_val)[:, 1]
            oof_probs[fold_val_idx] = fold_probs
            fold_data.append((y_fold_val, fold_probs))
        cv_label = f"{label} (5-fold CV)" if label else "5-fold CV"
        _save_calibration_plot(y_tr_np, oof_probs, name=name, label=cv_label, fold_data=fold_data)
        print("*Calibration plot saved.\n\n")

    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
        min_child_weight = 3
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

    calibrated_model = None
    if X_cal is not None and y_cal is not None:
        X_cal_pd = X_cal.to_pandas()
        if not cbsa:
            X_cal_pd = X_cal_pd.drop(columns=["OMB13CBSA"], errors="ignore")
        if cat_cols:
            effective_cat_cols = [c for c in cat_cols if c != "OMB13CBSA"] if not cbsa else cat_cols
            X_cal_pd = prepare_cat_cols(X_cal_pd, effective_cat_cols)
        calibrated_model = calibrate_model(model, X_cal_pd, y_cal, w_cal)
        print("*Platt scaling calibrator fit on held-out calibration set.")

        if name is not None:
            y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test)
            y_prob_cal = calibrated_model.predict_proba(X_test_pd)[:, 1]
            _save_calibration_plot(
                y_test_np, y_pred_proba,
                name=f"{name}_pre_vs_post_cal",
                label=f"{label} — test set" if label else "test set",
                y_prob_cal=y_prob_cal,
                model_name="XGBoost",
            )
            print("*Pre-vs-post calibration plot saved.\n\n")

    return model, calibrated_model, X_test_pd


def _save_calibration_plot(y_true, y_prob, name, label="", n_bins=10, strategy="quantile",
                           fold_data=None, model_name="XGBoost", y_prob_cal=None, cal_label=None):
    """Save a calibration (reliability) diagram from pre-computed probabilities.

    y_prob_cal: optional Platt-scaled probabilities to overlay as a second curve.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    # Bin counts for ECE and per-point annotations — filter empty bins to match
    # what calibration_curve returns (it drops empty bins from both arrays).
    if strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    counts_all, _ = np.histogram(y_prob, bins=bin_edges)
    counts = counts_all[counts_all > 0][: len(prob_pred)]

    brier = brier_score_loss(y_true, y_prob)
    ece = float(np.sum(np.abs(prob_true - prob_pred) * counts / counts.sum()))

    header = f" — {label}" if label else ""
    print(f"\n\nCALIBRATION{header}")
    print(f"Brier score: {brier:.4f}  ECE: {ece:.4f}")

    # Compute post-calibration metrics if provided
    brier_cal = ece_cal = prob_true_cal = prob_pred_cal = None
    if y_prob_cal is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=n_bins, strategy=strategy)
        if strategy == "quantile":
            cal_edges = np.quantile(y_prob_cal, np.linspace(0, 1, n_bins + 1))
            cal_edges = np.unique(cal_edges)
        else:
            cal_edges = np.linspace(0, 1, n_bins + 1)
        counts_cal_all, _ = np.histogram(y_prob_cal, bins=cal_edges)
        counts_cal = counts_cal_all[counts_cal_all > 0][:len(prob_pred_cal)]
        brier_cal = brier_score_loss(y_true, y_prob_cal)
        ece_cal = float(np.sum(np.abs(prob_true_cal - prob_pred_cal) * counts_cal / counts_cal.sum()))
        print(f"Platt-scaled Brier score: {brier_cal:.4f}  ECE: {ece_cal:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Per-fold curves in the background to show fold-to-fold variability
    if fold_data:
        for y_true_fold, y_prob_fold in fold_data:
            try:
                pt, pp = calibration_curve(y_true_fold, y_prob_fold, n_bins=n_bins, strategy=strategy)
                ax.plot(pp, pt, color="steelblue", alpha=0.2, linewidth=1)
            except ValueError:
                pass

    # Histogram of predicted probabilities on a secondary y-axis (pushed to bottom)
    ax2 = ax.twinx()
    hist_vals, _, _ = ax2.hist(y_prob, bins=30, alpha=0.15, color="steelblue", zorder=0)
    ax2.set_ylim(0, hist_vals.max() * 5)
    ax2.set_ylabel("Prediction count", color="steelblue", alpha=0.7, fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=8)

    # Keep calibration curve on top of histogram
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    # Aggregate calibration curve with per-bin sample-count annotations
    ax.plot(prob_pred, prob_true, marker="o", color="steelblue", linewidth=2, label=f"{model_name} (raw)")
    for x, y_pt, n in zip(prob_pred, prob_true, counts):
        ax.annotate(f"n={n:,}", (x, y_pt), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="gray")

    # Platt-scaled overlay curve
    if y_prob_cal is not None and prob_true_cal is not None:
        curve_label = cal_label or f"{model_name} (Platt scaled)"
        ax.plot(prob_pred_cal, prob_true_cal, marker="s", color="darkorange",
                linewidth=2, label=curve_label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Plot{header}")
    ax.legend(loc="upper left", fontsize=9)

    # Metrics in a tidy annotation box (bottom-right)
    if brier_cal is not None:
        metrics_text = (
            f"Before — Brier: {brier:.4f}  ECE: {ece:.4f}\n"
            f"After  — Brier: {brier_cal:.4f}  ECE: {ece_cal:.4f}"
        )
    else:
        metrics_text = f"Brier: {brier:.4f}\nECE:   {ece:.4f}"
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9))

    plt.tight_layout()
    os.makedirs("output/calibration", exist_ok=True)
    plt.savefig(f"output/calibration/{name}_calibration.png", dpi=150)
    plt.close()


def run_calibration_plot(model, X_test_pd, y_test, name, label="", n_bins=10, calibrated_model=None):
    y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
    y_prob = model.predict_proba(X_test_pd)[:, 1]
    y_prob_cal = calibrated_model.predict_proba(X_test_pd)[:, 1] if calibrated_model is not None else None
    _save_calibration_plot(y_true, y_prob, name=name, label=label, n_bins=n_bins, y_prob_cal=y_prob_cal)


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


