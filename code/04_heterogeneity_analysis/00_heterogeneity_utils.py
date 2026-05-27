"""
heterogeneity_utils.py
======================
Shared utility module for all heterogeneity analyses in 04_heterogeneity_analysis/.

Two loaders, depending on what you need:

  load_df()   — reads the CSV only.  ``energy_deprivation`` is already there;
                no tree model required.  Use for observed-outcome steps only.

  load_data() — also loads the model and runs predict_proba() to add
                ``pred_prob``.  Required for predicted-probability and SHAP steps.

Typical usage — one call does everything (predicted + observed + SHAP):

    from heterogeneity_utils import run_heterogeneity
    run_heterogeneity("HHRACE")

If you need to pre-process the data first (e.g. bin a continuous variable),
load it yourself and pass it in so nothing is loaded twice:

    from heterogeneity_utils import load_data, run_heterogeneity
    df, X_pd, model = load_data()
    df = df.with_columns(...)   # your pre-processing
    run_heterogeneity("my_col", df=df, X_pd=X_pd, model=model)

For observed-outcome analysis only (no tree model needed):

    from heterogeneity_utils import load_df, step1_true_outcome, step2_true_outcome
    df = load_df()
    step1_true_outcome(df, "HHRACE")
    step2_true_outcome(df, "HHRACE")

All outputs land under tree_model_output/heterogeneity/{group_col}/.
"""

import os
import sys
import re

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from scipy.stats import gaussian_kde

sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, prepare_cat_cols, filter_temp_vars, get_feature_names


# ---------------------------------------------------------------------------
# Easily-editable constants
# ---------------------------------------------------------------------------

# Conditioning variables for Step 2 — edit this list to add/remove variables
CONDITIONING_VARS: list[str] = ["WALLCRACK", "ACPRIMARY", "FUSEBLOW", "ROACH", "DISHH"]

# Default file paths (relative to project root — run scripts from there)
DEFAULT_MODEL_PATH    = "data/processed/models/current_climate_xgboost_no_cbsa_with_weights_calibrated.pkl"
DEFAULT_CURR_CLM_PATH = "data/processed/current_climate/basic_ready_for_trees_ahs_climate_no_cbsa.csv"
DEFAULT_OUTPUT_DIR    = "tree_model_output/heterogeneity/"

# Categorical columns for XGBoost encoding (must match training pipeline)
CAT_COLS_NO_CBSA: list[str] = [
    "TENURE", "BLD", "INTLANG", "DIVISION",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD",
]

# Columns to drop before building the feature matrix (non-feature metadata)
COLS_TO_DROP: list[str] = ["energy_deprivation", "year", "WEIGHT", "CONTROL"]

# ---------------------------------------------------------------------------
# Visual style — change once, applies everywhere
# ---------------------------------------------------------------------------
PALETTE      = list(plt.cm.tab10.colors)   # 10-color consistent cycle
OVERALL_COLOR = "#333333"                  # dark gray for the "Overall" curve/bar
FIGSIZE_DIST = (13, 6)                     # distribution / density plots
FIGSIZE_SHAP = (10, 8)                     # SHAP beeswarm / bar plots
DPI          = 150
TITLE_SIZE   = 13
LABEL_SIZE   = 11
TICK_SIZE    = 9
LEGEND_SIZE  = 9
KDE_POINTS   = 400                         # resolution of KDE curves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_df(curr_clim_path: str = DEFAULT_CURR_CLM_PATH) -> pl.DataFrame:
    """Read the current-climate CSV and return it as a Polars DataFrame.

    ``energy_deprivation`` is already a column in the CSV — no model needed.
    Use this when you only want to run the observed-outcome steps
    (``step1_true_outcome``, ``step2_true_outcome``).

    Use ``load_data()`` instead when you also need predicted probabilities
    (``pred_prob``) or SHAP values.
    """
    print("Reading current-climate CSV …")
    return pl.read_csv(curr_clim_path)


def load_data(
    curr_clim_path: str = DEFAULT_CURR_CLM_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple:
    """Load the calibrated model, compute predicted probabilities, and return
    the full analysis frame alongside the SHAP-ready pandas feature matrix.

    Only needed when running the predicted-probability or SHAP steps.
    For observed-outcome analysis only, use ``load_df()`` — no tree required.

    Returns
    -------
    df_pl : pl.DataFrame
        Original CSV columns + ``pred_prob`` (Platt-scaled deprivation
        probability).  Includes CONTROL, grouping variables, conditioning
        variables, and ``energy_deprivation``.
    X_pd : pd.DataFrame
        Pandas feature matrix in training-column order, categorical-encoded.
        Row order matches ``df_pl`` exactly.
    model : _PlattScaler
        The loaded calibrated XGBoost model.
    """
    print("Loading model …")
    model = load_model(model_path)
    feature_names = get_feature_names(model)

    curr_clim = load_df(curr_clim_path)   # plain CSV read — no model needed here

    # Build pandas feature matrix (same pipeline used during training)
    drop_cols = [c for c in COLS_TO_DROP if c in curr_clim.columns]
    X_raw = curr_clim.drop(drop_cols).to_pandas()
    X_raw = filter_temp_vars(X_raw)
    X_pd  = prepare_cat_cols(X_raw, CAT_COLS_NO_CBSA)
    X_pd  = X_pd[feature_names]

    print("Computing Platt-scaled predicted probabilities …")
    pred_probs = model.predict_proba(X_pd)[:, 1]

    df_pl = curr_clim.with_columns(pl.Series("pred_prob", pred_probs))
    print(f"  Dataset: {len(df_pl):,} rows  |  mean pred_prob = {pred_probs.mean():.4f}")
    return df_pl, X_pd, model


def run_heterogeneity(
    group_col: str,
    df: pl.DataFrame = None,
    X_pd: pd.DataFrame = None,
    model=None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_shap_samples: int = 2000,
    true_col: str = "energy_deprivation",
) -> tuple:
    """Run all heterogeneity steps for a grouping variable.

    Runs Steps 1–3 for **predicted** probabilities (``pred_prob`` column) and
    Steps 1–2 for the **observed** binary outcome (``true_col`` column) in
    parallel, so both analyses land in the same output directory.

    Parameters
    ----------
    group_col : str
        Column in ``df`` to group by (e.g. "HHRACE", "TENURE", "age_group").
    df, X_pd, model : optional
        Pre-loaded data and model.  If any are None, ``load_data()`` is called
        automatically — useful when the caller already has these in memory and
        wants to avoid loading them a second time.
    output_dir : str
        Root output directory; subdirectories are created as needed.
    max_shap_samples : int
        Per-subgroup SHAP row cap (randomly subsampled, seed=42).
    true_col : str
        Column holding the observed binary outcome.  Set to None to skip the
        observed-outcome analysis.

    Returns
    -------
    (df, X_pd, model) — the loaded objects, in case the caller needs them.
    """
    if df is None or X_pd is None or model is None:
        df, X_pd, model = load_data()
    # Predicted probability analysis (Steps 1–3)
    step1_distribution(df, group_col, output_dir=output_dir)
    step2_conditional(df, group_col, output_dir=output_dir)
    step3_shap(df, group_col, X_pd, model, output_dir=output_dir, max_samples=max_shap_samples)
    # Observed outcome analysis (Steps 1–2; SHAP is model-based and doesn't apply)
    if true_col is not None:
        step1_true_outcome(df, group_col, true_col=true_col, output_dir=output_dir)
        step2_true_outcome(df, group_col, true_col=true_col, output_dir=output_dir)
    return df, X_pd, model


# ---------------------------------------------------------------------------
# Step 1: Distribution of predicted probabilities
# ---------------------------------------------------------------------------

def step1_distribution(
    df: pl.DataFrame,
    group_col: str,
    prob_col: str = "pred_prob",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Compute and export summary statistics and a density plot for the
    predicted deprivation probability, overall and by subgroup.

    Outputs (written to ``{output_dir}/{group_col}/step1/``):
    - ``stats.csv``         — per-group + Overall summary statistics
    - ``distribution.png``  — overlapping KDE density curves
    """
    out_dir = os.path.join(output_dir, group_col, "step1")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Step 1] {group_col}")

    # --- Summary statistics --------------------------------------------------
    stats = _compute_stats(df, group_col, prob_col)
    csv_path = os.path.join(out_dir, "stats.csv")
    stats.write_csv(csv_path)
    print(f"  Stats CSV → {csv_path}")

    # --- Density plot ---------------------------------------------------------
    png_path = os.path.join(out_dir, "distribution.png")
    _density_plot(
        df=df,
        group_col=group_col,
        prob_col=prob_col,
        title=f"Predicted Deprivation Probability — by {group_col}",
        out_path=png_path,
    )
    print(f"  Distribution plot → {png_path}")


# ---------------------------------------------------------------------------
# Step 2: Conditional subgroup comparisons
# ---------------------------------------------------------------------------

def step2_conditional(
    df: pl.DataFrame,
    group_col: str,
    prob_col: str = "pred_prob",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """For each conditioning variable, stratify the sample by that variable's
    values and recompute Step 1 outputs within each stratum.

    Outputs (written to ``{output_dir}/{group_col}/step2/{cond_var}/``):
    - ``{cond_var}_{cond_val}_stats.csv``
    - ``{cond_var}_{cond_val}_distribution.png``
    """
    print(f"\n[Step 2] {group_col}")

    for cond_var in CONDITIONING_VARS:
        if cond_var not in df.columns:
            print(f"  Warning: conditioning variable '{cond_var}' not in DataFrame — skipping.")
            continue

        cond_out = os.path.join(output_dir, group_col, "step2", cond_var)
        os.makedirs(cond_out, exist_ok=True)

        unique_vals = (
            df.select(pl.col(cond_var).drop_nulls().unique())
            .to_series()
            .to_list()
        )
        unique_vals = sorted(unique_vals)

        print(f"  Conditioning on {cond_var} ({len(unique_vals)} strata) …")

        for val in unique_vals:
            stratum = df.filter(pl.col(cond_var) == val)
            if len(stratum) < 5:
                # Too few rows for meaningful density estimation
                continue

            safe_val = _safe_filename(str(val))
            stem     = f"{cond_var}_{safe_val}"

            # Stats CSV
            stats = _compute_stats(stratum, group_col, prob_col)
            stats.write_csv(os.path.join(cond_out, f"{stem}_stats.csv"))

            # Density plot
            _density_plot(
                df=stratum,
                group_col=group_col,
                prob_col=prob_col,
                title=f"Pred. Deprivation Prob. — {group_col} | {cond_var} = {val}",
                out_path=os.path.join(cond_out, f"{stem}_distribution.png"),
            )

        print(f"    → {cond_out}/")


# ---------------------------------------------------------------------------
# Steps 1–2: Observed binary outcome (energy_deprivation)
# ---------------------------------------------------------------------------

def step1_true_outcome(
    df: pl.DataFrame,
    group_col: str,
    true_col: str = "energy_deprivation",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Compute and export observed deprivation prevalence by subgroup.

    Unlike Step 1 for predicted probabilities, the true outcome is binary so a
    KDE density plot is not meaningful.  Instead this produces a prevalence bar
    chart with 95 % confidence intervals.

    Outputs (written to ``{output_dir}/{group_col}/step1/``):
    - ``observed_stats.csv``      — same schema as predicted stats; mean = prevalence
    - ``observed_prevalence.png`` — horizontal bar chart sorted by prevalence
    """
    if true_col not in df.columns:
        print(f"\n[Step 1 - Observed] skipped — '{true_col}' not in DataFrame.")
        return

    out_dir = os.path.join(output_dir, group_col, "step1")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Step 1 - Observed] {group_col}")

    stats = _compute_stats(df, group_col, true_col)
    csv_path = os.path.join(out_dir, "observed_stats.csv")
    stats.write_csv(csv_path)
    print(f"  Observed stats CSV → {csv_path}")

    png_path = os.path.join(out_dir, "observed_prevalence.png")
    _prevalence_plot(
        df=df,
        group_col=group_col,
        binary_col=true_col,
        title=f"Observed Energy Deprivation Rate — by {group_col}",
        out_path=png_path,
    )
    print(f"  Observed prevalence plot → {png_path}")


def step2_true_outcome(
    df: pl.DataFrame,
    group_col: str,
    true_col: str = "energy_deprivation",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Conditional subgroup comparisons using the observed binary outcome.

    Mirrors ``step2_conditional`` but produces prevalence bar charts instead of
    KDE density plots.

    Outputs (written to ``{output_dir}/{group_col}/step2/{cond_var}/``):
    - ``{cond_var}_{cond_val}_observed_stats.csv``
    - ``{cond_var}_{cond_val}_observed_prevalence.png``
    """
    if true_col not in df.columns:
        print(f"\n[Step 2 - Observed] skipped — '{true_col}' not in DataFrame.")
        return

    print(f"\n[Step 2 - Observed] {group_col}")

    for cond_var in CONDITIONING_VARS:
        if cond_var not in df.columns:
            print(f"  Warning: conditioning variable '{cond_var}' not in DataFrame — skipping.")
            continue

        cond_out = os.path.join(output_dir, group_col, "step2", cond_var)
        os.makedirs(cond_out, exist_ok=True)

        unique_vals = sorted(
            df.select(pl.col(cond_var).drop_nulls().unique()).to_series().to_list()
        )

        print(f"  Conditioning on {cond_var} ({len(unique_vals)} strata) …")

        for val in unique_vals:
            stratum = df.filter(pl.col(cond_var) == val)
            if len(stratum) < 5:
                continue

            safe_val = _safe_filename(str(val))
            stem     = f"{cond_var}_{safe_val}"

            stats = _compute_stats(stratum, group_col, true_col)
            stats.write_csv(os.path.join(cond_out, f"{stem}_observed_stats.csv"))

            _prevalence_plot(
                df=stratum,
                group_col=group_col,
                binary_col=true_col,
                title=f"Obs. Deprivation Rate — {group_col} | {cond_var} = {val}",
                out_path=os.path.join(cond_out, f"{stem}_observed_prevalence.png"),
            )

        print(f"    → {cond_out}/")


# ---------------------------------------------------------------------------
# Step 3: SHAP beeswarm plots by subgroup
# ---------------------------------------------------------------------------

def step3_shap(
    df: pl.DataFrame,
    group_col: str,
    X_pd: pd.DataFrame,
    model,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_samples: int = 2000,
) -> None:
    """Generate SHAP beeswarm and mean-absolute-SHAP bar charts for the full
    sample and for each subgroup defined by ``group_col``.

    Outputs (written to ``{output_dir}/{group_col}/step3/``):
    - ``overall_beeswarm.png`` / ``overall_bar.png``
    - ``{group_val}_beeswarm.png`` / ``{group_val}_bar.png``  (per subgroup)

    Parameters
    ----------
    max_samples : int
        Maximum rows passed to the SHAP explainer per subgroup.  Large datasets
        are randomly subsampled (seed=42).  Set to None to use all rows.
    """
    out_dir = os.path.join(output_dir, group_col, "step3")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Step 3] {group_col} — SHAP")

    # SHAP requires the raw XGBoost booster, not the Platt-scaling wrapper
    raw_model = model.estimator if hasattr(model, "estimator") else model
    explainer = shap.TreeExplainer(raw_model)

    # Rebuild a positional index array aligned with df (both built from same CSV)
    group_series = df[group_col].to_numpy()  # numpy array for fast masking

    # --- Overall baseline ---------------------------------------------------
    print("  Computing overall SHAP values …")
    X_overall = _subsample(X_pd, max_samples, seed=42)
    _shap_beeswarm_and_bar(
        explainer=explainer,
        X_sub=X_overall,
        label="Overall",
        out_dir=out_dir,
        stem="overall",
    )

    # --- Per-subgroup -------------------------------------------------------
    unique_vals = sorted(
        v for v in df[group_col].drop_nulls().unique().to_list()
    )
    for val in unique_vals:
        mask  = group_series == val
        X_grp = X_pd.iloc[mask]

        if len(X_grp) < 5:
            print(f"  Skipping {group_col}={val} (n={len(X_grp)} < 5)")
            continue

        X_sub = _subsample(X_grp, max_samples, seed=42)
        safe  = _safe_filename(str(val))
        print(f"  {group_col}={val}  n={len(X_grp):,}  (SHAP on {len(X_sub):,} rows)")

        _shap_beeswarm_and_bar(
            explainer=explainer,
            X_sub=X_sub,
            label=f"{group_col} = {val}",
            out_dir=out_dir,
            stem=safe,
        )

    print(f"  → {out_dir}/")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_stats(df: pl.DataFrame, group_col: str, prob_col: str) -> pl.DataFrame:
    """Return a Polars DataFrame of summary statistics per group + Overall row.

    Columns: group_val (str), mean, std, median, p25, p75, p10, p90, n
    ``group_val`` holds each subgroup's label (cast to string) plus "Overall".
    """
    aggs = [
        pl.col(prob_col).mean().alias("mean"),
        pl.col(prob_col).std().alias("std"),
        pl.col(prob_col).median().alias("median"),
        pl.col(prob_col).quantile(0.25).alias("p25"),
        pl.col(prob_col).quantile(0.75).alias("p75"),
        pl.col(prob_col).quantile(0.10).alias("p10"),
        pl.col(prob_col).quantile(0.90).alias("p90"),
        pl.len().alias("n"),
    ]

    # Per-group rows — cast the group key to string so it merges cleanly with "Overall"
    by_group = (
        df.group_by(group_col)
        .agg(aggs)
        .with_columns(pl.col(group_col).cast(pl.Utf8).alias("group_val"))
        .drop(group_col)
        .select(["group_val", "mean", "std", "median", "p25", "p75", "p10", "p90", "n"])
        .sort("group_val")
    )

    # Overall row
    overall = (
        df.select(aggs)
        .with_columns(pl.lit("Overall").alias("group_val"))
        .select(["group_val", "mean", "std", "median", "p25", "p75", "p10", "p90", "n"])
    )

    return pl.concat([overall, by_group])


def _prevalence_plot(
    df: pl.DataFrame,
    group_col: str,
    binary_col: str,
    title: str,
    out_path: str,
) -> None:
    """Horizontal bar chart of observed deprivation prevalence per subgroup.

    Bars are sorted descending by prevalence (highest risk at top).
    Error bars show the 95 % normal-approximation confidence interval.
    A vertical dashed line marks the overall prevalence as a reference.
    """
    groups = sorted(df[group_col].drop_nulls().unique().to_list())

    rows = []
    for val in groups:
        arr = (
            df.filter(pl.col(group_col) == val)[binary_col]
            .drop_nulls()
            .cast(pl.Float64)
            .to_numpy()
        )
        n = len(arr)
        p = float(arr.mean()) if n > 0 else 0.0
        se = float(np.sqrt(p * (1 - p) / n)) if n > 1 else 0.0
        rows.append({"val": val, "p": p, "ci": 1.96 * se, "n": n})

    # Sort descending by prevalence so the highest-risk group is at the top
    rows.sort(key=lambda r: r["p"], reverse=True)

    overall_arr = df[binary_col].drop_nulls().cast(pl.Float64).to_numpy()
    overall_prev = float(overall_arr.mean())

    labels = [str(r["val"]) for r in rows]
    ps     = [r["p"]  for r in rows]
    cis    = [r["ci"] for r in rows]
    ns     = [r["n"]  for r in rows]
    # Colour by original sorted position so colours are stable across subgroups
    group_order = {v: i for i, v in enumerate(groups)}
    colors = [PALETTE[group_order[r["val"]] % len(PALETTE)] for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE_DIST)
    y = np.arange(len(rows))

    ax.barh(
        y, ps, xerr=cis,
        color=colors, alpha=0.85, capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "dimgray"},
    )

    # Annotate each bar with prevalence % and n.
    # Offset y by -0.3 (axis is inverted, so -0.3 = slightly above the bar on screen)
    # so the label clears the horizontal error bar line and its caps.
    x_max = max(p + c for p, c in zip(ps, cis)) if ps else 0.1
    for i, (p, ci, n) in enumerate(zip(ps, cis, ns)):
        ax.text(
            p + ci + x_max * 0.01,   # just past this bar's own error bar cap
            i - 0.3,                  # slightly above the bar (inverted axis)
            f"{p:.1%}  (n={n:,})",
            va="center", fontsize=TICK_SIZE,
        )

    # Overall reference line — capture handle for legend
    vline = ax.axvline(
        overall_prev, color=OVERALL_COLOR, linestyle="--", linewidth=1.8,
        label=f"Overall = {overall_prev:.1%}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.invert_yaxis()   # highest prevalence at the top
    ax.set_xlabel("Share energy deprived (observed)", fontsize=LABEL_SIZE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(axis="x", color="lightgray", linewidth=0.6, zorder=0)

    # Legend: one coloured patch per subgroup + the overall reference line
    bar_patches = [
        mpatches.Patch(color=colors[i], alpha=0.85, label=labels[i])
        for i in range(len(rows))
    ]
    ax.legend(handles=bar_patches + [vline], fontsize=LEGEND_SIZE,
              loc="lower right", frameon=True)

    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _density_plot(
    df: pl.DataFrame,
    group_col: str,
    prob_col: str,
    title: str,
    out_path: str,
) -> None:
    """Overlapping KDE density curves — one per subgroup plus an Overall curve."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DIST)

    x_grid = np.linspace(0, 1, KDE_POINTS)

    # Overall curve (drawn last so it sits on top)
    all_probs = df[prob_col].drop_nulls().to_numpy()

    groups = sorted(df[group_col].drop_nulls().unique().to_list())

    # Per-group curves
    for i, val in enumerate(groups):
        probs = df.filter(pl.col(group_col) == val)[prob_col].drop_nulls().to_numpy()
        if len(probs) < 5:
            continue
        kde = gaussian_kde(probs)
        ax.plot(
            x_grid,
            kde(x_grid),
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.8,
            label=f"{val}  (n={len(probs):,})",
            alpha=0.85,
        )

    # Overall dashed curve on top
    kde_all = gaussian_kde(all_probs)
    ax.plot(
        x_grid,
        kde_all(x_grid),
        color=OVERALL_COLOR,
        linewidth=2.2,
        linestyle="--",
        label=f"Overall  (n={len(all_probs):,})",
        zorder=10,
    )

    ax.set_xlabel("Predicted deprivation probability", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)

    legend = ax.legend(
        fontsize=LEGEND_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _shap_beeswarm_and_bar(
    explainer: shap.TreeExplainer,
    X_sub: pd.DataFrame,
    label: str,
    out_dir: str,
    stem: str,
) -> None:
    """Compute SHAP values for X_sub and save a beeswarm + mean-abs bar chart.

    SHAP's plot functions draw onto whatever figure is current, so we use
    ``plt.figure()`` (matching the pattern in analysis_functions.run_shap)
    rather than the object-oriented ``fig, ax`` API.
    """
    explanation = explainer(X_sub)

    # --- Beeswarm -----------------------------------------------------------
    plt.figure(figsize=FIGSIZE_SHAP)
    shap.plots.beeswarm(explanation, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {label}", fontsize=TITLE_SIZE)
    plt.tick_params(labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}_beeswarm.png"), dpi=DPI, bbox_inches="tight")
    plt.clf()
    plt.close("all")

    # --- Mean absolute SHAP bar chart ---------------------------------------
    plt.figure(figsize=FIGSIZE_SHAP)
    shap.plots.bar(explanation.abs.mean(0), show=False, max_display=20)
    plt.title(f"Mean |SHAP| — {label}", fontsize=TITLE_SIZE)
    plt.tick_params(labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}_bar.png"), dpi=DPI, bbox_inches="tight")
    plt.clf()
    plt.close("all")


def _subsample(X: pd.DataFrame, max_samples: int | None, seed: int = 42) -> pd.DataFrame:
    """Return a random subsample of X if it exceeds max_samples rows."""
    if max_samples is None or len(X) <= max_samples:
        return X
    return X.sample(n=max_samples, random_state=seed)


def _safe_filename(s: str) -> str:
    """Convert an arbitrary string into a filesystem-safe filename stem."""
    return re.sub(r"[^\w\-]", "_", s).strip("_")
