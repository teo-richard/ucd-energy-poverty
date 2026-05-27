"""
heterogeneity_utils.py
======================
Shared utility module for all heterogeneity analyses in 04_heterogeneity_analysis/.

Two loaders, depending on what you need:

  load_df()   — reads the CSV only.  ``energy_deprivation`` is already there;
                no tree model required.  Use for observed-outcome steps only.

  load_data() — also loads the model and runs predict_proba() to add
                ``pred_prob``.  Required for the step 1 predicted-probability plot.

Typical usage — one call does step 1 (predicted + observed) and step 2 (observed):

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

All outputs land under output/heterogeneity/{group_col}/.
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
from scipy.stats import gaussian_kde

sys.path.insert(0, "code/00_shared")
from analysis_functions import load_model, prepare_cat_cols, filter_temp_vars, get_feature_names


# ---------------------------------------------------------------------------
# Easily-editable constants
# ---------------------------------------------------------------------------

# Conditioning variables for Step 2 — edit this list to add/remove variables
CONDITIONING_VARS: list[str] = ["WALLCRACK", "ACPRIMARY", "FUSEBLOW", "ROACH", "DISHH", "OMB13CBSA"]

# CBSA conditioning produces a single heatmap rather than per-value density plots
CBSA_COL        = "OMB13CBSA"
CBSA_MIN_N_CELL = 50       # cells with fewer observations are grayed out in heatmap

# Default file paths (relative to project root — run scripts from there)
DEFAULT_MODEL_PATH    = "data/processed/models/current_climate_xgboost_no_cbsa_with_weights_calibrated.pkl"
DEFAULT_CURR_CLM_PATH = "data/processed/current_climate/basic_ready_for_trees_ahs_climate.csv"
DEFAULT_OUTPUT_DIR    = "output/heterogeneity/"

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
DPI          = 150
TITLE_SIZE   = 13
LABEL_SIZE   = 11
TICK_SIZE    = 9
LEGEND_SIZE  = 9
KDE_POINTS   = 400                         # resolution of KDE curves


# ---------------------------------------------------------------------------
# Human-readable labels for coded variables
# ---------------------------------------------------------------------------

LABEL_MAPS: dict[str, dict] = {
    "DISHH": {
        1: "At least 1 disabled person",
        2: "No disabled persons",
    },
    "DIVISION": {
        1: "New England",
        2: "Middle Atlantic",
        3: "East North Central",
        4: "West North Central",
        5: "South Atlantic",
        6: "East South Central",
        7: "West South Central",
        8: "Mountain",
        9: "Pacific",
    },
    "ACPRIMARY": {
        1: "Electric central AC",
        2: "Gas central AC",
        3: "LP gas central AC",
        4: "Other fuel central AC",
        5: "1 room unit",
        6: "2 room units",
        7: "3 room units",
        8: "4 room units",
        9: "5 room units",
        10: "6 room units",
        11: "7+ room units",
        12: "No AC",
    },
    "FUSEBLOW": {
        1: "1 blown (last 3 months)",
        2: "2 blown",
        3: "3 blown",
        4: "4+ blown",
        5: "None blown",
    },
    "ROACH": {
        1: "Seen daily (last 12 months)",
        2: "Seen weekly",
        3: "Seen monthly",
        4: "Seen few times/year",
        5: "None seen",
    },
    "WALLCRACK": {
        1: "Yes (holes/cracks present)",
        2: "No holes/cracks",
    },
    "HHRACE": {
        1: "White alone",
        2: "Black alone",
        3: "American Indian alone",
        4: "Asian alone",
        5: "Native Hawaiian / Pacific Islander",
        6: "Other / multi-racial",
    },
    "TENURE": {
        1: "Own / being bought",
        2: "Rent for cash",
        3: "No cash rent",
    },
    "HEATTYPE": {
        1: "Forced air furnace",
        2: "Steam / hot water",
        3: "Heat pump",
        4: "Electric baseboard",
        5: "Pipeless furnace",
        6: "Portable electric",
        7: "Cooking stove for heat",
        8: "No heating system",
        999: "Other",
    },
    "age_group": {
        "1_Under35": "Under 35",
        "2_35to49": "35–49",
        "3_50to64": "50–64",
        "4_65plus": "65+",
    },
}


def _label(val, label_map: dict | None) -> str:
    """Return the human-readable label for val, falling back to str(val)."""
    if label_map is None:
        return str(val)
    return label_map.get(val, str(val))


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
    true_col: str = "energy_deprivation",
) -> tuple:
    """Run all heterogeneity steps for a grouping variable.

    Produces a step 1 predicted-probability density plot plus step 1–2
    observed-outcome outputs (prevalence bar charts and conditioning heatmaps).

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
    true_col : str
        Column holding the observed binary outcome.  Set to None to skip the
        observed-outcome analysis.

    Returns
    -------
    (df, X_pd, model) — the loaded objects, in case the caller needs them.
    """
    if df is None or X_pd is None or model is None:
        df, X_pd, model = load_data()
    # Step 1: predicted probability distribution
    step1_distribution(df, group_col, output_dir=output_dir)
    # Steps 1–2: observed binary outcome only (no predicted-prob steps after step 1)
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
        label_map=LABEL_MAPS.get(group_col),
    )
    print(f"  Distribution plot → {png_path}")


# ---------------------------------------------------------------------------
# Step 2: Conditional subgroup comparisons
# ---------------------------------------------------------------------------

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
        label_map=LABEL_MAPS.get(group_col),
    )
    print(f"  Observed prevalence plot → {png_path}")


def step2_true_outcome(
    df: pl.DataFrame,
    group_col: str,
    true_col: str = "energy_deprivation",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Conditional subgroup comparisons using the observed binary outcome.

    For each conditioning variable produces a single heatmap: rows = strata of
    the conditioning variable (sorted by overall mean, highest at top), columns =
    group_col subgroups, cells = observed deprivation rate (%).

    Outputs (written to ``{output_dir}/{group_col}/step2/{cond_var}/``):
    - ``{cond_var}_observed_heatmap.png``
    """
    if true_col not in df.columns:
        print(f"\n[Step 2 - Observed] skipped — '{true_col}' not in DataFrame.")
        return

    print(f"\n[Step 2 - Observed] {group_col}")

    lmap = LABEL_MAPS.get(group_col)

    for cond_var in CONDITIONING_VARS:
        if cond_var not in df.columns:
            print(f"  Warning: conditioning variable '{cond_var}' not in DataFrame — skipping.")
            continue
        if cond_var == group_col:
            print(f"  Skipping {cond_var} — same as analysis variable.")
            continue

        cond_out = os.path.join(output_dir, group_col, "step2", cond_var)
        os.makedirs(cond_out, exist_ok=True)

        min_n    = CBSA_MIN_N_CELL if cond_var == CBSA_COL else 5
        out_path = os.path.join(cond_out, f"{cond_var}_observed_heatmap.png")
        _heatmap(
            df=df,
            group_col=group_col,
            cond_col=cond_var,
            value_col=true_col,
            title=f"Obs. Deprivation Rate — {group_col} by {cond_var}",
            out_path=out_path,
            group_label_map=lmap,
            cond_label_map=LABEL_MAPS.get(cond_var),
            min_n_cell=min_n,
        )
        print(f"    → {out_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _heatmap(
    df: pl.DataFrame,
    group_col: str,
    cond_col: str,
    value_col: str,
    title: str,
    out_path: str,
    group_label_map: dict | None = None,
    cond_label_map: dict | None = None,
    min_n_cell: int = 5,
) -> None:
    """Heatmap of mean value_col: cond_col strata (rows) × group_col subgroups (cols).

    Rows are sorted by overall stratum mean (highest at top). Cells with fewer
    than min_n_cell observations are grayed out.
    """
    df = df.filter(pl.col(group_col).is_not_null() & pl.col(cond_col).is_not_null())

    # Per-cell mean and count
    cell_stats = (
        df.group_by([cond_col, group_col])
        .agg([
            pl.col(value_col).drop_nulls().cast(pl.Float64).mean().alias("_mean"),
            pl.col(value_col).drop_nulls().len().alias("_n"),
        ])
    )

    # Row order: cond_col values sorted by overall mean (highest at top)
    cond_order = (
        df.group_by(cond_col)
        .agg(pl.col(value_col).drop_nulls().cast(pl.Float64).mean().alias("_overall"))
        .sort("_overall", descending=True)
        .select(cond_col)
        .to_series()
        .to_list()
    )

    # Column order and labels
    group_vals = sorted(df[group_col].drop_nulls().unique().to_list())
    col_labels = [_label(v, group_label_map) for v in group_vals]
    row_labels = [_label(v, cond_label_map) for v in cond_order]

    # Build value and count matrices
    nrows, ncols = len(cond_order), len(group_vals)
    mat_val  = np.full((nrows, ncols), np.nan)
    mat_n    = np.zeros((nrows, ncols), dtype=int)
    cond_idx  = {v: i for i, v in enumerate(cond_order)}
    group_idx = {v: i for i, v in enumerate(group_vals)}
    for row in cell_stats.to_dicts():
        r = cond_idx[row[cond_col]]
        c = group_idx[row[group_col]]
        mat_val[r, c] = row["_mean"]
        mat_n[r, c]   = row["_n"]

    # Mask cells below minimum n — they will render as gray
    masked = np.ma.masked_where((mat_n < min_n_cell) | np.isnan(mat_val), mat_val)

    # Scale figure: wider for more columns; taller per row for small matrices,
    # compressed for large ones (e.g. many CBSAs)
    max_row_label_len = max((len(l) for l in row_labels), default=5)
    left_pad = max(2.5, max_row_label_len * 0.11)
    fig_w = max(8, ncols * 1.8 + left_pad)
    row_h = 0.5 if nrows > 30 else 0.8
    fig_h = max(4, nrows * row_h + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("lightgray")
    vmax = float(np.nanmax(mat_val)) if not np.all(np.isnan(mat_val)) else 1.0
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    # Annotation font scales down for large matrices
    annot_fs = max(7, min(12, 100 // max(nrows, ncols)))
    threshold = 0.5 * vmax
    for r in range(nrows):
        for c in range(ncols):
            if not np.ma.is_masked(masked[r, c]):
                v = mat_val[r, c]
                color = "white" if v > threshold else "black"
                ax.text(c, r, f"{v:.1%}\n(n={mat_n[r, c]:,})", ha="center", va="center",
                        fontsize=annot_fs, color=color)

    row_tick_fs = 7 if nrows > 30 else TICK_SIZE
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=row_tick_fs)
    ax.set_xlabel(group_col, fontsize=LABEL_SIZE)
    ax.set_ylabel(cond_col, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


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
    label_map: dict | None = None,
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

    labels = [_label(r["val"], label_map) for r in rows]
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
            i - 0.2,                  # slightly above the bar (inverted axis)
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
    label_map: dict | None = None,
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
            label=f"{_label(val, label_map)}  (n={len(probs):,})",
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


def _safe_filename(s: str) -> str:
    """Convert an arbitrary string into a filesystem-safe filename stem."""
    return re.sub(r"[^\w\-]", "_", s).strip("_")
