# Energy Poverty ML Project — Full Notes

## Project Overview

**Dataset:** American Housing Survey (AHS) 2023
- 26,700 households across 15 US metro areas (CBSAs)
- 107 variables: housing characteristics, utilities, demographics, neighborhood quality, financial variables
- Joined with NClimGrid 2023 climate data (county-level, aggregated to CBSA)
- Climate projection data: CMIP6-LOCA2 (1950–2100, scenarios aligned with 1.5°C, 2°C, 3°C warming)

**Target Variable:** Binary energy poverty — household spends >10% of income on utilities
- ~17% of households are energy poor (4,581 of 26,700)
- Baseline energy poverty rate: 17.16%

**Overall Goal:** Identify which households are at risk of energy poverty as climate changes. Train ML model on current AHS + climate data, then run inference with projected climate data to see if the distribution of energy-poor households changes.

**Key Assumption:** Survey characteristics (building age, AC ownership, income, etc.) stay fixed while only climate variables change. This is a known limitation — households most at risk in 2050 may not exist in the 2023 survey sample. Sensitivity analysis planned by varying distributions of key characteristics.

---

## Model Choice: Why Tree-Based Models

Tree-based models are well-suited to this problem because:
- Mixed variable types (categorical housing characteristics + continuous climate variables + binary flags)
- Likely nonlinear interactions (e.g., high CDD only matters for households without AC)
- Class imbalance (17% positive) — trees handle this more gracefully than linear models
- Feature importance / SHAP values for interpretation

**No need to worry about with tree models:**
- Multicollinearity
- Outliers (splits are rank-based)
- Normalization/scaling

**Still need to handle:**
- Missing values (LightGBM handles natively; XGBoost has limitations for categoricals with NAs)
- Categorical encoding (depends on library)
- Target leakage (biggest risk — biggest mistake you can make)

---

## LightGBM vs XGBoost

| | LightGBM | XGBoost |
|---|---|---|
| Tree growth | Leaf-wise | Level-wise |
| Speed | Faster | Slightly slower |
| Native categoricals | Yes, robust | Yes, but newer and less mature |
| Native NA handling | Yes, for all column types | Yes for numeric; limited for categoricals |
| Categorical encoding | `category` dtype, auto-detected | `category` dtype, must set `enable_categorical=True` and `tree_method='hist'` |

**LightGBM native categorical handling:** Sorts categories by mean target value and finds the best binary cut on that sorted order. Efficiently approximates optimal categorical split. Requires columns to be cast to `category` dtype. Default max of 32 categories per variable (`max_cat_threshold=32`); above that it falls back to cruder method.

**Bottom line:** LightGBM is the better default choice for this dataset because of robust native categorical and NA handling. XGBoost produces very similar results but requires more preprocessing.

---

## Two Dataset Strategy

**Dataset 1 — LightGBM native:**
- Minimal preprocessing
- Collapsed categoricals (where needed), cast to `category` dtype
- NAs kept as-is (LightGBM learns from missingness)
- Best for raw predictive performance

**Dataset 2 — One-hot encoded:**
- Collapsed categoricals → one-hot encoded
- Works with any model (XGBoost, sklearn RandomForest, logistic regression)
- NAs imputed
- Enables fair model comparison

**Key rule:** Collapsing decisions should be identical across both datasets. The only difference is the final encoding step and NA handling.

**Workflow:**
```
raw data
    ↓
feature engineering        ← create new variables here
    ↓
categorical collapsing
    ↓
    ├── LightGBM dataset   (category dtypes, NAs kept)
    └── Encoded dataset    (one-hot encoded, NAs imputed)
```

---

## NA Handling

**LightGBM:** Keep NAs as-is. LightGBM learns which direction to send missing values at each split based on what reduces loss. Many NAs are structurally informative (e.g., missing AC type likely means no AC system). Imputing would destroy this signal.

**XGBoost with categoricals:** Cannot handle NaN in categorical columns natively. Fix:
```python
for col in cat_cols:
    X_train_pd[col] = X_train_pd[col].fillna(-1).astype(int).astype('category')
    X_test_pd[col] = X_test_pd[col].fillna(-1).astype(int).astype('category')
```
Using -1 as sentinel: XGBoost treats it as a distinct "missing" category and learns patterns from missingness. Not identical to true NA handling but reasonable in practice.

**Why pandas category dtype causes float issues with XGBoost:** When a column has NaN values, pandas stores it as float64 (because integers can't represent NaN). When cast to `category`, the categories array contains floats. XGBoost's categorical split-finding algorithm requires integer or string labels and explicitly rejects float category indices. Fix is to fill NAs before casting so pandas can use integer dtype.

**Variables with >50% missingness:** Consider dropping rather than relying on NA handling.

---

## Categorical Variable Handling

### Nominal vs Ordinal vs Continuous

**Nominal (unordered categories):** Cast to `category` dtype for LightGBM/XGBoost, or one-hot encode. Integer codes are arbitrary labels with no numeric meaning (e.g., HEATTYPE where code 1 = forced air furnace, code 8 = portable heater — 8 is not "more" than 1).

**Ordinal (ordered categories):** Map to integers that preserve ordering, treat as numeric. Do NOT cast to `category` — this would throw away ordering information. Do NOT one-hot encode — this throws away ordering too. Tree models find threshold splits (e.g., "education level < 3") which is meaningful for ordinals.

Example:
```python
edu_map = {
    "less than HS": 0,
    "HS diploma": 1,
    "some college": 2,
    "bachelors": 3,
    "graduate": 4
}
df['education'] = df['education'].map(edu_map)
```

**Why OLS handles ordinals differently:** OLS assumes equal spacing between ordinal levels (1→2 = 2→3), which is wrong, so dummy-coding is required. Tree models don't assume any functional form — they find thresholds empirically — so ordinals can stay as integers.

**Continuous:** Leave as-is.

### Collapsing Categories

**Frequency threshold approach:** Flag categories below ~1% of observations as candidates for collapsing. At 26,700 households, 1% = 267 households. Below this, mean target estimates per category become noisy.

**But frequency is only a starting heuristic.** Domain knowledge drives the actual decision:

- Ask: does this variable have a plausible direct mechanism with energy poverty?
- Ask: are rare categories semantically distinct from existing categories?
- Ask: what is the energy poverty rate in rare categories vs the 17% baseline? If near 17%, collapsing loses almost no signal. If far from 17%, be careful.

**Two types of rare categories:**
1. Genuinely uncommon situation (e.g., wood stove heating) → collapse to Other
2. One concept split across many codes (e.g., 1 room AC, 2 room ACs, 3 room ACs...) → collapse together into their own group, NOT into Other

**Red flags:**
- "Other" becomes the largest category after collapsing → threshold too aggressive or drop variable
- All categories have ~17% EP rate → variable has no signal, consider dropping
- Categories that are actually missing data codes (e.g., "not applicable", "don't know") → treat as NAs

**Collapsing in Polars:**
```python
heattype_map = {1: 1, 2: 2, 3: 3, ...}
df = df.with_columns(
    pl.col("HEATTYPE")
    .replace(heattype_map, default=999)
    .alias("HEATTYPE")
)
```
`default=999` handles any codes not explicitly listed. Unused keys are silently ignored — fine if some codes don't appear in your data.

### Collapsed Variable Codebook

**ACPRIMARY — Type of Primary Air Conditioning**

| New Code | Description | Original Codes |
|---|---|---|
| 1 | Central AC | 1, 2, 3, 4 |
| 2 | Room AC | 5, 6, 7, 8, 9, 10, 11 |
| 3 | No AC | 12 |
| 999 | Other/Unknown | all others |

Rationale: The mechanistically meaningful distinction for energy poverty is central AC vs room AC vs no AC. Number of room units (codes 5–11) is irrelevant — what matters is the household type. This is a "one concept split across many codes" case — do not collapse room AC codes into Other.

**HEATTYPE — Type of Main Heating Equipment**

| New Code | Description | Original Codes |
|---|---|---|
| 1 | Forced air furnace | 1 |
| 2 | Steam or hot water system | 2 |
| 3 | Electric heat pump | 3 |
| 4 | Electric baseboard/coils | 4 |
| 5 | Floor/wall/pipeless furnace | 5 |
| 6 | Portable electric heaters | 8 |
| 7 | Cooking stove used for heating | 14 |
| 8 | No heating system | 13 |
| 999 | Other (vented/unvented room heaters, wood, fireplace) | 6, 7, 9, 10, 11, 12 |

Rationale: Codes 8 (portable electric), 13 (no heat), and 14 (cooking stove) are strong poverty signals — households using these can't afford proper heating. Keep them separate. Codes 6, 7, 9–12 are various room heaters and fireplaces — rare, mixed mechanisms, collapse to 999.

**SUPP1HEAT — First Type of Supplemental Heating**

| New Code | Description | Original Codes |
|---|---|---|
| 1 | No supplemental heating | 14 |
| 2 | Forced air / heat pump | 3, 4 |
| 3 | Steam/hot water | 8 |
| 4 | Cookstove/oven used for heat | 1, 5 |
| 5 | Portable electric heaters | 7 |
| 6 | Wood/biomass | 9, 12 |
| 7 | Pipeless furnace | 6 |
| 8 | Room heaters (vented/unvented) | 10, 11 |
| 9 | Built-in electric | 2 |
| 999 | Other/Not reported | 13, -9 |

Rationale: Codes 1, 5 (cookstove/oven for heat) and code 7 (portable electric) are strong poverty signals. Do not auto-collapse by frequency — would lose this signal.

**BLD — Type of Housing Unit**

| New Code | Description | Original Codes |
|---|---|---|
| 1 | Mobile home or trailer | 1 |
| 2 | Single family detached | 2 |
| 3 | Single family attached | 3 |
| 4 | Small multifamily (2–4 units) | 4, 5 |
| 5 | Medium multifamily (5–19 units) | 6, 7 |
| 6 | Large multifamily (20+ units) | 8, 9 |
| 999 | Other (boat, RV, van, etc.) | 10 |

Rationale: Mobile homes kept separate (28% EP rate, poor insulation). Single family detached vs attached kept separate (different exterior surface area → different heating/cooling loss). Building size groupings reflect shared utilities and efficiency standards. Boat/RV/van collapsed to 999.

**HHRACE — Race of Householder**

| New Code | Description | Original Codes |
|---|---|---|
| 1 | White only | 1 |
| 2 | Black only | 2 |
| 3 | American Indian / Alaska Native only | 3 |
| 4 | Asian only | 4 |
| 5 | Hawaiian / Pacific Islander only | 5 |
| 6 | Multiracial | 6–21 |
| 999 | Not applicable | -6 |

Rationale: Keep codes 3 (AIAN) and 5 (Hawaiian/PI) separate despite small samples — distinct historical relationships with housing policy. Small sample size means noisy estimates, which is an honest reflection of the data.

**HHFNTVTY — Householder's Birth Country**
Collapsed to regions: US-born (1), Europe (2), Mexico/Central America (3), Caribbean (4), South America (5), East/Southeast Asia (6), South/West Asia (7), Africa (8), Canada/Oceania/Other (9), Not applicable (999). Note: code 210 (India) belongs in South/West Asia (7), not East Asia.

### Variables NOT Collapsed

**Low cardinality (under ~10 categories):** `TENURE`, `INTLANG`, `HHMAR`, `HHCITSHP`, `HSHLDTYPE`, `COOKFUEL`, `DRYER`, `FIREPLACE`, `MULTIGEN`

**Geography (kept distinct):** `DIVISION`, `OMB13CBSA`

**Ordinal (mapped to integer):** `HHGRAD`, `YRBUILT`, `UNITSIZE`, `ROACH`, `FUSEBLOW`, `NUM*` disability/asthma vars

**Already binary:** `PARFOREIGNCOUNTRY`, `HHSAMECOUNTRY`, `PARSAMECOUNTRY`

### Variables Dropped

- `SEWTYPE`: 91.5% of households are code 1, everything else tiny and near 17% baseline EP rate — no signal
- `ACSECNDRY`: Decided to drop; ACPRIMARY already captures AC situation
- `CONTROL`: Household ID — meaningless arbitrary number, must not be included as feature
- `INTMONTH`: Interview month — no causal mechanism for energy poverty, likely picking up seasonal utility cost artifact

### Special Variable Handling

**PERPOVLVL:** Codes 2–500 are literal percentages. Code 1 = "≤0%" should be recoded to 0. Code 501 = "501%+" is fine as-is.
```python
df = df.with_columns(
    pl.col("PERPOVLVL").replace({1: 0, 501: 501})
)
```

---

## Feature Engineering

Create engineered features before splitting into the two datasets so both benefit. Examples:

**Binary flags:**
- `native_born_same_as_parent` — immigrant generation status, cleaner signal than birth country alone

**Interaction terms encoding known mechanisms:**
- `no_ac_AND_high_cdd` — households without AC in hot climates are specifically vulnerable
- `old_building_AND_cold_climate` — poor insulation matters more with high HDD
- `income_to_hdd_ratio` — climate burden relative to income

**Derived continuous variables:**
- Utility cost per square foot
- HDD + CDD as total climate burden

**Caution:** Do not engineer features that encode the target. `utility_cost_as_pct_income` IS the target — never include it as a feature.

---

## Leakage Audit

**Critical — do this before modeling.** Variables derived from or directly encoding utility spending must be dropped from features.

**Confirmed leakage — drop:**
- `yearly_utils_cost` — direct leakage, target is derived from this
- `TOTHCAMT` — total housing costs, includes utilities

**Checked and confirmed clean:**
- `PERPOVLVL` vs `yearly_utils_cost`: correlation 0.30 — legitimate, poverty level is genuinely related to energy poverty
- `HINCP` vs `yearly_utils_cost`: correlation 0.29 — legitimate, higher income households spend more on utilities in absolute terms
- `POVLVLINC` vs `yearly_utils_cost`: correlation 0.39 — legitimate

**Highest correlation with `energy_poverty`:** `PERPOVLVL` at 0.598 — not leakage, this is a real causal relationship (low income relative to poverty line → more likely to spend >10% on utilities).

**Diagnostic:** After removing confirmed leakage variables, AUC-ROC should drop from ~0.9999 to a realistic range (~0.90–0.97). If still near 1.0, more leakage exists.

---

## Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = ahs_climate.drop(["energy_poverty", "yearly_utils_cost", "TOTHCAMT", 
                       "CONTROL", "WEIGHT"])
y = ahs_climate["energy_poverty"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**`stratify=y`:** Ensures 17%/83% class split is preserved in both train and test sets. Essential for imbalanced classification.

**Survey weights:** Extract before splitting, attach temporarily to X so they split in parallel:
```python
weights = ahs_climate["WEIGHT"]
X_with_weight = X.with_column(weights.alias("WEIGHT"))

X_train_w, X_test_w, y_train, y_test = train_test_split(
    X_with_weight, y, test_size=0.2, random_state=42, stratify=y
)

w_train = X_train_w["WEIGHT"]
w_test = X_test_w["WEIGHT"]
X_train = X_train_w.drop("WEIGHT")
X_test = X_test_w.drop("WEIGHT")
```

Then pass to `.fit()`:
```python
model.fit(
    X_train, y_train,
    sample_weight=w_train.to_numpy(),
    eval_set=[(X_test, y_test)],
    ...
)
```

**Note:** Do NOT pass `sample_weight` to `eval_set`. Evaluation set is for monitoring convergence only — keep it unweighted.

---

## LightGBM Model

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",  # handles 17% class imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train.to_numpy(), y_train.to_numpy(),
    eval_set=[(X_test.to_numpy(), y_test.to_numpy())],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)
```

**Note on `.to_numpy()`:** Required due to version conflict between LightGBM and sklearn. Save feature names separately before converting:
```python
feature_names = X_train.columns  # Polars returns plain list, no .to_list() needed
```

**`class_weight="balanced"`:** Automatically adjusts for class imbalance. Equivalent to setting `scale_pos_weight = 17695/3665 ≈ 4.83` manually.

**Early stopping:** Stops training when validation AUC stops improving for 50 rounds. Prevents overfitting. Do NOT pass `early_stopping_rounds` to `.fit()` in newer versions — put it in the constructor or use callbacks.

**Results:**
- Early stopped at iteration 648
- AUC-ROC: 0.96
- Precision (energy poor class): 0.69
- Recall (energy poor class): 0.83
- Accuracy: 0.91

**On precision of 0.68–0.69:** Not as bad as it looks given 17% class imbalance. For policy targeting (identifying at-risk households), high recall is more important than high precision — missing a vulnerable household is worse than investigating a non-vulnerable one. Precision/recall tradeoff can be adjusted via classification threshold:
```python
threshold = 0.6
y_pred_adjusted = (y_pred_proba > threshold).astype(int)
```

---

## XGBoost Model

```python
import xgboost as xgb

# Convert to pandas for XGBoost
X_train_pd = X_train.to_pandas()
X_test_pd = X_test.to_pandas()

# Cast nominal categoricals to category dtype
# Must fill NAs first — pandas stores columns with NaN as float64,
# and XGBoost's categorical handling requires integer codes
cat_cols = [
    "TENURE", "BLD", "INTLANG", "DIVISION", "OMB13CBSA",
    "HHMAR", "HHRACE", "HHCITSHP", "HSHLDTYPE", "MILHH", "PARTNER",
    "COOKFUEL", "DRYER", "HOTWATER", "HEATFUEL", "HEATTYPE",
    "ACPRIMARY", "SUPP1HEAT", "FIREPLACE", "MULTIGEN", "SAMEHHLD"
]

for col in cat_cols:
    X_train_pd[col] = X_train_pd[col].fillna(-1).astype(int).astype('category')
    X_test_pd[col] = X_test_pd[col].fillna(-1).astype(int).astype('category')

model_xgb = xgb.XGBClassifier(
    enable_categorical=True,
    tree_method="hist",        # required for native categorical support
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,               # level-wise growth; ~equivalent to num_leaves=64
    scale_pos_weight=4.83,     # 17695/3665 = 4.83 (negatives/positives)
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=50   # in constructor for newer XGBoost versions
)

model_xgb.fit(
    X_train_pd, y_train.to_numpy(),
    eval_set=[(X_test_pd, y_test.to_numpy())],
    verbose=50
)

y_pred_proba_xgb = model_xgb.predict_proba(X_test_pd)[:, 1]
y_pred_xgb = model_xgb.predict(X_test_pd)
```

**Results:**
- Early stopped at iteration 610
- AUC-ROC: 0.9613
- Precision (energy poor class): 0.71
- Recall (energy poor class): 0.82
- Accuracy: 0.91

**Key differences from LightGBM:**
- `max_depth` (level-wise) vs `num_leaves` (leaf-wise): depth=6 → max 64 leaves, roughly comparable to LightGBM's default 31 leaves
- `scale_pos_weight` vs `class_weight="balanced"`: mathematically equivalent
- Feature importance uses "gain" by default (average loss improvement per split) vs LightGBM's split count — not directly comparable numbers, but rankings should be broadly similar

---

## Model Comparison

Both models produce nearly identical results:

| Metric | LightGBM | XGBoost |
|---|---|---|
| AUC-ROC | 0.96 | 0.9613 |
| Precision (EP class) | 0.69 | 0.71 |
| Recall (EP class) | 0.83 | 0.82 |
| Accuracy | 0.91 | 0.91 |
| Converged at iteration | 648 | 610 |

**Top features (both models agree):** `PERPOVLVL` and `HINCP` dominate. After that, models diverge somewhat due to different feature importance metrics (split count vs gain).

**Feature importance divergence is partly methodological:** Gain favors features making fewer but higher-quality splits (biases toward continuous variables). Split count favors frequently used features. Rankings of the *same* features can differ substantially for this reason. SHAP values are more comparable across models.

**That both models agree on AUC-ROC ≈ 0.96 is a reassuring sign** — suggests genuine patterns in the data rather than algorithm-specific artifacts.

---

## Evaluation

```python
from sklearn.metrics import roc_auc_score, classification_report

print("AUC-ROC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print(classification_report(y_test, y_pred))
```

**Use AUC-ROC, not accuracy.** At 17% positive rate, a model predicting everyone as non-poor gets 83% accuracy but is useless.

**Feature importance (Polars):**
```python
importance = pl.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort("importance", descending=True)
print(importance.head(20))
```

---

## SHAP Values

SHAP (SHapley Additive exPlanations) gives directional effects and accounts for feature interactions — more useful than raw feature importance for identifying which household characteristics drive risk.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.to_numpy())
shap.summary_plot(shap_values, X_test.to_numpy(), feature_names=feature_names)
```

**How to read SHAP summary plot:**
- Each row is a feature, ordered by mean absolute SHAP value (most important at top)
- Each dot is one household
- X-axis: SHAP value (positive = pushes toward energy poverty, negative = pushes away)
- Color: feature value (red = high, blue = low)
- Gray dots: categorical variables where color gradient doesn't apply cleanly

**How to read SHAP waterfall plot:**
- Shows how model arrived at prediction $f(x)$ for one specific household
- Starts from baseline $E[f(X)]$ (average model output across all households)
- Each bar shows how much that feature's value pushed the prediction up (red) or down (blue)
- Output is in log-odds scale, not probability. Convert: $p = \frac{1}{1 + e^{-f(x)}}$

**Key SHAP findings:**
- `PERPOVLVL` and `HINCP` dominate both models — low values push strongly toward energy poverty. Classic pattern: blue dots far right, pink dots far left
- `OMB13CBSA` and `BLD` appear gray — categorical variables, directionality can't be read from color. Need dependence plots for those
- `mintemp` (minimum temperature) shows clear gradient in no-CBSA model — low values (cold) push toward energy poverty. Intuitive: colder minimum temperatures = higher heating demand
- `DRYER` and `SOLAR` appear with positive SHAP values (pushing toward EP) — worth investigating. May be proxies for other characteristics rather than direct mechanisms
- Variables showing as `nan = VARIABLE` in waterfall: missingness itself is informative — model learned that missing values for certain variables are associated with higher EP risk

**`class_weight="balanced"` shifts baseline:** With balanced weighting, $E[f(X)] \approx 0.27$ in log-odds, corresponding to ~57% probability. The true base rate is 17%. This is expected behavior — the model's internal scale is adjusted. The *relative* comparisons between households and between climate scenarios are still valid.

---

## Climate Projection Analysis

### Approach

Use projected climate variables (min/max/avg temperature) from CMIP6-LOCA2. Swap projected temperatures into the AHS dataset while keeping all household characteristics fixed. Run inference with trained model. Compare predicted energy poverty rates.

**Recommended structure:**
```
For each warming scenario (1.5°C, 2°C, 3°C):
    For each year (2030, 2040, 2050):
        Swap in projected temps for that scenario/year
        Run model inference on full AHS dataset
        Record predicted EP rate overall and by household subgroup
```

### Critical Bug: Unit Mismatch

**Current climate temps:** Fahrenheit (mean mintemp ~50°F, mean maxtemp ~70°F)

**Projected climate temps:** Celsius (mean proj_tasmin ~11°C, mean proj_tasmax ~23°C)

These are actually similar values once converted — the data join was correct, but the model was trained on Fahrenheit and receiving Celsius values at inference time. A value of 11 for mintemp during training would be interpreted as 11°F (-12°C) — an extreme outlier producing nonsensical predictions.

**Fix — convert projected temps to Fahrenheit before inference:**
```python
df_proj = df_proj.with_columns([
    ((pl.col("proj_tasmin") * 9/5) + 32).alias("proj_tasmin"),
    ((pl.col("proj_tasmax") * 9/5) + 32).alias("proj_tasmax"),
    ((pl.col("proj_tas") * 9/5) + 32).alias("proj_tas"),
])
```

### CBSA vs No-CBSA for Projection

**With CBSA model:** Predicted EP rate went from 0.256 (current) to 0.187 (projected) — a decrease, which is wrong directionally. Cause: CBSA and temperature are highly correlated during training. When you swap in projected temperatures while keeping CBSA fixed, the model receives an out-of-distribution combination it has never seen — a specific metro area paired with temperatures that don't match what that metro had during training. Produces nonsensical predictions.

**Without CBSA model:** Predicted EP rate went from 0.20 (current) to 0.22 (projected) — a ~10% relative increase, correct direction. Temperature variables do real work because CBSA is no longer absorbing their signal.

**Conclusion: Use no-CBSA model as primary result for climate projection analysis.**

Keep CBSA model for understanding current-climate drivers of energy poverty, but exclude from projection analysis. Note the reason in methods section.

### Results

| Model | Current EP Rate | Projected EP Rate | Change |
|---|---|---|---|
| No-CBSA | 0.20 | 0.22 | +10% relative |
| With-CBSA | 0.256 | 0.187 | -27% (invalid) |

**Interpretation of no-CBSA result:** Under projected warming, predicted energy poverty prevalence increases from ~20% to ~22% — a ~10% relative increase. This is the core finding of the climate projection step.

**`mintemp` (minimum temperature) is more predictive than `maxtemp`** in the no-CBSA model. `proj_tasmin` ranks 12th in SHAP importance; `maxtemp` doesn't appear in top 20. This suggests heating demand drives more energy poverty than cooling demand in your metro areas, or that cooling costs are more easily avoided through behavioral adaptation than heating costs.

### Geographic Interpretation

With 15 metro areas spanning diverse climates, findings will be heterogeneous. A result showing "under moderate warming, energy poverty decreases in cold-climate metros because heating costs fall, but low-income households without AC remain exposed to extreme heat" is more nuanced and defensible than a simple "climate change makes things worse." Check which metros are driving the effect.

---

## Key Limitations

1. **Fixed household characteristics:** Survey characteristics held constant while only climate changes. Households most at risk in 2050 may not exist in 2023 sample.
2. **Training data extrapolation:** Model trained on 2023 climate conditions. Projected 2050 values (especially for 3°C scenario) may be outside the range seen during training, making predictions less reliable.
3. **Survey weights:** AHS is a complex probability sample — ignoring weights may over-represent certain groups. Run weighted and unweighted versions and compare.
4. **`INTMONTH` artifact:** Interview month had spurious correlation with energy poverty (likely seasonal utility bill timing). Should be excluded from model.
5. **`DRYER` and `SOLAR` signals:** Both show positive SHAP values (associated with higher EP risk) despite intuitive expectations otherwise. Investigate before reporting.
6. **No-CBSA model loses local context:** DIVISION captures broad regional variation but loses local energy prices, building codes, and utility infrastructure that CBSA encoded.
7. **Missing climate variables:** HDD and CDD were not available; using min/max/avg temperature instead. These are related but not identical measures of climate burden.

---

## Common Bugs and Fixes

| Bug | Fix |
|---|---|
| Python silent string concatenation (missing comma between adjacent string literals in list) | Always check list definitions carefully; `"INTMONTH""HHMAR"` → `"INTMONTHHHMAR"` |
| `tolist()` on Polars columns | Polars `.columns` already returns a plain list — use directly |
| LightGBM/sklearn version conflict (`feature_names_in_` setter error) | Use `.to_numpy()` on all inputs to `.fit()` and `.predict()` |
| XGBoost `early_stopping_rounds` unexpected keyword argument | Move to model constructor in newer XGBoost versions |
| XGBoost "Category index has floating point dtype" | Fill NAs before casting: `.fillna(-1).astype(int).astype('category')` |
| SHAP waterfall and summary plot rendering on top of each other | Add `plt.figure()` before each plot and `plt.close()` after saving |
| Projected climate temps in wrong units (Celsius vs Fahrenheit) | Convert: `(celsius * 9/5) + 32` before running inference |
| `CONTROL` (household ID) included as feature | Drop explicitly before defining X |
| Survey `WEIGHT` variable included as feature | Extract separately, drop from X, pass as `sample_weight` to `.fit()` |

---

## Installation Notes

**LightGBM on Mac (OpenMP error):**
```bash
brew install libomp
# or
conda install -c conda-forge lightgbm
```

**Version conflicts between LightGBM and sklearn:**
```bash
conda install -n energyPoverty -c conda-forge lightgbm scikit-learn --force-reinstall
```
Installing from the same channel at the same time ensures compatibility.

---

## File Structure Reference

```
data/
  processed/
    LightGBM_data/
      01_02_04_ready_for_lightgbm_ahs_climate.csv
    01_02_04_generally_ready_for_trees_ahs_climate.csv

code/
  01_current_climate/
    03_analysis/
      01_lightgbm_analysis.py
      03_xgboost_analysis.py
```
