# Energy Poverty ML Research Notes
## Comprehensive Analysis Summary

---

## 1. Project Overview

**Goal:** Predict household energy poverty (EP) using the 2023 American Housing Survey (AHS) joined with climate data, then apply projected 2050 CMIP6-LOCA2 temperatures to estimate how EP rates shift under warming scenarios.

**EP Definition:** A household spends more than 10% of household income on utilities.

**Dataset:** ~26,700 households across 15 metro areas (CBSA oversample). Approximately 17% positive rate (EP households).

**Primary Model:** XGBoost, no-CBSA specification (see Section 4 for rationale).

---

## 2. Data Pipeline

### 2.1 Pipeline Order

1. Drop unwanted columns
2. Drop columns with too many nulls (true nulls)
3. Convert columns to numeric
4. Replace sentinel values with null
5. Re-check and drop columns with too many nulls post-sentinel replacement
6. `df.describe()` and correlation checks
7. Filter out invalid rows
8. Feature engineering

### 2.2 Sentinel Values

AHS uses the following sentinel codes for missing/inapplicable responses:

- `-6`, `-7`, `-8`, `-9` — refused, don't know, not applicable (string and numeric columns)
- `9`, `99`, `999`, `9999`, `99999` — not applicable (numeric columns, variable-dependent)

All sentinel values replaced with `null` before modeling. Caution: `9` and `99` may be legitimate values for some variables (e.g. room counts, household size). Verify against AHS codebook per variable.

### 2.3 Data Quality Issues Identified and Addressed

| Issue | Count | Action |
|---|---|---|
| Homeowners with $0 utility cost | 112 | Excluded — likely data error |
| Renters with $0 utility cost (utilities included in rent) | 1,444 | Excluded — mechanically coded non-EP but may mask hardship |
| Utility cost >$10,800/year | 266 | Retained — mostly large wealthy homes, plausible |
| Extreme outlier: CONTROL 11034668 ($18,960 utility / $38k income) | 1 | Verify against TOTHCAMT |

Excluding the zero-utility rows improved Brier score.

### 2.4 PERPOVLVL Top-Coding

The published AHS `PERPOVLVL` variable is top-coded at 501 — 9,622 rows (36% of sample) are collapsed to exactly 501 regardless of true poverty ratio. This means `PERPOVLVL` has zero discriminating power across households with incomes ranging from $71k to $6.4M.

**Fix:** Constructed `constr_PERPOVLVL` using official Census 2023 poverty thresholds by household size and number of children under 18:

$$\text{constr\_PERPOVLVL} = \frac{\text{HINCP}}{\text{poverty\_threshold}_{(\text{size, children})}} \times 100$$

**Validation:** Mean absolute error of 6.8 vs published `PERPOVLVL` for non-top-coded rows (median 4.4, max 42.8). Max error attributable to elderly householder threshold adjustment not captured in lookup table.

**Result for top-coded households:** `constr_PERPOVLVL` ranges from 460 to 14,473 (mean 1,094, median 798) for households previously stuck at 501.

`constr_PERPOVLVL` replaces `PERPOVLVL` in all model specifications.

### 2.5 Climate Variables

Original features: `mintemp`, `maxtemp`, `meantemp` (annual averages, °F).

Including all three simultaneously is redundant since `meantemp = (mintemp + maxtemp) / 2`.

**Final specification:** `meantemp` only. Brier score comparison:
- `meantemp` only: 0.0610
- `HDD_approx` + `CDD_approx`: 0.0611
- Performance difference is negligible (0.0001); `meantemp` preferred for simplicity.

**Current climate data:** NClimGrid 2023 annual averages joined to AHS by CBSA.

**Projected climate data:** CMIP6-LOCA2 projected 2050 temperatures joined by CBSA.

---

## 3. Feature Engineering

| Feature | Description |
|---|---|
| `constr_PERPOVLVL` | Uncapped poverty ratio constructed from Census 2023 thresholds |
| `meantemp` | Annual mean temperature = (mintemp + maxtemp) / 2 |
| `energy_poverty` | Binary target: 1 if utility cost > 10% of HINCP |

---

## 4. Model Specifications

### 4.1 Two Model Variants

**CBSA model:** Includes `OMB13CBSA` as a feature. Problem: CBSA and temperature are collinear — the model uses CBSA as a geographic shortcut and temperature variables lose predictive power. Projection results are nonsensical (negative EP change under warming).

**No-CBSA model:** Excludes `OMB13CBSA` from features but retains it in the dataframe for post-hoc analysis. Temperature does real predictive work. Projection results are directionally correct. **This is the primary specification for all projection and sensitivity analyses.**

`OMB13CBSA` is retained in the dataframe and used for CBSA-level aggregation after inference.

### 4.2 Hyperparameters

- `max_depth`: 5 (reduced from default 6 — improved calibration)
- Other parameters: document current settings here

### 4.3 Train/Test Split

- 80% training, 20% test
- Test set locked away until final evaluation
- All sensitivity analyses and calibration assessment run on training set via 5-fold CV

### 4.4 Cross-Validation

5-fold CV used for calibration assessment. Out-of-fold (OOF) predictions generated for all training households — each household's predicted probability comes from a model that never saw it during training. CV models are temporary diagnostic tools; the final model is trained on the full training set.

### 4.5 XGBoost Categorical Handling

XGBoost 2.x enforces strict validation of unseen categories. Alignment loop applied to ensure test set and projection data category dtypes exactly match training set. Unseen categories treated as missing rather than causing crashes.

---

## 5. Model Performance

| Metric | Value |
|---|---|
| Brier score (5-fold CV OOF) | 0.0611 |
| ECE (Expected Calibration Error) | 0.0370 |

### 5.1 Calibration

Calibration assessed via 5-fold CV OOF predictions with quantile binning (10 bins, ~1,238 households each).

**Pattern:** Well-calibrated below 0.3 predicted probability (where the majority of households reside). Systematic overprediction in the 0.3–0.8 range — when the model predicts 0.60, true EP rate is approximately 0.39.

**Likely causes:**
1. Label noise from self-reported utility expenditure data — households near the 10% threshold may have miscoded EP status due to reporting error
2. Known miscalibration tendency of gradient boosted trees with imbalanced classes

**Implication:** Raw predicted probabilities in the 0.3–0.8 range are inflated. All reported results use relative comparisons between scenarios rather than absolute probability values. Relative comparisons are robust to this systematic bias since it affects all scenarios equally.

### 5.2 CBSA-Level Calibration

| Metro | Predicted EP | Actual EP | Gap |
|---|---|---|---|
| Riverside-San Bernardino, CA | 0.291 | 0.263 | +0.028 |
| Miami-Fort Lauderdale, FL | 0.262 | 0.220 | +0.042 |
| Detroit, MI | 0.257 | 0.193 | +0.064 |
| Boston, MA | 0.239 | 0.195 | +0.045 |
| Houston, TX | 0.235 | 0.195 | +0.040 |
| Los Angeles, CA | 0.226 | 0.189 | +0.037 |
| Atlanta, GA | 0.224 | 0.179 | +0.044 |
| Philadelphia, PA | 0.223 | 0.183 | +0.039 |
| New York, NY | 0.207 | 0.158 | +0.049 |
| Chicago, IL | 0.198 | 0.166 | +0.033 |
| Dallas-Fort Worth, TX | 0.187 | 0.148 | +0.038 |
| San Francisco, CA | 0.183 | 0.159 | +0.024 |
| Phoenix, AZ | 0.171 | 0.143 | +0.028 |
| Seattle, WA | 0.130 | 0.102 | +0.028 |
| Washington, DC | 0.127 | 0.104 | +0.023 |

Calibration gap is consistently positive across all metros — systematic overprediction. Detroit shows the largest gap (+0.064).

---

## 6. SHAP Analysis

### 6.1 Feature Importance — All Specifications

`constr_PERPOVLVL` and `HINCP` dominate every model specification. Both show:
- High values (red) → strong negative SHAP (reduces EP probability)
- Low values (blue) → strong positive SHAP (increases EP probability)

SHAP spread for these two features dwarfs all others.

### 6.2 Top Features — No-CBSA Model (Current Climate)

Rank 1: `constr_PERPOVLVL`
Rank 2: `HINCP`
Rank 3: `BLD`
Rank 4: `DIVISION`
Rank 5: `DRYER`
Rank 6: `BEDROOMS`
Rank 7: `HHAGE`
Rank 8: `TENURE`
Rank 9: `TOTROOMS`
Rank 10: `YRBUILT`
Rank 11: `meantemp` (formerly `mintemp`)

### 6.3 Feature Ranking Stability

Feature ranking is stable across all income sensitivity scenarios. The only notable change: at 130% income, `HINCP` flips to rank #1 above `constr_PERPOVLVL`, suggesting the continuous income variable does more work at the upper tail of the income distribution.

### 6.4 Unresolved SHAP Signals

- **`DRYER`:** Consistently positive SHAP (pushes toward EP). Likely a proxy for larger households or higher energy consumption rather than a direct mechanism.
- **`nan = SOLAR`:** Missing solar panel data associated with higher EP risk. Plausible (no solar = no cost offset) but worth confirming.
- **`nan = CELLPHONE`:** Missing cellphone data associated with lower EP risk in CBSA model waterfalls. Demographic proxy or data quality issue — unresolved.

---

## 7. Climate Projection Results

### 7.1 Aggregate Results

| Scenario | Mean EP probability | EP rate (threshold 0.5) |
|---|---|---|
| 2023 baseline (current climate) | 0.20 | — |
| 2050 projected climate only | 0.22 | +2pp |

The +2pp absolute increase represents a +10% relative increase from baseline.

### 7.2 CBSA-Level Climate Results

Projected temperature changes and EP deltas by metro:

| Metro | Δmintemp | Δmaxtemp | EP delta |
|---|---|---|---|
| Chicago, IL | +3.82°F | +3.93°F | +1.9pp |
| Detroit, MI | +3.45°F | +4.42°F | +0.4pp |
| Dallas-Fort Worth, TX | +1.03°F | +2.78°F | +0.1pp |
| Atlanta, GA | +2.39°F | +2.81°F | +0.07pp |
| Los Angeles, CA | +2.32°F | +4.61°F | +0.07pp |
| Philadelphia, PA | +3.02°F | +3.50°F | +0.04pp |
| New York, NY | +2.67°F | +2.89°F | +0.03pp |
| Miami-Fort Lauderdale, FL | +1.09°F | +0.76°F | ~0pp |
| Washington, DC | +3.96°F | +3.83°F | −0.02pp |
| San Francisco, CA | +2.72°F | +4.18°F | −0.03pp |
| Phoenix, AZ | +1.46°F | +4.89°F | −0.1pp |
| Seattle, WA | +1.25°F | +2.71°F | ~0pp |
| Boston, MA | +2.60°F | +3.84°F | −0.4pp |
| Riverside-San Bernardino, CA | +1.71°F | +5.84°F | −1.3pp |

**Key finding:** The climate effect is heterogeneous across metros — not uniformly negative. Several metros show EP improvement under projected warming. The cross-metro ranking of absolute EP levels is driven by income and housing vulnerability rather than climate exposure.

---

## 8. Income Sensitivity Analysis

### 8.1 Uniform Income Shifts (Projected Climate + HINCP only)

Both `HINCP` and `constr_PERPOVLVL` shifted proportionally together.

| HINCP multiplier | Mean predicted EP probability |
|---|---|
| 0.90 | 0.2257 |
| 1.00 | 0.2154 |
| 1.10 | 0.2067 |
| 1.20 | 0.1984 |
| 1.30 | 0.1908 |

Approximately −0.9pp per 10% income increase. Relationship is roughly linear across this range.

A ~20% income increase would offset the +2pp climate effect entirely.

### 8.2 Historical Quintile Shifts

Grounded in CE Survey real income growth by quintile, 2000–2023 (CPI-adjusted, base year 2000):

| Quintile | Real income growth 2000–2023 |
|---|---|
| Q1 (0–20%) | +6.3% |
| Q2 (21–40%) | +25.3% |
| Q3 (41–60%) | +22.0% |
| Q4 (61–80%) | +19.8% |
| Q5 (81–100%) | +28.2% |

Q1 real income growth was nearly flat over 23 years — substantially below all other quintiles.

### 8.3 Quintile-Level Results — Historical Shifts (Projected Climate)

| Quintile | Shift % | n | Baseline EP prob | Shifted EP prob | Change |
|---|---|---|---|---|---|
| Q1 | +6.3% | 3,114 | 0.6185 | 0.5929 | −2.6pp |
| Q2 | +25.3% | 3,075 | 0.2427 | 0.1341 | −10.9pp |
| Q3 | +22.0% | 3,108 | 0.0544 | 0.0178 | −3.7pp |
| Q4 | +19.8% | 3,085 | 0.0076 | 0.0030 | −0.5pp |
| Q5 | +28.2% | 3,091 | 0.0012 | 0.0009 | ~0pp |

### 8.4 Aggregate Results — Historical Shifts (Projected Climate)

| Scenario | Mean EP probability | EP rate (threshold 0.5) |
|---|---|---|
| 2050 projected climate only | 0.1854 | 0.1731 |
| 2050 projected + historical income shifts | 0.1503 | 0.1411 |

BAU income growth reduces predicted EP by ~3.2pp — larger in magnitude than the +2pp climate effect.

### 8.5 Q1-Only Income Sweep (Projected Climate)

Q1 baseline EP probability: 0.6968. Sweep of Q1-only income shifts holding all other quintiles at projected climate baseline:

| Q1 shift | Q1 EP prob | Overall EP prob | EP rate (t=0.5) |
|---|---|---|---|
| +10% | 0.6901 | 0.2078 | 0.1950 |
| +20% | 0.6817 | 0.2061 | 0.1938 |
| +30% | 0.6747 | 0.2047 | 0.1932 |
| +50% | 0.6598 | 0.2017 | 0.1903 |
| +70% | 0.6452 | 0.1988 | 0.1878 |
| +100% | 0.6221 | 0.1941 | 0.1836 |
| +150% | 0.5912 | 0.1878 | 0.1769 |
| +200% | 0.5645 | 0.1824 | 0.1714 |

Even at +200% income (tripling Q1 incomes), Q1 EP probability only drops from 0.697 to 0.565 — never approaching the 0.5 threshold.

**Important caveat:** This analysis shifts `HINCP` and `constr_PERPOVLVL` proportionally but holds all housing characteristics fixed. In reality, income gains would likely trigger residential mobility and housing upgrades. The flat Q1 response curve reflects both a genuine structural finding and a model limitation — the model was trained on data where high income and old inefficient housing rarely co-occur, so predictions in that region are extrapolations beyond the training distribution.

---

## 9. Key Findings

### 9.1 Energy poverty is primarily a poverty problem, not a climate problem

`constr_PERPOVLVL` and `HINCP` dominate every model specification across all scenarios. Climate variables are real but secondary.

### 9.2 Projected warming adds modest aggregate pressure

+2pp absolute increase in predicted EP under 2050 projected conditions. This is contingent on a model trained on 2023 conditions extrapolating to projected temperatures.

### 9.3 Q1 households are structurally embedded in energy poverty

~62–70% predicted EP probability under both current and projected conditions (note: raw probabilities are inflated due to model miscalibration — true EP rate likely closer to 40–50% based on calibration curve). Q1 EP probability barely responds even to implausible income increases of 200%. This reflects both a genuine structural finding and model limitations in the out-of-distribution region.

### 9.4 The aggregate climate effect is driven by middle quintiles

Q2 households (baseline EP ~0.24) are the most responsive to both income and climate changes. Q2 EP probability nearly halves under historical income growth. They are the marginal population — households that could tip either way depending on economic and climate trajectories.

### 9.5 BAU income growth more than offsets projected climate effects in aggregate

Under historically-grounded income growth, aggregate EP falls by ~3.2pp despite projected warming — a net improvement from today's baseline. However this masks severe Q1 heterogeneity.

### 9.6 Climate effects are heterogeneous across metros

EP change under projected warming ranges from −1.3pp (Riverside) to +1.9pp (Chicago). Cross-metro differences in absolute EP levels are driven by income and housing vulnerability rather than climate exposure. The metro ranking does not simply follow hot vs cold climate patterns.

---

## 10. Limitations

### 10.1 Fixed-household assumption

The projection analysis holds all household characteristics fixed at 2023 values and swaps in 2050 temperatures. This does not model adaptation, housing turnover, income changes, or behavioral response. Results represent a counterfactual: "what would 2023 households face under 2050 temperatures."

### 10.2 Out-of-distribution extrapolation

The model was trained on 2023 temperature and income distributions. Both climate projections and sensitivity income shifts push the model into regions of feature space with limited training data. Predictions become less reliable the further inputs deviate from the training distribution.

### 10.3 Label noise

EP status is derived from self-reported AHS utility expenditure data. Households near the 10% threshold may have miscoded EP status due to reporting errors (seasonal variation, utilities-included confusion, annual vs monthly reporting). This is the primary suspected cause of systematic model miscalibration in the 0.3–0.8 probability range.

### 10.4 Target variable construction

1,444 renters with utilities included in rent are excluded from analysis. These households are not necessarily energy-secure — they may face energy hardship without direct utility payment obligations. Results apply to households with direct utility payment responsibility.

### 10.5 15-metro sample

Results are based on the 15 AHS oversample CBSAs. Generalizability to other metros or rural areas is uncertain.

### 10.6 PERPOVLVL construction

`constr_PERPOVLVL` is reconstructed from Census 2023 poverty thresholds using household size and children count. Mean absolute error of 6.8 vs published values for non-top-coded rows. Elderly householder threshold adjustments are not fully captured.

### 10.7 Temperature variable approximation

`meantemp` is an annual average proxy for heating and cooling demand. Proper heating degree days (HDD) and cooling degree days (CDD) would require daily temperature data and would be more causally proximate to utility costs.

### 10.8 Model calibration

Systematic overprediction in the 0.3–0.8 predicted probability range (ECE = 0.037). Raw predicted probabilities should not be interpreted as true EP rates. All reported results use relative comparisons between scenarios.

### 10.9 Income sensitivity interpretation

The Q1 income sweep holds housing characteristics fixed. In reality, income gains would likely trigger residential mobility and housing quality improvements. The shallow Q1 response curve likely represents a lower bound on the true income effect rather than a precise causal estimate.

---

## 11. Technical Notes

### 11.1 Why no-CBSA model for projections

The CBSA model produced a projected EP change of approximately −27% under warming — a nonsensical result. CBSA and temperature are collinear: the model uses CBSA as a geographic shortcut during training, absorbing temperature signal. When projected temperatures are swapped in, the CBSA feature anchors predictions to 2023 geography while the temperature change goes unregistered. The no-CBSA model forces temperature variables to do real predictive work.

### 11.2 Unit mismatch bug (resolved)

Early projected temperature runs used Celsius values in a model trained on Fahrenheit data. A projected temperature of 11°C was interpreted as 11°F ≈ −12°C, producing extreme outlier predictions. Resolved by ensuring consistent Fahrenheit units throughout the pipeline.

### 11.3 Model saving and loading

Model saved using `model.save_model("model.json")` (preferred over pickle for XGBoost version compatibility). Feature list saved separately alongside model. When loading for inference, feature columns are aligned to training set column order and category dtypes before prediction.

### 11.4 constr_PERPOVLVL shifting in sensitivity analysis

When shifting income by factor $k$, both `HINCP` and `constr_PERPOVLVL` are shifted by the same factor:

$$\text{constr\_PERPOVLVL\_shifted} = \text{constr\_PERPOVLVL} \times k$$

This is internally consistent because:

$$\frac{\text{HINCP} \times k}{\text{poverty\_threshold}} \times 100 = \text{constr\_PERPOVLVL} \times k$$

The poverty threshold denominator depends on household composition, not income, so it remains fixed.

---

## 12. Outstanding Items

- [ ] Run optimistic quintile shift scenario (projected climate + optimistic income shifts) and record aggregate EP rate
- [ ] Run combined climate + optimistic income scenario
- [ ] Run HEATTYPE sensitivity analysis for Q1 households
- [ ] Resolve DRYER, SOLAR, CELLPHONE SHAP signals
- [ ] Verify CONTROL 11034668 extreme outlier against TOTHCAMT
- [ ] Investigate why Detroit has the largest CBSA calibration gap (+0.064)
- [ ] Investigate why Chicago has the largest positive climate EP delta (+1.9pp) despite large Δmintemp — expected direction is ambiguous
- [ ] Consider whether to add additional AHS years (2021/2022) — requires variable harmonization check against 2023 codebook
- [ ] Document final hyperparameter settings

---

*Last updated: May 2026*
