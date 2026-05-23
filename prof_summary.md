# UCD Energy Poverty Research — Findings Summary

## Overview

This project predicts **energy deprivation** among US households using American Housing Survey (AHS 2023) data merged with historical climate data (NClimGrid) and 2050 climate projections (CMIP6-LOCA2) across 15 metropolitan areas. The outcome variable (`energy_deprivation`) is a binary indicator equal to 1 if a household reports experiencing dangerous heat (`HOT = 1`) or dangerous cold (`COLD = 1`).

The trained model is then applied to 2050 climate projections to forecast how energy deprivation prevalence may change under warming, and a series of sensitivity analyses explore the role of income growth and heating system type.

---

## 1. Model Performance

The primary model is a **survey-weighted XGBoost classifier** excluding Core-Based Statistical Area (CBSA) codes, calibrated using Platt scaling on a held-out calibration set (10% of the full data). CBSA was excluded because including it at training time causes a distribution-shift artifact when the model is applied to 2050 projected data — it sees CBSA codes never observed at inference time, producing incorrect directional predictions.

**Data splits:** 70% train / 10% calibration / 20% test

| Metric | Before Calibration | After Platt Scaling |
|---|---|---|
| Brier Score | 0.1480 | 0.1291 |
| ECE (Expected Calibration Error) | 0.1119 | 0.0151 |
| AUC-ROC | 0.7253 | — |

Platt scaling dramatically improves probability calibration (ECE drops ~93%), while the AUC is unchanged since calibration does not alter rank ordering.

**XGBoost outperforms Logistic Regression** on both Brier score and ECE.

---

## 2. Most Important Predictors (SHAP)

The top features driving predicted energy deprivation risk are:

| Feature | Interpretation |
|---|---|
| `WALLCRACK` | Cracks or holes in interior walls — housing quality signal |
| `ACPRIMARY` | Primary air conditioning type — cooling access |
| `FUSEBLOW` | Blown fuses or tripped breakers — electrical system quality |
| `ROACH` | Evidence of cockroaches — housing condition proxy |
| `DISHH` | Disability status of householder |

These are predominantly **housing quality and household vulnerability markers**, not climate variables. This foreshadows the headline finding on climate.

---

## 3. Climate Change Is Not a Meaningful Driver

**Headline finding: climate variables have low SHAP importance relative to housing quality and socioeconomic features.**

Applying the 2050 CMIP6-projected temperatures to the current AHS household population:

- **2023 (current climate) mean EP probability: ~20%**
- **2050 (projected climate) mean EP probability: ~22%**

This is approximately a **10% relative increase** (2 percentage points absolute). While directionally consistent with warming, it is a modest change — households' energy deprivation risk is dominated by structural and economic factors, not temperature exposure.

---

## 4. Income

### 4a. Simultaneous Income Shifts (All Households)

Scaling all household incomes (`HINCP`) simultaneously by a uniform factor produces almost no change in predicted EP probability:

| Income Factor | Mean Predicted EP Prob |
|---|---|
| ×0.90 (−10%) | 0.1843 |
| ×1.00 (baseline) | 0.1844 |
| ×1.10 (+10%) | 0.1826 |
| ×1.20 (+20%) | 0.1814 |
| ×1.30 (+30%) | 0.1803 |

A **30% income increase for all households** reduces mean EP probability by only 0.41 percentage points. This indicates the model is picking up structural features of poverty that income alone cannot address.

### 4b. Quintile-Specific Realistic Income Shifts

Using historically-informed income growth rates by quintile (applied to the 2050 projected dataset):

| Quintile | Income Shift | Baseline EP Prob | Post-Shift EP Prob | Change |
|---|---|---|---|---|
| Q1 (lowest) | +6.3% | 0.1913 | 0.1901 | −0.0012 |
| Q2 | +25.3% | 0.1869 | 0.1843 | −0.0026 |
| Q3 | +22.0% | 0.1834 | 0.1820 | −0.0014 |
| Q4 | +19.8% | 0.1813 | 0.1760 | −0.0053 |
| Q5 (highest) | +28.2% | 0.1792 | 0.1731 | −0.0061 |

**Key observation:** Q1 barely moves despite an income shift. Higher-income quintiles respond more strongly to the same proportional income gains. The lowest-income households are the most at risk and the least responsive to income growth.

### 4c. Q1-Only Income Shifts (Stress Test)

To understand how much Q1 income would need to grow to meaningfully reduce their risk, income for Q1 alone was scaled from +10% to +200%:

| Q1 Income Shift | Q1 EP Prob | Overall EP Prob |
|---|---|---|
| +10% | 0.1899 | 0.1841 |
| +50% | 0.1906 | 0.1843 |
| +100% | 0.1879 | 0.1837 |
| +200% | 0.1821 | 0.1826 |

Even **doubling Q1 household income** reduces Q1 mean EP probability by only ~1 percentage point (from 0.1913 to 0.1821). This strongly suggests that income is not the binding constraint for the most vulnerable households — housing quality, neighborhood conditions, and structural deprivation markers captured in features like `WALLCRACK`, `FUSEBLOW`, and `ROACH` are more determinative.

### 4d. Optimistic Income Scenario

An optimistic scenario where all quintiles receive larger income boosts (Q1: +15%, Q2–Q4: +20%, Q5: +25%) still produces only marginal changes:

| Quintile | Income Shift | Baseline EP Prob | Post-Shift EP Prob |
|---|---|---|---|
| Q1 | +15% | 0.1913 | 0.1895 |
| Q2 | +20% | 0.1869 | 0.1850 |
| Q3 | +20% | 0.1834 | 0.1816 |
| Q4 | +20% | 0.1813 | 0.1759 |
| Q5 | +25% | 0.1792 | 0.1732 |

The pattern holds: Q1 is inelastic to income growth even under optimistic assumptions.

---

## 5. Heating Type Does Not Drive Energy Deprivation Risk

A counterfactual analysis was run for each income quintile by forcing all households to each heating system type and observing the change in predicted EP probability:

**Results for Q1 (baseline EP prob: 0.1913):**

| Heating System | Counterfactual EP Prob | Delta |
|---|---|---|
| Heat pump | 0.1797 | −0.0115 |
| Forced air furnace | 0.1859 | −0.0054 |
| Electric baseboard | 0.1838 | −0.0075 |
| Steam/hot water | 0.1893 | −0.0020 |
| No heating system | 0.2041 | +0.0128 |
| Portable electric | 0.1921 | +0.0009 |

The **maximum achievable reduction** from switching to the best heating system (heat pump) is about 1.2 percentage points. This is small relative to the baseline risk level, and consistent across income quintiles.

**Conclusion: Changing heating type alone is not a meaningful lever for reducing energy deprivation risk.**

---

## Summary of Key Findings

| Question | Finding |
|---|---|
| What predicts energy deprivation? | Housing quality markers (wall cracks, blown fuses, pests) and household vulnerability — not climate. |
| Does climate change worsen energy deprivation? | Modestly: ~10% relative increase by 2050, but climate is a weak predictor compared to structural factors. |
| Does income growth help? | Minimally. Even large income increases produce small reductions in EP probability, especially for the lowest quintile. |
| Are low-income households responsive to income growth? | No. Q1 is highly inelastic — doubling their income barely moves predicted EP probability. |
| Does heating system type matter? | Slightly. Heat pumps reduce risk the most, but the effect is small across all quintiles. |

**The overarching conclusion:** Energy deprivation is driven primarily by structural housing quality deficits and deep household vulnerability — not by climate exposure or income levels alone. Policies targeting housing quality improvements are likely to be more effective than income transfers or heating system mandates.

---

## Data & Methods Note

- **Outcome:** Binary energy deprivation (`HOT = 1` or `COLD = 1` from AHS)
- **Training data:** AHS 2023 + NClimGrid historical climate
- **Projection data:** CMIP6-LOCA2 for 15 US metro CBSAs (2050 horizon)
- **Model:** XGBoost (gradient-boosted trees), survey-weighted, no CBSA features
- **Calibration:** Platt scaling on a held-out calibration set
- **Class imbalance handling:** `scale_pos_weight = 4.83` (ratio of negatives to positives)
- **Explainability:** SHAP TreeExplainer (summary and waterfall plots)
- **Sensitivity analyses:** Uniform income scaling, quintile-specific income shifts, Q1-only shifts, optimistic scenario, heating type counterfactual sweep

---

## Future

I am going to try to get CBSA-level utility prices. I just downloaded the data from EIA yesterday and it looks doable to get prices by CBSA. I can extract a good estimate of prices based on revenue divided by sales and then use various crosswalks to eventually map everything to CBSA. 

This seems like quite an important omitted variable.

---

## Policy

The finding that the at-risk population is essentially stable between 2023 and 2050 implies that waiting for climate impacts to manifest before targeting assistance is the wrong approach — the households who will be most vulnerable under 2050 conditions are largely identifiable and reachable today. 