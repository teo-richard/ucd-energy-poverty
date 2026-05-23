**XGBoost Vs. Logistic Regression**
XGBoost outperforms Logistic Regression:

XGBoost: 
- Brier = 0.148, ECE = 0.1119
- After Platt Scaling: Brier = 0.1291, ECE = 0.0151

AUC-ROC = 0.7253
Precision

Most important features: WALLCRACK, ACPRIMARY, FUSEBLOW, ROACH, DISHH


**Climate Change**
Biggest finding: Climate is not a meaningful driver of energy poverty. 



**Income**
Biggest finding: Income actually drives risk.

Split income into quintiles:
- Q1 hardly movies with 6.3% income shift (EP prob decreases from .61 to .58)
- Q2 respond more strongly (see `sensitivity_hincp_quintiles_per_quintile.csv`)
- Even with optimistic growth, we do not see much of a shift in Q1

**Heating Type Heterogeneity**

We do not see that changing heating type will decrease liklihood of EP.