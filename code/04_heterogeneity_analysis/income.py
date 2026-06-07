"""
income.py — Heterogeneity analysis by household income quintile (HINCP).

HINCP is continuous, so heterogeneity_utils.load_df() bins it into quintiles
(income_quintile) before analysis — see CONDITIONING_VARS / LABEL_MAPS there.

Income quintiles (income_quintile):
  1_Q1 — lowest income quintile
  2_Q2 — second quintile
  3_Q3 — third (middle) quintile
  4_Q4 — fourth quintile
  5_Q5 — highest income quintile

Run from the project root:
    python code/04_heterogeneity_analysis/income.py
"""

import sys
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import run_heterogeneity

run_heterogeneity("income_quintile")
