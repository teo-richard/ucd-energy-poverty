"""
race.py — Heterogeneity analysis by householder race (HHRACE).

HHRACE codes:
  1 = White alone            4 = Asian alone
  2 = Black alone            5 = Native Hawaiian / Pacific Islander
  3 = American Indian alone  6 = Other / multi-racial

Run from the project root:
    python code/04_heterogeneity_analysis/race.py
"""

import sys
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import run_heterogeneity

run_heterogeneity("HHRACE")
