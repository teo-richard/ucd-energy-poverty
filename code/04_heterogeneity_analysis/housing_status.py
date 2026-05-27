"""
housing_status.py — Heterogeneity analysis by housing tenure (TENURE).

TENURE codes:
  1 = Owned or being bought by someone in the household
  2 = Rented for cash
  3 = Occupied without payment of cash rent

Run from the project root:
    python code/04_heterogeneity_analysis/housing_status.py
"""

import sys
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import run_heterogeneity

run_heterogeneity("TENURE")
