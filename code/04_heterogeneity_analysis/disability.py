"""
disability.py — Heterogeneity analysis by disability status (DISHH).

DISHH codes:
  1 = At least one household member has a disability
  2 = No household members have a disability

Run from the project root:
    python code/04_heterogeneity_analysis/disability.py
"""

import sys
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import run_heterogeneity

run_heterogeneity("DISHH")
