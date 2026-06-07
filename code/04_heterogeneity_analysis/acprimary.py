"""
acprimary.py — Heterogeneity analysis by primary air conditioning type (ACPRIMARY).

ACPRIMARY codes (collapsed):
  1 = Central AC
  2 = Room AC
  3 = No AC
  999 = Other/Unknown

Run from the project root:
    python code/04_heterogeneity_analysis/acprimary.py
"""

import sys
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import run_heterogeneity

run_heterogeneity("ACPRIMARY")
