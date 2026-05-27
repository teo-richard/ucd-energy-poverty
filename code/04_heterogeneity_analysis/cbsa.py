"""
cbsa.py — Heterogeneity analysis by geography (Census DIVISION).

The project uses the no-CBSA model variant (OMB13CBSA excluded from features).
Geographic heterogeneity is therefore analysed using Census DIVISION (9 regions),
which is included in the current-climate dataset and was a model feature.

DIVISION codes:
  1 = New England          6 = East South Central
  2 = Middle Atlantic      7 = West South Central
  3 = East North Central   8 = Mountain
  4 = West North Central   9 = Pacific
  5 = South Atlantic

Run from the project root:
    python code/04_heterogeneity_analysis/cbsa.py
"""

import sys
sys.path.insert(0, "code/04_heterogeneity_analysis")
from heterogeneity_utils import run_heterogeneity

run_heterogeneity("DIVISION")
