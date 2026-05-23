import subprocess, sys
from pathlib import Path

script_dir   = Path(__file__).parent
project_root = script_dir.parents[1]

scripts = [
    "01_hincp_simultaneous/01_xgboost_analysis.py",
    "02_hincp_shift_quintiles/01_xgboost_analysis.py",
    "03_hincp_shift_quintiles_optimistic/01_xgboost_analysis.py",
    "04_q1_shifts_only/01_xgboost_analysis.py",
    "05_heattype_counterfactual/01_xgboost_analysis.py",
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run([sys.executable, script_dir / script], check=True, cwd=project_root)
