import subprocess, sys
from pathlib import Path

script_dir   = Path(__file__).parent
project_root = script_dir.parents[2]

scripts = [
    "00.1_analysis_module.py",
    "01_lightgbm_analysis.py",
    "02_xgboost_analysis.py",
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run([sys.executable, script_dir / script], check=True, cwd=project_root)
