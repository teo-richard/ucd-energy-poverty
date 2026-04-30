import subprocess, sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parents[2]

scripts = [
    "01_basic_cleaning.py",
    "02_data_checks.py",
    "03_feature_engineering.py",
    "04_collapse_categories.py",
    "05_last_minute_cleanup.py",
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run([sys.executable, script_dir / script], check=True, cwd=project_root)
