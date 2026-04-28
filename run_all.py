import subprocess
import sys

def run_script(script_path):
    print(f"\nRunning {script_path}...\n")
    result = subprocess.run([sys.executable, script_path])

    if result.returncode != 0:
        print(f"Error while running {script_path}")
        exit(1)

# --- Run all models ---
run_script("models/vader_model.py")
run_script("models/roberta_model.py")
run_script("analysis/comparison_dashboard.py")

print("\nAll tasks completed successfully.")
