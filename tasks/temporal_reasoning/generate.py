#!/usr/bin/env python3
"""
Master control script to run the four data generation scripts in the correct order:
- joint_sorting_question.py must run before joint_sorting_context.py
- visit_cloze_question.py must run before visit_cloze_context.py
"""

import subprocess
import sys
from pathlib import Path

# Directory where this script and the other scripts reside
BASE_DIR = Path(__file__).parent

def run_script(script_name: str, description: str = ""):
    """Run a given Python script and print its status."""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    print(f"\n=== Starting: {description or script_name} ===")
    print(f"Executing: {script_path}")

    # Use the same Python interpreter to run the sub‑script
    result = subprocess.run([sys.executable, str(script_path)])

    if result.returncode != 0:
        print(f"[ERROR] Script {script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"=== Finished: {description or script_name} ===\n")

if __name__ == "__main__":
    # 1. joint sorting task
    run_script("joint_sorting_question.py", description="Generate joint sorting questions")
    run_script("joint_sorting_context.py",   description="Generate joint sorting context")

    # 2. Visit cloze task
    run_script("visit_cloze_question.py", description="Generate visit cloze questions")
    run_script("visit_cloze_context.py",   description="Generate visit cloze context")
    
    run_script("visit_sorting_question.py", description="Generate visit sorting questions")
    run_script("visit_sorting_context.py",   description="Generate visit sorting context")

    print("All scripts completed successfully!")