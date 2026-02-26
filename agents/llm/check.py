import os
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_LABEL = "deepseek-v3.2"
TASK_TYPES = ["joint_sorting", "visit_cloze"]
CONTEXT_DIR = ROOT_DIR / "context_data"
RUN_DIR = ROOT_DIR / "run_llm"
OUTPUT_FILE = ROOT_DIR / "agents" / "llm" / "missing_patients.yaml"

def find_missing_patients():
    missing = set()
    for task in TASK_TYPES:
        src_dir = CONTEXT_DIR / task
        if not src_dir.exists():
            print(f"Warning: source directory not found: {src_dir}")
            continue
        src_pids = {f.stem for f in src_dir.glob("P*.jsonl")}
        res_dir = RUN_DIR / task / MODEL_LABEL
        if not res_dir.exists():
            missing.update(src_pids)
            print(f"Warning: result directory missing: {res_dir}, marking all {len(src_pids)} patients as missing for task {task}")
            continue
        for pid in src_pids:
            if not (res_dir / f"{pid}.jsonl").exists():
                missing.add(pid)
                print(f"Missing: {task}/{pid}")
    return missing

def main():
    print(f"Checking missing result files for model: {MODEL_LABEL}")
    missing = find_missing_patients()
    if missing:
        missing_list = sorted(missing)
        print(f"\nTotal missing patients: {len(missing_list)}")
        print("\n# Copy the following into config.yaml under 'specific_patients':")
        print("specific_patients:")
        for pid in missing_list:
            print(f"  - {pid}")

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            yaml.dump({"specific_patients": missing_list}, f, allow_unicode=True, indent=2)
        print(f"\nMissing list saved to: {OUTPUT_FILE} (YAML format, can be directly included in config.yaml)")
    else:
        print("All patients have result files. No missing patients.")

if __name__ == "__main__":
    main()