import pandas as pd
from pathlib import Path
import json

# Config
ROOT_DIR = Path(__file__).resolve().parent.parent  # 脚本在 LongTermBench/scripts/ 下
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

ADMISSIONS_FILE = RAW_DATA_DIR / "admissions.csv"
PATIENT_INDEX_FILE = BENCH_DATA_DIR / "patient_index.csv"

DEMO_MODE = True   # True 时只处理前 N 个病人
DEMO_N = 5

# Load data
admissions = pd.read_csv(ADMISSIONS_FILE, parse_dates=["admittime", "dischtime"])
patient_index = pd.read_csv(PATIENT_INDEX_FILE)

# Optional: demo mode
if DEMO_MODE:
    patient_index = patient_index.head(DEMO_N)

# Build visits skeleton
for _, row in patient_index.iterrows():
    pid = row["patient_id"]
    subj = row["subject_id"]
    visit_ids = row["visit_ids"].split(";")

    # Filter admissions for this patient
    patient_adm = admissions[admissions["subject_id"] == subj].copy()
    patient_adm = patient_adm.sort_values("admittime").reset_index(drop=True)

    # Create visits directory
    visits_dir = BENCH_DATA_DIR / "patients" / pid / "visits"
    visits_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over admissions and create skeleton
    for i, adm_row in enumerate(patient_adm.itertuples()):
        # visit_id P000123-V1, V2 ...
        visit_id = f"{pid}-V{i+1}"

        visit_json = {
            "visit_id": visit_id,
            "hadm_id": int(adm_row.hadm_id),
            "admit_time": adm_row.admittime.strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_time": adm_row.dischtime.strftime("%Y-%m-%d %H:%M:%S")
            # admission_info, event_stream, discharge_info, summary, ground_truth_note
            # 都留空 / 后续步骤填充
        }

        # Save visit JSON
        visit_file = visits_dir / f"V{i+1}.json"
        with open(visit_file, "w") as f:
            json.dump(visit_json, f, indent=2)

print(f"Step 2 complete: visits skeleton built{' (demo mode)' if DEMO_MODE else ''}.")
