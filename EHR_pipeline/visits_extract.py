import pandas as pd
from pathlib import Path
import json

# Config
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

ADMISSIONS_FILE = RAW_DATA_DIR / "hosp"/ "admissions.csv"
NOTES_FILE = RAW_DATA_DIR / "note" / "discharge.csv"
PATIENT_INDEX_FILE = BENCH_DATA_DIR / "patient_index.csv"

DEMO_MODE = True
DEMO_N = 5

# Load data
admissions = pd.read_csv(
    ADMISSIONS_FILE,
    parse_dates=["admittime", "dischtime"]
)
patient_index = pd.read_csv(PATIENT_INDEX_FILE)
notes = pd.read_csv(NOTES_FILE)

# Optional demo mode
if DEMO_MODE:
    patient_index = patient_index.head(DEMO_N)

# Build visits skeleton
for _, row in patient_index.iterrows():
    pid = row["patient_id"]
    subj = row["subject_id"]

    # Filter admissions for this patient
    patient_adm = admissions[admissions["subject_id"] == subj].copy()
    patient_adm = patient_adm.sort_values("admittime").reset_index(drop=True)

    # Create visits directory
    visits_dir = BENCH_DATA_DIR / "patients" / pid / "visits"
    visits_dir.mkdir(parents=True, exist_ok=True)

    visit_list = []
    for i, adm_row in enumerate(patient_adm.itertuples()):
        visit_id = f"{pid}-V{i+1}"
        hadm_id = int(adm_row.hadm_id)

        # Extract discharge summary as ground truth note
        note_rows = notes[notes["hadm_id"] == hadm_id]
        if len(note_rows) > 0:
            ground_truth_note = note_rows.iloc[0]["text"]
            has_note = True
        else:
            ground_truth_note = None
            has_note = False

        visit_json = {
            "visit_id": visit_id,
            "hadm_id": hadm_id,
            "admit_time": adm_row.admittime.strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_time": adm_row.dischtime.strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_note": ground_truth_note,
            "has_note": has_note
            # admission_info, discharge_info, summary -> 后续 Step (LLM)
            # event_stream -> Step 3
        }

        visit_file = visits_dir / f"V{i+1}.json"
        with open(visit_file, "w") as f:
            json.dump(visit_json, f, indent=2)

        visit_list.append(visit_json)

    # Build patient-level support flags
    patient_json_path = BENCH_DATA_DIR / "patients" / pid / "patient.json"

    # Load existing patient.json generated in Step 1
    with open(patient_json_path) as f:
        patient_json = json.load(f)

    # Add support flags
    patient_json["support"] = {
        "event_stream": True,  # 所有 patient 默认可用
        "has_full_note": all(v["has_note"] for v in visit_list)  # 全部 visit 有 note 才为 True
    }

    # Save updated patient.json
    with open(patient_json_path, "w") as f:
        json.dump(patient_json, f, indent=2)

print(
    f"Step 2 complete: visits skeleton + ground truth + flags built"
    f"{f' (demo mode: first {DEMO_N} samples)' if DEMO_MODE else ''}."
)
