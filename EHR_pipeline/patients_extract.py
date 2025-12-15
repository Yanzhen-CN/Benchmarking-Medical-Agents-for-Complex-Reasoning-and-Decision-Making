import pandas as pd
from pathlib import Path
import json

# Config
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

ADMISSIONS_FILE = RAW_DATA_DIR / "hosp" / "admissions.csv"
PATIENTS_FILE = RAW_DATA_DIR / "hosp" / "patients.csv"

MIN_VISITS = 3

# Optional Demo mode
DEMO_MODE = True
DEMO_N = 5

# Load data
admissions = pd.read_csv(ADMISSIONS_FILE)
patients = pd.read_csv(PATIENTS_FILE)

# Aggregate visits
visit_group = (
    admissions
    .groupby("subject_id")["hadm_id"]
    .agg(list)
    .reset_index()
)
admissions_sorted = admissions.sort_values("admittime")
race_map = (
    admissions_sorted
    .groupby("subject_id")["race"]
    .first()
)

visit_group["n_visits"] = visit_group["hadm_id"].apply(len)

# Filter cohort
cohort = visit_group[visit_group["n_visits"] >= MIN_VISITS].copy()
cohort = cohort.sort_values("subject_id").reset_index(drop=True)

# Apply demo mode filter after cohort is defined
if DEMO_MODE:
    cohort = cohort.head(DEMO_N)

# Assign bench patient_id
cohort["patient_id"] = [
    f"P{i:06d}" for i in range(1, len(cohort) + 1)
]

cohort["visit_ids"] = cohort["hadm_id"].apply(
    lambda x: ";".join(str(v) for v in x)
)

patient_index = cohort[[
    "patient_id",
    "subject_id",
    "n_visits",
    "visit_ids"
]]

# Save patient_index.csv
BENCH_DATA_DIR.mkdir(exist_ok=True)
(BENCH_DATA_DIR / "patients").mkdir(exist_ok=True)

patient_index_path = BENCH_DATA_DIR / "patient_index.csv"
patient_index.to_csv(patient_index_path, index=False)

# Build patient.json (patient_info)
patient_meta = patients.set_index("subject_id")

for _, row in patient_index.iterrows():
    pid = row["patient_id"]
    subj = row["subject_id"]

    if subj not in patient_meta.index:
        continue

    meta = patient_meta.loc[subj]

    patient_json = {
        "patient_id": pid,
        "gender": meta.get("gender"),
        "race": race_map.get(subj),
        "age_first_visit": int(meta.get("anchor_age")),
    }

    patient_dir = BENCH_DATA_DIR / "patients" / pid
    patient_dir.mkdir(parents=True, exist_ok=True)

    with open(patient_dir / "patient.json", "w") as f:
        json.dump(patient_json, f, indent=2)

mode_suffix = f" (demo mode: first {DEMO_N} samples)" if DEMO_MODE else ""
print(f"Step 1 complete: patient_index.csv and patient_info built{mode_suffix}.")