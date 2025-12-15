import pandas as pd
from pathlib import Path
import json

# Config
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

ADMISSIONS_FILE = RAW_DATA_DIR / "hosp" / "admissions.csv"
NOTES_FILE = RAW_DATA_DIR / "note" / "discharge.csv"
PATIENT_INDEX_FILE = BENCH_DATA_DIR / "patient_index.csv"

DEMO_MODE = True
DEMO_N = 5

# Load data
print("Loading source data...")
admissions = pd.read_csv(
    ADMISSIONS_FILE,
    parse_dates=["admittime", "dischtime"]
)
patient_index = pd.read_csv(PATIENT_INDEX_FILE)
notes = pd.read_csv(NOTES_FILE)

# Optional demo mode
if DEMO_MODE:
    print(f"[DEMO MODE] Processing first {DEMO_N} patients.")
    patient_index = patient_index.head(DEMO_N)

# Build visits and integrate into patient JSON
print("Building visits and updating patient JSON files...")
patients_processed = 0
total_visits_created = 0
patients_with_full_notes = 0

for _, row in patient_index.iterrows():
    pid = row["patient_id"]
    subj = row["subject_id"]
    
    # Path to the patient's primary JSON file
    patient_json_path = BENCH_DATA_DIR / "patients" / f"{pid}.json"
    
    # Load existing patient.json generated in Step 1
    if not patient_json_path.exists():
        print(f"Warning: Patient file {patient_json_path} not found. Skipping.")
        continue
        
    with open(patient_json_path) as f:
        patient_data = json.load(f)
    
    # Filter admissions for this patient and sort by time
    patient_adm = admissions[admissions["subject_id"] == subj].copy()
    patient_adm = patient_adm.sort_values("admittime").reset_index(drop=True)
    
    # Build the visits list for this patient
    visits_list = []
    has_full_note = True  # Initialize to True, will be set to False if any visit lacks a note
    
    for i, adm_row in enumerate(patient_adm.itertuples(), start=1):
        visit_id = f"{pid}-V{i}"
        hadm_id = int(adm_row.hadm_id)
        
        # Extract discharge summary as ground truth note
        note_rows = notes[notes["hadm_id"] == hadm_id]
        if len(note_rows) > 0:
            ground_truth_note = note_rows.iloc[0]["text"]
            current_visit_has_note = True
        else:
            ground_truth_note = None
            current_visit_has_note = False
            has_full_note = False  # At least one visit lacks a note
        
        # Create the visit object according to our target schema
        visit_obj = {
            "visit_id": visit_id,
            "hadm_id": hadm_id,
            "admit_time": adm_row.admittime.strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_time": adm_row.dischtime.strftime("%Y-%m-%d %H:%M:%S"),
            "admission_type": str(adm_row.admission_type) if hasattr(adm_row, 'admission_type') else None,
            "admission_info": {
                # Populate with fields from admissions.csv as needed
                "admission_location": str(adm_row.admission_location) if hasattr(adm_row, 'admission_location') else None,
                "insurance": str(adm_row.insurance) if hasattr(adm_row, 'insurance') else None,
                # Add other admission-related fields here
            },
            "discharge_info": {
                # Will be populated in a later step if needed
            },
            "event_stream": [],  # To be populated by Step 3
            "summary": {},       # To be populated by a later summarization step
            "ground_truth_note": ground_truth_note
        }
        
        visits_list.append(visit_obj)
        total_visits_created += 1
    
    # Update the patient data structure
    patient_data["visits"] = visits_list
    
    # Add support flags at the patient level
    patient_data["support"] = {
        "event_stream": True,  # All patients have event streams (to be populated)
        "has_full_note": has_full_note
    }
    
    # Save the updated patient data back to the same JSON file
    with open(patient_json_path, 'w') as f:
        json.dump(patient_data, f, indent=2, ensure_ascii=False)
    
    patients_processed += 1
    if has_full_note:
        patients_with_full_notes += 1

# Summary
print("\n" + "="*50)
print("STEP 2 COMPLETE")
print(f"  Patients processed: {patients_processed}")
print(f"  Total visits created: {total_visits_created}")
print(f"  Patients with notes for all visits: {patients_with_full_notes}")
print(f"  Output updated in: {BENCH_DATA_DIR / 'patients'}")
if DEMO_MODE:
    print(f"  Mode: DEMO (first {DEMO_N} patients)")
print("="*50)