import pandas as pd
from pathlib import Path
import json

# ==================== CONFIGURATION ====================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

ADMISSIONS_FILE = RAW_DATA_DIR / "hosp" / "admissions.csv"
PATIENTS_FILE = RAW_DATA_DIR / "hosp" / "patients.csv"

MIN_VISITS = 3
DEMO_MODE = True
DEMO_N = 5
# =======================================================

def main():
    print("Loading source data...")
    admissions = pd.read_csv(ADMISSIONS_FILE, parse_dates=["admittime", "dischtime"])
    patients = pd.read_csv(PATIENTS_FILE)

    # Create mapping for patient-level static attributes from their FIRST admission
    print("Creating static attribute mappings from first admissions...")
    first_admissions = admissions.sort_values(['subject_id', 'admittime']).groupby('subject_id').first()
    static_attribute_map = first_admissions[['race', 'language', 'marital_status']].to_dict('index')

    # Build patient cohort with required minimum visits
    print(f"Building cohort (min_visits={MIN_VISITS})...")
    visit_group = admissions.groupby("subject_id")["hadm_id"].agg(list).reset_index()
    visit_group["n_visits"] = visit_group["hadm_id"].apply(len)
    cohort = visit_group[visit_group["n_visits"] >= MIN_VISITS].copy()
    cohort = cohort.sort_values("subject_id").reset_index(drop=True)

    if DEMO_MODE:
        print(f"[DEMO MODE] Limiting to first {DEMO_N} patients.")
        cohort = cohort.head(DEMO_N)

    # Assign benchmark patient IDs
    cohort["patient_id"] = [f"P{i:06d}" for i in range(1, len(cohort) + 1)]

    # Prepare output directories
    BENCH_DATA_DIR.mkdir(exist_ok=True)
    patients_dir = BENCH_DATA_DIR / "patients"
    patients_dir.mkdir(exist_ok=True)

    # Data structures for supporting index files
    patient_index_rows = []
    hadm_mapping_rows = []
    patient_meta = patients.set_index("subject_id")

    print("Building patient JSON files...")
    for idx, row in cohort.iterrows():
        pid = row["patient_id"]
        subj = row["subject_id"]
        hadm_list = row["hadm_id"]

        # Get base patient info from patients.csv
        meta = patient_meta.loc[subj] if subj in patient_meta.index else pd.Series()
        
        # Get static attributes from the patient's FIRST admission record
        static_attrs = static_attribute_map.get(subj, {})
        
        # ===== 1. Build PATIENT_INFO (Static) =====
        patient_info = {
            "patient_id": pid,
            "subject_id": int(subj),
            "gender": str(meta.get("gender", "UNKNOWN")),
            "race": str(static_attrs.get("race", "UNKNOWN")),
            "age_first_visit": int(meta.get("anchor_age", 0)),
            "language": str(static_attrs.get("language", "UNKNOWN")),
            "marital_status": str(static_attrs.get("marital_status", "UNKNOWN"))
        }

        # Get all admissions for this patient, sorted by time
        patient_admissions = admissions[admissions["subject_id"] == subj].copy()
        patient_admissions = patient_admissions.sort_values("admittime").reset_index(drop=True)

        visits_list = []
        
        # ===== 2. Build VISITS skeleton (Dynamic) =====
        for visit_idx, (_, adm_row) in enumerate(patient_admissions.iterrows(), start=1):
            hadm_id = int(adm_row.hadm_id)
            
            # Record mapping for Step 3 (event_stream integration)
            hadm_mapping_rows.append({
                "hadm_id": hadm_id,
                "patient_id": pid,
                "visit_index": visit_idx - 1  # 0-based for list indexing
            })
            
            # Create the visit skeleton with the FINAL structure
            visit_obj = {
                "visit_id": f"{pid}-V{visit_idx}",
                "hadm_id": hadm_id,
                "admit_time": adm_row.admittime.strftime("%Y-%m-%d %H:%M:%S"),
                "discharge_time": adm_row.dischtime.strftime("%Y-%m-%d %H:%M:%S"),
                
                # Admission Info: table data + placeholder for Part B extraction
                "admission_info": {
                    # From admissions.csv (management context)
                    "admission_type": str(adm_row.admission_type),
                    "admission_location": str(adm_row.admission_location),
                    "insurance": str(adm_row.insurance),
                    # Placeholder for Part B (LLM/rule extraction from note)
                    "admission_note": None
                },
                
                # Discharge Info: table data + placeholder for Part B extraction
                "discharge_info": {
                    # From admissions.csv
                    "discharge_location": str(adm_row.discharge_location),
                    # Placeholder for Part B (LLM/rule extraction from note)
                    "discharge_note": None
                },
                
                # Core data streams (to be populated by subsequent steps)
                "event_stream": [],      # Will be populated by Step 3 (Python pipeline)
                "summary": {},           # Will be populated by Part B (LLM extraction)
                "ground_truth_note": None  # Will be populated by Step 2 (note linkage)
            }
            
            visits_list.append(visit_obj)

        # ===== 3. Assemble complete patient record =====
        patient_record = {
            "patient_info": patient_info,
            "visits": visits_list
        }
        
        # Save the complete patient to a single JSON file
        patient_file = patients_dir / f"{pid}.json"
        with open(patient_file, 'w') as f:
            json.dump(patient_record, f, indent=2, ensure_ascii=False)
        
        # Add to patient index
        patient_index_rows.append({
            "patient_id": pid,
            "subject_id": int(subj),
            "n_visits": len(visits_list),
            "visit_ids": ";".join(str(h) for h in hadm_list)
        })

    # ===== 4. Save supporting index files =====
    print("Saving supporting index files...")
    
    # Save patient_index.csv
    patient_index_df = pd.DataFrame(patient_index_rows)
    patient_index_path = BENCH_DATA_DIR / "patient_index.csv"
    patient_index_df.to_csv(patient_index_path, index=False)
    
    # Save hadm_mapping.csv (critical for Step 3 performance)
    hadm_mapping_df = pd.DataFrame(hadm_mapping_rows)
    mapping_path = BENCH_DATA_DIR / "hadm_mapping.csv"
    hadm_mapping_df.to_csv(mapping_path, index=False)
    
    # ===== 5. Print summary =====
    print("\n" + "="*60)
    print("STEP 1 COMPLETE: Patient Framework Built")
    print("="*60)
    print(f"Patients Processed: {len(cohort)}")
    print(f"Total Visits in Cohort: {cohort['n_visits'].sum()}")
    print(f"\nOutput Structure:")
    print(f"  • Primary Data: {len(cohort)} files in '{patients_dir}/'")
    print(f"     - Format: Single JSON per patient")
    print(f"     - Contains: patient_info + visits[] skeleton")
    print(f"  • Index Files:")
    print(f"     - {patient_index_path}")
    print(f"     - {mapping_path}")
    print(f"\nPlaceholders for Subsequent Steps:")
    print(f"  • admission_info.admission_note: For Part B extraction")
    print(f"  • discharge_info.discharge_note: For Part B extraction")
    print(f"  • event_stream[]: For Step 3 (Python pipeline)")
    print(f"  • summary{{}}: For Part B (LLM extraction)")
    print(f"  • ground_truth_note: For Step 2 (note linkage)")
    
    if DEMO_MODE:
        print(f"\nMode: DEMO (First {DEMO_N} patients)")

if __name__ == "__main__":
    main()