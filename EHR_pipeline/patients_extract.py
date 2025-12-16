import pandas as pd
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

ADMISSIONS_FILE = RAW_DATA_DIR / "hosp" / "admissions.csv"
PATIENTS_FILE = RAW_DATA_DIR / "hosp" / "patients.csv"
DIAGNOSES_FILE = RAW_DATA_DIR / "hosp" / "diagnoses_icd.csv"
D_ICD_FILE = RAW_DATA_DIR / "hosp" / "d_icd_diagnoses.csv"

MIN_VISITS = 3
DEMO_MODE = True
DEMO_N = 5

def main():
    print("Loading source data...")
    admissions = pd.read_csv(ADMISSIONS_FILE, parse_dates=["admittime", "dischtime"])
    patients = pd.read_csv(PATIENTS_FILE)

    diagnoses = pd.read_csv(DIAGNOSES_FILE)
    d_icd = pd.read_csv(D_ICD_FILE)

    diagnoses = diagnoses.merge(
        d_icd,
        on=["icd_code", "icd_version"],
        how="left"
    )

    print("Creating static attribute mappings...")
    first_admissions = admissions.sort_values(
        ["subject_id", "admittime"]
    ).groupby("subject_id").first()

    static_attribute_map = first_admissions[
        ["race", "language", "marital_status"]
    ].to_dict("index")

    print(f"Building cohort (min_visits={MIN_VISITS})...")
    visit_group = admissions.groupby("subject_id")["hadm_id"].agg(list).reset_index()
    visit_group["n_visits"] = visit_group["hadm_id"].apply(len)

    cohort = visit_group[visit_group["n_visits"] >= MIN_VISITS].copy()
    cohort = cohort.sort_values("subject_id").reset_index(drop=True)

    if DEMO_MODE:
        cohort = cohort.head(DEMO_N)

    cohort["patient_id"] = [f"P{i:06d}" for i in range(1, len(cohort) + 1)]

    BENCH_DATA_DIR.mkdir(exist_ok=True)
    patients_dir = BENCH_DATA_DIR / "patients"
    patients_dir.mkdir(exist_ok=True)

    patient_index_rows = []
    hadm_mapping_rows = []

    patient_meta = patients.set_index("subject_id")

    print("Building patient JSON files...")
    for _, row in cohort.iterrows():
        pid = row["patient_id"]
        subj = row["subject_id"]
        hadm_list = row["hadm_id"]

        meta = patient_meta.loc[subj] if subj in patient_meta.index else pd.Series()
        static_attrs = static_attribute_map.get(subj, {})

        patient_info = {
            "patient_id": pid,
            "subject_id": int(subj),
            "gender": str(meta.get("gender", "UNKNOWN")),
            "race": str(static_attrs.get("race", "UNKNOWN")),
            "age_first_visit": int(meta.get("anchor_age", 0)),
            "language": str(static_attrs.get("language", "UNKNOWN")),
            "marital_status": str(static_attrs.get("marital_status", "UNKNOWN"))
        }

        patient_admissions = admissions[
            admissions["subject_id"] == subj
        ].sort_values("admittime").reset_index(drop=True)

        visits_list = []

        for visit_idx, (_, adm_row) in enumerate(patient_admissions.iterrows(), start=1):
            hadm_id = int(adm_row.hadm_id)

            hadm_mapping_rows.append({
                "hadm_id": hadm_id,
                "patient_id": pid,
                "visit_index": visit_idx - 1
            })

            diag_rows = diagnoses[diagnoses["hadm_id"] == hadm_id]
            diag_rows = diag_rows.sort_values("seq_num")

            diagnosis = [
                {
                    "seq_num": int(r.seq_num),
                    "icd_version": int(r.icd_version),
                    "description": str(r.long_title)
                    if pd.notnull(r.long_title) else None
                }
                for _, r in diag_rows.iterrows()
            ]

            visit_obj = {
                "visit_id": f"{pid}-V{visit_idx}",
                "hadm_id": hadm_id,

                "admission_info": {
                    "admission_time": adm_row.admittime.strftime("%Y-%m-%d %H:%M:%S"),
                    "admission_type": str(adm_row.admission_type),
                    "admission_location": str(adm_row.admission_location),
                    "insurance": str(adm_row.insurance),
                    "admission_note": None
                },

                "discharge_info": {
                    "discharge_time": adm_row.dischtime.strftime("%Y-%m-%d %H:%M:%S"),
                    "discharge_location": str(adm_row.discharge_location),
                    "discharge_note": None
                },

                "event_stream": [],
                "diagnosis": diagnosis,
                "summary": {},
                "ground_truth_note": None
            }

            visits_list.append(visit_obj)

        patient_record = {
            "patient_info": patient_info,
            "visits": visits_list
        }

        patient_file = patients_dir / f"{pid}.json"
        with open(patient_file, "w", encoding="utf-8") as f:
            json.dump(patient_record, f, indent=2, ensure_ascii=False)

        patient_index_rows.append({
            "patient_id": pid,
            "subject_id": int(subj),
            "n_visits": len(visits_list),
            "visit_ids": ";".join(str(h) for h in hadm_list)
        })

    patient_index_df = pd.DataFrame(patient_index_rows)
    patient_index_df.to_csv(BENCH_DATA_DIR / "patient_index.csv", index=False)

    hadm_mapping_df = pd.DataFrame(hadm_mapping_rows)
    hadm_mapping_df.to_csv(BENCH_DATA_DIR / "hadm_mapping.csv", index=False)

    print("STEP 1 COMPLETE")

if __name__ == "__main__":
    main()
