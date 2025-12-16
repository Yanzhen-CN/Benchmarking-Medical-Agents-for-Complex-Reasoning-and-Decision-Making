import pandas as pd
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_EVENT_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

DEMO_MODE = True
DEMO_N = 5


def load_patient_files():
    patients_dir = BENCH_DATA_DIR / "patients"
    patient_files = sorted(patients_dir.glob("P*.json"))
    if DEMO_MODE:
        patient_files = patient_files[:DEMO_N]
    return patient_files


def load_events():
    print("Loading extracted event tables...")

    lab_df = pd.read_csv(
        RAW_EVENT_DIR / "labevents_extract.csv",
        parse_dates=["charttime"]
    )

    vital_df = pd.read_csv(
        RAW_EVENT_DIR / "chartevents_extract.csv",
        parse_dates=["charttime"]
    )

    med_df = pd.read_csv(
        RAW_EVENT_DIR / "hosp"/ "prescriptions.csv",
        parse_dates=["starttime"]
    )

    img_df = pd.read_csv(
        RAW_EVENT_DIR / "note"/ "radiology_extract.csv",
        parse_dates=["charttime"]
    )

    proc_df = pd.read_csv(
        RAW_EVENT_DIR / "hosp"/ "procedures_extract.csv",
        parse_dates=["charttime"]
    )

    return lab_df, vital_df, med_df, img_df, proc_df


def build_lab_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    events = []

    for _, r in rows.iterrows():
        events.append({
            "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
            "type": "lab",
            "name": r["label"],
            "category": r.get("category", None),
            "value": r["valuenum"],
            "unit": r.get("valueuom", None),
            "flag": r.get("flag", None)
        })

    return events


def build_vital_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    events = []

    for _, r in rows.iterrows():
        events.append({
            "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
            "type": "vital",
            "name": r["label"],
            "value": r["valuenum"],
            "unit": r.get("unitname", None),
            "flag": "warning"
        })

    return events


def build_med_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    events = []

    for _, r in rows.iterrows():
        events.append({
            "timestamp": r["starttime"].strftime("%Y-%m-%d %H:%M:%S"),
            "type": "medication",
            "name": r["drug"],
            "dose": r.get("dose_val_rx", None),
            "unit": r.get("dose_unit_rx", None),
            "route": r.get("route", None)
        })

    return events


def build_imaging_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    events = []

    for _, r in rows.iterrows():
        events.append({
            "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
            "type": "imaging",
            "exam": r.get("exam", None),
            "content": r["text"]
        })

    return events


def build_procedure_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    events = []

    for _, r in rows.iterrows():
        events.append({
            "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
            "type": "procedure",
            "name": r["long_title"],
            "code_system": f"ICD-{r.get('icd_version', '')}"
        })

    return events


def main():
    patient_files = load_patient_files()
    lab_df, vital_df, med_df, img_df, proc_df = load_events()

    print(f"Building event_stream for {len(patient_files)} patients...")

    for patient_file in patient_files:
        with open(patient_file) as f:
            patient = json.load(f)

        for visit in patient["visits"]:
            hadm_id = visit["hadm_id"]
            events = []

            events.extend(build_lab_events(lab_df, hadm_id))
            events.extend(build_vital_events(vital_df, hadm_id))
            events.extend(build_med_events(med_df, hadm_id))
            events.extend(build_imaging_events(img_df, hadm_id))
            events.extend(build_procedure_events(proc_df, hadm_id))

            events = sorted(events, key=lambda x: x["timestamp"])
            visit["event_stream"] = events

        with open(patient_file, "w", encoding="utf-8") as f:
            json.dump(patient, f, indent=2, ensure_ascii=False)

        print(f"Updated {patient_file.name}")

    print("Complete: event_stream written to patient JSON files.")


if __name__ == "__main__":
    main()
