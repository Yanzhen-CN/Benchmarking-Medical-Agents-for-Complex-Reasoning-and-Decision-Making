import pandas as pd
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

DEMO_MODE = True
DEMO_N = 5


def load_patient_files():
    patients_dir = BENCH_DATA_DIR / "patients"
    files = sorted(patients_dir.glob("P*.json"))
    return files[:DEMO_N] if DEMO_MODE else files


def load_events():
    print("Loading event tables...")
    # lab
    lab_df = pd.read_csv(
        RAW_DATA_DIR / "labevents_extract.csv",
        parse_dates=["charttime"]
    )

    d_lab = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "d_labitems.csv"
    )

    lab_df = lab_df.merge(
        d_lab[["itemid", "label", "category", "fluid"]],
        on="itemid",
        how="left"
    )

    # vital
    vital_df = pd.read_csv(
        RAW_DATA_DIR / "chartevents_extract.csv",
        parse_dates=["charttime"]
    )

    d_items = pd.read_csv(
        RAW_DATA_DIR / "icu" / "d_items.csv"
    )

    vital_df = vital_df.merge(
        d_items[
            ["itemid", "label", "unitname", "lownormalvalue", "highnormalvalue"]
        ],
        on="itemid",
        how="left"
    )
    
    # med
    med_df = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "prescriptions.csv",
        parse_dates=["starttime"]
    )

    # image
    img_df = pd.read_csv(
        RAW_DATA_DIR / "note" / "radiology.csv",
        parse_dates=["charttime"]
    )

    # procedure
    proc_df = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "procedures_icd.csv"
    )

    d_proc = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "d_icd_procedures.csv"
    )

    proc_df = proc_df.merge(
        d_proc,
        on=["icd_code", "icd_version"],
        how="left"
    )

    return lab_df, vital_df, med_df, img_df, proc_df


def build_lab_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    return [{
        "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
        "type": "lab",
        "name": r["label"],
        "category": r.get("category"),
        "value": r["valuenum"],
        "unit": r.get("valueuom"),
        "flag": r.get("flag")
    } for _, r in rows.iterrows()]


def build_vital_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    return [{
        "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
        "type": "vital",
        "name": r["label"],
        "value": r["valuenum"],
        "unit": r.get("unitname"),
        "flag": "warning" if r.get("warning") == 1 else None
    } for _, r in rows.iterrows()]


def build_med_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    return [{
        "timestamp": r["starttime"].strftime("%Y-%m-%d %H:%M:%S"),
        "type": "medication",
        "name": r["drug"],
        "dose": r.get("dose_val_rx"),
        "unit": r.get("dose_unit_rx"),
        "route": r.get("route")
    } for _, r in rows.iterrows()]


def build_imaging_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    return [{
        "timestamp": r["charttime"].strftime("%Y-%m-%d %H:%M:%S"),
        "type": "imaging",
        "content": r["text"]
    } for _, r in rows.iterrows()]


def build_procedure_events(df, hadm_id):
    rows = df[df["hadm_id"] == hadm_id]
    return [{
        "timestamp": None,
        "type": "procedure",
        "name": r["long_title"]
    } for _, r in rows.iterrows() if pd.notnull(r.get("long_title"))]


def main():
    patient_files = load_patient_files()
    lab_df, vital_df, med_df, img_df, proc_df = load_events()

    for pf in patient_files:
        with open(pf, "r", encoding="utf-8") as f:
            patient = json.load(f)

        for visit in patient["visits"]:
            hadm_id = visit["hadm_id"]
            events = []

            events.extend(build_lab_events(lab_df, hadm_id))
            events.extend(build_vital_events(vital_df, hadm_id))
            events.extend(build_med_events(med_df, hadm_id))
            events.extend(build_imaging_events(img_df, hadm_id))
            events.extend(build_procedure_events(proc_df, hadm_id))

            events = [e for e in events if e["timestamp"] is not None]
            events.sort(key=lambda x: x["timestamp"])

            visit["event_stream"] = events

        with open(pf, "w", encoding="utf-8") as f:
            json.dump(patient, f, indent=2, ensure_ascii=False)

        print(f"Updated {pf.name}")

    print("STEP 2 COMPLETE: event_stream populated.")


if __name__ == "__main__":
    main()
