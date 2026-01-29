import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

DEMO_MODE = True
DEMO_N = 5


# ============================================================
# Utilities
# ============================================================
def _fmt_ts(x) -> Optional[str]:
    """Convert datetime-like value to a normalized timestamp string."""
    if x is None or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def _nan_to_none(x):
    """Convert NaN/NA to None for strict JSON serialization."""
    return None if pd.isna(x) else x


def load_patient_files() -> List[Path]:
    """Load patient JSON files from bench_data/patients."""
    patients_dir = BENCH_DATA_DIR / "patients"
    files = sorted(patients_dir.glob("P*.json"))
    return files[:DEMO_N] if DEMO_MODE else files


def collect_hadm_ids(patient_files: List[Path]) -> Set[int]:
    """Collect the set of hadm_id values referenced by selected patient JSONs."""
    hadm_ids: Set[int] = set()
    for pf in patient_files:
        with open(pf, "r", encoding="utf-8") as f:
            patient = json.load(f)
        for v in patient.get("visits", []):
            hid = v.get("hadm_id")
            if hid is None:
                continue
            try:
                hadm_ids.add(int(hid))
            except Exception:
                pass
    return hadm_ids


def _filter_by_hadm(df: pd.DataFrame, hadm_ids: Set[int]) -> pd.DataFrame:
    """Filter a dataframe to the cohort hadm_ids, with robust casting."""
    if df.empty or "hadm_id" not in df.columns:
        return df

    df = df.copy()
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
    df = df.dropna(subset=["hadm_id"])
    df["hadm_id"] = df["hadm_id"].astype(int)
    return df[df["hadm_id"].isin(hadm_ids)].copy()


def assign_visit_ids(visits: List[Dict[str, Any]]) -> None:
    """
    Assign visit_id (V1, V2, ...) in-place.

    Rule:
      - Use the original order in patient["visits"] for stability.
      - If you want chronological visit_id later, sort visits by admittime first.
    """
    for i, v in enumerate(visits, start=1):
        v["visit_id"] = f"V{i}"


def assign_event_ids(events: List[Dict[str, Any]], patient_id: str, visit_id: str) -> None:
    """
    Assign event_id (P0001-V1-E001) in-place, based on the final sorted event_stream.

    Rule:
      event_id = {patient_id}-{visit_id}-E{seq}
      where seq is 1..N within the visit.
    """
    for i, e in enumerate(events, start=1):
        width = max(2, len(str(i)))  # E01..E99
        e["event_id"] = f"{patient_id}-{visit_id}-E{i:0{width}d}"

def load_lab(hadm_ids: Set[int]):
    # ----------------------------
    # Lab events (extracted)
    # ----------------------------
    lab_df = pd.read_csv(
        RAW_DATA_DIR / "labevents_extract.csv",
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "value", "valueuom", "flag"],
        parse_dates=["charttime"],
        dtype={
            "hadm_id": "Int64",
            "itemid": "Int64",
            "value": "string",
            "valueuom": "string",
            "flag": "string",
        },
        low_memory=False,
    )
    lab_df = _filter_by_hadm(lab_df, hadm_ids)

    d_lab = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "d_labitems.csv",
        usecols=["itemid", "label", "category", "fluid"],
        dtype={"itemid": "Int64", "label": "string", "category": "string", "fluid": "string"},
        low_memory=False,
    )
    lab_df = lab_df.merge(d_lab, on="itemid", how="left")
    return lab_df

def load_vital(hadm_ids: Set[int]):
    # ----------------------------
    # Vital events (extracted)
    # ----------------------------
    vital_df = pd.read_csv(
        RAW_DATA_DIR / "chartevents_extract.csv",
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "value", "warning"],
        parse_dates=["charttime"],
        dtype={
            "hadm_id": "Int64",
            "itemid": "Int64",
            "value": "string",
            # Read as string first; normalize to int to handle mixed representations.
            "warning": "string",
        },
        low_memory=False,
    )
    vital_df = _filter_by_hadm(vital_df, hadm_ids)
    vital_df["warning_int"] = pd.to_numeric(vital_df["warning"], errors="coerce").fillna(0).astype(int)

    d_items = pd.read_csv(
        RAW_DATA_DIR / "icu" / "d_items.csv",
        usecols=["itemid", "label", "unitname", "lownormalvalue", "highnormalvalue"],
        dtype={"itemid": "Int64", "label": "string", "unitname": "string"},
        low_memory=False,
    )
    vital_df = vital_df.merge(
        d_items[["itemid", "label", "unitname", "lownormalvalue", "highnormalvalue"]],
        on="itemid",
        how="left",
    )
    return vital_df

def load_prescriptions(hadm_ids: Set[int]) -> pd.DataFrame:
    # ----------------------------
    # Prescriptions (prefer extracted if available)
    # ----------------------------
    pres_path = RAW_DATA_DIR / "prescriptions_extract.csv"
    if not pres_path.exists():
        pres_path = RAW_DATA_DIR / "hosp" / "prescriptions.csv"

    pres_df = pd.read_csv(
        pres_path,
        usecols=["hadm_id", "starttime", "drug", "dose_val_rx", "dose_unit_rx", "route"],
        parse_dates=["starttime"],
        dtype={
            "hadm_id": "Int64",
            "drug": "string",
            # Mixed-type column in MIMIC; enforce string to avoid dtype warnings.
            "dose_val_rx": "string",
            "dose_unit_rx": "string",
            "route": "string",
        },
        low_memory=False,
    )
    pres_df = _filter_by_hadm(pres_df, hadm_ids)
    return pres_df

def load_emar(hadm_ids: Set[int]) -> pd.DataFrame:
    """
    Load eMAR administration records and join with eMAR_detail to get dose/route.
    """
    emar_path = RAW_DATA_DIR / "emar_extract.csv"
    if not emar_path.exists():
        emar_path = RAW_DATA_DIR / "hosp" / "emar.csv"

    emar_df = pd.read_csv(
        emar_path,
        usecols=["hadm_id", "emar_id", "charttime", "scheduletime", "medication", "event_txt"],
        parse_dates=["charttime", "scheduletime"],
        dtype={
            "hadm_id": "Int64",
            "emar_id": "string",
            "medication": "string",
            "event_txt": "string",
        },
        low_memory=False,
    )
    emar_df = _filter_by_hadm(emar_df, hadm_ids)
    print("load emar susccess")
    emar_detail_path = RAW_DATA_DIR / "emar_detail_extract.csv"
    if not emar_detail_path.exists():
        emar_detail_path = RAW_DATA_DIR / "hosp" / "emar_detail.csv"

    emar_detail_df = pd.read_csv(
        emar_detail_path,
        usecols=[
            "emar_id", "dose_given", "dose_given_unit", 
            "dose_due", "dose_due_unit", "route"
        ],
        dtype={
            "emar_id": "string",
            "dose_given": "string",
            "dose_given_unit": "string",
            "dose_due": "string",
            "dose_due_unit": "string",
            "route": "string",
        },
        low_memory=False,
    )
    emar_detail_df = _filter_by_hadm(emar_detail_df, hadm_ids)

    # merge
    emar_df = emar_df.merge(emar_detail_df, on="emar_id", how="left")

    # Event time: prefer charttime, fallback to scheduletime
    emar_df["event_time"] = emar_df["charttime"].fillna(emar_df["scheduletime"])

    emar_df = emar_df.drop_duplicates(
        subset=[
            "hadm_id", "emar_id", "event_time", "medication", "event_txt",
            "dose_given", "dose_given_unit", "dose_due", "dose_due_unit", "route"
        ]
    )
    emar_df = _filter_by_hadm(emar_df, hadm_ids)
    return emar_df

def load_integrated_medications(hadm_ids: Set[int]) -> pd.DataFrame:
    """
    Load Prescriptions and eMAR, then merge them into a single 'med_df'.
    
    Logic:
    1. Load raw tables.
    2. Aggregate eMAR events by 'poe_id' (e.g., merge "Administered", "Held").
    3. Left Join eMAR info onto Prescriptions (Prescription is the master event).
    4. Fill missing eMAR status with 'Unknown'.
    """
    print("  - Loading and merging medication data...")
    
    pres_df = load_prescriptions(hadm_ids)
    emar_df = load_emar(hadm_ids)

    if pres_df.empty:
        return pd.DataFrame()

    # 1. Aggregate eMAR statuses
    # If a poe_id has multiple records, combine them (e.g., "Administered, Held")
    if not emar_df.empty:
        # Ensure string type for joining
        emar_df['event_txt'] = emar_df['event_txt'].astype(str)
        
        emar_agg = emar_df.groupby('poe_id')['event_txt'].apply(
            lambda x: ', '.join(sorted(set(x)))
        ).reset_index()
        emar_agg.rename(columns={'event_txt': 'execution_status'}, inplace=True)
    else:
        emar_agg = pd.DataFrame(columns=['poe_id', 'execution_status'])

    # 2. Merge (Left Join)
    # Ensure keys are strings to prevent mismatch
    pres_df['poe_id'] = pres_df['poe_id'].astype(str)
    emar_agg['poe_id'] = emar_agg['poe_id'].astype(str)

    med_df = pres_df.merge(emar_agg, on='poe_id', how='left')

    # 3. Handle missing status
    med_df['execution_status'] = med_df['execution_status'].fillna('Unknown')

    return med_df

def load_imaging(hadm_ids: Set[int]) -> pd.DataFrame:
    # ----------------------------
    # Imaging (prefer extracted if available)
    # ----------------------------
    img_path = RAW_DATA_DIR / "radiology_extract.csv"
    if not img_path.exists():
        img_path = RAW_DATA_DIR / "note" / "radiology.csv"

    img_df = pd.read_csv(
        img_path,
        usecols=["hadm_id", "charttime", "text"],
        parse_dates=["charttime"],
        dtype={"hadm_id": "Int64", "text": "string"},
        low_memory=False,
    )
    img_df = _filter_by_hadm(img_df, hadm_ids)
    return img_df


def load_procedures(hadm_ids: Set[int]) -> pd.DataFrame:
    # ----------------------------
    # Procedures (add timestamp via chartdate)
    # ----------------------------
    proc_df = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "procedures_icd.csv",
        usecols=["hadm_id", "icd_code", "icd_version", "chartdate"],
        parse_dates=["chartdate"],
        dtype={"hadm_id": "Int64", "icd_code": "string", "icd_version": "Int64"},
        low_memory=False,
    )
    proc_df = _filter_by_hadm(proc_df, hadm_ids)

    d_proc = pd.read_csv(
        RAW_DATA_DIR / "hosp" / "d_icd_procedures.csv",
        usecols=["icd_code", "icd_version", "long_title"],
        dtype={"icd_code": "string", "icd_version": "Int64", "long_title": "string"},
        low_memory=False,
    )
    proc_df = proc_df.merge(d_proc, on=["icd_code", "icd_version"], how="left")
    return proc_df



# ============================================================
# Event builders (merge by timestamp only)
# ============================================================
def build_lab_events(df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """
    Build lab events merged strictly by charttime.

    Output format:
      - One event per unique charttime
      - Each event contains an 'items' list of lab measurements
    """
    rows = df[df["hadm_id"] == hadm_id].copy()
    if rows.empty:
        return []

    rows = rows.dropna(subset=["charttime"])

    events: List[Dict[str, Any]] = []
    for ct, g in rows.groupby("charttime"):
        items = []
        for r in g.to_dict("records"):
            items.append(
                {
                    "name": _nan_to_none(r.get("label")),
                    "category": _nan_to_none(r.get("category")),
                    "fluid": _nan_to_none(r.get("fluid")),
                    "value_num": _nan_to_none(r.get("valuenum")),
                    "value_text": _nan_to_none(r.get("value")),
                    "unit": _nan_to_none(r.get("valueuom")),
                    "flag": _nan_to_none(r.get("flag")),
                }
            )

        events.append(
            {
                "timestamp": _fmt_ts(ct),
                "type": "lab",
                "items": items,
            }
        )

    return events


def build_vital_events(df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """
    Build vital events merged strictly by charttime.

    Notes:
      - Each item can carry a per-item flag.
      - The event-level flag is set to 'warning' if any item is warning.
    """
    rows = df[df["hadm_id"] == hadm_id].copy()
    if rows.empty:
        return []

    rows = rows.dropna(subset=["charttime"])

    events: List[Dict[str, Any]] = []
    for ct, g in rows.groupby("charttime"):
        any_warning = bool((g["warning_int"] == 1).any()) if "warning_int" in g.columns else False

        items = []
        for r in g.to_dict("records"):
            item_warning = (int(r.get("warning_int", 0)) == 1) if r.get("warning_int") is not None else False
            items.append(
                {
                    "name": _nan_to_none(r.get("label")),
                    "value_num": _nan_to_none(r.get("valuenum")),
                    "value_text": _nan_to_none(r.get("value")),
                    "unit": _nan_to_none(r.get("unitname")),
                    "flag": "warning" if item_warning else None,
                }
            )

        events.append(
            {
                "timestamp": _fmt_ts(ct),
                "type": "vital",
                "flag": "warning" if any_warning else None,
                "items": items,
            }
        )

    return events


def build_pres_events(df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """
    Build presscription events merged strictly by starttime.

    Output format:
      - One event per unique starttime
      - Each event contains an 'items' list of administered/started prescription
    """
    rows = df[df["hadm_id"] == hadm_id].copy()
    if rows.empty:
        return []

    rows = rows.dropna(subset=["starttime"])

    events: List[Dict[str, Any]] = []
    for st, g in rows.groupby("starttime"):
        items = []
        for r in g.to_dict("records"):
            items.append(
                {
                    "name": _nan_to_none(r.get("drug")),
                    "dose": _nan_to_none(r.get("dose_val_rx")),
                    "unit": _nan_to_none(r.get("dose_unit_rx")),
                    "route": _nan_to_none(r.get("route")),
                }
            )

        events.append(
            {
                "timestamp": _fmt_ts(st),
                "type": "prescription",
                "items": items,
            }
        )

    return events

def build_emar_events(df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """
    medication events: derived from hosp.emar (+ emar_detail) (administration)
    merged by event_time into items list.
    """
    rows = df[df["hadm_id"] == hadm_id].copy()
    if rows.empty:
        return []

    rows = rows.dropna(subset=["event_time"])

    events: List[Dict[str, Any]] = []

    for t, g in rows.groupby("event_time"):
        items = []
        for r in g.to_dict("records"):
            # Prefer dose_given; fallback to dose_due
            dose = r.get("dose_given")
            unit = r.get("dose_given_unit")
            if dose is None or pd.isna(dose):
                dose = r.get("dose_due")
                unit = r.get("dose_due_unit")

            items.append(
                {
                    "name": _nan_to_none(r.get("medication")),
                    "dose": _nan_to_none(dose),
                    "unit": _nan_to_none(unit),
                    # "route": _nan_to_none(r.get("route")),
                    "status": _nan_to_none(r.get("event_txt")),
                }
            )

        events.append(
            {
                "timestamp": _fmt_ts(t),
                "type": "emar",
                "items": items
            }
        )

    return events

def build_med_events(med_df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """
    Construct medication events from the integrated DataFrame.
    """
    subset = med_df[med_df["hadm_id"] == hadm_id]
    
    events = []
    for _, row in subset.iterrows():
        start_time = row.get("starttime")
        
        # Skip invalid times
        if pd.isna(start_time):
            continue

        # Format dose string (Val + Unit)
        dose_val = row.get("dose_val_rx")
        dose_unit = row.get("dose_unit_rx")
        dose_str = f"{dose_val} {dose_unit}" if pd.notna(dose_val) and pd.notna(dose_unit) else str(dose_val)
        if dose_str == "nan": 
            dose_str = None

        # Build event
        evt = {
            "type": "MEDICATION",
            "timestamp": _fmt_ts(start_time),
            "end_timestamp": _fmt_ts(row.get("stoptime")),
            "drug": _nan_to_none(row.get("drug")),
            "route": _nan_to_none(row.get("route")),
            "dose": _nan_to_none(dose_str),
            "poe_id": _nan_to_none(row.get("poe_id")),
            "status": row.get("execution_status")  # This is the integrated column
        }
        
        events.append(evt)
        
    return events

def build_imaging_events(df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """Build imaging events (one record -> one event)."""
    rows = df[df["hadm_id"] == hadm_id].copy()
    if rows.empty:
        return []

    rows = rows.dropna(subset=["charttime"])

    events = []
    for r in rows.to_dict("records"):
        events.append(
            {
                "timestamp": _fmt_ts(r.get("charttime")),
                "type": "imaging",
                "content": _nan_to_none(r.get("text")),
            }
        )
    return events


def build_procedure_events(df: pd.DataFrame, hadm_id: int) -> List[Dict[str, Any]]:
    """
    Build procedure events.

    Timestamp rule:
      - Use chartdate with default time 00:00:00 (date-level precision).
    """
    rows = df[df["hadm_id"] == hadm_id].copy()
    if rows.empty:
        return []

    events = []
    for r in rows.to_dict("records"):
        title = r.get("long_title")
        if title is None or pd.isna(title):
            continue

        chartdate = r.get("chartdate")
        ts = None
        if chartdate is not None and not pd.isna(chartdate):
            ts = pd.Timestamp(chartdate).strftime("%Y-%m-%d 00:00:00")

        events.append(
            {
                "timestamp": ts,
                "type": "procedure",
                "name": _nan_to_none(title),
            }
        )
    return events


# ============================================================
# Main
# ============================================================
def main():
    patient_files = load_patient_files()
    hadm_ids = collect_hadm_ids(patient_files)

    lab_df = load_lab(hadm_ids)
    vital_df = load_vital(hadm_ids)
    # pres_df = load_prescriptions(hadm_ids)
    # emar_df = load_emar(hadm_ids)
    med_df = load_integrated_medications(hadm_ids)
    img_df = load_imaging(hadm_ids)
    proc_df = load_procedures(hadm_ids)

    for pf in patient_files:
        with open(pf, "r", encoding="utf-8") as f:
            patient = json.load(f)

        # Keep patient_id aligned with the file naming convention: Pxxxx.json -> "Pxxxx"
        patient_id = patient.get("patient_id") or pf.stem
        patient["patient_id"] = patient_id

        visits = patient.get("visits", [])
        assign_visit_ids(visits)

        for visit in visits:
            hadm_id = int(visit["hadm_id"])
            events: List[Dict[str, Any]] = []

            # build event stream
            events.extend(build_lab_events(lab_df, hadm_id))
            events.extend(build_vital_events(vital_df, hadm_id))
            # events.extend(build_pres_events(pres_df, hadm_id))
            # events.extend(build_emar_events(emar_df,hadm_id))
            events.extend(build_med_events(med_df, hadm_id))
            events.extend(build_imaging_events(img_df, hadm_id))
            events.extend(build_procedure_events(proc_df, hadm_id))

            # Drop events without timestamps and sort deterministically.
            events = [e for e in events if e.get("timestamp") is not None]
            events.sort(key=lambda x: (x["timestamp"], str(x.get("type", ""))))

            # Assign Pxxxx-Vy-E01 style ids after sorting.
            new_events = []
            for i, e in enumerate(events, start=1):
                eid = f"{patient_id}-{visit['visit_id']}-E{i}"
                new_events.append({"event_id": eid, **e})  # id first, then original fields

            visit["event_stream"] = new_events

        # Enforce strict JSON output (do not allow NaN values).
        with open(pf, "w", encoding="utf-8") as f:
            json.dump(patient, f, indent=2, ensure_ascii=False, allow_nan=False)

        print(f"Updated {pf.name}")

    print("STEP 2 COMPLETE: event_stream populated with visit_id and event_id.")


if __name__ == "__main__":
    main()
