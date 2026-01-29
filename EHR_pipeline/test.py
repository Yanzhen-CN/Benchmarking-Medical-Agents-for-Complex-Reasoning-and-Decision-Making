import json
import os
import gc
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# ============================================================
# Configuration & Constants
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

DEMO_MODE = True
DEMO_N = 10
MAX_WORKERS = min(8, os.cpu_count() or 1) 

# ============================================================
# Utility Functions
# ============================================================

def _fmt_ts(x: Any) -> Optional[str]:
    """Normalize datetime to string format."""
    if x is None or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")

def _nan_to_none(x: Any) -> Any:
    """Convert NaN to None for valid JSON serialization."""
    if pd.isna(x):
        return None
    return x

def _find_data_file(filename_stem: str, raw_subfolder: str) -> Path:
    """
    Smart path finder:
    1. Checks for {stem}_extract.csv (Optimized file)
    2. Checks for {raw_subfolder}/{stem}.csv (Original MIMIC file)
    """
    # Priority 1: Extracted file in root or raw_data
    extract_name = f"{filename_stem}_extract.csv"
    if (RAW_DATA_DIR / extract_name).exists():
        return RAW_DATA_DIR / extract_name
    
    # Priority 2: Raw file in subfolder (hosp, icu, note)
    raw_name = f"{filename_stem}.csv"
    raw_path = RAW_DATA_DIR / raw_subfolder / raw_name
    if raw_path.exists():
        return raw_path
        
    raise FileNotFoundError(f"Could not find {extract_name} OR {raw_subfolder}/{raw_name}")

def load_patient_files() -> List[Path]:
    """Load patient JSON files."""
    patients_dir = BENCH_DATA_DIR / "patients"
    files = sorted(patients_dir.glob("P*.json"))
    if DEMO_MODE:
        return files[:DEMO_N]
    return files

def get_cohort_hadm_ids(patient_files: List[Path]) -> Set[int]:
    """Scan patient files to get relevant hadm_ids."""
    hadm_ids = set()
    for pf in patient_files:
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for v in data.get("visits", []):
                if v.get("hadm_id"):
                    hadm_ids.add(int(v["hadm_id"]))
        except Exception:
            continue
    return hadm_ids

def df_to_map(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group giant DataFrame by hadm_id into a dictionary for O(1) access."""
    if df is None or df.empty:
        return {}
    return {hid: group for hid, group in df.groupby("hadm_id")}

# ============================================================
# Data Loaders (With Fallback Logic)
# ============================================================

def load_lab_data(cohort_ids: Set[int]) -> pd.DataFrame:
    print("  [Loader] Loading Lab Events...")
    # Logic: Labs are usually extracted, but we check just in case
    fpath = _find_data_file("labevents", "hosp")
    
    df = pd.read_csv(
        fpath,
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "value", "valueuom", "flag"],
        parse_dates=["charttime"],
        low_memory=False
    )
    df = df[df["hadm_id"].isin(cohort_ids)].copy()
    
    # Load definitions
    d_path = _find_data_file("d_labitems", "hosp")
    d_lab = pd.read_csv(d_path, usecols=["itemid", "label", "category", "fluid"])
    
    return df.merge(d_lab, on="itemid", how="left")

def load_vital_data(cohort_ids: Set[int]) -> pd.DataFrame:
    print("  [Loader] Loading Vital Signs...")
    fpath = _find_data_file("chartevents", "icu")
    
    # Note: Raw chartevents is huge. Ensure usecols is strict.
    df = pd.read_csv(
        fpath,
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "value", "warning"],
        parse_dates=["charttime"],
        low_memory=False
    )
    df = df[df["hadm_id"].isin(cohort_ids)].copy()
    
    d_path = _find_data_file("d_items", "icu")
    d_items = pd.read_csv(d_path, usecols=["itemid", "label", "unitname"])
    
    return df.merge(d_items, on="itemid", how="left")

def load_med_data(cohort_ids: Set[int]) -> pd.DataFrame:
    print("  [Loader] Loading Medications...")
    
    # 1. Prescriptions (Try extract, then raw hosp/prescriptions.csv)
    pres_path = _find_data_file("prescriptions", "hosp")
    pres = pd.read_csv(
        pres_path,
        usecols=["hadm_id", "starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route", "poe_id"],
        parse_dates=["starttime", "stoptime"],
        low_memory=False
    )
    pres = pres[pres["hadm_id"].isin(cohort_ids)].copy()
    
    # 2. eMAR (Try extract, then raw hosp/emar.csv)
    # Note: Not all datasets use eMAR, so we wrap in try/except or simple check
    try:
        emar_path = _find_data_file("emar", "hosp")
        emar = pd.read_csv(emar_path, usecols=["poe_id", "event_txt"], low_memory=False)
        
        # Aggregate status
        emar["event_txt"] = emar["event_txt"].astype(str)
        emar_agg = emar.groupby("poe_id")["event_txt"].apply(lambda x: ", ".join(sorted(set(x)))).reset_index()
        emar_agg.rename(columns={"event_txt": "status"}, inplace=True)
        
        # Merge
        pres["poe_id"] = pres["poe_id"].astype(str)
        emar_agg["poe_id"] = emar_agg["poe_id"].astype(str)
        merged = pres.merge(emar_agg, on="poe_id", how="left")
        merged["status"] = merged["status"].fillna("Ordered")
        return merged
        
    except FileNotFoundError:
        print("  [Warning] eMAR file not found. Skipping eMAR merge.")
        pres["status"] = "Ordered"
        return pres

def load_proc_data(cohort_ids: Set[int]) -> pd.DataFrame:
    print("  [Loader] Loading Procedures...")
    # Likely raw: hosp/procedures_icd.csv
    fpath = _find_data_file("procedures_icd", "hosp")
    
    df = pd.read_csv(
        fpath,
        usecols=["hadm_id", "icd_code", "icd_version", "chartdate"],
        parse_dates=["chartdate"],
        low_memory=False
    )
    df = df[df["hadm_id"].isin(cohort_ids)].copy()
    
    d_path = _find_data_file("d_icd_procedures", "hosp")
    d_proc = pd.read_csv(d_path, usecols=["icd_code", "icd_version", "long_title"])
    
    return df.merge(d_proc, on=["icd_code", "icd_version"], how="left")

def load_img_data(cohort_ids: Set[int]) -> pd.DataFrame:
    print("  [Loader] Loading Radiology...")
    # Likely raw: note/radiology.csv
    fpath = _find_data_file("radiology", "note")
    
    df = pd.read_csv(
        fpath,
        usecols=["hadm_id", "charttime", "text"],
        parse_dates=["charttime"],
        low_memory=False
    )
    return df[df["hadm_id"].isin(cohort_ids)].copy()

# ============================================================
# Event Builders
# ============================================================

def build_lab_events(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty: return []
    events = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("charttime")): continue
        events.append({
            "type": "LAB",
            "timestamp": _fmt_ts(row["charttime"]),
            "name": _nan_to_none(row.get("label")),
            "value": _nan_to_none(row.get("valuenum")),
            "value_text": _nan_to_none(row.get("value")),
            "unit": _nan_to_none(row.get("valueuom")),
            "flag": _nan_to_none(row.get("flag")),
            "category": _nan_to_none(row.get("category"))
        })
    return events

def build_vital_events(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty: return []
    events = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("charttime")): continue
        # Warning might be '1', 1, or True depending on source
        warn_val = row.get("warning")
        is_warning = (str(warn_val) == "1") or (warn_val == 1)
        
        events.append({
            "type": "VITAL",
            "timestamp": _fmt_ts(row["charttime"]),
            "name": _nan_to_none(row.get("label")),
            "value": _nan_to_none(row.get("valuenum")),
            "unit": _nan_to_none(row.get("unitname")),
            "warning": is_warning
        })
    return events

def build_med_events(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty: return []
    events = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("starttime")): continue
        
        dose_val = row.get("dose_val_rx")
        dose_unit = row.get("dose_unit_rx")
        dose_str = f"{dose_val} {dose_unit}" if pd.notna(dose_val) else None

        events.append({
            "type": "MEDICATION",
            "timestamp": _fmt_ts(row["starttime"]),
            "end_timestamp": _fmt_ts(row.get("stoptime")),
            "drug": _nan_to_none(row.get("drug")),
            "route": _nan_to_none(row.get("route")),
            "dose": dose_str,
            "status": row.get("status")
        })
    return events

def build_proc_events(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty: return []
    events = []
    for row in df.to_dict("records"):
        ts = row.get("chartdate")
        if pd.isna(ts): continue
        ts_str = pd.Timestamp(ts).strftime("%Y-%m-%d 00:00:00")
        
        events.append({
            "type": "PROCEDURE",
            "timestamp": ts_str,
            "name": _nan_to_none(row.get("long_title")),
            "code": _nan_to_none(row.get("icd_code"))
        })
    return events

def build_img_events(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty: return []
    events = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("charttime")): continue
        events.append({
            "type": "IMAGING",
            "timestamp": _fmt_ts(row["charttime"]),
            "report": _nan_to_none(row.get("text"))
        })
    return events

# ============================================================
# Worker Process
# ============================================================

def process_patient(file_path: Path, data_maps: Dict[str, Dict]) -> str:
    """Worker function: Process one patient using shared data maps."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            patient = json.load(f)
        
        pid = patient.get("patient_id", file_path.stem)
        
        visits = patient.get("visits", [])
        for i, v in enumerate(visits, 1):
            if "visit_id" not in v:
                v["visit_id"] = f"V{i}"
        
        for visit in visits:
            hid = int(visit.get("hadm_id", -1))
            if hid == -1: continue
            
            # O(1) Data Retrieval
            events = []
            events.extend(build_lab_events(data_maps["lab"].get(hid)))
            events.extend(build_vital_events(data_maps["vital"].get(hid)))
            events.extend(build_med_events(data_maps["med"].get(hid)))
            events.extend(build_proc_events(data_maps["proc"].get(hid)))
            events.extend(build_img_events(data_maps["img"].get(hid)))
            
            # Sort & ID
            events = [e for e in events if e.get("timestamp")]
            events.sort(key=lambda x: (x["timestamp"], x["type"]))
            
            final_stream = []
            for idx, evt in enumerate(events, 1):
                pad = max(2, len(str(idx)))
                eid = f"{pid}-{visit['visit_id']}-E{idx:0{pad}d}"
                evt_with_id = {"event_id": eid}
                evt_with_id.update(evt)
                final_stream.append(evt_with_id)
            
            visit["event_stream"] = final_stream
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(patient, f, indent=2, ensure_ascii=False)
            
        return f"Success: {pid}"

    except Exception as e:
        return f"Error: {file_path.name} - {str(e)}"

# ============================================================
# Main Execution
# ============================================================

def main():
    print(">>> Starting Robust Event Extraction")
    
    # 1. Patients
    patient_files = load_patient_files()
    cohort_ids = get_cohort_hadm_ids(patient_files)
    print(f">>> Found {len(patient_files)} patients covering {len(cohort_ids)} admissions.")
    
    if not cohort_ids:
        print("No admissions found. Exiting.")
        return

    # 2. Parallel Loading (Handles Raw vs Extracted automatically)
    print(">>> Step 1: Loading Data (Parallel IO)...")
    with ThreadPoolExecutor(max_workers=5) as io_pool:
        f_lab = io_pool.submit(load_lab_data, cohort_ids)
        f_vital = io_pool.submit(load_vital_data, cohort_ids)
        f_med = io_pool.submit(load_med_data, cohort_ids)
        f_proc = io_pool.submit(load_proc_data, cohort_ids)
        f_img = io_pool.submit(load_img_data, cohort_ids)
        
        try:
            raw_lab = f_lab.result()
            raw_vital = f_vital.result()
            raw_med = f_med.result()
            raw_proc = f_proc.result()
            raw_img = f_img.result()
        except Exception as e:
            print(f"!!! Critical Loader Error: {e}")
            return

    # 3. Indexing
    print(">>> Step 2: Indexing Data Maps...")
    data_maps = {
        "lab": df_to_map(raw_lab),
        "vital": df_to_map(raw_vital),
        "med": df_to_map(raw_med),
        "proc": df_to_map(raw_proc),
        "img": df_to_map(raw_img)
    }
    
    del raw_lab, raw_vital, raw_med, raw_proc, raw_img
    gc.collect()
    
    # 4. Processing
    print(f">>> Step 3: Processing with {MAX_WORKERS} workers...")
    worker_fn = partial(process_patient, data_maps=data_maps)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(worker_fn, patient_files))
        
    success_count = sum(1 for r in results if r.startswith("Success"))
    print(f"\n>>> Done. Success: {success_count}/{len(results)}")

if __name__ == "__main__":
    main()