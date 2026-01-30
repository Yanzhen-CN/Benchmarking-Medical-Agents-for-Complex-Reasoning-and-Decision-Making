# -*- coding: utf-8 -*-
"""
es_extract.py (Polars + ThreadPoolExecutor)

Changes vs your original:
- Use Polars (pl) for ALL CSV loading + filtering + joins + eMAR aggregation (faster, lower memory).
- Use ThreadPoolExecutor for BOTH:
  1) parallel data loading
  2) per-patient processing
  (avoids ProcessPool pickling/copying huge data_maps)

Output stays the same:
- Each patient JSON gets visit["event_stream"] list with event_id, type, timestamp, etc.
- Builders still operate on pandas DataFrame slices (we convert polars->pandas at loader boundary).

Requirements:
- pip/conda install polars
"""

import json
import os
import gc
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import polars as pl

# Allow running from repo root or EHR_pipeline directory.
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import BuildConfig
from util.logUtil import setup_logger, get_logger

config = BuildConfig()

# ============================================================
# Configuration & Constants
# ============================================================
RAW_DATA_DIR = config.paths.RAW_DATA_DIR
BENCH_DATA_DIR = config.paths.BENCH_DATA_DIR

DEMO_MODE = config.run.DEMO_MODE
DEMO_N = config.run.DEMO_N
MAX_WORKERS = min(config.run.MAX_WORKERS, os.cpu_count() or 1)

# ============================================================
# Utility Functions
# ============================================================

def _fmt_ts(x: Any) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def _nan_to_none(x: Any) -> Any:
    try:
        if pd.isna(x):
            return None
    except Exception:
        return x
    return x


def _find_data_file(filename_stem: str, raw_subfolder: str) -> Path:
    extract_name = f"{filename_stem}_extract.csv"
    extract_path = RAW_DATA_DIR / extract_name
    if extract_path.exists():
        return extract_path

    raw_name = f"{filename_stem}.csv"
    raw_path = RAW_DATA_DIR / raw_subfolder / raw_name
    if raw_path.exists():
        return raw_path

    raise FileNotFoundError(f"Could not find {extract_name} OR {raw_subfolder}/{raw_name}")


def load_patient_files() -> List[Path]:
    patients_dir = BENCH_DATA_DIR / "patients"
    files = sorted(patients_dir.glob("P*.json"))
    return files[:DEMO_N] if DEMO_MODE else files


def get_cohort_hadm_ids(patient_files: List[Path]) -> Set[int]:
    hadm_ids: Set[int] = set()
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
    if df is None or df.empty:
        return {}
    return {int(hid): group for hid, group in df.groupby("hadm_id")}


# ---------------- Polars helpers ----------------

def _scan_csv(path: Path, columns: List[str]) -> pl.LazyFrame:
    """
    Use scan_csv (lazy) and only select the columns we need.
    ignore_errors=True helps tolerate rare dirty rows.
    """
    return pl.scan_csv(
        str(path),
        has_header=True,
        ignore_errors=True,
    ).select([c for c in columns])


def _to_pandas(lf: pl.LazyFrame) -> pd.DataFrame:
    """
    Collect in streaming mode if possible, then convert to pandas.
    """
    return lf.collect(streaming=True).to_pandas()


def _cast_hadm(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.col("hadm_id").cast(pl.Int64, strict=False))


# ============================================================
# Data Loaders (Polars)
# ============================================================

def load_lab_data(cohort_ids: Set[int]) -> pd.DataFrame:
    logger = get_logger()
    logger.info("[Loader-pl] Loading Lab Events...")

    cohort_list = list(cohort_ids)

    fpath = _find_data_file("labevents", "hosp")
    cols = ["hadm_id", "itemid", "charttime", "valuenum", "value", "valueuom", "flag"]
    lab_lf = (
        _scan_csv(fpath, cols)
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
            pl.col("charttime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("valuenum").cast(pl.Float64, strict=False),
        ])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    d_path = _find_data_file("d_labitems", "hosp")
    d_cols = ["itemid", "label", "category", "fluid"]
    d_lf = (
        _scan_csv(d_path, d_cols)
        .with_columns(pl.col("itemid").cast(pl.Int64, strict=False))
    )

    out_lf = lab_lf.join(d_lf, on="itemid", how="left")
    out = _to_pandas(out_lf)

    logger.info(f"[Loader-pl] Lab loaded: rows={len(out)} from {fpath.name}")
    return out


def load_vital_data(cohort_ids: Set[int]) -> pd.DataFrame:
    logger = get_logger()
    logger.info("[Loader-pl] Loading Vital Signs...")

    cohort_list = list(cohort_ids)

    fpath = _find_data_file("chartevents", "icu")
    cols = ["hadm_id", "itemid", "charttime", "valuenum", "value", "warning"]
    vit_lf = (
        _scan_csv(fpath, cols)
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
            pl.col("charttime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("valuenum").cast(pl.Float64, strict=False),
            # warning sometimes "1"/"0"/null; keep as-is (builder handles)
        ])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    d_path = _find_data_file("d_items", "icu")
    d_cols = ["itemid", "label", "unitname"]
    d_lf = (
        _scan_csv(d_path, d_cols)
        .with_columns(pl.col("itemid").cast(pl.Int64, strict=False))
    )

    out_lf = vit_lf.join(d_lf, on="itemid", how="left")
    out = _to_pandas(out_lf)

    logger.info(f"[Loader-pl] Vitals loaded: rows={len(out)} from {fpath.name}")
    return out


def load_med_data(cohort_ids: Set[int]) -> pd.DataFrame:
    logger = get_logger()
    logger.info("[Loader-pl] Loading Medications...")

    cohort_list = list(cohort_ids)

    pres_path = _find_data_file("prescriptions", "hosp")
    pres_cols = ["hadm_id", "starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route", "poe_id"]
    pres_lf = (
        _scan_csv(pres_path, pres_cols)
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("poe_id").cast(pl.Utf8, strict=False),
            pl.col("starttime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("stoptime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("dose_val_rx").cast(pl.Utf8, strict=False),  # keep string-ish (MIMIC has mixed)
            pl.col("dose_unit_rx").cast(pl.Utf8, strict=False),
            pl.col("drug").cast(pl.Utf8, strict=False),
            pl.col("route").cast(pl.Utf8, strict=False),
        ])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    # Optional eMAR merge
    try:
        emar_path = _find_data_file("emar", "hosp")
        emar_cols = ["poe_id", "event_txt"]
        emar_lf = (
            _scan_csv(emar_path, emar_cols)
            .with_columns([
                pl.col("poe_id").cast(pl.Utf8, strict=False),
                pl.col("event_txt").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars(),
            ])
            .filter((pl.col("event_txt") != "") & (pl.col("event_txt") != "nan") & (pl.col("event_txt") != "None"))
        )

        # group_by aggregate: unique + sort + concat into single status string
        emar_agg_lf = (
            emar_lf.group_by("poe_id")
            .agg(
                pl.col("event_txt").unique().sort().str.concat(", ").alias("status")
            )
        )

        merged_lf = (
            pres_lf.join(emar_agg_lf, on="poe_id", how="left")
            .with_columns(pl.col("status").fill_null("Ordered"))
        )

        merged = _to_pandas(merged_lf)
        logger.info(f"[Loader-pl] Med loaded: rows={len(merged)} from {pres_path.name} (+emar)")
        return merged

    except FileNotFoundError:
        pres = _to_pandas(pres_lf)
        pres["status"] = "Ordered"
        logger.warning("[Loader-pl] eMAR file not found. Skipping eMAR merge.")
        logger.info(f"[Loader-pl] Med loaded: rows={len(pres)} (no eMAR)")
        return pres


def load_proc_data(cohort_ids: Set[int]) -> pd.DataFrame:
    logger = get_logger()
    logger.info("[Loader-pl] Loading Procedures...")

    cohort_list = list(cohort_ids)

    fpath = _find_data_file("procedures_icd", "hosp")
    cols = ["hadm_id", "icd_code", "icd_version", "chartdate"]
    proc_lf = (
        _scan_csv(fpath, cols)
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("icd_code").cast(pl.Utf8, strict=False),
            pl.col("icd_version").cast(pl.Int64, strict=False),
            pl.col("chartdate").cast(pl.Utf8, strict=False).str.to_date(strict=False),
        ])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    d_path = _find_data_file("d_icd_procedures", "hosp")
    d_cols = ["icd_code", "icd_version", "long_title"]
    d_lf = (
        _scan_csv(d_path, d_cols)
        .with_columns([
            pl.col("icd_code").cast(pl.Utf8, strict=False),
            pl.col("icd_version").cast(pl.Int64, strict=False),
            pl.col("long_title").cast(pl.Utf8, strict=False),
        ])
    )

    out_lf = proc_lf.join(d_lf, on=["icd_code", "icd_version"], how="left")
    out = _to_pandas(out_lf)

    logger.info(f"[Loader-pl] Proc loaded: rows={len(out)} from {fpath.name}")
    return out


def load_img_data(cohort_ids: Set[int]) -> pd.DataFrame:
    logger = get_logger()
    logger.info("[Loader-pl] Loading Radiology...")

    cohort_list = list(cohort_ids)

    fpath = _find_data_file("radiology", "note")
    cols = ["hadm_id", "charttime", "text"]
    img_lf = (
        _scan_csv(fpath, cols)
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("charttime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("text").cast(pl.Utf8, strict=False),
        ])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    out = _to_pandas(img_lf)
    logger.info(f"[Loader-pl] Img loaded: rows={len(out)} from {fpath.name}")
    return out

# ============================================================
# Event Builders
# ============================================================

def build_lab_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    events: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("charttime")):
            continue
        events.append(
            {
                "type": "LAB",
                "timestamp": _fmt_ts(row["charttime"]),
                "name": _nan_to_none(row.get("label")),
                "value": _nan_to_none(row.get("valuenum")),
                "value_text": _nan_to_none(row.get("value")),
                "unit": _nan_to_none(row.get("valueuom")),
                "flag": _nan_to_none(row.get("flag")),
                "category": _nan_to_none(row.get("category")),
            }
        )
    return events


def build_vital_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    events: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("charttime")):
            continue
        warn_val = row.get("warning")
        is_warning = (str(warn_val) == "1") or (warn_val == 1)
        events.append(
            {
                "type": "VITAL",
                "timestamp": _fmt_ts(row["charttime"]),
                "name": _nan_to_none(row.get("label")),
                "value": _nan_to_none(row.get("valuenum")),
                "unit": _nan_to_none(row.get("unitname")),
                "warning": is_warning,
            }
        )
    return events


def build_med_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    df = df.dropna(subset=["starttime"]).copy()

    events: List[Dict[str, Any]] = []
    for start_ts, group in df.groupby("starttime"):
        items: List[Dict[str, Any]] = []
        for r in group.to_dict("records"):
            dose_val = r.get("dose_val_rx")
            dose_str = None
            if pd.notna(dose_val):
                dose_str = f"{dose_val} {r.get('dose_unit_rx', '')}".strip()

            items.append(
                {
                    "drug": _nan_to_none(r.get("drug")),
                    "route": _nan_to_none(r.get("route")),
                    "dose": _nan_to_none(dose_str),
                    "status": _nan_to_none(r.get("status")),
                    "end_timestamp": _fmt_ts(r.get("stoptime")),
                }
            )

        events.append(
            {
                "type": "MEDICATION",
                "timestamp": _fmt_ts(start_ts),
                "items": items,
            }
        )
    return events


def build_proc_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    events: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        ts = row.get("chartdate")
        if pd.isna(ts):
            continue
        ts_str = pd.Timestamp(ts).strftime("%Y-%m-%d 00:00:00")
        events.append(
            {
                "type": "PROCEDURE",
                "timestamp": ts_str,
                "name": _nan_to_none(row.get("long_title")),
                "code": _nan_to_none(row.get("icd_code")),
            }
        )
    return events


def build_img_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    events: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("charttime")):
            continue
        events.append(
            {
                "type": "IMAGING",
                "timestamp": _fmt_ts(row["charttime"]),
                "report": _nan_to_none(row.get("text")),
            }
        )
    return events

# ============================================================
# Worker
# ============================================================

def process_patient(file_path: Path, data_maps: Dict[str, Dict[int, pd.DataFrame]]) -> bool:
    logger = get_logger()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            patient = json.load(f)

        pid = patient.get("patient_id", file_path.stem)

        visits = patient.get("visits", [])
        for i, v in enumerate(visits, 1):
            if "visit_id" not in v:
                v["visit_id"] = f"V{i}"

        total_events_written = 0

        for visit in visits:
            hid = int(visit.get("hadm_id", -1))
            if hid == -1:
                continue

            events: List[Dict[str, Any]] = []
            events.extend(build_lab_events(data_maps["lab"].get(hid)))
            events.extend(build_vital_events(data_maps["vital"].get(hid)))
            events.extend(build_med_events(data_maps["med"].get(hid)))
            events.extend(build_proc_events(data_maps["proc"].get(hid)))
            events.extend(build_img_events(data_maps["img"].get(hid)))

            events = [e for e in events if e.get("timestamp")]
            events.sort(key=lambda x: (x["timestamp"], x["type"]))

            final_stream: List[Dict[str, Any]] = []
            for idx, evt in enumerate(events, 1):
                eid = f"{pid}-{visit['visit_id']}-E{idx}"
                evt_with_id = {"event_id": eid}
                evt_with_id.update(evt)
                final_stream.append(evt_with_id)

            visit["event_stream"] = final_stream
            total_events_written += len(final_stream)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(patient, f, indent=2, ensure_ascii=False)

        logger.info(f"Success: {pid} | visits={len(visits)} | events={total_events_written}")
        return True

    except Exception:
        logger.exception(f"Error processing {file_path.name}")
        return False

# ============================================================
# Main Execution
# ============================================================

def event_stream_extract():
    log_file = str(BENCH_DATA_DIR / "step2_event_stream.log")
    logger = setup_logger(level="INFO", log_file=log_file)

    logger.info(">>> Starting Robust Event Extraction (Polars + ThreadPool)")

    # 1) Patients
    patient_files = load_patient_files()
    cohort_ids = get_cohort_hadm_ids(patient_files)
    logger.info(f">>> Found {len(patient_files)} patients covering {len(cohort_ids)} admissions.")

    if not cohort_ids:
        logger.warning("No admissions found. Exiting.")
        return

    # 2) Parallel Loading (ThreadPool for IO)
    logger.info(">>> Step 1: Loading Data (Parallel IO, Polars)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as io_pool:
        futures = {
            "lab": io_pool.submit(load_lab_data, cohort_ids),
            "vital": io_pool.submit(load_vital_data, cohort_ids),
            "med": io_pool.submit(load_med_data, cohort_ids),
            "proc": io_pool.submit(load_proc_data, cohort_ids),
            "img": io_pool.submit(load_img_data, cohort_ids),
        }
        try:
            raw_lab = futures["lab"].result()
            raw_vital = futures["vital"].result()
            raw_med = futures["med"].result()
            raw_proc = futures["proc"].result()
            raw_img = futures["img"].result()
        except Exception:
            logger.exception("!!! Critical Loader Error")
            return

    # 3) Indexing
    logger.info(">>> Step 2: Indexing Data Maps...")
    data_maps: Dict[str, Dict[int, pd.DataFrame]] = {
        "lab": df_to_map(raw_lab),
        "vital": df_to_map(raw_vital),
        "med": df_to_map(raw_med),
        "proc": df_to_map(raw_proc),
        "img": df_to_map(raw_img),
    }

    # free big dfs
    del raw_lab, raw_vital, raw_med, raw_proc, raw_img
    gc.collect()

    # 4) Processing (ThreadPool) - NO pickling of data_maps
    logger.info(f">>> Step 3: Processing patients with {MAX_WORKERS} threads...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(process_patient, pf, data_maps) for pf in patient_files]
        for fut in as_completed(futs):
            try:
                ok = fut.result()
                success_count += 1 if ok else 0
            except Exception:
                logger.exception("Unhandled exception in worker future")

    logger.info(f">>> Done. Success: {success_count}/{len(patient_files)}")


if __name__ == "__main__":
    event_stream_extract()
