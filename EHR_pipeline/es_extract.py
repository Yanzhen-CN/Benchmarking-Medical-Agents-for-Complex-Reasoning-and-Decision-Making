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

def _available_cols(path: Path) -> List[str]:
    """
    Read header only to get existing columns (robust across different extracts/raw files).
    """
    try:
        return pl.read_csv(str(path), n_rows=0).columns
    except Exception:
        # fallback: try lazy schema (may fail on some malformed files)
        return pl.scan_csv(str(path), has_header=True, ignore_errors=True).schema.keys()


def _scan_csv_safe(path: Path, columns: List[str]) -> pl.LazyFrame:
    """
    scan_csv but only selects columns that actually exist in the file.
    """
    avail = set(_available_cols(path))
    use_cols = [c for c in columns if c in avail]
    if not use_cols:
        raise ValueError(f"No requested columns exist in {path}")
    return pl.scan_csv(str(path), has_header=True, ignore_errors=True).select(use_cols)


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
    """
    Unified Loader:
    1. Hosp (ICD): 只有 chartdate -> 设为 charttime (00:00:00), endtime = None
    2. ICU (Events): 有 starttime -> 设为 charttime, endtime = 真实结束时间
    """
    logger = get_logger()
    logger.info("[Loader-pl] Loading Procedures (ICD + ICU)...")

    cohort_list = list(cohort_ids)

    # ================= 1. 加载 HOSP ICD (模糊时间) =================
    fpath_icd = _find_data_file("procedures_icd", "hosp")
    cols_icd = ["hadm_id", "icd_code", "icd_version", "chartdate"]
    
    lf_icd = (
        _scan_csv(fpath_icd, cols_icd)
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            # chartdate 转为 datetime (默认 00:00:00)
            pl.col("chartdate").cast(pl.Utf8, strict=False).str.to_date(strict=False).cast(pl.Datetime),
            pl.col("icd_code").cast(pl.Utf8, strict=False),
            pl.col("icd_version").cast(pl.Int64, strict=False),
        ])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    # Join ICD 字典获取名称
    d_path_icd = _find_data_file("d_icd_procedures", "hosp")
    lf_d_icd = _scan_csv(d_path_icd, ["icd_code", "icd_version", "long_title"])
    
    # 整理 ICD 结构
    lf_icd_final = (
        lf_icd.join(lf_d_icd, on=["icd_code", "icd_version"], how="left")
        .select([
            pl.col("hadm_id"),
            pl.col("chartdate").alias("charttime"), # 统一时间列
            pl.lit(None, dtype=pl.Datetime).alias("endtime"), # ICD 没有结束时间 -> NaN
            pl.col("long_title").alias("name"),     # 统一名称列
            pl.lit(1).alias("is_fuzzy")             # 标记：1代表是模糊时间
        ])
    )

    # ================= 2. 加载 ICU Events (精确时间) =================
    fpath_icu = _find_data_file("procedureevents", "icu")
    cols_icu = ["hadm_id", "itemid", "starttime", "endtime", "statusdescription"]
    
    lf_icu = (
        _scan_csv(fpath_icu, cols_icu)
        .filter(
            pl.col("hadm_id").is_in(cohort_list) & 
            (pl.col("statusdescription") != "Rewritten")
        )
        .with_columns([
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("starttime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("endtime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False),
            pl.col("itemid").cast(pl.Int64, strict=False),
        ])
    )

    # Join ICU 字典获取名称
    d_path_icu = _find_data_file("d_items", "icu")
    lf_d_icu = _scan_csv(d_path_icu, ["itemid", "label"]).with_columns(pl.col("itemid").cast(pl.Int64))

    # 整理 ICU 结构
    lf_icu_final = (
        lf_icu.join(lf_d_icu, on="itemid", how="left")
        .select([
            pl.col("hadm_id"),
            pl.col("starttime").alias("charttime"), # 统一时间列
            pl.col("endtime"),                      # 保留结束时间
            pl.col("label").alias("name"),          # 统一名称列
            pl.lit(0).alias("is_fuzzy")             # 标记：0代表精确时间
        ])
    )

    # ================= 3. 合并返回 =================
    # 使用 diagonal concat，自动对齐列名
    out_lf = pl.concat([lf_icd_final, lf_icu_final], how="diagonal")
    out = _to_pandas(out_lf)

    logger.info(f"[Loader-pl] Proc loaded (ICD+ICU): rows={len(out)}")
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

def load_micro_data(cohort_ids: Set[int]) -> pd.DataFrame:
    """
    Microbiology loader (Polars -> pandas).

    - Reads hosp/microbiologyevents.csv (or microbiologyevents_extract.csv if exists via _find_data_file)
    - Robust datetime parsing:
        charttime: try "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", date-only "%Y-%m-%d"
        chartdate: same
      event_time = coalesce(charttime_dt, chartdate_dt)
    - Filters by cohort hadm_id
    - Returns pandas DF with unified 'event_time' column (datetime64[ns])
    """
    logger = get_logger()
    logger.info("[Loader-pl] Loading Microbiology...")

    cohort_list = list(cohort_ids)

    # prefer *_extract.csv if present, else hosp/microbiologyevents.csv
    fpath = _find_data_file("microbiologyevents", "hosp")

    # columns we care about (will be intersected with existing)
    cols = [
        "hadm_id",
        "charttime",
        "chartdate",
        "micro_specimen_id",
        "spec_type_desc",
        "test_seq",
        "test_name",
        "org_name",
        "isolate_num",
        "ab_name",
        "interpretation",
        "dilution_text",
        "dilution_comparison",
        "dilution_value",
        "comments",
    ]

    # scan and select only existing columns
    lf0 = pl.scan_csv(str(fpath), has_header=True, ignore_errors=True)
    existing = set(lf0.collect_schema().names())
    use_cols = [c for c in cols if c in existing]
    if not use_cols:
        logger.warning(f"[Loader-pl] Microbiology: no usable columns in {fpath.name}")
        return pd.DataFrame()

    micro_lf = lf0.select(use_cols).with_columns(
        pl.col("hadm_id").cast(pl.Int64, strict=False)
    )

    def _parse_mimic_dt(col: pl.Expr) -> pl.Expr:
        """
        Robust datetime parser for MIMIC-ish strings.
        Returns pl.Datetime or null (never throws).
        """
        s = (
            col.cast(pl.Utf8, strict=False)
             .str.strip_chars()
             .replace("", None)
             .replace("nan", None)
             .replace("None", None)
        )

        p1 = s.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        p2 = s.str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
        p3 = s.str.strptime(pl.Date, format="%Y-%m-%d", strict=False).cast(pl.Datetime, strict=False)

        return pl.coalesce([p1, p2, p3])

    # add parsed datetime columns safely (even if charttime/chartdate missing)
    if "charttime" in use_cols:
        micro_lf = micro_lf.with_columns(
            _parse_mimic_dt(pl.col("charttime")).alias("charttime_dt")
        )
    else:
        micro_lf = micro_lf.with_columns(
            pl.lit(None, dtype=pl.Datetime).alias("charttime_dt")
        )

    if "chartdate" in use_cols:
        micro_lf = micro_lf.with_columns(
            _parse_mimic_dt(pl.col("chartdate")).alias("chartdate_dt")
        )
    else:
        micro_lf = micro_lf.with_columns(
            pl.lit(None, dtype=pl.Datetime).alias("chartdate_dt")
        )

    micro_lf = (
        micro_lf.with_columns([
            pl.coalesce([pl.col("charttime_dt"), pl.col("chartdate_dt")]).alias("event_time"),
        ])
        .drop(["charttime_dt", "chartdate_dt"])
        .filter(pl.col("hadm_id").is_in(cohort_list))
    )

    out = _to_pandas(micro_lf)

    # pandas side: ensure datetime dtype
    if "event_time" in out.columns:
        out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")

    logger.info(f"[Loader-pl] Microbiology loaded: rows={len(out)} from {fpath.name}")
    return out


# ============================================================
# Event Builders
# ============================================================

def build_lab_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    df = df.dropna(subset=["charttime"]).copy()
    events: List[Dict[str, Any]] = []

    for ts, group in df.groupby("charttime"):
        items = []
        for r in group.to_dict("records"):
            items.append({
                "name": _nan_to_none(r.get("label")),
                "category": _nan_to_none(r.get("category")),
                "fluid": _nan_to_none(r.get("fluid")),
                "value_num": _nan_to_none(r.get("valuenum")),
                "value_text": _nan_to_none(r.get("value")),
                "unit": _nan_to_none(r.get("valueuom")),
                "flag": _nan_to_none(r.get("flag"))
            })
        
        events.append({
            "type": "lab",
            "timestamp": _fmt_ts(ts),
            "items": items
        })
    return events


def build_vital_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    
    df = df.dropna(subset=["charttime"]).copy()
    events: List[Dict[str, Any]] = []

    for ts, group in df.groupby("charttime"):
        items = []
        any_warning = False
        
        for r in group.to_dict("records"):
            warn_val = r.get("warning")
            is_warning = (str(warn_val) == "1") or (warn_val == 1)
            if is_warning:
                any_warning = True
                
            items.append({
                "name": _nan_to_none(r.get("label")),
                "value_num": _nan_to_none(r.get("valuenum")),
                "value_text": _nan_to_none(r.get("value")),
                "unit": _nan_to_none(r.get("unitname")),
                "flag": "warning" if is_warning else None
            })
        
        events.append({
            "type": "vital",
            "timestamp": _fmt_ts(ts),
            "flag": "warning" if any_warning else None,
            "items": items
        })
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

    # 移除没有时间的数据
    df = df.dropna(subset=["charttime"]).copy()
    events: List[Dict[str, Any]] = []

    # 按 charttime 和 is_fuzzy 分组
    # 这样可以区分“某天发生的ICD手术”和“某天00:00:00发生的ICU操作”
    for (ts, is_fuzzy), group in df.groupby(["charttime", "is_fuzzy"]):
        
        items: List[Dict[str, Any]] = []
        
        for r in group.to_dict("records"):
            # 极简 Item 结构
            item_dict = {
                "name": _nan_to_none(r.get("name")),
                "has_fuzzy_timestamp": is_fuzzy,
            }
            
            
            if pd.notna(r.get("endtime")):
                item_dict["end_timestamp"] = _fmt_ts(r.get("endtime"))
            items.append(item_dict)


        if is_fuzzy == 1:
            # ICD: 只显示日期 YYYY-MM-DD
            ts_str = ts.strftime("%Y-%m-%d")
        else:
            # ICU: 显示精确时间 YYYY-MM-DD HH:MM:SS
            ts_str = _fmt_ts(ts)

        events.append({
            "type": "PROCEDURE",
            "timestamp": ts_str,
            "items": items
        })
            
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
def build_micro_events(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Build microbiology events from pandas slice (one hadm_id).
    Uses event_time (charttime fallback chartdate) as timestamp.
    Groups by (micro_specimen_id, test_seq) if present else event_time.
    """
    if df is None or df.empty:
        return []

    if "event_time" not in df.columns:
        return []

    df = df.dropna(subset=["event_time"]).copy()
    if df.empty:
        return []

    # choose grouping columns for "one specimen/test"
    group_cols: List[str] = []
    if "micro_specimen_id" in df.columns:
        group_cols.append("micro_specimen_id")
    if "test_seq" in df.columns:
        group_cols.append("test_seq")
    if not group_cols:
        group_cols = ["event_time"]

    df = df.sort_values(by=["event_time"], na_position="last")

    events: List[Dict[str, Any]] = []

    for _, g in df.groupby(group_cols, dropna=False):
        g_sorted = g.sort_values(by=["event_time"], na_position="last")
        first = g_sorted.iloc[0].to_dict()

        ts_str = _fmt_ts(first.get("event_time"))
        if ts_str is None:
            continue

        specimen_id = _nan_to_none(first.get("micro_specimen_id"))
        spec_type = _nan_to_none(first.get("spec_type_desc"))
        test_name = _nan_to_none(first.get("test_name"))
        test_seq = _nan_to_none(first.get("test_seq"))

        # organism existence: org_name non-null AND not a known negative token
        has_org = False
        if "org_name" in g_sorted.columns:
            org_series = g_sorted["org_name"].dropna()
            if not org_series.empty:
                # treat "NEGATIVE"/"NO GROWTH" etc. as negative signals (optional but safer)
                neg_tokens = {"NEGATIVE", "NO GROWTH", "NONE", "N/A"}
                cleaned = org_series.astype(str).str.strip()
                has_org = (~cleaned.str.upper().isin(neg_tokens)).any()

        organisms: List[Dict[str, Any]] = []

        if has_org:
            org_group_cols = ["org_name"]
            if "isolate_num" in g_sorted.columns:
                org_group_cols.append("isolate_num")

            for org_key, og in g_sorted.groupby(org_group_cols, dropna=False):
                # normalize key to tuple
                if not isinstance(org_key, tuple):
                    org_key = (org_key,)

                org_name = org_key[0]
                isolate_num = org_key[1] if (len(org_key) > 1) else None

                # skip negative tokens again (in case mixed)
                if org_name is None or (isinstance(org_name, str) and org_name.strip().upper() in {"NEGATIVE", "NO GROWTH", "NONE", "N/A"}):
                    continue

                antibiotics: List[Dict[str, Any]] = []
                if "ab_name" in og.columns:
                    for _, r in og.iterrows():
                        ab = r.get("ab_name")
                        if ab is None or pd.isna(ab):
                            continue
                        antibiotics.append(
                            {
                                "ab_name": _nan_to_none(ab),
                                "interpretation": _nan_to_none(r.get("interpretation")),
                                "dilution_text": _nan_to_none(r.get("dilution_text")),
                                "dilution_comparison": _nan_to_none(r.get("dilution_comparison")),
                                "dilution_value": _nan_to_none(r.get("dilution_value")),
                            }
                        )

                organisms.append(
                    {
                        "org_name": _nan_to_none(org_name),
                        "isolate_num": _nan_to_none(isolate_num),
                        "antibiotics": antibiotics,
                    }
                )

        # collect comments
        comments_out: Optional[List[str]] = None
        if "comments" in g_sorted.columns:
            cs: List[str] = []
            for c in g_sorted["comments"].dropna().unique().tolist():
                if isinstance(c, str):
                    t = c.strip()
                    if t and t != "___":
                        cs.append(t)
            if cs:
                comments_out = cs

        events.append(
            {
                "type": "microbiology",
                "timestamp": ts_str,
                "specimen": {
                    "specimen_id": specimen_id,
                    "spec_type": spec_type,
                    "test_name": test_name,
                    "test_seq": test_seq,
                },
                "results": {
                    "negative": (len(organisms) == 0),
                    "organisms": organisms,
                    "comments": comments_out,
                },
            }
        )

    events.sort(key=lambda x: (x["timestamp"], x["type"]))
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
            events.extend(build_micro_events(data_maps["micro"].get(hid))) 

            events = [e for e in events if e.get("timestamp")]
            events.sort(key=lambda x: (x["timestamp"], x["type"]))

            final_stream: List[Dict[str, Any]] = []
            for idx, evt in enumerate(events, 1):
                eid = f"{visit['visit_id']}-E{idx}"
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
            "micro": io_pool.submit(load_micro_data, cohort_ids),
        }
        try:
            raw_lab = futures["lab"].result()
            raw_vital = futures["vital"].result()
            raw_med = futures["med"].result()
            raw_proc = futures["proc"].result()
            raw_img = futures["img"].result()
            raw_micro = futures["micro"].result()
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
        "micro": df_to_map(raw_micro),
    }

    # free big dfs
    del raw_lab, raw_vital, raw_med, raw_proc, raw_img, raw_micro
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
    
    # for pf in patient_files:
    #     try:
    #         process_patient(pf, data_maps)
    #         success_count += 1
    #     except Exception:
    #         logger.exception(f"Unhandled exception processing {pf.name}")

    logger.info(f">>> Done. Success: {success_count}/{len(patient_files)}")


if __name__ == "__main__":
    event_stream_extract()
