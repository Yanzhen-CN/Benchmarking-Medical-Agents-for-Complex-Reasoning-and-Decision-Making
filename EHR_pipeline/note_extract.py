import os
import json
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import polars as pl
from tqdm import tqdm

# Allow running from repo root or EHR_pipeline directory.
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from EHR_pipeline.note_slicing import split_note_to_adm_discharge
from util.logUtil import setup_logger
from config import BuildConfig

logger = setup_logger()
config = BuildConfig()

PATIENT_DIR = Path(config.noteExtract.PATIENT_OUTPUT_PATH)
PATIENT_INDEXES = Path(config.noteExtract.PATIENT_INDEXES_PATH)
DISCHARGE_NOTE_PATH = Path(config.noteExtract.DISCHARGE_NOTES_PATH)


# ============================================================
# Polars discharge map builder (fast, avoids full sort)
# ============================================================
def build_discharge_map_polars(
    discharge_csv: Path,
    subject_ids: List[int],
) -> Dict[Tuple[int, int], str]:
    cols = pl.read_csv(str(discharge_csv), n_rows=1).columns
    has_charttime = "charttime" in cols

    use_cols = ["subject_id", "hadm_id", "text"] + (["charttime"] if has_charttime else [])

    lf = (
        pl.scan_csv(str(discharge_csv), ignore_errors=True)
        .select([c for c in use_cols if c in cols])
        .with_columns([
            pl.col("subject_id").cast(pl.Int64, strict=False),
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("text").cast(pl.Utf8, strict=False),
        ])
        .filter(
            pl.col("subject_id").is_in(subject_ids)
            & pl.col("subject_id").is_not_null()
            & pl.col("hadm_id").is_not_null()
            & pl.col("text").is_not_null()
            & (pl.col("text").str.len_chars() > 0)
        )
    )

    if has_charttime:
        lf = lf.with_columns(
            pl.col("charttime").cast(pl.Utf8, strict=False).str.to_datetime(strict=False)
        )
        # ✅ group 内按 charttime 排序取 last，避免全表 sort
        lf = (
            lf.group_by(["subject_id", "hadm_id"])
            .agg(pl.col("text").sort_by("charttime").last().alias("text"))
        )
    else:
        lf = (
            lf.group_by(["subject_id", "hadm_id"])
            .agg(pl.first("text").alias("text"))
        )

    df = lf.collect(streaming=True)

    out: Dict[Tuple[int, int], str] = {}
    for sid, hid, text in df.iter_rows():
        out[(int(sid), int(hid))] = text
    return out


# ============================================================
# Worker (thread-safe: only reads discharge_map + edits its own file)
# ============================================================
def process_one_patient_thread(
    subject_id: int,
    patient_id: str,
    discharge_map: Dict[Tuple[int, int], str],
) -> Tuple[str, int, int]:
    patient_file = PATIENT_DIR / f"{patient_id}.json"
    if not patient_file.exists():
        return f"{patient_id}.json", 0, 0

    with open(patient_file, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    try:
        file_subj = int(patient_data["patient_info"]["subject_id"])
    except Exception:
        return patient_file.name, 0, 0

    if file_subj != subject_id:
        return patient_file.name, 0, 0

    ok_visits = 0
    missing_visits = 0
    changed = False  # ✅ 没变化就不写，省 IO

    for v in patient_data.get("visits", []):
        hadm_raw = v.get("hadm_id")
        if hadm_raw is None:
            continue
        try:
            hadm_id = int(hadm_raw)
        except Exception:
            continue

        note_text = discharge_map.get((subject_id, hadm_id))

        if v.get("ground_truth_note") != note_text:
            v["ground_truth_note"] = note_text
            changed = True

        if not note_text:
            missing_visits += 1
            if v.get("admission_info", {}).get("admission_note") is not None:
                v.setdefault("admission_info", {})["admission_note"] = None
                changed = True
            if v.get("discharge_info", {}).get("discharge_note") is not None:
                v.setdefault("discharge_info", {})["discharge_note"] = None
                changed = True
            continue

        parsed = split_note_to_adm_discharge(note_text)
        adm = parsed.get("admission_info")
        dis = parsed.get("discharge_info")

        if v.get("admission_info", {}).get("admission_note") != adm:
            v.setdefault("admission_info", {})["admission_note"] = adm
            changed = True
        if v.get("discharge_info", {}).get("discharge_note") != dis:
            v.setdefault("discharge_info", {})["discharge_note"] = dis
            changed = True

        ok_visits += 1

    if changed:
        tmp_path = patient_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            # no indent for speed
            json.dump(patient_data, f, ensure_ascii=False, allow_nan=False,indent=2)
        tmp_path.replace(patient_file)

    return patient_file.name, ok_visits, missing_visits


# ============================================================
# Main
# ============================================================
def extract_notes():
    patient_indexes = pd.read_csv(PATIENT_INDEXES, usecols=["subject_id", "patient_id"])
    patient_indexes["subject_id"] = pd.to_numeric(patient_indexes["subject_id"], errors="coerce").astype("Int64")
    patient_indexes = patient_indexes.dropna(subset=["subject_id", "patient_id"])
    patient_indexes["subject_id"] = patient_indexes["subject_id"].astype(int)
    patient_indexes["patient_id"] = patient_indexes["patient_id"].astype(str).str.strip()

    subject_ids = patient_indexes["subject_id"].unique().tolist()
    logger.info(f"patient_indexes: {len(patient_indexes)} rows, unique subject_id={len(subject_ids)}")

    logger.info("Building discharge_map (polars scan_csv + group_by)...")
    discharge_map = build_discharge_map_polars(DISCHARGE_NOTE_PATH, subject_ids)
    logger.info(f"Discharge map size: {len(discharge_map)}")

    # ✅ threads: for IO heavy workloads it's great; for CPU heavy, modest threads is best
    total = len(patient_indexes)
    req_workers = int(getattr(getattr(config, "run", None), "NUM_WORKERS", 16) or 16)
    max_workers = max(1, min(req_workers, total, 32))  # cap 32 to avoid IO thrash
    logger.info(f"Concurrent processing: ThreadPoolExecutor max_workers={max_workers}")

    ok_patients = 0
    ok_visits = 0
    missing_visits = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for row in patient_indexes.itertuples(index=False):
            futs.append(
                ex.submit(process_one_patient_thread, int(row.subject_id), str(row.patient_id), discharge_map)
            )

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing patients (threads)"):
            try:
                fname, okv, miss = fut.result(timeout=300)
                ok_patients += 1
                ok_visits += okv
                missing_visits += miss
            except Exception as e:
                logger.error(f"Error processing patient: {e}")
    
    # for row in tqdm(patient_indexes.itertuples(index=False), total=len(patient_indexes), desc="Processing patients (threads)"):
    #     try:
    #         fname, okv, miss = process_one_patient_thread(int(row.subject_id), str(row.patient_id), discharge_map)
    #         ok_patients += 1
    #         ok_visits += okv
    #         missing_visits += miss
    #     except Exception as e:
    #         logger.error(f"Error processing patient {row.patient_id}: {e}")
    # logger.success("DONE")
    # logger.info(f"Patients processed: {ok_patients}")
    # logger.info(f"Visits updated (have note): {ok_visits}")
    # logger.info(f"Visits missing note: {missing_visits}")


if __name__ == "__main__":
    extract_notes()
