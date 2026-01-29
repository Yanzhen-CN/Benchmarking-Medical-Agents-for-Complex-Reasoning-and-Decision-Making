# STEP1: Extract patients and visits from MIMIC-IV
# - Build patient JSON files for benchmark
import json
from pathlib import Path

import pandas as pd
import polars as pl
from util.logUtil import setup_logger
from config import BuildConfig

logger = setup_logger()
config = BuildConfig()

# =========================
# Input files (MIMIC-IV)
# =========================
ADMISSIONS_FILE = config.patientExtract.ADMISSIONS_FILE
PATIENTS_FILE   = config.patientExtract.PATIENTS_FILE
DIAGNOSES_FILE  = config.patientExtract.DIAGNOSES_FILE
D_ICD_FILE      = config.patientExtract.D_ICD_FILE

# NOTE: 你需要在 config 里提供这个路径：note/discharge.csv
DISCHARGE_FILE  = config.patientExtract.DISCHARGE_NOTES_FILE

BENCH_DATA_DIR = config.paths.BENCH_DATA_DIR

# =========================
# Global configs
# =========================
MIN_VISITS    = config.run.MIN_VISITS
ANALYZE_MODE  = config.run.ANALYZE_MODE
DEMO_MODE     = config.run.DEMO_MODE
DEMO_N        = config.run.DEMO_N


# ============================================================
# Helpers: Full-coverage cohort by discharge notes
# ============================================================


def load_latest_discharge_notes(discharge_file: str | Path) -> pd.DataFrame:
    """
    读取 note/discharge.csv，并对同一 (subject_id, hadm_id) 多条记录取“最新”一条。
    最新规则：按 (charttime, storetime) 排序后取最后。
    """
    logger.info("Loading discharge notes table")
    discharge = pd.read_csv(
        discharge_file,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"],
        parse_dates=["charttime", "storetime"],
    )

    # 统一 dtype，避免 join 时类型不一致
    discharge["subject_id"] = pd.to_numeric(discharge["subject_id"], errors="coerce").astype("Int64")
    discharge["hadm_id"] = pd.to_numeric(discharge["hadm_id"], errors="coerce").astype("Int64")
    discharge = discharge.dropna(subset=["subject_id", "hadm_id"]).copy()
    discharge["subject_id"] = discharge["subject_id"].astype("int64")
    discharge["hadm_id"] = discharge["hadm_id"].astype("int64")

    discharge = discharge.sort_values(["subject_id", "hadm_id", "charttime", "storetime"])
    discharge_latest = discharge.drop_duplicates(subset=["subject_id", "hadm_id"], keep="last").reset_index(drop=True)
    logger.info(f"Discharge notes loaded: {len(discharge)} rows; latest-per-visit: {len(discharge_latest)} rows")
    return discharge_latest


def compute_full_coverage_subjects(
    admissions: pd.DataFrame,
    discharge_latest: pd.DataFrame,
    min_visits: int,
) -> tuple[pd.Index, pd.DataFrame, pd.DataFrame]:
    """
    返回：
      - eligible_subjects: 满足
            (1) admissions 的 visit 全部能在 discharge_latest 匹配
            (2) admissions visit 数 >= min_visits
      - coverage_df: 每个 subject 的 admissions_visits / matched_visits
      - missing_visits_df: admissions 中找不到 discharge note 的 visit 列表
    """
    # admissions key
    ad_keys = admissions[["subject_id", "hadm_id"]].drop_duplicates()
    dc_keys = discharge_latest[["subject_id", "hadm_id"]].drop_duplicates()

    matched_visits = ad_keys.merge(dc_keys, on=["subject_id", "hadm_id"], how="inner")

    ad_cnt = ad_keys.groupby("subject_id").size().rename("admissions_visits")
    m_cnt  = matched_visits.groupby("subject_id").size().rename("matched_visits")

    coverage_df = pd.concat([ad_cnt, m_cnt], axis=1).fillna(0)
    coverage_df["matched_visits"] = coverage_df["matched_visits"].astype(int)

    eligible_subjects = coverage_df[
        (coverage_df["admissions_visits"] >= min_visits) &
        (coverage_df["matched_visits"] == coverage_df["admissions_visits"])
    ].index

    # 缺失 visit（用于 debug / 导出）
    missing_visits_df = ad_keys.merge(dc_keys, on=["subject_id", "hadm_id"], how="left", indicator=True)
    missing_visits_df = missing_visits_df[missing_visits_df["_merge"] == "left_only"][["subject_id", "hadm_id"]]

    return eligible_subjects, coverage_df, missing_visits_df


# ============================================================
# Analyze-only: cohort 规模统计（不生成 JSON）
# ============================================================

def analyze_and_export_cohort(admissions: pd.DataFrame, discharge_latest: pd.DataFrame) -> None:
    """
    分析在不同 visit 阈值下，可用病人数量（subject 级别 full-coverage：所有 visit 都有出院 note）。
    输出目录: bench_data/cohort_analysis/
      - subject_index_all_full_coverage.csv
      - visit_threshold_stats_full_coverage.csv
      - subject_index_eligible_min_visits_full_coverage.csv
      - missing_visits_admissions_without_discharge_note.csv
    """
    out_dir = config.patientExtract.COHORT_ANALYSIS_OUT_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Analyzing cohort statistics with FULL-COVERAGE constraint")

    # 全覆盖 subject 列表（先用 min_visits=1 拿到“全覆盖全集”，再做阈值统计）
    eligible_subjects_all, coverage_df, missing_visits_df = compute_full_coverage_subjects(
        admissions=admissions,
        discharge_latest=discharge_latest,
        min_visits=1,
    )

    # 导出缺失 visits
    missing_visits_df.to_csv(
        out_dir / "missing_visits_admissions_without_discharge_note.csv",
        index=False
    )

    # 只保留 full-coverage admissions
    admissions_fc = admissions[admissions["subject_id"].isin(eligible_subjects_all)].copy()

    # subject -> hadm list
    visit_group = (
        admissions_fc
        .groupby("subject_id")["hadm_id"]
        .agg(list)
        .reset_index()
    )
    visit_group["n_visits"] = visit_group["hadm_id"].apply(len)
    visit_group = visit_group.sort_values("subject_id").reset_index(drop=True)

    # 1) 全覆盖全集索引
    subject_index_all = visit_group.copy()
    subject_index_all["visit_ids"] = subject_index_all["hadm_id"].apply(lambda xs: ";".join(str(x) for x in xs))
    subject_index_all = subject_index_all[["subject_id", "n_visits", "visit_ids"]]
    subject_index_all.to_csv(out_dir / "subject_index_all_full_coverage.csv", index=False)

    # 2) 不同 visit 阈值统计（在 full-coverage 子集内）
    total_patients = len(visit_group)
    max_visits = int(visit_group["n_visits"].max()) if total_patients else 0

    ks = list(range(1, min(max_visits, 20) + 1))
    for k in [MIN_VISITS, 25, 30, 40, 50, 75, 100]:
        if 1 <= k <= max_visits and k not in ks:
            ks.append(k)
    ks = sorted(ks)

    rows = []
    for k in ks:
        cnt = int((visit_group["n_visits"] >= k).sum())
        pct = (cnt / total_patients * 100) if total_patients else 0
        rows.append({"min_visits": k, "n_patients": cnt, "percent": round(pct, 4)})

    pd.DataFrame(rows).to_csv(out_dir / "visit_threshold_stats_full_coverage.csv", index=False)

    # 3) 导出：满足 MIN_VISITS 的 eligible cohort（full-coverage + min_visits）
    eligible = visit_group[visit_group["n_visits"] >= MIN_VISITS].copy()
    eligible.to_csv(out_dir / "subject_index_eligible_min_visits_full_coverage.csv", index=False)

    logger.success("ANALYZE_MODE completed (FULL-COVERAGE)")
    logger.info(f"Full-coverage subjects: {total_patients}")
    logger.info(f"Eligible subjects (full-coverage & >= {MIN_VISITS} visits): {len(eligible)}")


# ============================================================
# Main pipeline
# ============================================================

def extract_patient_data():
    logger.info("Loading admissions table")
    admissions = pd.read_csv(
        ADMISSIONS_FILE,
        parse_dates=["admittime", "dischtime"]
    )

    # 统一 dtype，避免 hadm_id/subject_id 被读成 float
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce").astype("Int64")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id"]).copy()
    admissions["subject_id"] = admissions["subject_id"].astype("int64")
    admissions["hadm_id"] = admissions["hadm_id"].astype("int64")

    # 读取并规整 discharge notes（最新一条）
    discharge_latest = load_latest_discharge_notes(DISCHARGE_FILE)

    # Analyze-only path
    if ANALYZE_MODE:
        analyze_and_export_cohort(admissions, discharge_latest)
        return

    # --------------------------------------------------------
    # FULL-COVERAGE + MIN_VISITS cohort
    # --------------------------------------------------------
    eligible_subjects, coverage_df, missing_visits_df = compute_full_coverage_subjects(
        admissions=admissions,
        discharge_latest=discharge_latest,
        min_visits=MIN_VISITS,
    )

    logger.info(
        f"Eligible subjects (FULL-COVERAGE & >= {MIN_VISITS} visits): {len(eligible_subjects)}"
    )
    if len(missing_visits_df) > 0:
        logger.warning(
            f"Found {len(missing_visits_df)} visits missing discharge note overall "
            f"(subjects with any missing visit are excluded)."
        )

    # discharge note text map: (subject_id, hadm_id) -> text
    discharge_note_map = {
        (int(r.subject_id), int(r.hadm_id)): r.text
        for _, r in discharge_latest.iterrows()
    }

    # --------------------------------------------------------
    # Full benchmark build path
    # --------------------------------------------------------
    logger.info("Loading patients and diagnosis tables")
    patients = pd.read_csv(PATIENTS_FILE)
    diagnoses = pd.read_csv(DIAGNOSES_FILE)
    d_icd = pd.read_csv(D_ICD_FILE)

    diagnoses = diagnoses.merge(
        d_icd,
        on=["icd_code", "icd_version"],
        how="left"
    )

    # --------------------------------------------------------
    # 静态属性（取首次住院）
    # --------------------------------------------------------
    logger.info("Extracting static patient attributes")
    first_admissions = (
        admissions
        .sort_values(["subject_id", "admittime"])
        .groupby("subject_id")
        .first()
    )

    static_attribute_map = first_admissions[
        ["race", "language", "marital_status"]
    ].to_dict("index")

    # --------------------------------------------------------
    # cohort（先按 eligible_subjects 过滤 admissions，再统计 visits）
    # --------------------------------------------------------
    admissions_eligible = admissions[admissions["subject_id"].isin(eligible_subjects)].copy()

    visit_group = (
        admissions_eligible
        .groupby("subject_id")["hadm_id"]
        .agg(list)
        .reset_index()
    )
    visit_group["n_visits"] = visit_group["hadm_id"].apply(len)

    cohort = (
        visit_group
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    if DEMO_MODE:
        logger.warning(f"DEMO_MODE enabled: using first {DEMO_N} patients only")
        cohort = cohort.head(DEMO_N)

    cohort["patient_id"] = [f"P{i:06d}" for i in range(1, len(cohort) + 1)]

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------
    BENCH_DATA_DIR.mkdir(exist_ok=True)
    patients_dir = config.patientExtract.PATIENT_OUTPUT_PATH
    patients_dir.mkdir(exist_ok=True)

    patient_index_rows = []
    hadm_mapping_rows = []

    patient_meta = patients.set_index("subject_id") if "subject_id" in patients.columns else pd.DataFrame()

    # --------------------------------------------------------
    # 为每个病人生成 JSON
    # --------------------------------------------------------
    logger.info("Building patient JSON files")

    for _, row in cohort.iterrows():
        pid = row["patient_id"]
        subj = int(row["subject_id"])
        hadm_list = row["hadm_id"]

        meta = patient_meta.loc[subj] if (not patient_meta.empty and subj in patient_meta.index) else pd.Series()
        static_attrs = static_attribute_map.get(subj, {})

        patient_info = {
            "patient_id": pid,
            "subject_id": subj,
            "gender": str(meta.get("gender", "UNKNOWN")),
            "race": str(static_attrs.get("race", "UNKNOWN")),
            "age_first_visit": int(meta.get("anchor_age", 0)) if pd.notnull(meta.get("anchor_age", None)) else 0,
            "language": str(static_attrs.get("language", "UNKNOWN")),
            "marital_status": str(static_attrs.get("marital_status", "UNKNOWN"))
        }

        patient_admissions = (
            admissions_eligible[admissions_eligible["subject_id"] == subj]
            .sort_values("admittime")
            .reset_index(drop=True)
        )

        visits_list = []

        for visit_idx, (_, adm_row) in enumerate(patient_admissions.iterrows(), start=1):
            hadm_id = int(adm_row.hadm_id)

            hadm_mapping_rows.append({
                "hadm_id": hadm_id,
                "patient_id": pid,
                "visit_index": visit_idx - 1
            })

            diag_rows = (
                diagnoses[diagnoses["hadm_id"] == hadm_id]
                .sort_values("seq_num")
            )

            diagnosis = [
                {
                    "seq_num": int(r.seq_num),
                    "icd_version": int(r.icd_version),
                    "description": str(r.long_title) if pd.notnull(r.long_title) else None
                }
                for _, r in diag_rows.iterrows()
            ]

            # FULL-COVERAGE 保证这里一定能取到 note（若取不到，说明数据/过滤有 bug）
            discharge_note_text = discharge_note_map.get((subj, hadm_id), None)

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
                    "discharge_note": discharge_note_text
                },
                "event_stream": [],
                "diagnosis": diagnosis,
                "summary": {},
                "ground_truth_note": None
            }

            visits_list.append(visit_obj)

        patient_record = {"patient_info": patient_info, "visits": visits_list}

        with open(patients_dir / f"{pid}.json", "w", encoding="utf-8") as f:
            json.dump(patient_record, f, indent=2, ensure_ascii=False)

        patient_index_rows.append({
            "patient_id": pid,
            "subject_id": subj,
            "n_visits": len(visits_list),
            "visit_ids": ";".join(str(h) for h in hadm_list)
        })

    # --------------------------------------------------------
    # 导出索引表
    # --------------------------------------------------------
    patient_index_path = config.patientExtract.PATIENT_INDEX_OUTPUT_PATH
    patient_index_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(patient_index_rows).to_csv(patient_index_path / "patient_index.csv", index=False)
    pd.DataFrame(hadm_mapping_rows).to_csv(patient_index_path / "hadm_mapping.csv", index=False)

    logger.success("STEP 1 COMPLETE (FULL-COVERAGE COHORT)")


if __name__ == "__main__":
    extract_patient_data()
