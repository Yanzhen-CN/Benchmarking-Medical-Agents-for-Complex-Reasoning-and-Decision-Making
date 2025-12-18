import json
from pathlib import Path
import pandas as pd
from note_slicing import split_note_to_adm_discharge

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"
PATIENT_DIR = BENCH_DATA_DIR / "patients"
PATIENT_INDEXES = BENCH_DATA_DIR / "patient_index.csv"

DISCHARGE_NOTE_PATH = RAW_DATA_DIR / "note" / "discharge.csv"


def build_discharge_map(discharge_df: pd.DataFrame) -> dict[tuple[int, int], str]:
    """
    返回 (subject_id, hadm_id) -> text
    若同一 (subject_id, hadm_id) 有多条记录：
    - 有 charttime 列则按时间取最后一条
    - 否则保留第一条
    """
    df = discharge_df.copy()

    # 统一类型，避免 int/str 匹配失败
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["subject_id", "hadm_id", "text"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["hadm_id"] = df["hadm_id"].astype(int)

    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values("charttime")
        df = df.drop_duplicates(subset=["subject_id", "hadm_id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")

    return df.set_index(["subject_id", "hadm_id"])["text"].to_dict()


def main():
    patient_indexes = pd.read_csv(PATIENT_INDEXES)
    discharge_df = pd.read_csv(DISCHARGE_NOTE_PATH)

    discharge_map = build_discharge_map(discharge_df)

    for _, row in patient_indexes.iterrows():
        subject_id = (row["subject_id"])
        patient_id = str(row["patient_id"]).strip()
        patient_file = PATIENT_DIR / f"{patient_id}.json"
        if not patient_file.exists():
            print(f"Patient file {patient_file} does not exist. Skipping.")
            continue

        with open(patient_file, "r") as f:
            patient_data = json.load(f)

        if int(patient_data["patient_info"]["subject_id"]) != subject_id:
            print(f"Subject ID mismatch in {patient_file}. Skipping.")
            continue

        updated = 0
        for v in patient_data.get("visits", []):
            hadm_id = v.get("hadm_id")
            if hadm_id is None:
                continue
            hadm_id = int(hadm_id)

            note_text = discharge_map.get((subject_id, hadm_id))
            v["ground_truth_note"] = note_text  # 找不到则为 None
            if not note_text:
                # 没有 note，就保持 admission_note/discharge_note 为 None
                v.setdefault("admission_info", {})["admission_note"] = None
                v.setdefault("discharge_info", {})["discharge_note"] = None
                continue

            parsed = split_note_to_adm_discharge(note_text)

            # 关键：写回到你 JSON 里对应位置
            v.setdefault("admission_info", {})["admission_note"] = parsed.get("admission_note")
            v.setdefault("discharge_info", {})["discharge_note"] = parsed.get("discharge_note")

            if note_text is not None:
                updated += 1

        with open(patient_file, "w") as f:
            json.dump(patient_data, f, ensure_ascii=False, indent=2)

        print(f"{patient_file.name}: updated {updated} visit ground_truth_note")


if __name__ == "__main__":
    main()
