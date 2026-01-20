import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import sys

TS_FMT = "%Y-%m-%d %H:%M:%S"
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "bench_data" / "patients"
OUTPUT_DIR = ROOT / "bench_data" / "patients_sequence"

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, TS_FMT)
    except ValueError:
        return None


def _make_item(
    *,
    schema_version: str,
    section: str,
    patient_id: str,
    subject_id: int,
    visit_id: Optional[str],
    hadm_id: Optional[int],
    timestamp: Optional[str],
    event_id: Optional[str],
    event_type: Optional[str],
    has_note: Optional[bool],
    source: str,
    data: Any,
    item_id: str,
    itemnum: int
) -> Dict[str, Any]:
    # 保持原有逻辑：itemnum 在这里 +1
    itemnum = itemnum + 1
    return {
        "metadata": {
            "schema_version": schema_version,
            "item_id": item_id,
            "section": section,
            "patient_id": patient_id,
            "subject_id": subject_id,
            "visit_id": visit_id,
            "hadm_id": hadm_id,
            "timestamp": timestamp,
            "event_id": event_id,
            "event_type": event_type,
            "has_note": has_note,
            "source": source,
            "itemnum": itemnum
        },
        "data": data,
    }


def convert(input_json_path: str) -> Path:
    in_path = Path(input_json_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")
    
    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # --- 逻辑不变，提取字段 ---
    patient_info = raw.get("patient_info")
    if not isinstance(patient_info, dict):
        raise ValueError("Missing or invalid field: patient_info")

    patient_id = patient_info.get("patient_id")
    subject_id = patient_info.get("subject_id")
    visits = raw.get("visits", [])
    schema_version = "1"
    source = in_path.name

    items: List[Dict[str, Any]] = []

    # 1) patient_info
    items.append(_make_item(
        schema_version=schema_version, section="patient_info",
        patient_id=patient_id, subject_id=subject_id,
        visit_id=None, hadm_id=None, timestamp=None,
        event_id=None, event_type=None, has_note=None,
        source=source, data=patient_info,
        item_id=f"{patient_id}-patient_info", itemnum=0
    ))

    def event_sort_key(e: Dict[str, Any], original_index: int) -> Tuple[bool, datetime, int]:
        dt = _parse_ts(e.get("timestamp"))
        if dt is None: return (True, datetime.max, original_index)
        return (False, dt, original_index)

    # 2) For each visit
    for v_idx, visit in enumerate(visits):
        if not isinstance(visit, dict): continue
        visit_id = visit.get("visit_id")
        hadm_id = visit.get("hadm_id")

        # admission_info
        admission_info = visit.get("admission_info", {})
        if isinstance(admission_info, dict) and admission_info:
            items.append(_make_item(
                schema_version=schema_version, section="admission_info",
                patient_id=patient_id, subject_id=subject_id,
                visit_id=visit_id, hadm_id=hadm_id,
                timestamp=admission_info.get("admission_time"),
                event_id=None, event_type=None,
                has_note=admission_info.get("admission_note") is not None,
                source=source, data=admission_info,
                item_id=f"{patient_id}-{visit_id}-admission_info" if visit_id else f"{patient_id}-visit{v_idx}-admission_info",
                itemnum=len(items)
            ))

        # events
        event_stream = visit.get("event_stream", []) or []
        indexed_events = [(i, e) for i, e in enumerate(event_stream) if isinstance(e, dict)]
        indexed_events.sort(key=lambda pair: event_sort_key(pair[1], pair[0]))

        for e_idx, ev in indexed_events:
            items.append(_make_item(
                schema_version=schema_version, section="event",
                patient_id=patient_id, subject_id=subject_id,
                visit_id=visit_id, hadm_id=hadm_id,
                timestamp=ev.get("timestamp"),
                event_id=ev.get("event_id"),
                event_type=ev.get("type"),
                has_note=None, source=source, data=ev,
                item_id=ev.get("event_id") or f"{patient_id}-{visit_id}-event-{e_idx}",
                itemnum=len(items)
            ))

        # discharge_info
        discharge_info = visit.get("discharge_info", {})
        if isinstance(discharge_info, dict) and discharge_info:
            items.append(_make_item(
                schema_version=schema_version, section="discharge_info",
                patient_id=patient_id, subject_id=subject_id,
                visit_id=visit_id, hadm_id=hadm_id,
                timestamp=discharge_info.get("discharge_time"),
                event_id=None, event_type=None,
                has_note=discharge_info.get("discharge_note") is not None,
                source=source, data=discharge_info,
                item_id=f"{patient_id}-{visit_id}-discharge_info" if visit_id else f"{patient_id}-visit{v_idx}-discharge_info",
                itemnum=len(items)
            ))

    # --- 修改输出路径逻辑 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # 确保文件夹存在
    out_path = OUTPUT_DIR / f"{in_path.stem}_sequenced.json"
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return out_path


def batch_convert(input_dir: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # 获取所有 P*.json 文件
    candidates = sorted(input_dir.glob("P*.json"))
    # 过滤掉已经是 _sequenced 的文件（以防万一）
    to_convert = [p for p in candidates if not p.name.endswith("_sequenced.json")]

    if not to_convert:
        print(f"No matching files in: {input_dir}")
        return

    print(f"Found {len(to_convert)} files. Output directory: {OUTPUT_DIR}")

    ok, failed = 0, 0
    for p in to_convert:
        try:
            out = convert(str(p))
            print(f"OK: {p.name} -> {OUTPUT_DIR.name}/{out.name}")
            ok += 1
        except Exception as e:
            print(f"FAIL: {p.name}: {e}")
            failed += 1

    print(f"Done. OK={ok}, FAIL={failed}")


if __name__ == "__main__":
    batch_convert(INPUT_DIR)