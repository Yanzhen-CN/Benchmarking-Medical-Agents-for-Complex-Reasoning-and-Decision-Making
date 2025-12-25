import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import sys

TS_FMT = "%Y-%m-%d %H:%M:%S"
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "bench_data" / "patients"



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
) -> Dict[str, Any]:
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
        },
        "data": data,
    }


def convert(input_json_path: str) -> Path:
    """
    Convert a patient JSON file into a sequenced list of JSON items.

    Output file: <same_dir>/<stem>_sequenced.json
    """
    in_path = Path(input_json_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")
    if in_path.suffix.lower() != ".json":
        raise ValueError(f"Input file must be .json: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Required top-level fields
    patient_info = raw.get("patient_info")
    if not isinstance(patient_info, dict):
        raise ValueError("Missing or invalid field: patient_info (must be an object)")

    patient_id = patient_info.get("patient_id")
    subject_id = patient_info.get("subject_id")
    if not isinstance(patient_id, str) or not patient_id:
        raise ValueError("Missing or invalid patient_info.patient_id")
    if not isinstance(subject_id, int):
        raise ValueError("Missing or invalid patient_info.subject_id (must be int)")

    visits = raw.get("visits", [])
    if not isinstance(visits, list):
        raise ValueError("Missing or invalid field: visits (must be a list)")

    schema_version = "1"
    source = in_path.name

    items: List[Dict[str, Any]] = []

    # 1) patient_info
    items.append(
        _make_item(
            schema_version=schema_version,
            section="patient_info",
            patient_id=patient_id,
            subject_id=subject_id,
            visit_id=None,
            hadm_id=None,
            timestamp=None,
            event_id=None,
            event_type=None,
            has_note=None,
            source=source,
            data=patient_info,
            item_id=f"{patient_id}-patient_info",
        )
    )

    # Helper for sorting events within a visit
    def event_sort_key(e: Dict[str, Any], original_index: int) -> Tuple[bool, datetime, int]:
        dt = _parse_ts(e.get("timestamp"))
        if dt is None:
            return (True, datetime.max, original_index)
        return (False, dt, original_index)

    # 2) For each visit: admission -> events -> discharge
    for v_idx, visit in enumerate(visits):
        if not isinstance(visit, dict):
            continue

        visit_id = visit.get("visit_id")
        hadm_id = visit.get("hadm_id")
        if visit_id is not None and not isinstance(visit_id, str):
            raise ValueError(f"Invalid visits[{v_idx}].visit_id (must be str or null)")
        if hadm_id is not None and not isinstance(hadm_id, int):
            raise ValueError(f"Invalid visits[{v_idx}].hadm_id (must be int or null)")

        # admission_info
        admission_info = visit.get("admission_info", {})
        if isinstance(admission_info, dict) and admission_info:
            adm_time = admission_info.get("admission_time")
            has_note = admission_info.get("admission_note") is not None
            items.append(
                _make_item(
                    schema_version=schema_version,
                    section="admission_info",
                    patient_id=patient_id,
                    subject_id=subject_id,
                    visit_id=visit_id,
                    hadm_id=hadm_id,
                    timestamp=adm_time if isinstance(adm_time, str) else None,
                    event_id=None,
                    event_type=None,
                    has_note=bool(has_note),
                    source=source,
                    data=admission_info,
                    item_id=f"{patient_id}-{visit_id}-admission_info" if visit_id else f"{patient_id}-visit{v_idx}-admission_info",
                )
            )

        # events
        event_stream = visit.get("event_stream", [])
        if event_stream is None:
            event_stream = []
        if not isinstance(event_stream, list):
            raise ValueError(f"Invalid visits[{v_idx}].event_stream (must be list)")

        indexed_events: List[Tuple[int, Dict[str, Any]]] = []
        for e_idx, ev in enumerate(event_stream):
            if isinstance(ev, dict):
                indexed_events.append((e_idx, ev))

        indexed_events.sort(key=lambda pair: event_sort_key(pair[1], pair[0]))

        for e_idx, ev in indexed_events:
            ev_id = ev.get("event_id")
            ev_type = ev.get("type")
            ev_ts = ev.get("timestamp")

            if ev_id is not None and not isinstance(ev_id, str):
                raise ValueError(f"Invalid event_id at visits[{v_idx}].event_stream[{e_idx}]")
            if ev_type is not None and not isinstance(ev_type, str):
                raise ValueError(f"Invalid type at visits[{v_idx}].event_stream[{e_idx}]")

            items.append(
                _make_item(
                    schema_version=schema_version,
                    section="event",
                    patient_id=patient_id,
                    subject_id=subject_id,
                    visit_id=visit_id,
                    hadm_id=hadm_id,
                    timestamp=ev_ts if isinstance(ev_ts, str) else None,
                    event_id=ev_id if isinstance(ev_id, str) else None,
                    event_type=ev_type if isinstance(ev_type, str) else None,
                    has_note=None,
                    source=source,
                    data=ev,
                    item_id=ev_id if isinstance(ev_id, str) and ev_id else f"{patient_id}-{visit_id}-event-{e_idx}",
                )
            )

        # discharge_info
        discharge_info = visit.get("discharge_info", {})
        if isinstance(discharge_info, dict) and discharge_info:
            dis_time = discharge_info.get("discharge_time")
            has_note = discharge_info.get("discharge_note") is not None
            items.append(
                _make_item(
                    schema_version=schema_version,
                    section="discharge_info",
                    patient_id=patient_id,
                    subject_id=subject_id,
                    visit_id=visit_id,
                    hadm_id=hadm_id,
                    timestamp=dis_time if isinstance(dis_time, str) else None,
                    event_id=None,
                    event_type=None,
                    has_note=bool(has_note),
                    source=source,
                    data=discharge_info,
                    item_id=f"{patient_id}-{visit_id}-discharge_info" if visit_id else f"{patient_id}-visit{v_idx}-discharge_info",
                )
            )

    out_path = in_path.with_name(f"{in_path.stem}_sequenced.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return out_path


def batch_convert(input_dir: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    candidates = sorted(input_dir.glob("P*.json"))
    to_convert = [p for p in candidates if not p.name.endswith("_sequenced.json")]

    if not to_convert:
        print(f"No matching files in: {input_dir}")
        return

    print(f"Found {len(to_convert)} files to convert in: {input_dir}")

    ok = 0
    failed = 0
    for p in to_convert:
        try:
            out = convert(str(p))
            print(f"OK: {p.name} -> {out.name}")
            ok += 1
        except Exception as e:
            print(f"FAIL: {p.name}: {e}")
            failed += 1

    print(f"Done. OK={ok}, FAIL={failed}")


if __name__ == "__main__":
    batch_convert(INPUT_DIR)