#!/usr/bin/env python3
"""
Select events per visit and attach the visit-local index within each event_type.

Default rules:
- Always include ADMISSION and DISCHARGE (if present), with visit_type_index = null.
- For selected event types (default: LAB/MEDICATION/IMAGING), include up to K per
  type per visit (default K=1), selecting randomly within the visit.

Input: bench_data/patients_sequence/*_sequenced.json
Output: tasks/factual_questions/events_selected/{patient_id}.jsonl

Notes:
- To change which event types are selected, edit SELECT_EVENT_TYPES below.
- To include additional types (e.g., PROCEDURE or MICROBIOLOGY), add them to the set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_in = repo_root / "bench_data" / "patients_sequence"
    default_out = Path(__file__).resolve().parent / "events_selected"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in, help="patients_sequence dir")
    p.add_argument("--output", type=Path, default=default_out, help="output dir")
    p.add_argument("--per-type", type=int, default=1, help="max events per type per visit")
    p.add_argument("--seed", type=int, default=7, help="random seed for sampling")
    p.add_argument(
        "--sort-by",
        choices=["event_id", "timestamp"],
        default="event_id",
        help="sort selected events within each visit",
    )
    return p.parse_args()


def iter_patient_files(inp: Path) -> Iterable[Path]:
    if inp.is_dir():
        yield from sorted(inp.glob("*_sequenced.json"))
    else:
        yield inp


EVENT_ID_RE = re.compile(r"^(?P<pid>P\\d+)-(?P<visit>V\\d+)-(?:E(?P<idx>\\d+)|(?P<tag>adm|dis))$")

# Selected event types (besides ADMISSION/DISCHARGE). Add/remove to toggle.
SELECT_EVENT_TYPES = {
    "LAB",
    "MEDICATION",
    "IMAGING",
    # "PROCEDURE",
    # "MICROBIOLOGY",
    # "VITAL",
}


def _order_from_event_id(event_id: str) -> int:
    m = EVENT_ID_RE.match(event_id or "")
    if not m:
        return 0
    idx = m.group("idx")
    tag = m.group("tag")
    if idx is not None:
        return int(idx)
    if tag == "adm":
        return 0
    if tag == "dis":
        return 1_000_000_000
    return 0


def _parse_ts(ts: str | None) -> datetime:
    if not ts:
        return datetime.min
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.min


def main() -> int:
    args = parse_args()
    in_path: Path = args.input
    out_dir: Path = args.output
    k = max(0, args.per_type)
    rng = random.Random(args.seed)

    total_written = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for pf in iter_patient_files(in_path):
        try:
            events = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(events, list) or not events:
            continue

        patient_id = events[0].get("event_id", "").split("-")[0] or pf.stem
        patient_out = out_dir / f"{patient_id}.jsonl"

        # Group by visit_ref preserving order
        visits: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
        for idx, ev in enumerate(events):
            vr = ev.get("visit_ref") or "UNKNOWN_VISIT"
            visits[vr].append((idx, ev))

        with patient_out.open("w", encoding="utf-8") as out_f:
            for visit_ref, items in visits.items():
                # Compute visit-local order (1-based) per event_type
                ordered_events = [ev for _, ev in items]
                type_counters: Dict[str, int] = defaultdict(int)
                visit_type_index_map: Dict[str, int] = {}
                for ev in ordered_events:
                    et = ev.get("event_type") or "UNKNOWN"
                    type_counters[et] += 1
                    ev_id = ev.get("event_id") or f"{patient_id}-{visit_ref}-IDX{type_counters[et]}"
                    visit_type_index_map[ev_id] = type_counters[et]

                # Bucket by event_type for selection
                type_to_events: Dict[str, List[Dict]] = defaultdict(list)
                for ev in ordered_events:
                    et = ev.get("event_type") or "UNKNOWN"
                    type_to_events[et].append(ev)

                selected: List[Dict] = []

                # Always include admission/discharge if present
                for et in ("ADMISSION", "DISCHARGE"):
                    for ev in type_to_events.get(et, []):
                        selected.append(ev)

                # Select up to K per other type (random)
                for et, evs in type_to_events.items():
                    if et in ("ADMISSION", "DISCHARGE", "PATIENT_DEMOGRAPHICS"):
                        continue
                    if et not in SELECT_EVENT_TYPES:
                        continue
                    if not evs or k <= 0:
                        continue
                    if len(evs) <= k:
                        selected.extend(evs)
                    else:
                        selected.extend(rng.sample(evs, k))

                # De-duplicate while preserving order
                seen = set()
                deduped: List[Dict] = []
                for ev in selected:
                    ev_id = ev.get("event_id") or ""
                    if ev_id in seen:
                        continue
                    seen.add(ev_id)
                    deduped.append(ev)

                if args.sort_by == "timestamp":
                    deduped.sort(
                        key=lambda e: (_parse_ts(e.get("timestamp")), _order_from_event_id(e.get("event_id") or "")),
                    )
                else:
                    deduped.sort(key=lambda e: _order_from_event_id(e.get("event_id") or ""))

                # Force ADMISSION first and DISCHARGE last within each visit
                adm = [e for e in deduped if (e.get("event_type") or "") == "ADMISSION"]
                dis = [e for e in deduped if (e.get("event_type") or "") == "DISCHARGE"]
                core = [e for e in deduped if (e.get("event_type") or "") not in ("ADMISSION", "DISCHARGE")]
                deduped = adm + core + dis

                for ev in deduped:
                    ev_id = ev.get("event_id") or ""
                    et = ev.get("event_type") or ""
                    vt_index = None if et in ("ADMISSION", "DISCHARGE") else visit_type_index_map.get(ev_id)
                    out_obj = {
                        "patient_id": patient_id,
                        "visit_ref": visit_ref,
                        "visit_type_index": vt_index,
                        "event_id": ev.get("event_id"),
                        "event_type": ev.get("event_type"),
                        "timestamp": ev.get("timestamp"),
                        "content": ev.get("content"),
                    }
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    total_written += 1

    print(f"wrote_dir={out_dir}")
    print(f"rows={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
