#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stats_longmedbench.py

Compute:
- average inpatient visits per patient
- average time-series medical events per visit

Assumes each patient JSON has:
{
  "patient_info": {...},
  "visits": [
     {"visit_id": "...", "event_stream": [ ... ]},
     ...
  ]
}
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median

def safe_len(x):
    return len(x) if isinstance(x, list) else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing patient JSON files, or a glob like '/path/*.json'",
    )
    ap.add_argument("--recursive", action="store_true", help="Recursively search subfolders (directory input only).")
    args = ap.parse_args()

    inp = Path(args.input)

    if any(ch in args.input for ch in ["*", "?", "["]):  # glob pattern
        files = sorted(Path().glob(args.input))
    elif inp.is_dir():
        files = sorted(inp.rglob("*.json") if args.recursive else inp.glob("*.json"))
    elif inp.is_file():
        files = [inp]
    else:
        raise SystemExit(f"Input not found: {args.input}")

    if not files:
        raise SystemExit("No JSON files found.")

    num_patients = 0
    total_visits = 0
    total_events = 0

    visits_per_patient = []
    events_per_visit = []

    bad_files = 0

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            bad_files += 1
            continue

        visits = data.get("visits", [])
        v_cnt = safe_len(visits)
        if v_cnt == 0:
            # still count this patient (up to you). Here we count it as 0 visits.
            num_patients += 1
            visits_per_patient.append(0)
            continue

        num_patients += 1
        total_visits += v_cnt
        visits_per_patient.append(v_cnt)

        for v in visits:
            es = v.get("event_stream", [])
            e_cnt = safe_len(es)
            total_events += e_cnt
            events_per_visit.append(e_cnt)

    avg_visits = (total_visits / num_patients) if num_patients else 0.0
    avg_events = (total_events / total_visits) if total_visits else 0.0

    print("========== LongMedBench Stats ==========")
    print(f"Patients: {num_patients} (bad/unreadable files skipped: {bad_files})")
    print(f"Total visits: {total_visits}")
    print(f"Total event_stream events: {total_events}")
    print("--------------------------------------")
    print(f"Avg inpatient visits per patient: {avg_visits:.3f}")
    print(f"Avg time-series medical events per visit: {avg_events:.3f}")

    if visits_per_patient:
        print(f"Median visits per patient: {median(visits_per_patient):.3f}")
    if events_per_visit:
        print(f"Median events per visit: {median(events_per_visit):.3f}")

if __name__ == "__main__":
    main()