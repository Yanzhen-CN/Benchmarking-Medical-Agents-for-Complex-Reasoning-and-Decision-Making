#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stats_longmedbench.py

Compute:
- average inpatient visits per patient
- average time-series medical events per visit
- variance of visits per patient
- variance of events per visit
- save visualizations as a PDF

Assumes each patient JSON has:
{
  "patient_info": {...},
  "visits": [
     {"visit_id": "...", "event_stream": [ ... ]},
     ...
  ]
}
"""
import numpy as np
import argparse
import json
from pathlib import Path
from statistics import mean, median, variance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PDF file path.",
    )
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

    # Variance calculations
    visit_std = np.std(visits_per_patient) if len(visits_per_patient) > 1 else 0.0
    event_std = np.std(events_per_visit) if len(events_per_visit) > 1 else 0.0

    print("========== LongMedBench Stats ==========")
    print(f"Patients: {num_patients} (bad/unreadable files skipped: {bad_files})")
    print(f"Total visits: {total_visits}")
    print(f"Total event_stream events: {total_events}")
    print("--------------------------------------")
    print(f"Avg inpatient visits per patient: {avg_visits:.3f}")
    print(f"Avg time-series medical events per visit: {avg_events:.3f}")
    print(f"Std of visits per patient: {visit_std:.3f}")
    print(f"Std of events per visit: {event_std:.3f}")

    if visits_per_patient:
        print(f"Median visits per patient: {median(visits_per_patient):.3f}")
    if events_per_visit:
        print(f"Median events per visit: {median(events_per_visit):.3f}")

    # Visualizing data
    with PdfPages(args.output) as pdf:
        # Plot visits per patient
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(visits_per_patient, bins=range(min(visits_per_patient), max(visits_per_patient) + 2), color='skyblue', edgecolor='black')
        ax.set_title('Visits per Patient')
        ax.set_xlabel('Number of Visits')
        ax.set_ylabel('Frequency')
        pdf.savefig(fig)  # saves the current figure into the PDF
        plt.close(fig)

        # Plot events per visit
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(events_per_visit, bins=range(min(events_per_visit), max(events_per_visit) + 2), color='lightgreen', edgecolor='black')
        ax.set_title('Events per Visit')
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Frequency')
        pdf.savefig(fig)  # saves the current figure into the PDF
        plt.close(fig)

    print(f"Visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
