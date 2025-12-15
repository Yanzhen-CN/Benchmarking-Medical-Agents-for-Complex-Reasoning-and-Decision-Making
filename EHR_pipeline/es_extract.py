import pandas as pd
import json
from pathlib import Path
from collections import defaultdict


# Config
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

DEMO_MODE = True
DEMO_N = 5
CHUNK_SIZE = 200_000


# Load patient index
patient_index = pd.read_csv(BENCH_DATA_DIR / "patient_index.csv")
if DEMO_MODE:
    patient_index = patient_index.head(DEMO_N)

# Collect all hadm_id used in current run
hadm_ids = set()
for v in patient_index["visit_ids"]:
    hadm_ids.update(int(x) for x in v.split(";"))


# Load dictionary tables
d_items = pd.read_csv(
    RAW_DATA_DIR / "icu/d_items.csv",
    usecols=["itemid", "label", "unitname"]
)

d_labitems = pd.read_csv(
    RAW_DATA_DIR / "hosp/d_labitems.csv",
    usecols=["itemid", "label", "fluid", "category"]
)


# Container for events grouped by hadm_id
events_by_hadm = defaultdict(list)


# Vital events from ICU chartevents
print("Loading ICU vital events (warning == 1)...")

for chunk in pd.read_csv(
    RAW_DATA_DIR / "icu/chartevents.csv",
    usecols=["hadm_id", "charttime", "itemid", "valuenum", "warning"],
    parse_dates=["charttime"],
    chunksize=CHUNK_SIZE
):
    chunk = chunk[
        (chunk["hadm_id"].isin(hadm_ids)) &
        (chunk["warning"] == 1)
    ]
    if chunk.empty:
        continue

    chunk = chunk.merge(d_items, on="itemid", how="left")

    for r in chunk.itertuples():
        events_by_hadm[r.hadm_id].append({
            "time": r.charttime.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "vital",
            "itemid": int(r.itemid),
            "label": r.label,
            "value": r.valuenum,
            "unit": r.unitname,
            "warning": 1
        })


# Lab events from hospital labevents
print("Loading lab events (flag == abnormal)...")

for chunk in pd.read_csv(
    RAW_DATA_DIR / "hosp/labevents.csv",
    usecols=["hadm_id", "charttime", "itemid", "valuenum", "valueuom", "flag"],
    parse_dates=["charttime"],
    chunksize=CHUNK_SIZE
):
    chunk = chunk[
        (chunk["hadm_id"].isin(hadm_ids)) &
        (chunk["flag"] == "abnormal")
    ]
    if chunk.empty:
        continue

    chunk = chunk.merge(d_labitems, on="itemid", how="left")

    for r in chunk.itertuples():
        events_by_hadm[r.hadm_id].append({
            "time": r.charttime.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "lab",
            "itemid": int(r.itemid),
            "label": r.label,
            "category": r.category,
            "fluid": r.fluid,
            "value": r.valuenum,
            "unit": r.valueuom,
            "flag": "abnormal"
        })


# Imaging events from radiology notes
print("Loading imaging notes...")

radiology = pd.read_csv(
    RAW_DATA_DIR / "note/radiology.csv",
    usecols=["hadm_id", "charttime", "text"],
    parse_dates=["charttime"]
)

radiology = radiology[radiology["hadm_id"].isin(hadm_ids)]

for r in radiology.itertuples():
    events_by_hadm[r.hadm_id].append({
        "time": r.charttime.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "imaging",
        "content": r.text
    })


# Write event_stream back to visit files
print("Writing event_stream into visit.json files...")

for _, row in patient_index.iterrows():
    pid = row["patient_id"]
    visits_dir = BENCH_DATA_DIR / "patients" / pid / "visits"

    for visit_file in visits_dir.glob("V*.json"):
        with open(visit_file) as f:
            visit = json.load(f)

        hadm_id = visit["hadm_id"]
        events = events_by_hadm.get(hadm_id, [])

        events = sorted(events, key=lambda x: x["time"])
        visit["event_stream"] = events

        with open(visit_file, "w") as f:
            json.dump(visit, f, indent=2)


print(
    f"Step 3 complete: event_stream built "
    f"{'(demo mode)' if DEMO_MODE else ''}"
)
