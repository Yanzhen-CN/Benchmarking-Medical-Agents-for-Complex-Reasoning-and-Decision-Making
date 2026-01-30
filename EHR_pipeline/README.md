# EHR Pipeline

This folder contains the end-to-end pipeline for building the EHR benchmark dataset and LLM-ready context.

## Dependencies

Use the repo-level requirements file:

```bash
pip install -r ../requirements.txt
```

Key libraries used in this pipeline include:
- pandas
- polars
- tqdm
- loguru
- openai (only if LLM-assisted steps are enabled)

## Data layout (expected)

Paths are configured in `../config.py` and default to:

```
raw_data/
  hosp/
    admissions.csv
    patients.csv
    diagnoses_icd.csv
    d_icd_diagnoses.csv
    labevents.csv
    prescriptions.csv
    procedures_icd.csv
    d_icd_procedures.csv
    microbiologyevents.csv
    emar.csv
    emar_detail.csv
  icu/
    chartevents.csv
    d_items.csv
  note/
    discharge.csv
    radiology.csv

bench_data/
  patients/
  patients_sequence/
  context/
```

## Configuration

All runtime parameters are configured in `../config.py`:

- Cohort selection:
  - `BuildConfig.run.MIN_VISITS`
  - `BuildConfig.run.ANALYZE_MODE`
  - `BuildConfig.run.DEMO_MODE`
  - `BuildConfig.run.DEMO_N`
- Input/output paths:
  - `BuildConfig.paths.RAW_DATA_DIR`
  - `BuildConfig.paths.BENCH_DATA_DIR`
- LLM settings (used in note slicing and context generation):
  - `BuildConfig.noteExtract.USE_LLM`
  - `ContextConfig.USE_LLM_FOR_IMAGE_DESC`
  - `ContextConfig.USE_LLM_FOR_REASON`
  - `LLMConfig.provider`, `LLMConfig.api_key`, `LLMConfig.base_url`

## Build steps (end-to-end)

Run these in order from this directory:

```bash
python events_preprocess.py
python patients_extract.py
python es_extract.py
python note_extract.py
python convert.py
python build_context.py
```

### Step details

1) `events_preprocess.py`
- Extracts abnormal vitals and labs from MIMIC-IV.
- Outputs: `raw_data/chartevents_extract.csv`, `raw_data/labevents_extract.csv`.

2) `patients_extract.py`
- Builds cohort and patient JSON files.
- Outputs:
  - `bench_data/patients/P*.json`
  - `bench_data/patient_index.csv`
  - `bench_data/hadm_mapping.csv`

3) `es_extract.py`
- Builds event stream for each visit.
- Updates: `bench_data/patients/P*.json` (adds `event_stream`).

4) `note_extract.py`
- Injects discharge notes and splits admission/discharge sections.
- Updates: `bench_data/patients/P*.json` (adds `admission_note`, `discharge_note`, `ground_truth_note`).

5) `convert.py`
- Converts patient JSON to sequenced event list.
- Outputs: `bench_data/patients_sequence/*_sequenced.json`.

6) `build_context.py`
- Builds LLM-style multi-turn context for each visit.
- Outputs: `bench_data/context/*.json`.

## Notes

- If LLM assistance is enabled, ensure your API key is available via environment variables (see `../config.py`).
- `build_sequence.py` at repo root is an optional runner; currently only the final steps are active.
