import pandas as pd
from pathlib import Path
import time

# ==================== CONFIGURATION ====================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

# Adjust CHUNK_SIZE based on available memory (e.g., 2 million rows)
CHUNK_SIZE = 2_000_000
# =======================================================

def main():
    print("=" * 70)
    print("STEP 3a: FULL-SCALE PRE-EXTRACTION OF EVENTS")
    print("One-time processing for entire cohort")
    print("=" * 70)
    
    # 1. Load the complete cohort mapping
    start_time = time.time()
    print("1. Loading full cohort information...")
    
    hadm_mapping_path = BENCH_DATA_DIR / "hadm_mapping.csv"
    if not hadm_mapping_path.exists():
        print(f"ERROR: {hadm_mapping_path} not found. Run Step 1 first.")
        return
    
    hadm_mapping = pd.read_csv(hadm_mapping_path)
    target_hadms = set(hadm_mapping["hadm_id"].tolist())
    print(f"   Total admissions in cohort: {len(target_hadms):,}")
    print(f"   Chunk size for processing: {CHUNK_SIZE:,} rows")
    
    # 2. Prepare output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = BENCH_DATA_DIR / f"pre_extracted_events_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    print(f"2. Output directory: {output_dir}")
    
    # 3. Extract vital signs from chartevents (warning == 1)
    print("\n3. EXTRACTING VITAL SIGNS from chartevents.csv")
    print("   This will process the full chartevents table (approx. 30GB)...")
    
    vital_chunks = []
    chart_path = RAW_DATA_DIR / "icu/chartevents.csv"
    cumulative_rows = 0
    cumulative_filtered = 0
    
    if chart_path.exists():
        for i, chunk in enumerate(pd.read_csv(
            chart_path,
            usecols=["hadm_id", "charttime", "itemid", "valuenum", "valueuom", "warning"],
            parse_dates=["charttime"],
            chunksize=CHUNK_SIZE,
            low_memory=False
        )):
            rows_in_chunk = len(chunk)
            cumulative_rows += rows_in_chunk
            
            # Filter: 1) in our cohort, 2) warning flag = 1
            chunk = chunk[chunk["hadm_id"].isin(target_hadms)]
            chunk = chunk[chunk["warning"] == 1]
            
            filtered_in_chunk = len(chunk)
            cumulative_filtered += filtered_in_chunk
            
            if not chunk.empty:
                chunk["event_type"] = "vital"
                vital_chunks.append(chunk)
            
            # Progress update - FIXED ratio calculation
            if i % 10 == 0:
                ratio = (cumulative_filtered / cumulative_rows * 100) if cumulative_rows > 0 else 0
                print(f"   Chunk {i+1}: Read {cumulative_rows:,} total rows | "
                      f"Filtered: {cumulative_filtered:,} events | "
                      f"Ratio: {ratio:.3f}%")
        
        # Save all vital events
        if vital_chunks:
            vital_df = pd.concat(vital_chunks, ignore_index=True)
            vital_output = output_dir / "vital_events.parquet"
            vital_df.to_parquet(vital_output, engine='pyarrow', compression='snappy')
            print(f"\n   SAVED: {len(vital_df):,} vital events")
            print(f"   File: {vital_output}")
        else:
            print("\n   No vital events found for the cohort")
    else:
        print(f"   File not found: {chart_path}")
    
    # 4. Extract lab results from labevents (flag == 'abnormal')
    print("\n4. EXTRACTING LAB RESULTS from labevents.csv")
    
    lab_chunks = []
    lab_path = RAW_DATA_DIR / "hosp/labevents.csv"
    cumulative_rows = 0
    cumulative_filtered = 0
    
    if lab_path.exists():
        for i, chunk in enumerate(pd.read_csv(
            lab_path,
            usecols=["hadm_id", "charttime", "itemid", "valuenum", "valueuom", "flag"],
            parse_dates=["charttime"],
            chunksize=CHUNK_SIZE,
            low_memory=False
        )):
            rows_in_chunk = len(chunk)
            cumulative_rows += rows_in_chunk
            
            # Filter: 1) in our cohort, 2) abnormal flag
            chunk = chunk[chunk["hadm_id"].isin(target_hadms)]
            chunk = chunk[chunk["flag"] == "abnormal"]
            
            filtered_in_chunk = len(chunk)
            cumulative_filtered += filtered_in_chunk
            
            if not chunk.empty:
                chunk["event_type"] = "lab"
                lab_chunks.append(chunk)
            
            if i % 10 == 0:
                print(f"   Chunk {i+1}: Read {cumulative_rows:,} total rows | "
                      f"Filtered: {cumulative_filtered:,} events")
        
        if lab_chunks:
            lab_df = pd.concat(lab_chunks, ignore_index=True)
            lab_output = output_dir / "lab_events.parquet"
            lab_df.to_parquet(lab_output, engine='pyarrow', compression='snappy')
            print(f"\n   SAVED: {len(lab_df):,} lab events")
            print(f"   File: {lab_output}")
        else:
            print("\n   No lab events found for the cohort")
    else:
        print(f"   File not found: {lab_path}")
    
    # 5. Extract other event types (smaller files)
    print("\n5. EXTRACTING OTHER EVENT TYPES")
    
    # 5a. Imaging events
    imaging_path = RAW_DATA_DIR / "note/radiology.csv"
    if imaging_path.exists():
        imaging_df = pd.read_csv(
            imaging_path,
            usecols=["hadm_id", "charttime", "text"],
            parse_dates=["charttime"]
        )
        imaging_df = imaging_df[imaging_df["hadm_id"].isin(target_hadms)]
        imaging_df["event_type"] = "imaging"
        imaging_output = output_dir / "imaging_events.parquet"
        imaging_df.to_parquet(imaging_output, engine='pyarrow', compression='snappy')
        print(f"   Imaging events: {len(imaging_df):,} saved")
    else:
        print(f"   Imaging file not found: {imaging_path}")
    
    # 5b. Procedure events
    proc_path = RAW_DATA_DIR / "hosp/procedures_icd.csv"
    if proc_path.exists():
        proc_df = pd.read_csv(
            proc_path,
            usecols=["hadm_id", "chartdate", "icd_code", "icd_version"]
        )
        proc_df = proc_df[proc_df["hadm_id"].isin(target_hadms)]
        proc_df["event_type"] = "procedure"
        proc_output = output_dir / "procedure_events.parquet"
        proc_df.to_parquet(proc_output, engine='pyarrow', compression='snappy')
        print(f"   Procedure events: {len(proc_df):,} saved")
    else:
        print(f"   Procedures file not found: {proc_path}")
    
    # 5c. Medication events
    med_path = RAW_DATA_DIR / "hosp/prescriptions.csv"
    if med_path.exists():
        med_df = pd.read_csv(
            med_path,
            usecols=["hadm_id", "starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route"],
            parse_dates=["starttime", "stoptime"]
        )
        med_df = med_df[med_df["hadm_id"].isin(target_hadms)]
        med_df["event_type"] = "medication"
        med_output = output_dir / "medication_events.parquet"
        med_df.to_parquet(med_output, engine='pyarrow', compression='snappy')
        print(f"   Medication events: {len(med_df):,} saved")
    else:
        print(f"   Medications file not found: {med_path}")
    
    # 6. Generate summary statistics
    print("\n6. GENERATING SUMMARY")
    
    event_counts = {}
    total_extracted_events = 0
    
    for parquet_file in output_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        event_type = parquet_file.stem.replace("_events", "")
        event_count = len(df)
        event_counts[event_type] = event_count
        total_extracted_events += event_count
    
    # 7. Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PRE-EXTRACTION COMPLETE")
    print("=" * 70)
    
    print(f"\nSUMMARY")
    print(f"  Total processing time: {total_time/60:.1f} minutes")
    print(f"  Cohort size: {len(target_hadms):,} admissions")
    print(f"  Total events extracted: {total_extracted_events:,}")
    
    print(f"\nEVENT BREAKDOWN:")
    for event_type, count in event_counts.items():
        print(f"  {event_type:12} {count:>12,} events")
    
    # Calculate storage efficiency
    if chart_path.exists() and 'vital' in event_counts:
        chart_size = chart_path.stat().st_size / (1024**3)  # GB
        vital_size = (output_dir / "vital_events.parquet").stat().st_size / (1024**2)  # MB
        compression_ratio = (chart_size * 1024) / vital_size if vital_size > 0 else 0
        
        print(f"\nSTORAGE EFFICIENCY")
        print(f"  Original chartevents.csv: {chart_size:.1f} GB")
        print(f"  Extracted vital events:   {vital_size:.1f} MB")
        print(f"  Approximate compression:  {compression_ratio:.0f}x smaller")
    
    print(f"\nNEXT STEPS:")
    print(f"  1. Update Step 3b script to read from: {output_dir.name}")
    print(f"  2. Run Step 3b to build event streams in patient JSON files.")
    print(f"\nNOTE: This extraction is performed once.")
    print("=" * 70)

if __name__ == "__main__":
    main()