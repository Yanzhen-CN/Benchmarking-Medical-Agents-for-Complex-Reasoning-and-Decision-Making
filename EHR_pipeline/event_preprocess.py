import pandas as pd
from pathlib import Path
import time

# ==================== CONFIGURATION ====================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"
BENCH_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "bench_data"

CHUNK_SIZE = 500_000  # Adjust based on your server's memory
# =======================================================

def main():
    print("=" * 70)
    print("STEP 3a: FULL-SCALE PRE-EXTRACTION OF EVENTS")
    print("(One-time processing of entire cohort)")
    print("=" * 70)
    
    # 1. Load the complete cohort mapping (NO DEMO MODE)
    start_time = time.time()
    print("1. Loading full cohort information...")
    
    hadm_mapping_path = BENCH_DATA_DIR / "hadm_mapping.csv"
    if not hadm_mapping_path.exists():
        print(f"ERROR: {hadm_mapping_path} not found. Run Step 1 first.")
        return
    
    hadm_mapping = pd.read_csv(hadm_mapping_path)
    target_hadms = set(hadm_mapping["hadm_id"].tolist())
    print(f"   Total admissions in cohort: {len(target_hadms):,}")
    
    # 2. Prepare output directory (with timestamp for versioning)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = BENCH_DATA_DIR / f"pre_extracted_events_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"2. Output directory: {output_dir}")
    
    # 3. Extract vital signs from chartevents (warning == 1)
    print("\n3. EXTRACTING VITAL SIGNS from chartevents.csv")
    print("   (This is the largest file, will take significant time...)")
    
    vital_chunks = []
    chart_path = RAW_DATA_DIR / "icu/chartevents.csv"
    total_vital_rows = 0
    total_vital_filtered = 0
    
    if chart_path.exists():
        # First, get approximate total rows for progress tracking
        # (This is optional and can be skipped if file is too large)
        print("   Scanning file size...")
        
        for i, chunk in enumerate(pd.read_csv(
            chart_path,
            usecols=["hadm_id", "charttime", "itemid", "valuenum", "valueuom", "warning"],
            parse_dates=["charttime"],
            chunksize=CHUNK_SIZE,
            low_memory=False
        )):
            total_vital_rows += len(chunk)
            
            # Filter: 1) in our cohort, 2) warning flag = 1
            chunk = chunk[chunk["hadm_id"].isin(target_hadms)]
            chunk = chunk[chunk["warning"] == 1]
            
            if not chunk.empty:
                chunk["event_type"] = "vital"
                vital_chunks.append(chunk)
                total_vital_filtered += len(chunk)
            
            # Progress update every N chunks
            if i % 20 == 0:
                print(f"   Processed {(i+1)*CHUNK_SIZE:,} rows | "
                      f"Filtered: {total_vital_filtered:,} events | "
                      f"Ratio: {total_vital_filtered/total_vital_rows*100:.2f}%")
        
        # Save all vital events
        if vital_chunks:
            vital_df = pd.concat(vital_chunks, ignore_index=True)
            vital_output = output_dir / "vital_events.parquet"
            vital_df.to_parquet(vital_output, engine='pyarrow', compression='snappy')
            print(f"\n   ✓ Saved {len(vital_df):,} vital events")
            print(f"     File: {vital_output}")
        else:
            print("\n   ⚠ No vital events found for the cohort")
    else:
        print(f"   ⚠ File not found: {chart_path}")
    
    # 4. Extract lab results from labevents (flag == 'abnormal')
    print("\n4. EXTRACTING LAB RESULTS from labevents.csv")
    
    lab_chunks = []
    lab_path = RAW_DATA_DIR / "hosp/labevents.csv"
    total_lab_rows = 0
    total_lab_filtered = 0
    
    if lab_path.exists():
        for i, chunk in enumerate(pd.read_csv(
            lab_path,
            usecols=["hadm_id", "charttime", "itemid", "valuenum", "valueuom", "flag"],
            parse_dates=["charttime"],
            chunksize=CHUNK_SIZE,
            low_memory=False
        )):
            total_lab_rows += len(chunk)
            
            # Filter: 1) in our cohort, 2) abnormal flag
            chunk = chunk[chunk["hadm_id"].isin(target_hadms)]
            chunk = chunk[chunk["flag"] == "abnormal"]
            
            if not chunk.empty:
                chunk["event_type"] = "lab"
                lab_chunks.append(chunk)
                total_lab_filtered += len(chunk)
            
            if i % 20 == 0:
                print(f"   Processed {(i+1)*CHUNK_SIZE:,} rows | "
                      f"Filtered: {total_lab_filtered:,} events")
        
        if lab_chunks:
            lab_df = pd.concat(lab_chunks, ignore_index=True)
            lab_output = output_dir / "lab_events.parquet"
            lab_df.to_parquet(lab_output, engine='pyarrow', compression='snappy')
            print(f"\n   ✓ Saved {len(lab_df):,} lab events")
            print(f"     File: {lab_output}")
        else:
            print("\n   ⚠ No lab events found for the cohort")
    else:
        print(f"   ⚠ File not found: {lab_path}")
    
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
        print(f"   ✓ Imaging: {len(imaging_df):,} events")
    else:
        print(f"   ⚠ Imaging file not found: {imaging_path}")
    
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
        print(f"   ✓ Procedures: {len(proc_df):,} events")
    else:
        print(f"   ⚠ Procedures file not found: {proc_path}")
    
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
        print(f"   ✓ Medications: {len(med_df):,} events")
    else:
        print(f"   ⚠ Medications file not found: {med_path}")
    
    # 6. Create a manifest file for tracking
    print("\n6. GENERATING MANIFEST")
    
    manifest = {
        "extraction_timestamp": timestamp,
        "cohort_admissions": len(target_hadms),
        "source_files": {
            "chartevents": str(chart_path) if chart_path.exists() else "NOT_FOUND",
            "labevents": str(lab_path) if lab_path.exists() else "NOT_FOUND",
            "radiology": str(imaging_path) if imaging_path.exists() else "NOT_FOUND",
            "procedures_icd": str(proc_path) if proc_path.exists() else "NOT_FOUND",
            "prescriptions": str(med_path) if med_path.exists() else "NOT_FOUND"
        }
    }
    
    # Count events in each saved file
    event_counts = {}
    total_extracted_events = 0
    
    for parquet_file in output_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        event_type = parquet_file.stem.replace("_events", "")
        event_count = len(df)
        event_counts[event_type] = event_count
        total_extracted_events += event_count
        
        # Add sample of hadm_ids for verification
        if event_count > 0:
            sample_hadms = df["hadm_id"].unique()[:5].tolist()
            manifest[f"{event_type}_sample_hadms"] = sample_hadms
    
    manifest["event_counts"] = event_counts
    manifest["total_extracted_events"] = total_extracted_events
    
    # Save manifest as JSON
    import json
    manifest_path = output_dir / "extraction_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # 7. Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PRE-EXTRACTION COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 EXTRACTION SUMMARY")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Cohort size: {len(target_hadms):,} admissions")
    print(f"   Total events extracted: {total_extracted_events:,}")
    
    print(f"\n📁 OUTPUT FILES ({output_dir}):")
    for event_type, count in event_counts.items():
        print(f"   {event_type:12} {count:>12,} events")
    
    # Calculate and display compression ratio
    if chart_path.exists() and 'vital' in event_counts:
        chart_size = chart_path.stat().st_size / (1024**3)  # GB
        vital_size = (output_dir / "vital_events.parquet").stat().st_size / (1024**2)  # MB
        
        print(f"\n💾 STORAGE EFFICIENCY")
        print(f"   chartevents.csv:      {chart_size:.1f} GB")
        print(f"   vital_events.parquet: {vital_size:.1f} MB")
        print(f"   Compression ratio:    {chart_size*1024/vital_size:.0f}x smaller")
    
    print(f"\n✅ NEXT STEPS:")
    print(f"   1. Run Step 3b to build event streams: python step3b_build_streams.py")
    print(f"   2. Update Step 3b to read from: {output_dir.name}")
    print(f"\n⚠  IMPORTANT: This is a ONE-TIME extraction.")
    print("   Future pipeline runs will use these pre-extracted files.")
    print("=" * 70)

if __name__ == "__main__":
    main()