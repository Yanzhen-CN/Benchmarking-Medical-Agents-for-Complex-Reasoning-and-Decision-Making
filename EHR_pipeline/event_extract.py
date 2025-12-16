import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "EHR_pipeline" / "raw_data"

def extract_abnormal_data():
    """Global preprocessing: extract only abnormal records from chartevents and labevents"""
    print("Starting preprocessing, extracting abnormal records...")
    
    # 1. Process vital signs data (chartevents.csv) - warning == 1
    print("\n1. Processing vital signs data (warning=1)...")
    vital_path = RAW_DATA_DIR / "icu/chartevents.csv"
    
    if vital_path.exists():
        vital_chunks = []
        total_rows = 0
        processed_rows = 0
        
        # Process in chunks for memory efficiency
        for chunk in pd.read_csv(
            vital_path,
            chunksize=2_000_000,
            low_memory=False
        ):
            total_rows += len(chunk)
            
            # Handle potential data type issues with warning column
            # Convert to string and strip whitespace for consistent comparison
            if 'warning' in chunk.columns:
                chunk['warning_clean'] = chunk['warning'].astype(str).str.strip()
                
                # Filter for warning=1 (including variants like '1', '1.0')
                filtered = chunk[
                    (chunk['warning_clean'] == '1') | 
                    (chunk['warning_clean'] == '1.0')
                ]
                filtered = filtered.drop(columns=['warning_clean'])
                
                if not filtered.empty:
                    vital_chunks.append(filtered)
                    processed_rows += len(filtered)
            else:
                print("Warning: 'warning' column not found in chartevents.csv")
                break
        
        if vital_chunks:
            vital_df = pd.concat(vital_chunks, ignore_index=True)
            output_path = RAW_DATA_DIR / "chartevents_extract.csv"
            vital_df.to_csv(output_path, index=False)
            print(f"   Original rows: {total_rows:,}")
            print(f"   Extracted {len(vital_df):,} rows with warning=1")
            print(f"   Saved to: {output_path}")
        else:
            print(f"   No records with warning=1 found in {total_rows:,} rows")
            # Create empty extract file for consistency
            pd.DataFrame(columns=pd.read_csv(vital_path, nrows=0).columns).to_csv(
                RAW_DATA_DIR / "chartevents_extract.csv", index=False
            )
    else:
        print(f"   File not found: {vital_path}")
    
    # 2. Process lab data (labevents.csv) - flag == 'abnormal'
    print("\n2. Processing lab data (flag='abnormal')...")
    lab_path = RAW_DATA_DIR / "hosp/labevents.csv"
    
    if lab_path.exists():
        lab_chunks = []
        total_rows = 0
        processed_rows = 0
        
        for chunk in pd.read_csv(
            lab_path,
            chunksize=2_000_000,
            low_memory=False
        ):
            total_rows += len(chunk)
            
            # Handle potential data type issues with flag column
            if 'flag' in chunk.columns:
                # Convert to string, lowercase, and strip whitespace
                chunk['flag_clean'] = chunk['flag'].astype(str).str.lower().str.strip()
                
                # Filter for flag='abnormal'
                filtered = chunk[chunk['flag_clean'] == 'abnormal']
                filtered = filtered.drop(columns=['flag_clean'])
                
                if not filtered.empty:
                    lab_chunks.append(filtered)
                    processed_rows += len(filtered)
            else:
                print("Warning: 'flag' column not found in labevents.csv")
                break
        
        if lab_chunks:
            lab_df = pd.concat(lab_chunks, ignore_index=True)
            output_path = RAW_DATA_DIR / "labevents_extract.csv"
            lab_df.to_csv(output_path, index=False)
            print(f"   Original rows: {total_rows:,}")
            print(f"   Extracted {len(lab_df):,} rows with flag='abnormal'")
            print(f"   Saved to: {output_path}")
        else:
            print(f"   No records with flag='abnormal' found in {total_rows:,} rows")
            # Create empty extract file for consistency
            pd.DataFrame(columns=pd.read_csv(lab_path, nrows=0).columns).to_csv(
                RAW_DATA_DIR / "labevents_extract.csv", index=False
            )
    else:
        print(f"   File not found: {lab_path}")
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("Extracted files are saved in:", RAW_DATA_DIR)
    print("Naming convention: original_filename + '_extract.csv'")
    print("="*50)

if __name__ == "__main__":
    extract_abnormal_data()