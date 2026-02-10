from config import BuildConfig
from util.logUtil import setup_logger
logger = setup_logger()
from EHR_pipeline import (
    events_preprocess, note_extract, patients_extract, 
    es_extract, convert, get_lab_panels#, build_context
)
    
# from context_builder import build_context
config = BuildConfig()

def main():
    if config.run.ANALYZE_MODE:
        logger.info("ANALYZE MODE is ON - No dataset will be built.")
        patients_extract.extract_patient_data()
        return
    logger.info("Starting dataset building process")
    logger.info("Step 0: Preprocessing events")
    events_preprocess.extract_abnormal_data()
    logger.info("Step 1: Constructing patient data")
    patients_extract.extract_patient_data()
    logger.info("Step 2: Extracting Event Stream")
    es_extract.event_stream_extract()
    logger.info("Step 3: Extracting Discharge Notes")
    note_extract.extract_notes()
    logger.info("Step 4: Data cleaning and finalizing")
    convert.batch_convert()
    # logger.info("Step 5: Building context for each patient")
    # build_context.build_context()
    logger.info("Step 5: Extracting lab panels")
    get_lab_panels.get_lab_panels()
    logger.info("Dataset building process completed successfully")

if __name__ == "__main__":
    main()