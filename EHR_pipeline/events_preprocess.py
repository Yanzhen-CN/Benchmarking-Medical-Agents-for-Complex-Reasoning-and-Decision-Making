# STEP 0 / 预处理：只抽取异常生命体征 & 异常化验结果


from __future__ import annotations

from pathlib import Path
import sys
from pathlib import Path as _Path

# Allow running from repo root or EHR_pipeline directory.
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import polars as pl
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from util.logUtil import setup_logger
logger = setup_logger()
from config import BuildConfig
config = BuildConfig()

def _write_empty_like(input_csv: Path, output_csv: Path) -> None:
    """
    Create an empty CSV with the same header as input_csv.
    Useful when no rows match filters (keeps downstream consistent).
    """
    header_df = pd.read_csv(input_csv, nrows=0)
    header_df.to_csv(output_csv, index=False)




def extract_abnormal_data() -> None:
    logger.info("Starting preprocessing (polars): extracting abnormal records")

    # ============================================================
    # 1) chartevents: warning == 1
    # ============================================================
    logger.info("1) Processing vital signs: icu/chartevents.csv (warning==1)")
    
    vital_path = config.eventExtract.CHARTEVENTS_IN_PATH
    vital_out = config.eventExtract.CHARTEVENTS_OUT_PATH

    if not vital_path.exists():
        logger.error(f"File not found: {vital_path}")
    else:
        # 删除旧文件，避免重复 append
        if vital_out.exists():
            vital_out.unlink()

        # 用 scan_csv 懒加载；streaming=True 让执行尽量流式
        lf = (
            pl.scan_csv(vital_path, ignore_errors=True)
            .with_columns(
                pl.col("warning").cast(pl.Utf8).str.strip_chars().alias("warning_clean")
            )
            .filter(pl.col("warning_clean").is_in(["1", "1.0"]))
            .drop("warning_clean")
        )

        # 关键：sink_csv 会边执行边写出（更接近你 pandas chunks 的效果）
        try:
            lf.sink_csv(vital_out)  # Polars 会自动并行解析/执行
            # 检查是否写出为空（文件可能存在但只有 header 或甚至没写）
            if vital_out.exists() and vital_out.stat().st_size > 0:
                logger.success(f"Vital extract done: saved={vital_out}")
            else:
                logger.warning("Vital extract seems empty; writing empty extract file")
                _write_empty_like(vital_path, vital_out)
        except pl.exceptions.ColumnNotFoundError:
            logger.error("Column 'warning' not found in chartevents.csv; aborting vital extraction")

    # ============================================================
    # 2) labevents: flag == 'abnormal'
    # ============================================================
    logger.info("2) Processing labs: hosp/labevents.csv (flag=='abnormal')")
    lab_path = config.eventExtract.LABEVENTS_IN_PATH
    lab_out = config.eventExtract.LABEVENTS_OUT_PATH

    if not lab_path.exists():
        logger.error(f"File not found: {lab_path}")
    else:
        if lab_out.exists():
            lab_out.unlink()

        lf = (
            pl.scan_csv(lab_path, ignore_errors=True)
            .with_columns(
                pl.col("flag").cast(pl.Utf8).str.to_lowercase().str.strip_chars().alias("flag_clean")
            )
            .filter(pl.col("flag_clean") == "abnormal")
            .drop("flag_clean")
        )

        try:
            lf.sink_csv(lab_out)
            if lab_out.exists() and lab_out.stat().st_size > 0:
                logger.success(f"Lab extract done: saved={lab_out}")
            else:
                logger.warning("Lab extract seems empty; writing empty extract file")
                _write_empty_like(lab_path, lab_out)
        except pl.exceptions.ColumnNotFoundError:
            logger.error("Column 'flag' not found in labevents.csv; aborting lab extraction")

    logger.success("Preprocessing complete (polars)")
    logger.info(f"Extracted files saved under: {config.paths.RAW_DATA_DIR}")

if __name__ == "__main__":
    extract_abnormal_data()
