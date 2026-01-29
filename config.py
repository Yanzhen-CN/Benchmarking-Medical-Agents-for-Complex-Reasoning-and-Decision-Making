import dotenv
import os
dotenv.load_dotenv()
from pathlib import Path

from typing import Optional, Any, Dict

class BuildConfig:
    class Paths:
        def __init__(self):
            # 原始 MIMIC-IV 数据目录

            self.RAW_DATA_DIR: Path = Path("./raw_data")
            # 生成的 benchmark 数据目录
            self.BENCH_DATA_DIR: Path = Path("./bench_data")
        
    class Run:
        def __init__(self):
            # 至少需要多少次住院，才算一个 longitudinal patient
            self.MIN_VISITS: int = 10
            # True：只分析 cohort，不生成 JSON
            self.ANALYZE_MODE: bool = False
            # Demo 模式：只取前 N 个病人，方便调试
            self.DEMO_MODE: bool = True
            self.DEMO_N: int = 5
            self.MAX_WORKERS: int = 16  # 并行处理时的最大线程数
        
    # PREPROCESSING: 事件抽取
    class EventExtract:
        def __init__(self, paths):
            # 生命体征 & 化验结果 异常值阈值文件
            self.CHARTEVENTS_IN_PATH: Path = paths.RAW_DATA_DIR / "icu" / "chartevents.csv"
            self.CHARTEVENTS_OUT_PATH: Path   = paths.RAW_DATA_DIR / "chartevents_extract.csv"
            
            self.LABEVENTS_IN_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "labevents.csv"
            self.LABEVENTS_OUT_PATH: Path   = paths.RAW_DATA_DIR / "labevents_extract.csv"
    
    # STEP1: 构建 Benchmark 患者数据
    class PatientExtract:
        def __init__(self, paths):
             # 原始数据文件路径
            self.ADMISSIONS_FILE: Path = paths.RAW_DATA_DIR / "hosp" / "admissions.csv"
            self.PATIENTS_FILE: Path   = paths.RAW_DATA_DIR / "hosp" / "patients.csv"
            self.DIAGNOSES_FILE: Path  = paths.RAW_DATA_DIR / "hosp" / "diagnoses_icd.csv"
            self.D_ICD_FILE: Path      = paths.RAW_DATA_DIR / "hosp" / "d_icd_diagnoses.csv"
            
            # 分析报告路径
            self.COHORT_ANALYSIS_OUT_PATH: Path = paths.BENCH_DATA_DIR / "cohort_analysis"
            
            # 患者数据输出路径
            self.PATIENT_OUTPUT_PATH: Path = paths.BENCH_DATA_DIR / "patients"
            self.PATIENT_INDEX_OUTPUT_PATH: Path = paths.BENCH_DATA_DIR
            
            # 出院笔记文件路径
            self.DISCHARGE_NOTES_FILE: Path = paths.RAW_DATA_DIR / "note" / "discharge.csv"
    # STEP2: 事件流构建
    class EventStreamExtract:
        def __init__(self, paths, patients, events):
            # 病人文件目录
            self.PATIENT_PATH: Path = patients.PATIENT_OUTPUT_PATH
            self.PATIENT_SEQUENCE_PATH: Path = paths.BENCH_DATA_DIR / "patients_sequence"
            # 生命体征 & 化验结果 抽取文件路径
            self.CHARTEVENTS_EXTRACT_PATH: Path = events.CHARTEVENTS_OUT_PATH
            self.LABEVENTS_EXTRACT_PATH: Path   = events.LABEVENTS_OUT_PATH
            self.D_ITEMS_PATH: Path             = paths.RAW_DATA_DIR / "icu" / "d_items.csv"
            
            # 用药记录文件路径
            self.PRESCRIPTIONS_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "prescriptions.csv" # RAW_DATA_DIR / "prescriptions_extract.csv"
            # 影像学报告文件路径
            self.RADIOLOGY_REPORTS_PATH: Path = paths.RAW_DATA_DIR / "note" / "radiology.csv" # RAW_DATA_DIR / "radiology_extract.csv"
            # 手术记录文件路径
            self.PROCEDURES_ICD_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "procedures_icd.csv"
            # 手术代码描述文件路径
            self.D_PROCEDURES_ICD_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "d_icd_procedures.csv"
            
            self.MICROBIOLOGYEVENTS_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "microbiologyevents.csv"
            
            self.EMAR_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "emar.csv"
            self.EMAR_DETAIL_PATH: Path = paths.RAW_DATA_DIR / "hosp" / "emar_detail.csv"
            
    # STEP3: 笔记抽取
    class NoteExtract:
        def __init__(self, paths):
            # 出院笔记文件路径
            self.DISCHARGE_NOTES_PATH: Path = paths.RAW_DATA_DIR / "note" / "discharge.csv"
            # 患者索引文件路径
            self.PATIENT_INDEXES_PATH: Path = paths.BENCH_DATA_DIR / "patient_index.csv"
            self.PATIENT_OUTPUT_PATH: Path = paths.BENCH_DATA_DIR / "patients"
            
            # 是否使用LLM辅助切分笔记
            self.USE_LLM: bool = True
            
    def __init__(self):
        self.paths = BuildConfig.Paths()
        self.run = BuildConfig.Run()
        self.eventExtract = BuildConfig.EventExtract(self.paths)
        self.noteExtract = BuildConfig.NoteExtract(self.paths)
        self.patientExtract = BuildConfig.PatientExtract(self.paths)
        self.eventStreamExtract = BuildConfig.EventStreamExtract(self.paths, self.patientExtract, self.eventExtract)
        

class LLMConfig:
    def __init__(self):
        self.provider: str = "qwen"  # qwen | openai | compatible
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")  # or OPENAI_API_KEY for openai provider
        self.base_url: Optional[str] = os.getenv("OPENAI_API_BASE_URL")  # for openai provider, if using custom base URL

        # Default models (override per call as needed)
        self.chat_model: str = "qwen-turbo"
        self.embed_model: str = "text-embedding-v3"  # Qwen embedding model name may differ; override if needed

        # Retry and robustness
        self.timeout_s: Optional[float] = None  # OpenAI SDK handles timeouts differently; keep for future extension
        self.max_retries: int = 10
        self.retry_backoff_base_s: float = 0.6
        self.retry_backoff_jitter_s: float = 0.2

        # Default generation params
        self.temperature: float = 0.0
        self.top_p: Optional[float] = None
        self.max_tokens: Optional[int] = None
        
class ContextConfig:
    def __init__(self):
        build = BuildConfig()
        self.MAX_WORKERS: int = 16
        self.SEQUENCE_IN_PATH: Path = build.eventStreamExtract.PATIENT_SEQUENCE_PATH
        self.CONTEXT_OUT_DIR: Path = build.paths.BENCH_DATA_DIR / "context"
        
        self.USE_LLM_FOR_IMAGE_DESC: bool = True
        self.IMAGE_DESC_MODEL: str = "qwen-turbo"
        self.IMAGE_DESC_THRESHOLD: float = 0.8
        
        self.USE_LLM_FOR_REASON: bool = True
        self.REASON_MODEL: str = "qwen-turbo"
        
        self.llm_config: LLMConfig = LLMConfig()
        
        