import dotenv
import os
dotenv.load_dotenv()
from pathlib import Path

from typing import Optional, Any, Dict

class LoggerConfig:
    def __init__(self):
        self.level: str = "INFO"
        self.log_file: Optional[str] = None  # e.g. "logs/app.log"

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
            self.MIN_VISITS: int = 15
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
            self.USE_LLM: bool = False
    
    class LabPanelExtract:
        def __init__(self, paths):
            self.INPUT_DIR: Path = paths.BENCH_DATA_DIR / "patients"
            self.OUTPUT_DIR: Path = paths.BENCH_DATA_DIR / "lab_panels"
            self.BATCH_SIZE: int = 120  # 每次给 LLM 的测试数量
            
    def __init__(self):
        self.paths = BuildConfig.Paths()
        self.run = BuildConfig.Run()
        self.eventExtract = BuildConfig.EventExtract(self.paths)
        self.noteExtract = BuildConfig.NoteExtract(self.paths)
        self.patientExtract = BuildConfig.PatientExtract(self.paths)
        self.eventStreamExtract = BuildConfig.EventStreamExtract(self.paths, self.patientExtract, self.eventExtract)
        self.labPanelExtract = BuildConfig.LabPanelExtract(self.paths)
        

class LLMConfig:
    def __init__(self, provider: str = 'qwen', chat_model: str = "qwen-turbo", 
                 embed_model: str = 'text-embedding-v3',
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider: str = provider  # qwen | openai | compatible
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")  # or OPENAI_API_KEY for openai provider
        self.base_url: Optional[str] = os.getenv("OPENAI_API_BASE_URL")  # for openai provider, if using custom base URL

        # Default models (override per call as needed)
        self.chat_model: str = chat_model
        self.embed_model: str = embed_model  # Qwen embedding model name may differ; override if needed
        self.model: str = os.getenv("LLM_MODEL", self.chat_model)

        # Retry and robustness
        self.timeout_s: Optional[float] = None  # OpenAI SDK handles timeouts differently; keep for future extension
        self.max_retries: int = 10
        self.retry_backoff_base_s: float = 0.6
        self.retry_backoff_jitter_s: float = 0.2
        self.max_inflight: int = int(os.getenv("LLM_MAX_INFLIGHT", "8"))
        self.qps: float = float(os.getenv("LLM_QPS", "5"))

        # Default generation params
        if provider == "qwen":
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


class AgentConfig:
    def __init__(self):
        # Agent identity (used by memory providers like mem0)
        self.agent_id: str = os.getenv("AGENT_ID", "bench-agent")
        self.app_id: Optional[str] = os.getenv("AGENT_APP_ID")
        self.run_id: Optional[str] = os.getenv("AGENT_RUN_ID")

        # Memory backend selection
        # mem0 | in_memory | custom
        self.memory_provider: str = os.getenv("MEMORY_PROVIDER", "mem0")

        # Retrieval / context settings
        self.memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "5"))
        self.max_recent_turns: int = int(os.getenv("AGENT_MAX_RECENT_TURNS", "6"))

        # What to store
        self.store_dialog: bool = os.getenv("AGENT_STORE_DIALOG", "1") == "1"
        self.store_observations: bool = os.getenv("AGENT_STORE_OBS", "1") == "1"
        self.include_memory_in_prompt: bool = os.getenv("AGENT_INCLUDE_MEMORY", "1") == "1"

        # Observation extractor settings (optional)
        self.observation_model: Optional[str] = os.getenv("AGENT_OBS_MODEL")

        # System prompt override (optional)
        self.system_prompt: Optional[str] = os.getenv("AGENT_SYSTEM_PROMPT")


class FactQGenConfig:
    def __init__(self):
        build = BuildConfig()
        self.EVENTS_SELECTED_DIR: Path = Path("./tasks/factual_questions/events_selected")
        self.QUESTIONS_OUT_DIR: Path = Path("./tasks/factual_questions/questions_generated")
        self.MODEL: str = os.getenv("QGEN_MODEL", LLMConfig().chat_model)
        self.BATCH_SIZE_LAB: int = int(os.getenv("QGEN_BATCH_LAB", "10"))
        self.BATCH_SIZE_MED: int = int(os.getenv("QGEN_BATCH_MED", "10"))
        self.RANDOM_SEED: int = int(os.getenv("QGEN_SEED", "7"))
        
        
class TimelineGenConfig:
    def __init__(self):
        # 继承基础配置 (如果有的话)
        # build = BuildConfig() 
    
        base = Path(__file__).resolve().parent  # 假设config.py在项目根目录
        self.PATIENTS_SEQ_DIR = base / "bench_data" / "patients_sequence"

        # --- 输出路径 ---
        # 建议把两个子任务分开存放，方便评测脚本读取
        self.TASK_ROOT: Path = Path("./question_data")
        self.MICRO_CLOZE_DIR: Path = self.TASK_ROOT / "visit_cloze"
        self.TRAJECTORY_DIR: Path = self.TASK_ROOT / "trajectory_sorting"

        # --- 参数配置 ---
        # 随机种子，保证每次生成的结果一致
        self.RANDOM_SEED: int = int(os.getenv("TIMELINE_SEED", "42"))
        
        # 排序任务：
        # 窗口 (默认 5 个 Visit)
        self.TRAJECTORY_WINDOW_SIZE: int = int(os.getenv("TIMELINE_WINDOW", "5"))
        # 步长 (默认2)
        self.TRAJECTORY_STRIDE: int = int(os.getenv("TIMELINE_STRIDE", "2"))
        
        # 填空任务：至少需要几个 targets 才能构成一个有效题目
        self.MIN_TARGETS_FOR_CLOZE: int = 4

class AgentQaGenConfig:
    def __init__(self):
        self.INPUT_DIR: Path = Path("./bench_data/patients")
        self.OUTPUT_PATH: Path = Path("./tasks/agentic_decision/questions_generated/")
        self.INDICATOR_PANEL_MAP: Path = Path("./bench_data/lab_panels/panel_to_indicators.json")
        self.K_ACTION: int = 6
        self.K_PARAM: int = 10
        self.K_MED: int = 10
        self.ENABLE_DISCHARGE_Q: bool = True
        self.DISCHARGE_XH: float = 6.0
        self.DISCHARGE_ONLY_WITHIN_H: float = 48.0
        self.RANDOM_SEED: int = 42
        
        self.DEMO_MODE: bool = False
        self.DEMO_N: int = 5
        
        self.AGENT_TASK_STARTING_VISIT: int = 9
        
        self.MAX_WORKERS: int = 16
        
        self.DISCHARGE_YES_RATIO = 0.7          # 目标 yes 比例
        self.DISCHARGE_MIN_GAP_H = 0.5          # 离出院至少多少小时，避免贴太近（可选）
        self.DISCHARGE_NO_MARGIN_H = 6.0        # No 采样在 (X, X+margin] 区间内（靠近但为 No）
        self.DISCHARGE_SAMPLE_K = 1             # 每个 visit 生成几个 T3-D
        
class AgentTaskConfig:
    def __init__(self):
        self.QUESTIONS_DIR: Path = Path("./tasks/agentic_decision/questions_generated")
        self.PATIENTS_DIR: Path = Path("./bench_data/patients")
        self.EVENT_SEQ_DIR: Path = Path("./bench_data/patients_sequence")
        
        self.MAXWORKERS: int = 16
        self.MAX_VISIBALE_VISITS: int = 10
        self.MAX_KNOWN_FACTS: int = 10
        self.MEMORY_TYPE: str = "report" # report | event_stream
        self.MAX_EVENTS_PER_VISIT = 9999
        
        self.CONTEXT_DIR: Path = Path("./tasks/agentic_decision/context")
        
        self.DEMO_MODE: bool = True
        self.DEMO_N: int = 5