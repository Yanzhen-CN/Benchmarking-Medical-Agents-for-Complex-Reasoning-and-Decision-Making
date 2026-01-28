import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# --- 配置部分 ---
TS_FMT = "%Y-%m-%d %H:%M:%S"
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "bench_data" / "patients"
OUTPUT_DIR = ROOT / "bench_data" / "patients_sequence"

# 黑名单字段（这些会被清洗掉，不进入 Content）
BLACKLIST_KEYS = {
    "subject_id", "hadm_id", "stay_id", "transfer_id", 
    "event_id", "charttime", "type" 
}

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts: return None
    try:
        return datetime.strptime(ts, TS_FMT)
    except ValueError:
        return None

def _clean_content(data: Dict[str, Any]) -> Dict[str, Any]:
    """清洗 content，剔除黑名单里的 ID"""
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k not in BLACKLIST_KEYS}

def _make_event(
    event_id_str: str,
    event_type: str,
    timestamp: Optional[str],
    content: Any,
    visit_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    构造最终 Event
    """
    return {
        "event_id": event_id_str, # 直接使用传入的格式化ID
        "event_type": event_type,
        "timestamp": timestamp,
        "visit_ref": visit_label,
        "content": content
    }

def convert_single_patient(json_path: Path) -> Optional[Path]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取失败 {json_path}: {e}")
        return None

    patient_info = data.get("patient_info") or {}
    patient_id = patient_info.get("patient_id", "Unknown")
    visits = data.get("visits") or []
    
    event_stream = []

    # ==========================================
    # 1. 静态背景 -> ID: {PID}-V0
    # ==========================================
    p_content = _clean_content(patient_info)
    # 移除 patient_id 本身，因为 ID 已经体现在 event_id 里了
    if "patient_id" in p_content: del p_content["patient_id"]

    event_stream.append(_make_event(
        event_id_str=f"{patient_id}-V0", # <--- 修改点：V0
        event_type="PATIENT_DEMOGRAPHICS",
        timestamp=None, 
        content=p_content,
        visit_label="V0" # 标记为 V0
    ))

    # ==========================================
    # 2. 动态轨迹 -> ID: {PID}-{Vn}-{Seq}
    # ==========================================
    for v_idx, visit in enumerate(visits):
        visit_label = f"V{v_idx+1}"
        
        # 每个 Visit 内部单独计数，从 1 开始
        visit_internal_counter = 0 
        
        # --- A. ADMISSION (头部) ---
        adm_info = visit.get("admission_info") or {}
        adm_note = adm_info.get("admission_note") or {}
        
        adm_content = {
            "location": adm_info.get("admission_location"),
            "admission_type": adm_info.get("admission_type"),
            "chief_complaint": adm_note.get("chief_complaint"),
            "history_of_present_illness": adm_note.get("history_of_present_illness"),
            "insurance": adm_info.get("insurance")
        }

        event_stream.append(_make_event(
            event_id_str=f"{patient_id}-{visit_label}-adm", # <--- 修改点
            event_type="ADMISSION",
            timestamp=adm_info.get("admission_time"),
            content=adm_content,
            visit_label=visit_label
        ))

        # --- B. EVENT STREAM (中间) ---
        raw_events = visit.get("event_stream") or []
        # 按时间排序
        raw_events.sort(key=lambda x: _parse_ts(x.get("charttime")) or datetime.min)

        for event in raw_events:
            e_type = str(event.get("type", "UNKNOWN")).upper()
            e_content = _clean_content(event)
            
            visit_internal_counter += 1
            event_stream.append(_make_event(
                event_id_str=f"{patient_id}-{visit_label}-E{visit_internal_counter}", # <--- 修改点
                event_type=e_type,
                timestamp=event.get("charttime"),
                content=e_content,
                visit_label=visit_label
            ))

        # --- C. DISCHARGE (尾部) ---
        dis_info = visit.get("discharge_info")
        if dis_info:
            dis_time = dis_info.get("discharge_time")
            # 时间补全逻辑
            if not dis_time and raw_events:
                dis_time = raw_events[-1].get("charttime")

            dis_content = _clean_content(dis_info)

            event_stream.append(_make_event(
                event_id_str=f"{patient_id}-{visit_label}-dis", # <--- 修改点
                event_type="DISCHARGE",
                timestamp=dis_time,
                content=dis_content,
                visit_label=visit_label
            ))

    # ==========================================
    # 3. 写入文件
    # ==========================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{json_path.stem}_sequenced.json"
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(event_stream, f, indent=2, ensure_ascii=False)
    
    return out_path

def batch_convert():
    if not INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在: {INPUT_DIR}")
        return

    files = sorted(INPUT_DIR.glob("P*.json"))
    # 重新生成所有文件，因为 ID 格式变了，覆盖旧的 sequenced
    # target_files = [p for p in files if "_sequenced" not in p.name] 
    # 这里建议全量重新跑一遍，去掉过滤条件，或者手动删除旧文件
    target_files = [p for p in files if "_sequenced" not in p.name]

    print(f"找到 {len(target_files)} 个源文件，准备转换...")
    
    count = 0
    for p in target_files:
        out = convert_single_patient(p)
        if out:
            print(f"  -> 生成: {out.name}")
            count += 1
            
    print(f"\n全部完成，共转换 {count} 个病人文件。")

if __name__ == "__main__":
    batch_convert()