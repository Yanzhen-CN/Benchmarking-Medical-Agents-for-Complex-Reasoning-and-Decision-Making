import sys
import os
from pathlib import Path

# ================= 路径修复 =================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ===========================================

import random
import copy
import argparse
import json
import glob
from tqdm import tqdm
from datetime import datetime

# 引入配置
try:
    from config import TimelineGenConfig
except ImportError:
    print("TimelineGenConfig not found, creat now")
    class TimelineGenConfig:
        def __init__(self):
            self.RANDOM_SEED = 42
            self.TRAJECTORY_WINDOW_SIZE = 5
            self.TRAJECTORY_STRIDE = 1
<<<<<<< HEAD
            self.PATIENTS_SEQ_DIR = PROJECT_ROOT / "bench_data" / "patients_sequence"
=======
            self.PATIENTS_SEQ_DIR = PROJECT_ROOT / "EHR_pipeline" / "bench_data" / "patients_sequence"
>>>>>>> d3eae85cdfc9f31ff4e3c7c9fc8dabb65610a363
            self.TRAJECTORY_DIR = PROJECT_ROOT / "question_data" / "trajectory_sorting"

def parse_ts(ts_str):
    if not ts_str: return None
    try: return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except: return None

def is_valid_text(text):
    if not text: return False
    s = str(text).strip()
    if not s or s.upper() in ["N/A", "NONE", "NULL", "UNKNOWN"]:
        return False
    return len(s) > 5

def extract_split_items(visit_events, visit_ref):
    """
    拆分 Admission/Discharge 并携带原始 event_id
    """
    sorted_events = sorted(
        visit_events, 
        key=lambda x: x.get('timestamp') or "9999-12-31"
    )

    items = []
    
    # --- 1. Admission ---
    adm_event = next((e for e in sorted_events if e.get('event_type') == 'ADMISSION'), None)
    if adm_event:
        start_time = parse_ts(adm_event.get('timestamp'))
        original_id = adm_event.get('event_id', f"{visit_ref}_ADM_UNK")
        
        c = adm_event.get('content', {})
        if isinstance(c, dict):
            parts = []
            if is_valid_text(c.get('chief_complaint')): 
                parts.append(f"Admission Chief Complaint: {c['chief_complaint']}")
            if is_valid_text(c.get('history_of_present_illness')): 
                parts.append(f"Admission History of Present Illness: {c['history_of_present_illness']}")
            
            if parts:
                items.append({
                    "ref": f"{visit_ref}_ADM",
                    "original_event_id": original_id,
                    "content": "\n".join(parts),
                    "timestamp": start_time,
                    "type": "ADMISSION"
                })

    # --- 2. Discharge ---
    dis_event = next((e for e in reversed(sorted_events) if e.get('event_type') == 'DISCHARGE'), None)
    if dis_event:
        end_time = parse_ts(dis_event.get('timestamp'))
        original_id = dis_event.get('event_id', f"{visit_ref}_DIS_UNK")

        c = dis_event.get('content', {})
        if isinstance(c, dict):
            parts = []
            if is_valid_text(c.get('discharge_diagnosis')): 
                parts.append(f"Discharge Diagnosis: {c['discharge_diagnosis']}")
            if is_valid_text(c.get('discharge_instructions')): 
                parts.append(f"Discharge Instructions: {c['discharge_instructions']}")
            
            if parts:
                items.append({
                    "ref": f"{visit_ref}_DIS",
                    "original_event_id": original_id,
                    "content": "\n".join(parts),
                    "timestamp": end_time,
                    "type": "DISCHARGE"
                })

    return items

def generate_jsonl_content(patient_events, patient_id, config=None):
    if config is None: config = TimelineGenConfig()
    
    window_size = config.TRAJECTORY_WINDOW_SIZE
    stride = config.TRAJECTORY_STRIDE

    visits_map = {}
    for event in patient_events:
        v_ref = event.get('visit_ref')
        if not v_ref or v_ref == 'V0': continue
        if v_ref not in visits_map: visits_map[v_ref] = []
        visits_map[v_ref].append(event)
    
    temp_visits = []
    for v_ref, events in visits_map.items():
        ts_strings = [e.get('timestamp') for e in events if e.get('timestamp')]
        if not ts_strings: continue
        min_ts = min(ts_strings)
        temp_visits.append((v_ref, min_ts, events))
    
    temp_visits.sort(key=lambda x: x[1])
    
    chronological_items = []
    valid_visit_count = 0
    for v_ref, _, events in temp_visits:
        visit_items = extract_split_items(events, v_ref)
        if visit_items:
            chronological_items.extend(visit_items)
            valid_visit_count += 1

    total_items = len(chronological_items)
    if valid_visit_count < window_size:
        return []

    item_window_size = window_size * 2 
    item_stride = stride * 2
    
    start_indices = list(range(0, total_items - item_window_size + 1, item_stride))
    if not start_indices and total_items >= item_window_size:
        start_indices = [0]
    
    jsonl_lines = []
    
    # === 初始化 ID 计数器 (纯数字) ===
    id_counter = 0

    for start_idx in start_indices:
        end_idx = start_idx + item_window_size
        chunk = chronological_items[start_idx : end_idx]
        
        if len(chunk) < 4: continue
        
        for rank, item in enumerate(chunk):
            item['_real_rank'] = rank
            
        shuffled_items = copy.deepcopy(chunk)
        random.shuffle(shuffled_items)
        
        options_dict = {}
        display_to_real = {} 
        options_provenance = {}

        for display_idx, item in enumerate(shuffled_items):
            str_idx = str(display_idx)
            options_dict[str_idx] = item['content']
            display_to_real[display_idx] = item['_real_rank']
            # Meta: Option -> Original Event ID
            options_provenance[str_idx] = item.get('original_event_id')
            
        # 1. 生成 Fact
        fact_obj = {
            "type": "fact",
            "id": id_counter, # 纯数字
            "data": options_dict
        }
        jsonl_lines.append(fact_obj)
        id_counter += 1 # 自增

        # 计算 Ground Truth
        sorted_pairs = sorted(display_to_real.items(), key=lambda x: x[1])
        correct_sequence = [k for k, v in sorted_pairs]

        # 2. 生成 Question
        num_opts = len(chunk)
        question_text = (
            f"I have provided a dictionary of {num_opts} clinical snippets (labeled 0 to {num_opts-1}) "
            "representing disjoint parts of a patient's medical history (Admission details and Discharge details). "
            "Based on clinical logic (e.g., Admission comes before Discharge, disease progression), "
            "sort these keys into the correct chronological order.\n"
            "Output strictly a JSON list of integers representing the sequence of keys.\n"
            f"Example format: [0, 2, 1, ...]"
        )

        question_obj = {
            "type": "question",
            "id": id_counter, # 纯数字
            "data": question_text,
            "ground_truth": correct_sequence,
            "meta": {
                "options_map": options_provenance,
                "window_range": f"{start_idx}_to_{end_idx}"
            }
        }
        jsonl_lines.append(question_obj)
        id_counter += 1 # 自增

    return jsonl_lines

if __name__ == "__main__":
    cfg = TimelineGenConfig()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(cfg.PATIENTS_SEQ_DIR))
    parser.add_argument("--output_dir", type=str, default=str(cfg.TRAJECTORY_DIR))
    args = parser.parse_args()

    print(f"[Config] Seed: {cfg.RANDOM_SEED}")
    random.seed(cfg.RANDOM_SEED)

    if not os.path.exists(args.input_dir):
        print(args.input_dir)
        print("Input dir not found")
        exit(1)

    files = glob.glob(os.path.join(args.input_dir, "P*_sequenced.json"))
    os.makedirs(args.output_dir, exist_ok=True)
    
    count = 0
    for fpath in tqdm(files, desc="Trajectory Gen"):
        try:
            pid = os.path.basename(fpath).split('_')[0]
            with open(fpath, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
            
            lines = generate_jsonl_content(data, pid, cfg)
            
            if lines:
                out_name = f"{pid}.jsonl"
                out_path = os.path.join(args.output_dir, out_name)
                with open(out_path, 'w', encoding='utf-8') as f_out:
                    for line in lines:
                        f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    print(f"Generated {count} files in {args.output_dir}")