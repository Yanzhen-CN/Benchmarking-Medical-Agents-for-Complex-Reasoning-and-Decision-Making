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
import argparse
import json
import glob
from tqdm import tqdm
from datetime import datetime

try:
    from config import TimelineGenConfig
except ImportError:
    # Fallback
    class TimelineGenConfig:
        def __init__(self):
            self.RANDOM_SEED = 42
            self.TRAJECTORY_WINDOW_SIZE = 5
            self.TRAJECTORY_STRIDE = 1
            self.PATIENTS_SEQ_DIR = Path("./EHR_pipeline/bench_data/patients_sequence")
            self.TRAJECTORY_DIR = Path("./tasks/task2/trajectory_sorting")

# ... extract_visit_summary 保持不变 ...
def extract_visit_summary(visit_events):
    summary = {
        "start_info": "N/A", "end_info": "N/A", "admission_time": None, "discharge_time": None
    }
    def parse_ts(ts_str):
        if not ts_str: return None
        try: return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except: return None

    for event in visit_events:
        etype = event.get('event_type', '').upper()
        content = event.get('content', {})
        ts_str = event.get('timestamp')
        if etype == 'ADMISSION':
            summary['admission_time'] = parse_ts(ts_str)
            text = str(content)
            if isinstance(content, dict):
                hpi = content.get('history_of_present_illness', '')
                cc = content.get('chief_complaint', '')
                text = hpi if hpi else f"CC: {cc}"
            summary['start_info'] = text[:1500]
        elif etype == 'DISCHARGE':
            summary['discharge_time'] = parse_ts(ts_str)
            text = str(content)
            if isinstance(content, dict):
                diag = content.get('discharge_diagnosis', '')
                instr = content.get('discharge_instructions', '')
                text = f"Diagnosis: {diag}\nInstructions: {instr}"
            summary['end_info'] = text[:1500]
    if summary['admission_time'] and not summary['discharge_time']:
        summary['discharge_time'] = summary['admission_time']
    return summary

def generate_trajectory_tasks_for_patient(patient_events, patient_id, config=None):
    if config is None: config = TimelineGenConfig()
    
    window_size = config.TRAJECTORY_WINDOW_SIZE
    stride = config.TRAJECTORY_STRIDE

    # 1. 整理 Valid Visits
    visits = {}
    for event in patient_events:
        v_ref = event.get('visit_ref')
        if not v_ref or v_ref == 'V0': continue
        if v_ref not in visits: visits[v_ref] = []
        visits[v_ref].append(event)
    
    valid_visits = []
    for v_ref, events in visits.items():
        summ = extract_visit_summary(events)
        if summ['admission_time'] and summ['end_info'] != "N/A":
            valid_visits.append({
                "visit_ref": v_ref,
                "summary": summ,
                "timestamp": summ['admission_time']
            })
            
    valid_visits.sort(key=lambda x: x['timestamp'])
    
    n_visits = len(valid_visits)
    # 如果总长度都不够一个窗口，直接跳过
    if n_visits < window_size:
        return []

    # ================= 核心逻辑：计算所有需要的 Start Index =================
    # 1. 正常的滑动窗口索引
    start_indices = list(range(0, n_visits - window_size + 1, stride))
    
    # 2. 强制包含“最后5个” (The Last Window Guarantee)
    # 只有当正常的滑动窗口没有恰好覆盖到终点时才添加
    last_possible_start = n_visits - window_size
    if start_indices[-1] != last_possible_start:
        start_indices.append(last_possible_start)
    
    # 去重 (以防万一)
    start_indices = sorted(list(set(start_indices)))
    # ====================================================================

    generated_tasks = []
    
    for start_idx in start_indices:
        end_idx = start_idx + window_size
        chunk = valid_visits[start_idx : end_idx]
        
        # 3. 展开成 Items (Admission + Discharge)
        chronological_items = []
        for visit in chunk:
            chronological_items.append({
                "original_ref": f"{visit['visit_ref']}-ADM",
                "content": f"[ADMISSION SUMMARY] {visit['summary']['start_info']}",
                "type": "ADMISSION"
            })
            chronological_items.append({
                "original_ref": f"{visit['visit_ref']}-DIS",
                "content": f"[DISCHARGE INSTRUCTIONS] {visit['summary']['end_info']}",
                "type": "DISCHARGE"
            })
            
        num_items = len(chronological_items)
        total_pairs = (num_items * (num_items - 1)) // 2
        
        # 4. Shuffle & Generate Options
        shuffled_indices = list(range(num_items))
        random.shuffle(shuffled_indices)
        
        input_options = []
        for display_id, real_idx in enumerate(shuffled_indices):
            input_options.append({
                "option_id": display_id,
                "content": chronological_items[real_idx]['content']
            })
            
        real_to_display = {real_idx: display_id for display_id, real_idx in enumerate(shuffled_indices)}
        correct_sequence = []
        for real_idx in range(num_items):
            correct_sequence.append(real_to_display[real_idx])
            
        task = {
            "task_type": "trajectory_sorting",
            "patient_id": patient_id,
            "chunk_id": f"{start_idx}_to_{end_idx-1}", 
            "question_text": f"Below are {num_items} clinical snippets (Admission Summaries and Discharge Instructions) labeled 0 to {num_items-1}. Sort them into the correct chronological order.",
            "input_options": input_options,
            "ground_truth": {
                "correct_order": correct_sequence,
                "debug_info": [item['original_ref'] for item in chronological_items]
            },
            "meta": {
                "num_visits": window_size,
                "num_items": num_items,
                "total_pairs": total_pairs,
                "window_start_idx": start_idx,
                "is_last_window": (start_idx == last_possible_start) # 标记一下这是不是最后一段
            }
        }
        generated_tasks.append(task)
    
    return generated_tasks

if __name__ == "__main__":
    cfg = TimelineGenConfig()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(cfg.PATIENTS_SEQ_DIR))
    parser.add_argument("--output_dir", type=str, default=str(cfg.TRAJECTORY_DIR))
    args = parser.parse_args()

    print(f"[Config] Seed: {cfg.RANDOM_SEED}")
    print(f"[Config] Window Size: {cfg.TRAJECTORY_WINDOW_SIZE}")
    print(f"[Config] Stride: {cfg.TRAJECTORY_STRIDE}")
    
    random.seed(cfg.RANDOM_SEED)
    
    if not os.path.exists(args.input_dir):
        print("Input dir not found")
        exit(1)

    files = glob.glob(os.path.join(args.input_dir, "P*_sequenced.json"))
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "trajectory_tasks.jsonl")

    count = 0
    with open(out_file, 'w', encoding='utf-8') as f_out:
        for fpath in tqdm(files, desc="Trajectory Sorting"):
            try:
                pid = os.path.basename(fpath).split('_')[0]
                with open(fpath, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                tasks = generate_trajectory_tasks_for_patient(data, pid, cfg)
                for task in tasks:
                    f_out.write(json.dumps(task, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                print(f"Error: {e}")

    print(f"Saved {count} tasks to {out_file}")