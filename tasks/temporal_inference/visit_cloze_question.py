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

try:
    from config import TimelineGenConfig
except ImportError:
    print("TimelineGenConfig not found, creat now")
    class TimelineGenConfig:
        def __init__(self):
            self.RANDOM_SEED = 42
            self.MIN_TARGETS_FOR_CLOZE = 4
            self.PATIENTS_SEQ_DIR = PROJECT_ROOT / "EHR_pipeline" / "bench_data" / "patients_sequence"
            self.TRAJECTORY_DIR = PROJECT_ROOT / "question_data" / "visit_cloze"

def is_precise_timestamp(ts_str):
    if not ts_str: return False
    return len(str(ts_str)) > 10

def generate_cloze_lines(patient_events, patient_id, config=None):
    if config is None: config = TimelineGenConfig()
    
    jsonl_lines = []
    # === 初始化 ID 计数器 ===
    id_counter = 0
    
    visits = {}
    for event in patient_events:
        v_ref = event.get('visit_ref')
        if not v_ref or v_ref == 'V0': continue
        if v_ref not in visits: visits[v_ref] = []
        visits[v_ref].append(event)
        
    for v_ref, events in visits.items():
        events.sort(key=lambda x: x.get('timestamp') or "")
        
        timeline_stream = [] 
        chronological_targets = [] 
        gap_counter = 0

        for event in events:
            etype = event.get('event_type', '').upper()
            timestamp = event.get('timestamp')
            evt_id = event.get('event_id', 'UNK_ID')
            
            is_target_candidate = False
            if etype in ['MEDICATION', 'PROCEDURE', 'SURGERY']:
                if timestamp and is_precise_timestamp(timestamp):
                    is_target_candidate = True
            
            if is_target_candidate:
                # Gap (挖空)
                timeline_stream.append({
                    "type": "GAP",
                    "gap_index": gap_counter,
                    "original_event_id": evt_id, # Meta: 记录原始ID
                    "note": "Missing clinical event here"
                })
                gap_counter += 1
                
                target_content = copy.deepcopy(event)
                target_content['_original_ts'] = timestamp
                if 'timestamp' in target_content: del target_content['timestamp']
                chronological_targets.append(target_content)
                
            else:
                # Anchor (保留)
                anchor_content = {
                    "event_type": etype,
                    "event_id": evt_id, # 显式保留
                    "timestamp": timestamp,
                    "content": event.get("content"),
                    "type": "ANCHOR"
                }
                timeline_stream.append(anchor_content)
        
        if len(chronological_targets) < config.MIN_TARGETS_FOR_CLOZE:
            continue
            
        for rank, item in enumerate(chronological_targets):
            item['_real_rank'] = rank
            
        shuffled_targets = copy.deepcopy(chronological_targets)
        random.shuffle(shuffled_targets)
        
        options_dict = {}
        display_to_real = {}
        options_meta_map = {}
        
        for display_idx, item in enumerate(shuffled_targets):
            real_rank = item.pop('_real_rank')
            if '_original_ts' in item: del item['_original_ts']
            
            str_idx = str(display_idx)
            options_dict[str_idx] = item
            display_to_real[display_idx] = real_rank
            # Meta: Option -> Event ID
            options_meta_map[str_idx] = item.get('event_id')

        # 1. 生成 Fact
        jsonl_lines.append({
            "type": "fact",
            "id": id_counter, # 纯数字
            "data": timeline_stream 
        })
        id_counter += 1 # 自增
        
        # Ground Truth
        real_to_display = {v: k for k, v in display_to_real.items()}
        correct_sequence = []
        for r in range(len(chronological_targets)):
            correct_sequence.append(real_to_display[r])

        # 2. 生成 Question
        num_opts = len(chronological_targets)
        question_text = (
            f"The clinical timeline (fact data) contains {num_opts} marked gaps (type 'GAP'). "
            f"Below are {num_opts} extracted events (Options 0 to {num_opts-1}) that belong to these gaps. "
            "Based on clinical logic and the surrounding anchors, match each option to its correct chronological gap.\n"
            "Output strictly a JSON list of integers, where the i-th integer is the Option ID for the i-th Gap.\n"
            "Example: [2, 0, 1] means Gap 0 takes Option 2, Gap 1 takes Option 0..."
        )
        
        jsonl_lines.append({
            "type": "question",
            "id": id_counter, # 纯数字
            "data": {
                "question": question_text,
                "options": options_dict
            },
            "ground_truth": correct_sequence,
            "meta": {
                "gap_count": num_opts,
                "options_map": options_meta_map # Option ID -> Event ID
            }
        })
        id_counter += 1 # 自增
        
    return jsonl_lines

if __name__ == "__main__":
    cfg = TimelineGenConfig()
    
    parser = argparse.ArgumentParser(description="Generate Micro-Cloze Tasks")
    parser.add_argument("--input_dir", type=str, default=str(cfg.PATIENTS_SEQ_DIR))
    parser.add_argument("--output_dir", type=str, default=str(cfg.MICRO_CLOZE_DIR))
    args = parser.parse_args()

    print(f"[Config] Seed: {cfg.RANDOM_SEED}")
    random.seed(cfg.RANDOM_SEED)

    if not os.path.exists(args.input_dir):
        print(f"Error: Input dir {args.input_dir} not found.")
        exit(1)

    files = glob.glob(os.path.join(args.input_dir, "P*_sequenced.json"))
    os.makedirs(args.output_dir, exist_ok=True)
    
    count = 0
    for fpath in tqdm(files, desc="Micro Cloze"):
        try:
            pid = os.path.basename(fpath).split('_')[0]
            with open(fpath, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
            
            lines = generate_cloze_lines(data, pid, cfg)
            
            if lines:
                out_name = f"{pid}.jsonl"
                out_path = os.path.join(args.output_dir, out_name)
                with open(out_path, 'w', encoding='utf-8') as f_out:
                    for line in lines:
                        f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    print(f"Saved tasks for {count} patients to {args.output_dir}")