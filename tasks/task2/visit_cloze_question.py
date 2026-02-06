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

# 引入配置
try:
    from config import TimelineGenConfig
except ImportError:
    class TimelineGenConfig:
        def __init__(self):
            self.RANDOM_SEED = 42
            self.MIN_TARGETS_FOR_CLOZE = 4
            self.PATIENTS_SEQ_DIR = Path("./EHR_pipeline/bench_data/patients_sequence")
            self.MICRO_CLOZE_DIR = Path("./tasks/task2/visit_cloze")

def generate_cloze_tasks_for_patient(patient_events, patient_id, config=None):
    """
    生成完形填空任务 (返回任务列表)
    """
    if config is None: config = TimelineGenConfig()
    
    tasks = []
    
    visits = {}
    for event in patient_events:
        v_ref = event.get('visit_ref')
        if not v_ref or v_ref == 'V0': continue
        if v_ref not in visits: visits[v_ref] = []
        visits[v_ref].append(event)
        
    for v_ref, events in visits.items():
        events.sort(key=lambda x: x.get('timestamp') or "")
        
        anchors = []
        targets = []
        
        for event in events:
            etype = event.get('event_type', '').upper()
            content = event.get('content', '')
            timestamp = event.get('timestamp')
            
            if etype in ['ADMISSION', 'DISCHARGE', 'LAB', 'MICROBIOLOGY', 'VITAL', 'IMAGING']:
                anchors.append({
                    "event_id": event['event_id'],
                    "type": "anchor",
                    "event_type": etype,
                    "timestamp": timestamp,
                    "content": content
                })
            elif etype in ['MEDICATION', 'PROCEDURE', 'SURGERY']:
                if not timestamp: continue
                targets.append({
                    "event_id": event['event_id'],
                    "type": "target",
                    "event_type": etype,
                    "content": content,
                    "original_timestamp": timestamp 
                })
        
        if len(targets) < config.MIN_TARGETS_FOR_CLOZE: 
            continue
            
        ground_truth_order = [t['event_id'] for t in targets] 
        shuffled_targets = copy.deepcopy(targets)
        random.shuffle(shuffled_targets)
        
        for t in shuffled_targets:
            del t['original_timestamp']
            
        task = {
            "task_type": "micro_cloze",
            "patient_id": patient_id,
            "visit_ref": v_ref,
            "question_text": "Analyze the clinical timeline provided in the 'anchors'. Determine the most logical chronological position for each item in the 'shuffled_targets'.",
            "input_context": {
                "anchors": anchors,
                "shuffled_targets": shuffled_targets
            },
            "ground_truth": {
                "correct_order": ground_truth_order
            }
        }
        tasks.append(task)
    return tasks

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
    out_file = os.path.join(args.output_dir, "micro_cloze_tasks.jsonl") # 改为 .jsonl

    print(f"Processing {len(files)} files -> {out_file}")
    
    count = 0
    with open(out_file, 'w', encoding='utf-8') as f_out:
        for fpath in tqdm(files, desc="Micro Cloze"):
            try:
                pid = os.path.basename(fpath).split('_')[0]
                with open(fpath, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                
                tasks = generate_cloze_tasks_for_patient(data, pid, cfg)
                for task in tasks:
                    f_out.write(json.dumps(task, ensure_ascii=False) + "\n") # Line by line
                    count += 1
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

    print(f"Saved {count} tasks to {out_file}")