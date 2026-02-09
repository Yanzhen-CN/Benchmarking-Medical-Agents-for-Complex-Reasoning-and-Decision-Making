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

def is_precise_timestamp(ts_str):
    """
    判断时间戳精度。
    MIMIC ETL 约定：
    - "YYYY-MM-DD" (len=10) -> 模糊时间 (Fuzzy/Date-level)
    - "YYYY-MM-DD HH:MM:SS" (len=19) -> 精确时间 (Precise/Minute-level)
    """
    if not ts_str: 
        return False
    return len(str(ts_str)) > 10

def generate_cloze_tasks_for_patient(patient_events, patient_id, config=None):
    """
    生成完形填空任务 (返回任务列表)
    逻辑升级：
    1. Anchors: 观测事件 + 只有日期的模糊手术 (作为背景)
    2. Targets: 只有具备精确时间戳的药物和手术 (作为考题)
    """
    if config is None: config = TimelineGenConfig()
    
    tasks = []
    
    # 按 Visit 分组
    visits = {}
    for event in patient_events:
        v_ref = event.get('visit_ref')
        if not v_ref or v_ref == 'V0': continue
        if v_ref not in visits: visits[v_ref] = []
        visits[v_ref].append(event)
        
    for v_ref, events in visits.items():
        # 预排序：保证 Anchors 在 Input 中是按时间顺序呈现的
        # Python 字符串比较： "2020-01-01" < "2020-01-01 10:00:00"
        # 意味着同日期的模糊背景会排在当天精确事件之前，符合阅读逻辑
        events.sort(key=lambda x: x.get('timestamp') or "")
        
        anchors = []
        targets = []
        
        for event in events:
            etype = event.get('event_type', '').upper()
            content = event.get('content', '')
            timestamp = event.get('timestamp')
            
            # 基础观测类事件 -> 永远是 Anchors
            if etype in ['ADMISSION', 'DISCHARGE', 'LAB', 'MICROBIOLOGY', 'VITAL', 'IMAGING']:
                anchors.append({
                    "event_id": event['event_id'],
                    "type": "anchor",
                    "event_type": etype,
                    "timestamp": timestamp,
                    "content": content
                })
            
            # 干预类事件 (MEDICATION, PROCEDURE) -> 根据精度分流
            elif etype in ['MEDICATION', 'PROCEDURE']:
                if not timestamp: continue

                # === 核心修改逻辑 ===
                if is_precise_timestamp(timestamp):
                    # 情况 A: 精确时间 -> 做成 Target (考题)
                    targets.append({
                        "event_id": event['event_id'],
                        "type": "target",
                        "event_type": etype,
                        "content": content,
                        "original_timestamp": timestamp # 暂时保留用于生成 Ground Truth
                    })
                else:
                    # 情况 B: 模糊时间 (只有日期) -> 降级为 Anchor (背景)
                    # 例如：Hosp Procedures (Billing ICD)
                    anchors.append({
                        "event_id": event['event_id'],
                        "type": "anchor", # 标记为背景
                        "event_type": etype,
                        "timestamp": timestamp,
                        "content": content
                    })
        
        # 检查 Target 数量是否足够生成一道题
        if len(targets) < config.MIN_TARGETS_FOR_CLOZE: 
            continue
            
        # 生成 Ground Truth (基于精确时间的顺序)
        ground_truth_order = [t['event_id'] for t in targets] 
        
        # 打乱 Targets
        shuffled_targets = copy.deepcopy(targets)
        random.shuffle(shuffled_targets)
        
        # 移除 Targets 中的时间戳 (防止泄题)
        for t in shuffled_targets:
            if 'original_timestamp' in t:
                del t['original_timestamp']
            # 注意：event 对象里本身可能有 timestamp 字段，如果是深拷贝过来的也要清理
            if 'timestamp' in t:
                del t['timestamp']
            
        task = {
            "task_type": "micro_cloze",
            "patient_id": patient_id,
            "visit_ref": v_ref,
            "question_text": "Analyze the clinical timeline provided in the 'anchors'. Determine the most logical chronological position for each item in the 'shuffled_targets'. Note that some procedures in anchors may only have date-level precision, serving as context.",
            "input_context": {
                "anchors": anchors,
                "shuffled_targets": shuffled_targets
            },
            "ground_truth": {
                "correct_order": ground_truth_order
            },
            "meta": {
                "num_anchors": len(anchors),
                "num_targets": len(targets)
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
    print(f"[Config] Min Targets: {cfg.MIN_TARGETS_FOR_CLOZE}")
    random.seed(cfg.RANDOM_SEED)

    if not os.path.exists(args.input_dir):
        print(f"Error: Input dir {args.input_dir} not found.")
        exit(1)

    files = glob.glob(os.path.join(args.input_dir, "P*_sequenced.json"))
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "micro_cloze_tasks.jsonl")

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
                    f_out.write(json.dumps(task, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

    print(f"Saved {count} tasks to {out_file}")