import sys
import os
from pathlib import Path

# ================= 路径修复 =================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ===========================================

import json
import glob
from tqdm import tqdm
import argparse
import random

# 引入配置
from config import TimelineGenConfig
try:
    import visit_cloze_question
    import trajectory_sorting_question
except ImportError:
    from tasks.task2 import visit_cloze_question
    from tasks.task2 import trajectory_sorting_question

def main():
    # 1. 加载配置
    cfg = TimelineGenConfig()
    
    parser = argparse.ArgumentParser(description="Generate Clinical Timeline Tasks (Master Script)")
    parser.add_argument("--input_dir", type=str, default=str(cfg.PATIENTS_SEQ_DIR))
    parser.add_argument("--output_root", type=str, default=str(cfg.TASK_ROOT))
    args = parser.parse_args()

    # 2. 设置随机种子
    print(f"Initializing with Random Seed: {cfg.RANDOM_SEED}")
    print(f"Using Min Targets for Cloze: {cfg.MIN_TARGETS_FOR_CLOZE}")
    random.seed(cfg.RANDOM_SEED)

    # 3. 准备目录和输出文件
    os.makedirs(cfg.MICRO_CLOZE_DIR, exist_ok=True)
    os.makedirs(cfg.TRAJECTORY_DIR, exist_ok=True)

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found.")
        return

    pattern = os.path.join(args.input_dir, "P*_sequenced.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} patient files in {args.input_dir}")

    # 定义输出文件路径 (.jsonl)
    cloze_file_path = cfg.MICRO_CLOZE_DIR / "all_micro_cloze_tasks.jsonl"
    traj_file_path = cfg.TRAJECTORY_DIR / "all_trajectory_tasks.jsonl"

    print(f"Streaming output to:")
    print(f" - {cloze_file_path}")
    print(f" - {traj_file_path}")

    cnt_cloze = 0
    cnt_traj = 0

    # 4. 循环处理并流式写入 (Streaming Write)
    # 同时打开两个文件进行写入，内存占用极低
    with open(cloze_file_path, 'w', encoding='utf-8') as f_cloze, \
         open(traj_file_path, 'w', encoding='utf-8') as f_traj:
        
        for file_path in tqdm(files, desc="Generating Tasks"):
            try:
                filename = os.path.basename(file_path)
                patient_id = filename.split('_')[0] 
                
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    patient_events = json.load(f_in)
                
                # --- Task A (Micro Cloze) ---
                cloze_tasks = visit_cloze_question.generate_cloze_tasks_for_patient(patient_events, patient_id, cfg)
                for task in cloze_tasks:
                    f_cloze.write(json.dumps(task, ensure_ascii=False) + "\n")
                    cnt_cloze += 1
                
                # --- Task B (Trajectory Sorting) ---
                traj_tasks = trajectory_sorting_question.generate_trajectory_tasks_for_patient(patient_events, patient_id, cfg)
                if traj_tasks:
                    for task in traj_tasks:
                        f_traj.write(json.dumps(task, ensure_ascii=False) + "\n")
                        cnt_traj += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # import traceback
                # traceback.print_exc()

    print(f"\nGeneration Complete!")
    print(f"Task A (Micro Cloze): {cnt_cloze} tasks generated.")
    print(f"Task B (Trajectory):  {cnt_traj} tasks generated.")

if __name__ == "__main__":
    main()