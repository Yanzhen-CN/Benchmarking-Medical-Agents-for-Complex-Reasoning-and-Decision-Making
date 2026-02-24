import os
import json
import glob
import time
import random
import threading
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 导入 Hulu-Med 模型 ====================
ROOT_DIR = Path(__file__).resolve().parents[2]          # 脚本位于 agents/llm/，向上两级到根目录
sys.path.insert(0, str(ROOT_DIR))                        # 将根目录加入 Python 路径
from util.hulu import TransformersLLMUtil                 # 导入队友写的类
# ============================================================

# ==================== 用户配置区域 ====================
MODEL_PATH = "/data/xzh/Hulu-Med-7B"          # 修改为您的模型路径
TASKS = ["trajectory_sorting", "visit_cloze", "visit_sorting"] # 要运行的任务列表
DEMO_N = 50                                   # 每个任务处理前 N 个患者，None 表示全部
SPECIFIC_PATIENTS = None                      # 指定患者 ID 列表，如 ["P001", "P002"]，None 表示全部
SPECIFIC_ITEMS = None                          # 精确到 item ID，格式见下方说明
ENABLE_THINKING = False                        # 是否启用思考模式
MAX_WORKERS = 1                                # 并发线程数（本地模型建议为1）
TEMPERATURE = 0.1                              # 生成温度
MAX_TOKENS = 512                               # 最大生成 token 数
# ====================================================

# 说明：SPECIFIC_ITEMS 格式示例
# [
#     {"patient": "P001", "ids": ["id1", "id2"]},   # 只处理 P001 的 id1 和 id2
#     {"patient": "P002"}                            # 处理 P002 所有 ids
# ]

# 全局模型实例（单例）
_model_instance = None
_model_lock = threading.Lock()

def get_model():
    """获取全局模型实例（线程安全）"""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                print(f"🚀 Loading Hulu-Med model from {MODEL_PATH}...")
                _model_instance = TransformersLLMUtil(
                    model_name_or_path=MODEL_PATH,
                    dtype="bfloat16",                # 可根据 GPU 调整
                    attn_implementation="eager",      # 若无 flash-attn 设为 eager
                    trust_remote_code=True,
                    add_system_prompt=True,
                )
                print("✅ Model loaded.")
    return _model_instance


def call_hulu_task(task_item):
    """调用本地 Hulu-Med 模型进行推理"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            model = get_model()

            # 调用 chat 方法
            response = model.chat(
                messages=task_item['messages'],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                enable_thinking=ENABLE_THINKING,
            )

            # 获取 token 使用情况（累计）
            usage = model.get_token_usage()

            result = {
                "status": "success",
                "task_type": task_item['task_type'],
                "pid": task_item['pid'],
                "id": task_item['id'],
                "prediction": response,
                "ground_truth": task_item['ground_truth'],
                "usage": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"]
                }
            }
            return result
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"⚠️ Error on attempt {attempt+1}/{max_retries}: {e}. Retrying in {wait_time:.2f}s")
            time.sleep(wait_time)

    return {
        "status": "error",
        "task_type": task_item['task_type'],
        "pid": task_item['pid'],
        "id": task_item['id'],
        "ground_truth": task_item['ground_truth'],
        "error": "Max retries exceeded."
    }


def build_task_list():
    """根据配置构建所有任务"""
    all_tasks = []
    expected_counts = defaultdict(int)  # (task, pid) -> 预期数量

    # 处理 specific_items
    if SPECIFIC_ITEMS:
        patient_whitelist = {}
        for item in SPECIFIC_ITEMS:
            pid = item.get("patient")
            if not pid:
                print(f"⚠️ 跳过缺少 patient 的条目: {item}")
                continue
            ids = item.get("ids")
            patient_whitelist[pid] = set(ids) if isinstance(ids, list) else None
        target_patients = set(patient_whitelist.keys())
    else:
        patient_whitelist = None
        target_patients = None

    for task in TASKS:
        input_dir = os.path.join(ROOT_DIR, "context_data", task)
        if not os.path.exists(input_dir):
            print(f"⚠️ 任务目录不存在: {input_dir}")
            continue

        files = sorted(glob.glob(os.path.join(input_dir, "P*.jsonl")))

        # 筛选患者文件
        if SPECIFIC_ITEMS:
            files = [f for f in files if os.path.basename(f).replace(".jsonl", "") in target_patients]
        elif SPECIFIC_PATIENTS:
            target_files = []
            for pid in SPECIFIC_PATIENTS:
                fpath = os.path.join(input_dir, f"{pid}.jsonl")
                if fpath in files:
                    target_files.append(fpath)
                else:
                    print(f"⚠️ 患者文件不存在: {fpath}")
            files = target_files
        elif DEMO_N is not None:
            files = files[:DEMO_N]

        for fpath in files:
            pid = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]

            if SPECIFIC_ITEMS:
                allowed_ids = patient_whitelist.get(pid)
                if allowed_ids is not None:
                    items = [it for it in items if it["id"] in allowed_ids]

            num_items = len(items)
            key = (task, pid)
            expected_counts[key] = num_items
            for item in items:
                all_tasks.append({
                    "task_type": task,
                    "pid": pid,
                    "id": item['id'],
                    "messages": item['messages'],
                    "ground_truth": item['ground_truth'],
                })

    return all_tasks, expected_counts


def main():
    # 构建任务列表
    all_tasks, expected_counts = build_task_list()
    print(f"📦 总任务数: {len(all_tasks)}")
    if not all_tasks:
        print("没有任务需要运行。")
        return

    # 线程锁（保证模型串行推理）
    inference_lock = threading.Lock()
    def wrapped_call(task_item):
        with inference_lock:
            return call_hulu_task(task_item)

    results = {}
    token_usage = defaultdict(lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0}))
    completed_counts = defaultdict(int)
    lock = threading.Lock()
    run_dir = os.path.join(ROOT_DIR, "run_llm")   # 与原始脚本一致
    failed_list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(wrapped_call, t): t for t in all_tasks}

        for future in tqdm(as_completed(future_map), total=len(all_tasks), desc="处理中"):
            res = future.result()
            key = (res['task_type'], res['pid'])
            data_to_write = None

            with lock:
                if res['status'] == "success":
                    entry = {
                        "id": res['id'],
                        "prediction": res['prediction'],
                        "ground_truth": res['ground_truth']
                    }
                    results.setdefault(key, []).append(entry)
                    completed_counts[key] += 1

                    if res.get('usage'):
                        model_label = "hulu-med"
                        task = res['task_type']
                        token_usage[model_label][task]["prompt"] += res['usage']['prompt_tokens']
                        token_usage[model_label][task]["completion"] += res['usage']['completion_tokens']
                        token_usage[model_label][task]["total"] += res['usage']['total_tokens']
                else:
                    failed_list.append({
                        "pid": res['pid'],
                        "id": res['id'],
                        "error": res['error']
                    })
                    # 写入错误占位
                    entry = {
                        "id": res['id'],
                        "prediction": f"ERROR: {res['error']}",
                        "ground_truth": res['ground_truth']
                    }
                    results.setdefault(key, []).append(entry)
                    completed_counts[key] += 1

                if completed_counts[key] == expected_counts[key]:
                    data_to_write = results.pop(key)
                    del completed_counts[key]

            if data_to_write:
                task, pid = key
                out_dir = os.path.join(run_dir, task, "hulu-med")
                os.makedirs(out_dir, exist_ok=True)
                data_to_write.sort(key=lambda x: x['id'])
                out_path = os.path.join(out_dir, f"{pid}.jsonl")
                with open(out_path, 'w', encoding='utf-8') as f:
                    for entry in data_to_write:
                        f.write(json.dumps({
                            "id": entry['id'],
                            "prediction": entry['prediction'],
                            "ground_truth": entry['ground_truth']
                        }, ensure_ascii=False) + "\n")

    # 处理剩余结果
    if results:
        print("\n📦 写入剩余结果...")
        for (task, pid), data in results.items():
            out_dir = os.path.join(run_dir, task, "hulu-med")
            os.makedirs(out_dir, exist_ok=True)
            data.sort(key=lambda x: x['id'])
            out_path = os.path.join(out_dir, f"{pid}.jsonl")
            with open(out_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps({
                        "id": entry['id'],
                        "prediction": entry['prediction'],
                        "ground_truth": entry['ground_truth']
                    }, ensure_ascii=False) + "\n")

    # 打印 token 统计
    print("\n🔢 Token 使用统计（累计）:")
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            print(f"  {model} | {task}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}")

    # 保存 token 统计
    stats_dir = os.path.join(ROOT_DIR, "agents", "llm", "usage")
    os.makedirs(stats_dir, exist_ok=True)
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            safe_model = model.replace("/", "_")
            safe_task = task.replace("/", "_")
            filename = f"{safe_model}_{safe_task}_usage_summary.json"
            filepath = os.path.join(stats_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model,
                    "task": task,
                    "prompt_tokens": tokens["prompt"],
                    "completion_tokens": tokens["completion"],
                    "total_tokens": tokens["total"]
                }, f, indent=2)
            print(f"📊 Token 统计已保存: {filepath}")

    # 保存失败日志
    failed_dir = os.path.join(ROOT_DIR, "agents", "llm", "failed")
    os.makedirs(failed_dir, exist_ok=True)
    if failed_list:
        filename = f"hulu-med_failed_summary.json"
        filepath = os.path.join(failed_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(failed_list, f, indent=2, ensure_ascii=False)
        print(f"❌ 失败任务已保存: {filepath} (共 {len(failed_list)} 个)")
    else:
        print("✅ 没有失败任务。")

    print(f"\n✅ 完成。结果保存在 '{run_dir}/hulu-med'，token 统计在 '{stats_dir}'，失败日志在 '{failed_dir}'。")


if __name__ == "__main__":
    main()