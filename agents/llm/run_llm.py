import os
import json
import glob
import yaml
import dotenv
import time
import random
import threading
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

def load_config(config_path):
    default = {
        "models": ["QWEN_TURBO"],
        "tasks": ["trajectory_sorting", "visit_cloze"],
        "demo_n": None,
        "max_workers": 10,
        "specific_patients": None
    }
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k in default:
            cfg.setdefault(k, default[k])
        return cfg
    print(f"⚠️ Config file not found: {config_path}. Using default settings.")
    return default

def call_llm_task(task_item, model_config, max_retries=5):
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['base_url']
            )
            # 环境变量名直接使用 env_prefix（可能包含点号）
            env_model_id_key = f"{model_config['env_prefix']}_MODEL_ID"
            model_id = os.getenv(env_model_id_key, model_config['default_model_id'])

            completion = client.chat.completions.create(
                model=model_id,
                messages=task_item['messages'],
                temperature=0.1,
                **model_config.get('params', {})
            )
            response = completion.choices[0].message.content
            usage = completion.usage

            result = {
                "status": "success",
                "llm_label": model_config['label'],
                "task_type": task_item['task_type'],
                "pid": task_item['pid'],
                "id": task_item['id'],
                "prediction": response,
                "ground_truth": task_item['ground_truth']
            }
            if usage:
                result["usage"] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            else:
                result["usage"] = None
            return result
        except Exception as e:
            status_code = None
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            elif hasattr(e, 'http_status'):
                status_code = e.http_status

            if status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Rate limit hit for {model_config['label']}, retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                return {
                    "status": "error",
                    "llm_label": model_config['label'],
                    "task_type": task_item['task_type'],
                    "pid": task_item['pid'],
                    "id": task_item['id'],
                    "ground_truth": task_item['ground_truth'],
                    "error": str(e)
                }
    return {
        "status": "error",
        "llm_label": model_config['label'],
        "task_type": task_item['task_type'],
        "pid": task_item['pid'],
        "id": task_item['id'],
        "ground_truth": task_item['ground_truth'],
        "error": "Max retries exceeded due to rate limit."
    }

def main():
    config = load_config(CONFIG_PATH)
    raw_models = config["models"]
    task_types = config["tasks"]
    demo_n = config["demo_n"]
    max_workers = config["max_workers"]
    specific_patients = config.get("specific_patients")

    # 将原始模型名称转换为带参数的配置
    model_list = []
    for m in raw_models:
        if m.endswith("_THINKING"):
            base_name = m[:-9]  # 去掉 "_THINKING"
            model_list.append({
                "name": m,
                "env_name": base_name,
                "params": {"extra_body": {"enable_thinking": True}}
            })
        else:
            model_list.append({
                "name": m,
                "env_name": m,
                "params": {}
            })

    # 构建有效模型配置
    model_configs = {}
    valid_models = []
    for m in model_list:
        name = m["name"]
        env_name = m["env_name"]
        params = m["params"]

        # 直接使用 env_name 构造环境变量名，不替换点号
        api_key = os.getenv(f"{env_name}_API_KEY")
        base_url = os.getenv(f"{env_name}_BASE_URL")
        if not api_key:
            print(f"⚠️ Skipping {name} (env name: {env_name}): API key missing.")
            continue

        # 输出目录使用的标签：将下划线替换为连字符，保留点号
        label = name.lower().replace("_", "-")
        # 默认模型 ID：将 env_name 中的下划线替换为连字符，保留点号
        default_model_id = env_name.lower().replace("_", "-")

        model_configs[name] = {
            "name": name,
            "label": label,
            "default_model_id": default_model_id,
            "api_key": api_key,
            "base_url": base_url,
            "params": params,
            "env_prefix": env_name  # 用于读取 MODEL_ID 环境变量
        }
        valid_models.append(name)

    if not valid_models:
        print("❌ No valid models. Exiting.")
        return

    # 构建任务列表
    all_tasks = []
    expected_counts = defaultdict(int)  # key: (task, model_label, pid) -> 条目数

    for task in task_types:
        input_dir = os.path.join(ROOT_DIR, "context_data", task)
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found: {input_dir}")
            continue

        files = sorted(glob.glob(os.path.join(input_dir, "P*.jsonl")))
        if specific_patients:
            target_files = []
            for pid in specific_patients:
                fpath = os.path.join(input_dir, f"{pid}.jsonl")
                if fpath in files:
                    target_files.append(fpath)
                else:
                    print(f"⚠️ Patient file not found: {fpath}")
            files = target_files
        elif demo_n is not None:
            files = files[:demo_n]

        for fpath in files:
            pid = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]
            num_items = len(items)
            for model in valid_models:
                key = (task, model_configs[model]['label'], pid)
                expected_counts[key] = num_items
                for item in items:
                    all_tasks.append({
                        "task_type": task,
                        "pid": pid,
                        "id": item['id'],
                        "messages": item['messages'],
                        "ground_truth": item['ground_truth'],
                        "llm_name": model
                    })

    print(f"🚀 Benchmarking {len(valid_models)} models on {len(task_types)} tasks.")
    print(f"📦 Total requests: {len(all_tasks)}")

    # 共享数据结构和锁（用于实时保存）
    results = {}
    token_usage = {}
    completed_counts = defaultdict(int)
    lock = threading.Lock()
    run_dir = os.path.join(ROOT_DIR, "run_llm")

    # 按模型和任务分组记录失败任务
    failed_by_model_task = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(call_llm_task, t, model_configs[t['llm_name']]): t
            for t in all_tasks
        }

        for future in tqdm(as_completed(future_map), total=len(all_tasks), desc="Processing Tasks"):
            res = future.result()
            key = (res['task_type'], res['llm_label'], res['pid'])
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
                        model = res['llm_label']
                        task = res['task_type']
                        token_usage.setdefault(model, {}).setdefault(task, {"prompt":0, "completion":0, "total":0})
                        token_usage[model][task]["prompt"] += res['usage']['prompt_tokens']
                        token_usage[model][task]["completion"] += res['usage']['completion_tokens']
                        token_usage[model][task]["total"] += res['usage']['total_tokens']
                else:
                    entry = {
                        "id": res['id'],
                        "prediction": f"ERROR: {res['error']}",
                        "ground_truth": res['ground_truth']
                    }
                    results.setdefault(key, []).append(entry)
                    completed_counts[key] += 1

                    model = res['llm_label']
                    task = res['task_type']
                    failed_by_model_task[(model, task)].append({
                        "pid": res['pid'],
                        "id": res['id'],
                        "error": res['error']
                    })

                if completed_counts[key] == expected_counts[key]:
                    data_to_write = results.pop(key)
                    del completed_counts[key]

            if data_to_write:
                task, model, pid = key
                out_dir = os.path.join(run_dir, task, model)
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

    # 处理未完成的结果
    if results:
        print("\n📦 Writing remaining results (incomplete patients)...")
        for (task, model, pid), data in results.items():
            out_dir = os.path.join(run_dir, task, model)
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

    # 打印并保存 token 统计
    print("\n🔢 Token Usage Summary:")
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            print(f"  {model} | {task}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}")

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
            print(f"📊 Token usage saved: {filepath}")

    # 保存失败任务日志
    if failed_by_model_task:
        failed_dir = os.path.join(ROOT_DIR, "agents", "llm", "failed")
        os.makedirs(failed_dir, exist_ok=True)
        for (model, task), failures in failed_by_model_task.items():
            safe_model = model.replace("/", "_")
            safe_task = task.replace("/", "_")
            filename = f"{safe_model}_{safe_task}_failed_summary.json"
            filepath = os.path.join(failed_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
            print(f"❌ Failed tasks saved: {filepath} (total {len(failures)} failures)")
    else:
        print("✅ No failed tasks.")

    print(f"\n✅ Done. Results in '{run_dir}', token stats in '{stats_dir}', failed logs in '{failed_dir}'.")

if __name__ == "__main__":
    main()